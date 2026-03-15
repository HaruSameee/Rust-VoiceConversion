import argparse
import sys
import types
from pathlib import Path


def patch_torch_load():
    import torch

    if getattr(torch.load, "_rustvc_safe_patch", False):
        return

    original_load = torch.load

    def _safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    _safe_load._rustvc_safe_patch = True  # type: ignore[attr-defined]
    torch.load = _safe_load


def configure_paths():
    base = Path(__file__).resolve().parent
    candidates = [
        base.parent,
        base,
        Path.cwd(),
        Path.cwd() / "scripts",
    ]
    for p in candidates:
        if p.exists():
            p_str = str(p.resolve())
            if p_str not in sys.path:
                sys.path.insert(0, p_str)

    infer_pack_ok = any((p / "infer_pack").is_dir() for p in candidates if p.exists())
    if not infer_pack_ok:
        raise RuntimeError("infer_pack フォルダが見つかりません")

    import infer_pack

    lib_mod = types.ModuleType("lib")
    lib_mod.infer_pack = infer_pack
    sys.modules["lib"] = lib_mod
    sys.modules["lib.infer_pack"] = infer_pack


def load_generator_checkpoint(pth_path: str):
    import torch

    patch_torch_load()
    configure_paths()
    from infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

    print(f"[generator] loading {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    sd = ckpt.get("weight", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    if sd is None:
        raise RuntimeError("checkpoint から state_dict を取得できませんでした")

    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("checkpoint に config が見つかりません")

    version = str(ckpt.get("version", "v2"))
    model = SynthesizerTrnMsNSFsidM(*cfg, version=version, is_half=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[generator] 警告: 不足キーがあります（先頭5件）: {missing[:5]}")
    if unexpected:
        print(f"[generator] 警告: 余剰キーがあります（先頭5件）: {unexpected[:5]}")
    model.eval().remove_weight_norm()
    return model, sd, version


def build_generator_inputs(sd, version, block_size_samples=24000):
    import torch

    n_frames = block_size_samples // 240
    if "enc_p.emb_phone.weight" in sd:
        hidden_dim = int(sd["enc_p.emb_phone.weight"].shape[1])
    else:
        hidden_dim = 768 if version == "v2" else 256

    phone = torch.zeros(1, n_frames, hidden_dim)
    phone_l = torch.LongTensor([n_frames])
    pitch = torch.zeros(1, n_frames, dtype=torch.long)
    pitchf = torch.zeros(1, n_frames)
    sid = torch.zeros(1, dtype=torch.long)
    rnd = torch.zeros(1, 192, n_frames)
    return phone, phone_l, pitch, pitchf, sid, rnd


def export_dynamic_generator(model, out_path: str, export_inputs):
    import torch

    import warnings

    phone, phone_l, pitch, pitchf, sid, rnd = export_inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            args=(phone, phone_l, pitch, pitchf, sid, rnd),
            f=out_path,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
            output_names=["audio"],
            dynamic_axes={
                "phone": {0: "batch", 1: "n_frames"},
                "phone_lengths": {0: "batch"},
                "pitch": {0: "batch", 1: "n_frames"},
                "pitchf": {0: "batch", 1: "n_frames"},
                "sid": {0: "batch"},
                "rnd": {0: "batch", 2: "n_frames"},
                "audio": {0: "batch", 2: "n_samples"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"[generator] dynamic -> {out_path}")


def export_stateful_generator(
    model, out_path: str, export_inputs, block_size_samples=24000
):
    import torch

    from infer_pack.models_onnx import StatefulSynthesizerTrnMsNSFsidM

    phone, phone_l, pitch, pitchf, sid, rnd = export_inputs
    stateful_model = StatefulSynthesizerTrnMsNSFsidM(
        model,
        output_start_samples=block_size_samples,
    )
    state_inputs = stateful_model.initial_state_tensors(
        batch_size=phone.shape[0],
        device=phone.device,
        dtype=phone.dtype,
    )
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"]
    input_names.extend([f"state_{name}_in" for name in stateful_model.state_names])
    output_names = ["audio"]
    output_names.extend([f"state_{name}_out" for name in stateful_model.state_names])
    dynamic_axes = {
        "phone": {0: "batch", 1: "n_frames"},
        "phone_lengths": {0: "batch"},
        "pitch": {0: "batch", 1: "n_frames"},
        "pitchf": {0: "batch", 1: "n_frames"},
        "sid": {0: "batch"},
        "rnd": {0: "batch", 2: "n_frames"},
        "audio": {0: "batch", 2: "n_samples"},
    }
    for name, state in zip(input_names[6:], state_inputs):
        if state.dim() >= 1:
            dynamic_axes[name] = {0: "batch"}
    for name, state in zip(output_names[1:], state_inputs):
        if state.dim() >= 1:
            dynamic_axes[name] = {0: "batch"}

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            stateful_model,
            args=(phone, phone_l, pitch, pitchf, sid, rnd, *state_inputs),
            f=out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"[generator] stateful -> {out_path}")


def export_generator(pth_path: str, out_path: str):
    model, sd, version = load_generator_checkpoint(pth_path)
    export_inputs = build_generator_inputs(sd, version)
    export_dynamic_generator(model, out_path, export_inputs)
    for block_size in (24000, 48000):
        stateful_inputs = build_generator_inputs(
            sd,
            version,
            block_size_samples=block_size,
        )
        stateful_out_path = str(
            Path(out_path).with_name(f"model_stateful_b{block_size}.onnx")
        )
        export_stateful_generator(
            model,
            stateful_out_path,
            stateful_inputs,
            block_size_samples=block_size,
        )

    legacy_stateful_out_path = str(Path(out_path).with_name("model_stateful.onnx"))
    export_stateful_generator(
        model,
        legacy_stateful_out_path,
        build_generator_inputs(sd, version, block_size_samples=24000),
        block_size_samples=24000,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_path")
    parser.add_argument("out_path")
    args = parser.parse_args()

    try:
        export_generator(args.pth_path, args.out_path)
        print(f"[generator] -> {args.out_path}")
    except Exception as e:
        print(f"エラー: generator の変換に失敗しました: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
