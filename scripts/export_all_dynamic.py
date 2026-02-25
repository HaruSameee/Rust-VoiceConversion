"""
Export HuBERT / RMVPE / RVC Generator to ONNX with dynamic axes.

This script is self-contained for this repository layout:
  - scripts/hubert_base.pt
  - scripts/rmvpe.pt
  - scripts/rv_e250_s36750.pth
  - scripts/infer_pack/*
  - scripts/rmvpe.py
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _torch_load(path: Path):
    # torch>=2.6 defaults to weights_only=True. Explicitly disable for legacy checkpoints.
    return torch.load(str(path), map_location="cpu", weights_only=False)

def _force_torch_load_weights_only_false() -> None:
    # fairseq internally calls torch.load without weights_only=..., so patch it once.
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def _add_scripts_to_path(project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _ensure_lib_infer_pack_alias() -> None:
    # models_onnx.py imports `lib.infer_pack`. Create lightweight alias.
    import infer_pack

    lib_mod = types.ModuleType("lib")
    lib_mod.infer_pack = infer_pack
    sys.modules["lib"] = lib_mod
    sys.modules["lib.infer_pack"] = infer_pack


class HubertWrapper(torch.nn.Module):
    def __init__(self, hubert_model: torch.nn.Module):
        super().__init__()
        self.hubert = hubert_model

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # Output is [B, T, C] where C=768 for HuBERT base.
        return self.hubert(
            source=source,
            padding_mask=None,
            mask=False,
            features_only=True,
        )["x"]


class RmvpeWaveformToF0(torch.nn.Module):
    def __init__(
        self,
        rmvpe_model: torch.nn.Module,
        mel_extractor: torch.nn.Module,
        threshold: float = 0.03,
    ):
        super().__init__()
        self.rmvpe_model = rmvpe_model
        self.mel_extractor = mel_extractor
        self.threshold = float(threshold)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, samples] @ 16kHz
        mel = self.mel_extractor(waveform, center=True)  # [B, 128, frames]
        n_frames = mel.shape[-1]
        padded_frames = 32 * ((n_frames - 1) // 32 + 1)
        if padded_frames > n_frames:
            mel = F.pad(mel, (0, padded_frames - n_frames), mode="reflect")
        salience = self.rmvpe_model(mel).float()[:, :n_frames, :]  # [B, frames, 360]
        conf, idx = torch.max(salience, dim=2)  # [B, frames], [B, frames]
        cents = 1997.3794 + idx.float() * 20.0
        f0 = 10.0 * torch.pow(torch.tensor(2.0, device=waveform.device), cents / 1200.0)
        return torch.where(conf > self.threshold, f0, torch.zeros_like(f0))


class RmvpeMelToF0(torch.nn.Module):
    def __init__(self, rmvpe_model: torch.nn.Module, threshold: float = 0.03):
        super().__init__()
        self.rmvpe_model = rmvpe_model
        self.threshold = float(threshold)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 128, n_frames]
        n_frames = mel.shape[-1]
        padded_frames = 32 * ((n_frames - 1) // 32 + 1)
        if padded_frames > n_frames:
            mel = F.pad(mel, (0, padded_frames - n_frames), mode="reflect")
        salience = self.rmvpe_model(mel).float()[:, :n_frames, :]  # [B, frames, 360]
        conf, idx = torch.max(salience, dim=2)
        cents = 1997.3794 + idx.float() * 20.0
        f0 = 10.0 * torch.pow(torch.tensor(2.0, device=mel.device), cents / 1200.0)
        return torch.where(conf > self.threshold, f0, torch.zeros_like(f0))


class RvcOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        rnd: torch.Tensor,
    ) -> torch.Tensor:
        # model output: [B, 1, n_samples]
        audio = self.model(phone, phone_lengths, pitch, pitchf, sid, rnd)
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio[:, 0, :]
        return audio


def _unwrap_state_dict(ckpt) -> Mapping[str, torch.Tensor]:
    if isinstance(ckpt, Mapping):
        for key in ("weight", "state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, Mapping):
                return value
    if isinstance(ckpt, Mapping):
        return ckpt
    raise RuntimeError("checkpoint does not contain a usable state_dict mapping")


def export_hubert(hubert_pt: Path, out_path: Path, opset: int, device: str) -> None:
    from fairseq import checkpoint_utils

    print(f"[hubert] loading {hubert_pt}")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(hubert_pt)])
    model = models[0].to(device).eval()
    wrapper = HubertWrapper(model).eval()
    dummy = torch.zeros(1, 16000, dtype=torch.float32, device=device)

    hubert_opset = max(opset, 18)
    rmvpe_opset = max(opset, 18)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["source"],
        output_names=["features"],
        dynamic_axes={
            "source": {0: "batch", 1: "samples"},
            "features": {0: "batch", 1: "n_frames"},
        },
        opset_version=hubert_opset,
        do_constant_folding=True,
        dynamo=True,
    )
    print(f"[hubert] exported -> {out_path}")


def export_rmvpe(rmvpe_pt: Path, out_path: Path, opset: int, device: str) -> None:
    print(f"[rmvpe] loading {rmvpe_pt}")
    from rmvpe import E2E, MelSpectrogram

    ckpt = _torch_load(rmvpe_pt)
    state = _unwrap_state_dict(ckpt)

    model = E2E(4, 1, (2, 2))
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    mel_extractor = MelSpectrogram(
        False,
        128,
        16000,
        1024,
        160,
        None,
        30,
        8000,
    ).to(device)
    rmvpe_opset = max(opset, 18)

    # Primary target: waveform input [B, samples] with dynamic samples.
    try:
        waveform_wrapper = RmvpeWaveformToF0(model, mel_extractor).eval()
        dummy_waveform = torch.zeros(1, 16000, dtype=torch.float32, device=device)
        torch.onnx.export(
            waveform_wrapper,
            (dummy_waveform,),
            str(out_path),
            input_names=["waveform"],
            output_names=["f0"],
            dynamic_axes={
                "waveform": {0: "batch", 1: "samples"},
                "f0": {0: "batch", 1: "n_frames"},
            },
            opset_version=rmvpe_opset,
            do_constant_folding=True,
            dynamo=True,
        )
        print(f"[rmvpe] exported waveform-dynamic -> {out_path}")
        return
    except Exception as err:
        print(
            "[rmvpe] warning: waveform-dynamic export failed; "
            f"falling back to mel-dynamic export. cause={err}"
        )

    # Fallback: mel input [B, 128, n_frames] with dynamic n_frames.
    mel_wrapper = RmvpeMelToF0(model).eval()
    dummy_mel = torch.zeros(1, 128, 128, dtype=torch.float32, device=device)
    torch.onnx.export(
        mel_wrapper,
        (dummy_mel,),
        str(out_path),
        input_names=["mel"],
        output_names=["f0"],
        dynamic_axes={
            "mel": {0: "batch", 2: "n_frames"},
            "f0": {0: "batch", 1: "n_frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[rmvpe] exported mel-dynamic fallback -> {out_path}")


def export_generator(generator_pth: Path, out_path: Path, opset: int, device: str) -> None:
    print(f"[generator] loading {generator_pth}")
    from infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

    ckpt = _torch_load(generator_pth)
    if not isinstance(ckpt, Mapping):
        raise RuntimeError("generator checkpoint must be a mapping")

    state = _unwrap_state_dict(ckpt)
    config = ckpt.get("config")
    if not isinstance(config, Sequence):
        raise RuntimeError("generator checkpoint missing list-like config")
    version = str(ckpt.get("version", "v2"))

    model = SynthesizerTrnMsNSFsidM(*config, version=version, is_half=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[generator] warning: missing keys={len(missing)} (showing first 5): {missing[:5]}")
    if unexpected:
        print(
            f"[generator] warning: unexpected keys={len(unexpected)} "
            f"(showing first 5): {unexpected[:5]}"
        )
    model = model.to(device).eval()
    model.remove_weight_norm()
    wrapper = RvcOnnxWrapper(model).eval()

    hidden_dim = 768 if version == "v2" else 256
    t = 100
    phone = torch.zeros(1, t, hidden_dim, dtype=torch.float32, device=device)
    phone_lengths = torch.tensor([t], dtype=torch.int64, device=device)
    pitch = torch.zeros(1, t, dtype=torch.int64, device=device)
    pitchf = torch.zeros(1, t, dtype=torch.float32, device=device)
    sid = torch.zeros(1, dtype=torch.int64, device=device)
    rnd = torch.randn(1, 192, t, dtype=torch.float32, device=device)

    torch.onnx.export(
        wrapper,
        (phone, phone_lengths, pitch, pitchf, sid, rnd),
        str(out_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
        output_names=["audio"],
        dynamic_axes={
            "phone": {0: "batch", 1: "n_frames"},
            "phone_lengths": {0: "batch"},
            "pitch": {0: "batch", 1: "n_frames"},
            "pitchf": {0: "batch", 1: "n_frames"},
            "sid": {0: "batch"},
            "rnd": {0: "batch", 2: "n_frames"},
            "audio": {0: "batch", 1: "n_samples"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[generator] exported -> {out_path}")


def _shape_summary(value_info) -> str:
    shape = value_info.type.tensor_type.shape
    dims = []
    for d in shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.dim_value:
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return f"[{', '.join(dims)}]"


def verify_dynamic_axes(onnx_paths: Iterable[Path]) -> None:
    import onnx

    print("\n[verify] ONNX symbolic shape summary")
    for path in onnx_paths:
        model = onnx.load(str(path))
        print(f"  {path}")
        for inp in model.graph.input:
            print(f"    input  {inp.name:14s} {_shape_summary(inp)}")
        for out in model.graph.output:
            print(f"    output {out.name:14s} {_shape_summary(out)}")


def main() -> None:
    _ensure_utf8_stdio()
    _force_torch_load_weights_only_false()
    parser = argparse.ArgumentParser(description="Export dynamic ONNX models for Rust-VC")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--hubert", type=Path, default=Path("scripts/hubert_base.pt"))
    parser.add_argument("--rmvpe", type=Path, default=Path("scripts/rmvpe.pt"))
    parser.add_argument("--generator", type=Path, default=Path("scripts/rv_e250_s36750.pth"))
    parser.add_argument("--out-dir", type=Path, default=Path("model"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    _add_scripts_to_path(project_root)
    _ensure_lib_infer_pack_alias()

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hubert_out = out_dir / "hubert_dynamic.onnx"
    rmvpe_out = out_dir / "rmvpe_dynamic.onnx"
    generator_out = out_dir / "model_dynamic.onnx"

    export_hubert((project_root / args.hubert).resolve(), hubert_out, args.opset, args.device)
    export_rmvpe((project_root / args.rmvpe).resolve(), rmvpe_out, args.opset, args.device)
    export_generator(
        (project_root / args.generator).resolve(),
        generator_out,
        args.opset,
        args.device,
    )

    verify_dynamic_axes([hubert_out, rmvpe_out, generator_out])
    if args.verify:
        print("[verify] completed")


if __name__ == "__main__":
    main()
