"""
Export multiple fixed-shape RVC generator ONNX models keyed by block_size.

This keeps HuBERT/RMVPE unchanged and generates:
  model/model_b8000.onnx
  model/model_b16000.onnx
  model/model_b24000.onnx
  model/model_b32000.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import torch

from export_all_dynamic import (
    RvcOnnxWrapper,
    _add_scripts_to_path,
    _ensure_lib_infer_pack_alias,
    _ensure_utf8_stdio,
    _force_torch_load_weights_only_false,
    _torch_load,
    _unwrap_state_dict,
)

BLOCK_SIZES = [8000, 16000, 24000, 32000]


def _generator_geometry(block_size: int, sample_rate: int = 48_000) -> tuple[int, int, int, int]:
    process_window = block_size * 2
    input_16k = (process_window * 16_000) // sample_rate
    phone_frames = max(1, input_16k // 160)
    output_samples = phone_frames * 480
    return process_window, input_16k, phone_frames, output_samples


def _load_generator(generator_pth: Path, device: str):
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
    return wrapper, hidden_dim


def export_generator_static(
    wrapper: torch.nn.Module,
    hidden_dim: int,
    block_size: int,
    out_path: Path,
    opset: int,
    device: str,
) -> None:
    process_window, input_16k, phone_frames, output_samples = _generator_geometry(block_size)

    phone = torch.zeros(1, phone_frames, hidden_dim, dtype=torch.float32, device=device)
    phone_lengths = torch.tensor([phone_frames], dtype=torch.int64, device=device)
    pitch = torch.zeros(1, phone_frames, dtype=torch.int64, device=device)
    pitchf = torch.zeros(1, phone_frames, dtype=torch.float32, device=device)
    sid = torch.zeros(1, dtype=torch.int64, device=device)
    rnd = torch.randn(1, 192, phone_frames, dtype=torch.float32, device=device)

    torch.onnx.export(
        wrapper,
        (phone, phone_lengths, pitch, pitchf, sid, rnd),
        str(out_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
        output_names=["audio"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(
        f"[generator] exported {out_path.name} "
        f"(block={block_size} process_window={process_window} input_16k={input_16k} "
        f"frames={phone_frames} expected_out={output_samples})"
    )


def main() -> None:
    _ensure_utf8_stdio()
    _force_torch_load_weights_only_false()
    parser = argparse.ArgumentParser(description="Export multi-static RVC generator ONNX models")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--generator", type=Path, default=Path("scripts/rv_e250_s36750.pth"))
    parser.add_argument("--out-dir", type=Path, default=Path("model"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=BLOCK_SIZES)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    _add_scripts_to_path(project_root)
    _ensure_lib_infer_pack_alias()

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generator_path = (project_root / args.generator).resolve()
    print(f"[generator] loading {generator_path}")
    wrapper, hidden_dim = _load_generator(generator_path, args.device)
    for block_size in args.block_sizes:
        out_path = out_dir / f"model_b{block_size}.onnx"
        export_generator_static(
            wrapper=wrapper,
            hidden_dim=hidden_dim,
            block_size=block_size,
            out_path=out_path,
            opset=args.opset,
            device=args.device,
        )

    print("[info] HuBERT/RMVPE are unchanged; use existing hubert/rmvpe ONNX files.")


if __name__ == "__main__":
    main()
