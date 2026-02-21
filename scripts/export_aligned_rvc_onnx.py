#!/usr/bin/env python3
"""
Create delay-compensated RVC ONNX by patching an existing decoder ONNX graph.

This script is self-contained in this project:
- No Mangio source import/use.
- No theoretical kernel/stride delay calculation.
- Optional checkpoint identity check via emb_g.weight.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, checker, helper, numpy_helper

try:
    import onnxruntime as ort
except Exception:
    ort = None


INT64_MAX = np.iinfo(np.int64).max


def _find_initializer(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init).astype(np.float32, copy=False)
    return None


def verify_checkpoint_identity(model: onnx.ModelProto, checkpoint_path: Path) -> None:
    cpt = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(cpt, dict):
        raise RuntimeError("checkpoint is not dict-like")
    weight = cpt.get("weight")
    if not isinstance(weight, dict):
        raise RuntimeError("checkpoint has no `weight` dict")
    if "emb_g.weight" not in weight:
        raise RuntimeError("checkpoint has no `weight['emb_g.weight']`")

    onnx_emb = _find_initializer(model, "emb_g.weight")
    if onnx_emb is None:
        raise RuntimeError("ONNX has no initializer `emb_g.weight`")

    pth_emb = weight["emb_g.weight"].detach().cpu().numpy().astype(np.float32, copy=False)
    if onnx_emb.shape != pth_emb.shape:
        raise RuntimeError(
            f"emb_g.weight shape mismatch: onnx={onnx_emb.shape} pth={pth_emb.shape}"
        )
    max_abs = float(np.max(np.abs(onnx_emb - pth_emb)))
    print(f"[identity] emb_g.weight shape={onnx_emb.shape} max_abs_diff={max_abs:.8e}")
    if max_abs != 0.0:
        print("[warn] ONNX and checkpoint weights are not bit-identical.")


def patch_output_head_crop(model: onnx.ModelProto, delay_samples: int) -> onnx.ModelProto:
    if delay_samples < 0:
        raise ValueError("delay_samples must be >= 0")

    graph = model.graph
    if not graph.output:
        raise RuntimeError("ONNX graph has no outputs")

    output = graph.output[0]
    original_output_name = output.name
    aligned_output_name = f"{original_output_name}_aligned"

    starts_name = "aligned_slice_starts"
    ends_name = "aligned_slice_ends"
    axes_name = "aligned_slice_axes"
    steps_name = "aligned_slice_steps"

    # Slice along time axis only: [B, C, T] -> [B, C, T-delay]
    starts = numpy_helper.from_array(np.array([delay_samples], dtype=np.int64), name=starts_name)
    ends = numpy_helper.from_array(np.array([INT64_MAX], dtype=np.int64), name=ends_name)
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name=axes_name)
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name)

    graph.initializer.extend([starts, ends, axes, steps])

    slice_node = helper.make_node(
        "Slice",
        inputs=[original_output_name, starts_name, ends_name, axes_name, steps_name],
        outputs=[aligned_output_name],
        name="DelayCompensateSlice",
    )
    graph.node.append(slice_node)

    # Redirect graph output to aligned tensor.
    output.name = aligned_output_name

    return model


def verify_runtime_shapes(original_onnx: Path, aligned_onnx: Path, delay_samples: int) -> None:
    if ort is None:
        print("[warn] onnxruntime not installed; skip runtime shape verification.")
        return

    s0 = ort.InferenceSession(str(original_onnx), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(aligned_onnx), providers=["CPUExecutionProvider"])

    inputs = {
        "phone": np.zeros((1, 200, 768), dtype=np.float32),
        "phone_lengths": np.array([200], dtype=np.int64),
        "pitch": np.full((1, 200), 128, dtype=np.int64),
        "pitchf": np.full((1, 200), 110.0, dtype=np.float32),
        "ds": np.array([0], dtype=np.int64),
        "rnd": np.zeros((1, 192, 200), dtype=np.float32),
    }

    y0 = s0.run(None, inputs)[0]
    y1 = s1.run(None, inputs)[0]
    t0 = int(y0.shape[-1])
    t1 = int(y1.shape[-1])
    print(f"[verify] original_output_len={t0}")
    print(f"[verify] aligned_output_len={t1}")
    print(f"[verify] len_delta={t0 - t1} (expected={delay_samples})")
    if t0 - t1 != delay_samples:
        raise RuntimeError("runtime length delta mismatch after slice patch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch decoder ONNX with fixed head-crop delay.")
    p.add_argument("--checkpoint", required=True, help="Path to .pth (identity check only)")
    p.add_argument("--output", required=True, help="Path to write aligned ONNX")
    p.add_argument("--project-root", default=".", help="Project root (default: .)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compatibility arg")
    p.add_argument("--delay-samples", type=int, required=True, help="Head-crop samples")
    p.add_argument(
        "--base-onnx",
        default=None,
        help="Base ONNX path. Default: <project-root>/model/model.onnx",
    )
    p.add_argument("--skip-verify", action="store_true", help="Skip ORT runtime length check")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.project_root).resolve()
    base_onnx = Path(args.base_onnx).resolve() if args.base_onnx else (root / "model" / "model.onnx")
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()

    if not base_onnx.exists():
        raise FileNotFoundError(f"base ONNX not found: {base_onnx}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.delay_samples < 0:
        raise ValueError("--delay-samples must be >= 0")

    model = onnx.load(str(base_onnx))
    verify_checkpoint_identity(model, checkpoint)
    model = patch_output_head_crop(model, args.delay_samples)

    checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    print(f"[ok] saved aligned ONNX: {output}")
    print(f"[ok] applied head crop: {args.delay_samples} samples")

    if not args.skip_verify:
        verify_runtime_shapes(base_onnx, output, args.delay_samples)


if __name__ == "__main__":
    main()
