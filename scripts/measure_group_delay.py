#!/usr/bin/env python3
"""
Measure group delay from an ONNX RVC decoder as a black box.

Method:
1) Run baseline inference with zero feature impulse.
2) Run impulse inference (single feature frame spike).
3) Compute response = impulse_out - baseline_out.
4) Delay = argmax(abs(response)) - expected_sample_index.

No theoretical kernel/stride formula is used in the delay computation.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import torch

try:
    import onnx
    from onnx import numpy_helper
except Exception as exc:  # pragma: no cover
    raise RuntimeError("onnx package is required: pip install onnx") from exc


def _load_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _get_input_names(sess: ort.InferenceSession) -> dict[str, str]:
    names = {i.name for i in sess.get_inputs()}
    required = {"phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"}
    missing = required - names
    if missing:
        raise RuntimeError(f"missing required ONNX inputs: {sorted(missing)}")
    return {k: k for k in required}


def _run(
    sess: ort.InferenceSession,
    output_name: str,
    phone: np.ndarray,
    phone_lengths: np.ndarray,
    pitch: np.ndarray,
    pitchf: np.ndarray,
    ds: np.ndarray,
    rnd: np.ndarray,
) -> np.ndarray:
    out = sess.run(
        [output_name],
        {
            "phone": phone,
            "phone_lengths": phone_lengths,
            "pitch": pitch,
            "pitchf": pitchf,
            "ds": ds,
            "rnd": rnd,
        },
    )[0]
    if out.ndim != 3:
        raise RuntimeError(f"expected audio shape [B,1,T], got {tuple(out.shape)}")
    return out.astype(np.float32, copy=False)


def _read_onnx_emb_g(onnx_path: Path) -> Optional[np.ndarray]:
    model = onnx.load(str(onnx_path))
    for init in model.graph.initializer:
        if init.name == "emb_g.weight":
            return numpy_helper.to_array(init).astype(np.float32, copy=False)
    return None


def _read_pth_emb_g(pth_path: Path) -> Optional[np.ndarray]:
    cpt = torch.load(str(pth_path), map_location="cpu")
    if not isinstance(cpt, dict):
        return None
    weight = cpt.get("weight")
    if not isinstance(weight, dict):
        return None
    emb = weight.get("emb_g.weight")
    if emb is None:
        return None
    return emb.detach().cpu().numpy().astype(np.float32, copy=False)


def _verify_checkpoint_match(onnx_path: Path, pth_path: Optional[Path]) -> None:
    if pth_path is None:
        return
    onnx_emb = _read_onnx_emb_g(onnx_path)
    pth_emb = _read_pth_emb_g(pth_path)
    if onnx_emb is None or pth_emb is None:
        print("[warn] skip pth/onnx identity check (emb_g.weight not found)")
        return
    if onnx_emb.shape != pth_emb.shape:
        raise RuntimeError(
            f"emb_g.weight shape mismatch: onnx={onnx_emb.shape} pth={pth_emb.shape}"
        )
    max_abs = float(np.max(np.abs(onnx_emb - pth_emb)))
    print(
        f"[identity] emb_g.weight shape={onnx_emb.shape} max_abs_diff={max_abs:.8e}"
    )
    if max_abs != 0.0:
        print("[warn] ONNX and PTH weights are not bit-identical.")


def measure_group_delay(
    onnx_path: Path,
    frames: int,
    hop_samples: int,
    impulse_amp: float,
    frame_start: int,
    frame_end: int,
    frame_step: int,
    pitch_id: int,
    pitchf_hz: float,
) -> None:
    sess = _load_session(onnx_path)
    _get_input_names(sess)
    out_name = sess.get_outputs()[0].name

    feature_dim = 768
    phone_lengths = np.array([frames], dtype=np.int64)
    pitch = np.full((1, frames), int(pitch_id), dtype=np.int64)
    pitchf = np.full((1, frames), float(pitchf_hz), dtype=np.float32)
    ds = np.array([0], dtype=np.int64)
    rnd = np.zeros((1, 192, frames), dtype=np.float32)

    base_phone = np.zeros((1, frames, feature_dim), dtype=np.float32)
    base_out = _run(sess, out_name, base_phone, phone_lengths, pitch, pitchf, ds, rnd)
    print(f"[measure] audio_shape={tuple(base_out.shape)}")

    used_frames = list(range(frame_start, frame_end + 1, frame_step))
    lags: List[int] = []

    for f in used_frames:
        phone = base_phone.copy()
        phone[0, f, :] = impulse_amp
        out = _run(sess, out_name, phone, phone_lengths, pitch, pitchf, ds, rnd)
        response = np.abs((out - base_out)[0, 0])
        peak = int(np.argmax(response))
        expected = int(f * hop_samples)
        lag = int(peak - expected)
        peak_abs = float(response[peak])
        lags.append(lag)
        print(
            f"[raw] frame={f:4d} expected={expected:7d} peak={peak:7d} "
            f"lag={lag:6d} peak_abs={peak_abs:.6f}"
        )

    median_lag = int(statistics.median(lags))
    mode_lag = max(set(lags), key=lags.count)
    print(f"[summary] lags={lags}")
    print(f"[summary] median_lag={median_lag} samples")
    print(f"[summary] mode_lag={mode_lag} samples")
    print(f"[summary] median_ms={median_lag / 48.0:.3f} ms @48kHz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black-box group-delay measurement for RVC ONNX.")
    p.add_argument("--onnx", required=True, help="Path to decoder ONNX (e.g. model/model.onnx)")
    p.add_argument(
        "--pth",
        default=None,
        help="Optional checkpoint to verify identity via emb_g.weight",
    )
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--hop-samples", type=int, default=480)
    p.add_argument("--impulse-amp", type=float, default=3.0)
    p.add_argument("--frame-start", type=int, default=20)
    p.add_argument("--frame-end", type=int, default=180)
    p.add_argument("--frame-step", type=int, default=10)
    p.add_argument("--pitch-id", type=int, default=128, help="Constant pitch id fed to decoder")
    p.add_argument("--pitchf-hz", type=float, default=110.0, help="Constant pitchf value (Hz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)
    pth_path = Path(args.pth) if args.pth else None
    if not onnx_path.exists():
        raise FileNotFoundError(f"onnx not found: {onnx_path}")
    if pth_path is not None and not pth_path.exists():
        raise FileNotFoundError(f"pth not found: {pth_path}")
    if args.frame_start < 1 or args.frame_end <= args.frame_start:
        raise ValueError("invalid frame range")

    _verify_checkpoint_match(onnx_path, pth_path)
    measure_group_delay(
        onnx_path=onnx_path,
        frames=args.frames,
        hop_samples=args.hop_samples,
        impulse_amp=args.impulse_amp,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=args.frame_step,
        pitch_id=args.pitch_id,
        pitchf_hz=args.pitchf_hz,
    )


if __name__ == "__main__":
    main()
