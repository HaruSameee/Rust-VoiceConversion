#!/usr/bin/env python3
"""
Measure sample-level lag between two WAV files via cross-correlation.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

def _to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x
    if x.dtype == np.float64:
        return x.astype(np.float32, copy=False)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = float(max(abs(info.min), info.max))
        return (x.astype(np.float32) / scale).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    if sf is not None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return _to_float32(data), int(sr)

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32, copy=False), int(sr)


def _xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.correlate(a, b, mode="full")


def verify_alignment(input_wav: Path, output_wav: Path, sample_rate: int | None) -> int:
    x, sr_x = load_audio_mono(input_wav)
    y, sr_y = load_audio_mono(output_wav)

    if sample_rate is not None:
        if sr_x != sample_rate or sr_y != sample_rate:
            raise RuntimeError(
                f"sample rate mismatch: input={sr_x}, output={sr_y}, expected={sample_rate}"
            )
    elif sr_x != sr_y:
        raise RuntimeError(f"sample rate mismatch: input={sr_x}, output={sr_y}")

    sr = sample_rate if sample_rate is not None else sr_x
    n = min(len(x), len(y))
    if n <= 8:
        raise RuntimeError("audio too short for alignment check")

    x = x[:n]
    y = y[:n]
    x = x / (np.max(np.abs(x)) + 1e-8)
    y = y / (np.max(np.abs(y)) + 1e-8)

    corr = _xcorr(y, x)
    lag = int(np.argmax(corr) - (n - 1))

    print(f"[verify] lag={lag} samples")
    print(f"[verify] lag_ms={lag / (sr / 1000.0):.3f} ms @ {sr}Hz")
    print(f"[verify] lag_frames_10ms={lag / (sr / 100.0):.3f}")
    print(f"[verify] direction={'output late' if lag > 0 else 'output early' if lag < 0 else 'aligned'}")

    if lag > 0:
        aligned = y[lag:]
        ref = x[: len(aligned)]
    elif lag < 0:
        aligned = y[: lag]
        ref = x[-lag : -lag + len(aligned)]
    else:
        aligned = y
        ref = x

    if len(aligned) > 0 and len(ref) > 0:
        num = float(np.dot(aligned, ref))
        den = float(np.linalg.norm(aligned) * np.linalg.norm(ref) + 1e-8)
        print(f"[verify] cosine_after_shift={num / den:.6f}")
    else:
        print("[verify] cosine_after_shift=nan (empty aligned segment)")

    return lag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample-level WAV alignment checker.")
    p.add_argument("--input", required=True, help="Reference input WAV")
    p.add_argument("--output", required=True, help="Converted/output WAV")
    p.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Expected sample rate (optional)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verify_alignment(Path(args.input), Path(args.output), args.sample_rate)


if __name__ == "__main__":
    main()
