#!/usr/bin/env python3
"""
Measure sample-level lag between two WAV files and suggest slice_offset_samples.

Usage example:
  python scripts/verify_alignment.py \
    --input debug_input.wav \
    --output debug_output.wav \
    --sample-rate 48000 \
    --current-slice-offset 6054
"""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

_SCIPY_CORRELATE = None
_SCIPY_CHECKED = False


def _get_scipy_correlate():
    global _SCIPY_CORRELATE, _SCIPY_CHECKED
    if _SCIPY_CHECKED:
        return _SCIPY_CORRELATE
    _SCIPY_CHECKED = True
    try:
        # Some environments print ABI warnings/traces for SciPy import.
        # Suppress noisy stderr and fall back to NumPy when SciPy is unusable.
        with contextlib.redirect_stderr(io.StringIO()):
            signal_mod = importlib.import_module("scipy.signal")
        _SCIPY_CORRELATE = getattr(signal_mod, "correlate", None)
    except Exception:
        _SCIPY_CORRELATE = None
    return _SCIPY_CORRELATE


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


def _xcorr(signal_out: np.ndarray, signal_in: np.ndarray) -> np.ndarray:
    scipy_correlate = _get_scipy_correlate()
    if scipy_correlate is not None:
        return scipy_correlate(signal_out, signal_in, mode="full", method="fft")
    return np.correlate(signal_out, signal_in, mode="full")


def _cosine_after_shift(ref: np.ndarray, out: np.ndarray, lag: int) -> float:
    if lag > 0:
        aligned = out[lag:]
        target = ref[: len(aligned)]
    elif lag < 0:
        aligned = out[: lag]
        target = ref[-lag : -lag + len(aligned)]
    else:
        aligned = out
        target = ref
    if aligned.size == 0 or target.size == 0:
        return float("nan")
    num = float(np.dot(aligned, target))
    den = float(np.linalg.norm(aligned) * np.linalg.norm(target) + 1e-8)
    return num / den


def verify_alignment(
    input_wav: Path,
    output_wav: Path,
    sample_rate: Optional[int],
    max_seconds: float,
    current_slice_offset: Optional[int],
) -> int:
    ref, sr_ref = load_audio_mono(input_wav)
    out, sr_out = load_audio_mono(output_wav)

    if sample_rate is not None:
        if sr_ref != sample_rate or sr_out != sample_rate:
            raise RuntimeError(
                f"sample rate mismatch: input={sr_ref}, output={sr_out}, expected={sample_rate}"
            )
        sr = sample_rate
    else:
        if sr_ref != sr_out:
            raise RuntimeError(f"sample rate mismatch: input={sr_ref}, output={sr_out}")
        sr = sr_ref

    n = min(len(ref), len(out))
    if n < 32:
        raise RuntimeError("audio is too short for robust correlation")

    if max_seconds > 0:
        limit = int(sr * max_seconds)
        n = min(n, limit)

    ref = ref[:n]
    out = out[:n]
    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    out = out / (np.max(np.abs(out)) + 1e-8)

    corr = _xcorr(out, ref)
    peak_idx = int(np.argmax(corr))
    lag_samples = peak_idx - (len(ref) - 1)
    peak_corr = float(corr[peak_idx])
    lag_ms = lag_samples * 1000.0 / float(sr)
    lag_ms_48k = lag_samples / 48.0
    lag_frames_10ms = lag_samples / (float(sr) / 100.0)
    direction = "output_late" if lag_samples > 0 else "output_early" if lag_samples < 0 else "aligned"
    cosine = _cosine_after_shift(ref, out, lag_samples)

    print(f"[verify] input={input_wav}")
    print(f"[verify] output={output_wav}")
    print(f"[verify] sr={sr} n={n} scipy={'yes' if _get_scipy_correlate() is not None else 'no'}")
    print(f"[verify] lag_samples={lag_samples}")
    print(f"[verify] lag_ms={lag_ms:.3f} ms @ {sr}Hz")
    print(f"[verify] lag_ms_48k={lag_ms_48k:.3f} ms (48kHz basis)")
    print(f"[verify] lag_frames_10ms={lag_frames_10ms:.3f}")
    print(f"[verify] direction={direction}")
    print(f"[verify] peak_corr={peak_corr:.6f}")
    print(f"[verify] cosine_after_shift={cosine:.6f}")

    if current_slice_offset is not None:
        suggested = max(0, current_slice_offset + lag_samples)
        print(f"[suggest] current_slice_offset_samples={current_slice_offset}")
        print(f"[suggest] new_slice_offset_samples={suggested}")
        print(
            "[suggest] formula: new_offset = current_offset + lag_samples "
            "(lag>0: shift right, lag<0: shift left)"
        )

    return lag_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-correlation WAV alignment checker.")
    p.add_argument("--input", required=True, help="Reference input WAV path")
    p.add_argument("--output", required=True, help="Converted output WAV path")
    p.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="Expected sample rate (default: 48000)",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=20.0,
        help="Max seconds to use for correlation (default: 20.0)",
    )
    p.add_argument(
        "--current-slice-offset",
        type=int,
        default=6_054,
        help="Current slice_offset_samples to compute recommendation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verify_alignment(
        input_wav=Path(args.input),
        output_wav=Path(args.output),
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        current_slice_offset=args.current_slice_offset,
    )


if __name__ == "__main__":
    main()
