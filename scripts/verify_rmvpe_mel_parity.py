#!/usr/bin/env python3
"""
Compare RMVPE mel spectrograms between:
1) Rust implementation (vc-signal::rmvpe_mel_from_audio) dumped as CSV
2) Python reference implementation (RVC-compatible)

This script helps identify parity gaps in STFT / mel filterbank / log scaling.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

N_FFT = 1024
HOP = 160
WIN = 1024
MELS = 128
FMIN = 30.0
FMAX = 8000.0
CLAMP = 1.0e-5
TARGET_SR = 16_000


def _require_librosa():
    try:
        import librosa  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "librosa is required. Install with: python -m pip install librosa"
        ) from exc
    return librosa


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    if sf is not None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32, copy=False), int(sr)

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


def _periodic_hann(win_length: int) -> np.ndarray:
    if win_length <= 1:
        return np.ones((win_length,), dtype=np.float32)
    # torch.hann_window(periodic=True)
    n = np.arange(win_length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)).astype(np.float32)


def _stft_mag_center_reflect(audio: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    if padded.shape[0] < N_FFT:
        return np.zeros((N_FFT // 2 + 1, 0), dtype=np.float32)

    frames = 1 + (padded.shape[0] - N_FFT) // HOP
    window = _periodic_hann(WIN)
    out = np.zeros((N_FFT // 2 + 1, frames), dtype=np.float32)

    for t in range(frames):
        start = t * HOP
        frame = padded[start : start + N_FFT] * window
        spec = np.fft.rfft(frame, n=N_FFT)
        out[:, t] = np.abs(spec).astype(np.float32)
    return out


def _python_reference_mel(audio_16k: np.ndarray) -> np.ndarray:
    librosa = _require_librosa()
    spec_mag = _stft_mag_center_reflect(audio_16k)
    mel_basis = librosa.filters.mel(
        sr=TARGET_SR,
        n_fft=N_FFT,
        n_mels=MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=True,
    ).astype(np.float32)
    mel_mag = mel_basis @ spec_mag
    mel_log = np.log(np.clip(mel_mag, CLAMP, None)).astype(np.float32)
    return mel_log


def _read_rust_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(line for line in f if not line.startswith("#"))
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        raise RuntimeError(f"no mel rows found in {path}")
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"unexpected rust mel shape from csv: {arr.shape}")
    return arr


def _stats(name: str, x: np.ndarray) -> str:
    return (
        f"{name}: shape={tuple(x.shape)} min={x.min():.6f} max={x.max():.6f} "
        f"mean={x.mean():.6f} std={x.std():.6f}"
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def _compare(py_mel: np.ndarray, rust_mel: np.ndarray) -> Tuple[np.ndarray, int]:
    frames = min(py_mel.shape[1], rust_mel.shape[1])
    if frames == 0:
        raise RuntimeError("mel has zero frames")
    py = py_mel[:, :frames]
    ru = rust_mel[:, :frames]
    diff = ru - py
    return diff, frames


def _run_rust_dump(input_wav: Path, rust_csv: Path, aligned: bool) -> None:
    cmd = [
        "cargo",
        "run",
        "-p",
        "vc-signal",
        "--bin",
        "dump_rmvpe_mel",
        "--",
        "--input",
        str(input_wav),
        "--output",
        str(rust_csv),
    ]
    if aligned:
        cmd.append("--aligned")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="WAV path (e.g. debug_input.wav)")
    p.add_argument(
        "--rust-csv",
        default="rust_mel.csv",
        help="Rust mel CSV path (from dump_rmvpe_mel)",
    )
    p.add_argument(
        "--run-rust-dump",
        action="store_true",
        help="Run Rust dumper before comparison",
    )
    p.add_argument(
        "--aligned",
        action="store_true",
        help="Compare using aligned frames (multiple of 32). Default is valid-only frames.",
    )
    p.add_argument(
        "--save-prefix",
        default="",
        help="Optional prefix to save py/rust/diff npy files",
    )
    args = p.parse_args()

    input_wav = Path(args.input)
    rust_csv = Path(args.rust_csv)

    if args.run_rust_dump:
        _run_rust_dump(input_wav, rust_csv, args.aligned)

    audio, sr = load_audio_mono(input_wav)
    if sr != TARGET_SR:
        librosa = _require_librosa()
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        sr = TARGET_SR

    py_mel = _python_reference_mel(audio)
    rust_mel = _read_rust_csv(rust_csv)

    if rust_mel.shape[0] != MELS:
        raise RuntimeError(
            f"unexpected rust mel bins: got {rust_mel.shape[0]}, expected {MELS}"
        )

    diff, used_frames = _compare(py_mel, rust_mel)
    py_used = py_mel[:, :used_frames]
    rust_used = rust_mel[:, :used_frames]

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    corr = _corr(py_used, rust_used)
    mean_shift = float(rust_used.mean() - py_used.mean())

    print(_stats("python_mel", py_used))
    print(_stats("rust_mel", rust_used))
    print(_stats("diff(rust-python)", diff))
    print(
        f"parity: frames={used_frames} mae={mae:.6f} rmse={rmse:.6f} "
        f"max_abs={max_abs:.6f} corr={corr:.6f} mean_shift={mean_shift:.6f}"
    )

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        np.save(str(prefix) + "_python_mel.npy", py_used)
        np.save(str(prefix) + "_rust_mel.npy", rust_used)
        np.save(str(prefix) + "_diff.npy", diff)
        print(f"saved: {prefix}_python_mel.npy / {prefix}_rust_mel.npy / {prefix}_diff.npy")

    print("hint: large mean_shift often indicates gain/log-floor mismatch; low corr indicates STFT/mel basis mismatch.")


if __name__ == "__main__":
    main()
