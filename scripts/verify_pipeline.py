"""
Verify dynamic ONNX pipeline for RVC (HuBERT -> RMVPE -> Generator).
Focus: shape consistency, dynamic axes, and numeric sanity checks.
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

# Windows console can choke on non-ASCII logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _tensor_stats(name: str, x: np.ndarray) -> str:
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={np.nanmin(x):.6f} max={np.nanmax(x):.6f} "
        f"mean={np.nanmean(x):.6f}"
    )


def _check_tensor(name: str, x: np.ndarray, eps: float = 1e-8) -> List[str]:
    issues = []
    if np.isnan(x).any():
        issues.append(f"{name}: NaN detected")
    if np.isinf(x).any():
        issues.append(f"{name}: Inf detected")
    if np.all(np.abs(x) < eps):
        issues.append(f"{name}: all-zero (silent) detected")
    return issues


def _adapt_phone_features(features: np.ndarray, target_dim: int) -> Tuple[np.ndarray, str]:
    """
    Adapt HuBERT features (B, T, 256) to target_dim (e.g., 768).
    This is a placeholder adapter for pipeline validation only.
    """
    b, t, c = features.shape
    if c == target_dim:
        return features, "phone features already match target_dim"
    if c < target_dim:
        # Repeat to reach target_dim, then truncate.
        reps = int(np.ceil(target_dim / c))
        tiled = np.tile(features, (1, 1, reps))
        return tiled[:, :, :target_dim], f"phone features tiled from {c} to {target_dim}"
    # c > target_dim
    return features[:, :, :target_dim], f"phone features truncated from {c} to {target_dim}"


def _coarse_f0(f0: np.ndarray) -> np.ndarray:
    # RVC coarse pitch is 0..255 int
    coarse = np.clip(np.round(f0), 0, 255).astype(np.int64)
    return coarse


def _session_inputs(session: ort.InferenceSession) -> Dict[str, ort.NodeArg]:
    return {i.name: i for i in session.get_inputs()}


def _log_inputs(title: str, session: ort.InferenceSession) -> None:
    print(f"\n[{title}] inputs:")
    for i in session.get_inputs():
        print(f"  - {i.name}: shape={i.shape} type={i.type}")


def _log_outputs(title: str, session: ort.InferenceSession) -> None:
    print(f"[{title}] outputs:")
    for o in session.get_outputs():
        print(f"  - {o.name}: shape={o.shape} type={o.type}")


def run_test(
    hubert_sess: ort.InferenceSession,
    rmvpe_sess: ort.InferenceSession,
    gen_sess: ort.InferenceSession,
    n_frames: int,
    sample_rate: int,
    seed: int,
) -> None:
    print("\n" + "=" * 80)
    print(f"[test] n_frames={n_frames} sample_rate={sample_rate}")

    rng = np.random.default_rng(seed)
    # Approximate HuBERT frame rate ~50 fps for 16kHz -> 320 samples per frame.
    n_samples = int(n_frames * 320)
    source = rng.standard_normal((1, n_samples), dtype=np.float32)

    # 1) HuBERT
    hubert_out = hubert_sess.run(None, {"source": source})[0]
    print(_tensor_stats("hubert.features", hubert_out))
    issues = _check_tensor("hubert.features", hubert_out)
    if issues:
        print("[warn] " + " | ".join(issues))

    # 2) RMVPE (mel input is external in real pipeline; use dummy mel for QA)
    # RMVPE requires n_frames multiple of 32
    mel_frames = int(np.ceil(n_frames / 32) * 32)
    mel = rng.standard_normal((1, 128, mel_frames), dtype=np.float32)
    rmvpe_out = rmvpe_sess.run(None, {"mel": mel})[0]
    # RMVPE output may be (B, T, bins). Reduce to (B, T) for generator.
    if rmvpe_out.ndim == 3 and rmvpe_out.shape[-1] > 1:
        f0_raw = np.max(rmvpe_out, axis=-1)
        print(
            _tensor_stats("rmvpe.f0_raw", rmvpe_out)
            + f" | reduced to (B, T) via max over bins"
        )
    else:
        f0_raw = rmvpe_out
        print(_tensor_stats("rmvpe.f0", rmvpe_out))
    issues = _check_tensor("rmvpe.f0", f0_raw)
    if issues:
        print("[warn] " + " | ".join(issues))

    # 3) Generator
    # Phone features from HuBERT -> adapt to generator input dim
    gen_inputs = _session_inputs(gen_sess)
    phone_dim = gen_inputs["phone"].shape[2]
    phone, adapt_msg = _adapt_phone_features(hubert_out, int(phone_dim))
    if phone.shape[1] != hubert_out.shape[1]:
        print(f"[warn] phone time length mismatch: hubert={hubert_out.shape[1]} phone={phone.shape[1]}")
    print(f"[info] {adapt_msg}")

    # Align pitch with phone frames
    phone_frames = phone.shape[1]
    # If rmvpe output length differs, resize by simple slice/pad for QA only.
    if f0_raw.shape[1] < phone_frames:
        pad = phone_frames - f0_raw.shape[1]
        f0 = np.pad(f0_raw, ((0, 0), (0, pad)), mode="constant")
        print(f"[info] f0 padded from {f0_raw.shape[1]} to {phone_frames}")
    else:
        f0 = f0_raw[:, :phone_frames]
        if f0_raw.shape[1] != phone_frames:
            print(f"[info] f0 truncated from {f0_raw.shape[1]} to {phone_frames}")

    pitch = _coarse_f0(f0)
    pitchf = f0.astype(np.float32)
    phone_lengths = np.array([phone_frames], dtype=np.int64)

    feed: Dict[str, np.ndarray] = {
        "phone": phone.astype(np.float32),
        "phone_lengths": phone_lengths,
        "pitch": pitch,
        "pitchf": pitchf,
        "sid": np.zeros((1,), dtype=np.int64),
    }

    # Optional inputs (if present in ONNX)
    if "ds" in gen_inputs:
        feed["ds"] = np.zeros((1,), dtype=np.int64)
    if "rnd" in gen_inputs:
        feed["rnd"] = rng.standard_normal((1, 192, phone_frames), dtype=np.float32)

    # Sanity log of generator inputs
    print("[generator] input shapes:")
    for k, v in feed.items():
        print(f"  - {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    audio = gen_sess.run(None, feed)[0]
    print(_tensor_stats("generator.audio", audio))
    issues = _check_tensor("generator.audio", audio)
    if issues:
        print("[warn] " + " | ".join(issues))

    # Final shape check
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 1:
        print(f"[warn] audio shape unexpected: {audio.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubert", default="model/dynamic/hubert_dynamic.onnx")
    ap.add_argument("--rmvpe", default="model/dynamic/rmvpe_dynamic.onnx")
    ap.add_argument("--generator", default="model/dynamic/model_dynamic.onnx")
    ap.add_argument("--frames", default="64,128,256")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    for p in [args.hubert, args.rmvpe, args.generator]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    providers = ["CPUExecutionProvider"]
    hubert_sess = ort.InferenceSession(args.hubert, providers=providers)
    rmvpe_sess = ort.InferenceSession(args.rmvpe, providers=providers)
    gen_sess = ort.InferenceSession(args.generator, providers=providers)

    print("[info] loaded ONNX sessions")
    _log_inputs("hubert", hubert_sess)
    _log_outputs("hubert", hubert_sess)
    _log_inputs("rmvpe", rmvpe_sess)
    _log_outputs("rmvpe", rmvpe_sess)
    _log_inputs("generator", gen_sess)
    _log_outputs("generator", gen_sess)

    frames_list = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    for i, n_frames in enumerate(frames_list):
        run_test(hubert_sess, rmvpe_sess, gen_sess, n_frames, args.sr, args.seed + i)

    print("\n[done] pipeline verification complete")


if __name__ == "__main__":
    main()
