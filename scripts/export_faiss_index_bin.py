#!/usr/bin/env python3
"""
Export all vectors from a FAISS .index file to raw float32 binary (.bin).

Usage:
  python scripts/export_faiss_index_bin.py
  python scripts/export_faiss_index_bin.py --index model/model.index
  python scripts/export_faiss_index_bin.py --index model/model.index --out model/model_vectors.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss  # type: ignore
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all vectors from FAISS index and save as raw float32 binary."
    )
    parser.add_argument(
        "index_positional",
        nargs="?",
        type=Path,
        help="Input FAISS index path (.index). If omitted, auto-detect from model/*.index.",
    )
    parser.add_argument(
        "--index",
        dest="index_option",
        type=Path,
        help="Input FAISS index path (.index)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output raw binary path (.bin). Default: <index_stem>_vectors.bin",
    )
    parser.add_argument(
        "--expect-dim",
        type=int,
        default=768,
        help="Expected vector dimension for Rust side (default: 768)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model"),
        help="Model directory used for auto detection (default: ./model)",
    )
    return parser.parse_args()


def auto_detect_index(model_dir: Path) -> Path:
    default = model_dir / "model.index"
    if default.exists():
        return default
    if model_dir.is_dir():
        candidates = sorted(model_dir.glob("*.index"))
        if candidates:
            return candidates[0]
    candidates = sorted(Path(".").glob("*.index"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"no .index found (searched: {default}, {model_dir}/*.index, ./*.index)"
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    index = args.index_option or args.index_positional
    if index is None:
        index = auto_detect_index(args.model_dir)
    index = index.resolve()
    if args.out is not None:
        out = args.out.resolve()
    else:
        out = index.with_name(f"{index.stem}_vectors.bin")
    return index, out


def main() -> int:
    args = parse_args()
    try:
        index_path, out_path = resolve_paths(args)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    expect_dim: int = args.expect_dim

    if not index_path.exists() or not index_path.is_file():
        print(f"[error] index file not found: {index_path}", file=sys.stderr)
        return 1

    print(f"[info] resolved index: {index_path}")
    print(f"[info] output bin: {out_path}")
    print(f"[info] loading index: {index_path}")
    index = faiss.read_index(str(index_path))
    total = int(index.ntotal)

    if total <= 0:
        print("[error] index.ntotal is 0 (no vectors).", file=sys.stderr)
        return 1

    print(f"[info] reconstruct_n(0, {total})")
    vectors = index.reconstruct_n(0, total)
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        print(f"[error] unexpected reconstructed shape: {vectors.shape}", file=sys.stderr)
        return 1

    n_vectors, dim = int(vectors.shape[0]), int(vectors.shape[1])
    print(f"[info] vectors: {n_vectors}")
    print(f"[info] dimension: {dim} (expected: {expect_dim})")
    if dim != expect_dim:
        print(
            f"[error] dimension mismatch: got {dim}, expected {expect_dim}",
            file=sys.stderr,
        )
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vectors.tofile(str(out_path))
    print(f"[info] wrote raw float32 binary: {out_path}")
    print(f"[info] bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
