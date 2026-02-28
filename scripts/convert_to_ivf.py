#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

MAGIC = b"RVCIVF01"
DEFAULT_INPUT = Path("model/model_vectors.bin")
DEFAULT_OUTPUT = Path("model/model_vectors_ivf.bin")
DEFAULT_DIMS = 768
DEFAULT_NLIST = 512
DEFAULT_NPROBE = 32


def prompt_str(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value if value else default


def prompt_int(label: str, default: int, minimum: int = 1) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"warning: invalid number, using default={default}")
        return default
    return max(minimum, value)


def resolve_dims(path: Path, explicit_dims: int | None, interactive: bool) -> int:
    byte_size = path.stat().st_size
    if byte_size % 4 != 0:
        raise RuntimeError(
            f"input file size is not divisible by 4: {path} ({byte_size} bytes)"
        )

    value_count = byte_size // 4
    if explicit_dims is not None:
        if explicit_dims <= 0:
            raise RuntimeError(f"dims must be >= 1: {explicit_dims}")
        return explicit_dims

    if value_count % DEFAULT_DIMS == 0:
        print(f"auto-detected dims={DEFAULT_DIMS}")
        return DEFAULT_DIMS

    if interactive:
        return prompt_int("Enter vector dims", DEFAULT_DIMS, minimum=1)

    raise RuntimeError("could not infer dims, please pass --dims")


def load_vectors(path: Path, dims: int) -> np.ndarray:
    flat = np.fromfile(path, dtype=np.float32)
    if flat.size == 0:
        raise RuntimeError(f"no vectors in input: {path}")
    if flat.size % dims != 0:
        raise RuntimeError(f"value_count={flat.size} is not divisible by dims={dims}")
    rows = flat.size // dims
    return np.ascontiguousarray(flat.reshape(rows, dims), dtype=np.float32)


def build_ivf(
    vectors: np.ndarray,
    dims: int,
    nlist: int,
    nprobe: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss not found. Run: pip install faiss-cpu") from exc

    kmeans = faiss.Kmeans(dims, nlist, niter=25, verbose=True, gpu=False)
    kmeans.train(vectors)
    centroids = np.ascontiguousarray(kmeans.centroids.reshape(nlist, dims), dtype=np.float32)

    _, labels = kmeans.index.search(vectors, 1)
    labels = labels.reshape(-1).astype(np.uint32, copy=False)
    order = np.argsort(labels, kind="stable")
    sorted_vectors = np.ascontiguousarray(vectors[order], dtype=np.float32)

    counts = np.bincount(labels[order].astype(np.int64), minlength=nlist).astype(np.uint64)
    offsets = np.zeros(nlist + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(counts, dtype=np.uint64)

    return centroids, offsets, sorted_vectors


def write_ivf_bin(
    out_path: Path,
    centroids: np.ndarray,
    offsets: np.ndarray,
    vectors: np.ndarray,
    default_nprobe: int,
) -> None:
    nlist, dims = centroids.shape
    rows = vectors.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", nlist, rows, dims, default_nprobe))
        f.write(np.ascontiguousarray(centroids, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(offsets, dtype="<u8").tobytes())
        f.write(np.ascontiguousarray(vectors, dtype="<f4").tobytes())


def load_ivf_bin(path: Path) -> dict[str, np.ndarray | int]:
    raw = path.read_bytes()
    if raw[:8] != MAGIC:
        raise RuntimeError(f"invalid IVF magic: {path}")

    nlist, rows, dims, default_nprobe = struct.unpack_from("<IIII", raw, 8)
    cursor = 24

    centroid_count = nlist * dims
    centroid_bytes = centroid_count * 4
    centroids = np.frombuffer(raw, dtype="<f4", count=centroid_count, offset=cursor).reshape(
        nlist, dims
    )
    cursor += centroid_bytes

    offsets_count = nlist + 1
    offsets = np.frombuffer(raw, dtype="<u8", count=offsets_count, offset=cursor)
    cursor += offsets_count * 8

    vector_count = rows * dims
    vectors = np.frombuffer(raw, dtype="<f4", count=vector_count, offset=cursor).reshape(
        rows, dims
    )
    cursor += vector_count * 4
    if cursor != len(raw):
        raise RuntimeError(f"unexpected trailing bytes in IVF file: {path}")

    return {
        "nlist": int(nlist),
        "rows": int(rows),
        "dims": int(dims),
        "default_nprobe": int(default_nprobe),
        "centroids": np.asarray(centroids, dtype=np.float32),
        "offsets": np.asarray(offsets, dtype=np.uint64),
        "vectors": np.asarray(vectors, dtype=np.float32),
    }


def native_ivf_search(
    ivf: dict[str, np.ndarray | int],
    query: np.ndarray,
    top_k: int,
    nprobe: int,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = ivf["centroids"]
    offsets = ivf["offsets"]
    vectors = ivf["vectors"]
    assert isinstance(centroids, np.ndarray)
    assert isinstance(offsets, np.ndarray)
    assert isinstance(vectors, np.ndarray)

    centroid_dists = np.sum((centroids - query[None, :]) ** 2, axis=1)
    probe_ids = np.argsort(centroid_dists)[: max(1, min(nprobe, centroids.shape[0]))]

    best: list[tuple[float, int]] = []
    for cluster in probe_ids:
        start = int(offsets[cluster])
        end = int(offsets[cluster + 1])
        if end <= start:
            continue
        block = vectors[start:end]
        dists = np.sum((block - query[None, :]) ** 2, axis=1)
        for local_idx, dist in enumerate(dists):
            best.append((float(dist), start + local_idx))

    best.sort(key=lambda item: item[0])
    best = best[:top_k]
    distances = np.array([item[0] for item in best], dtype=np.float32)
    labels = np.array([item[1] for item in best], dtype=np.int64)
    return distances, labels


def quality_check(
    ivf_path: Path,
    original_vectors: np.ndarray,
    sorted_vectors: np.ndarray,
    nprobe: int,
) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss not found. Run: pip install faiss-cpu") from exc

    ivf = load_ivf_bin(ivf_path)
    if int(ivf["rows"]) != original_vectors.shape[0]:
        raise RuntimeError(
            f"row mismatch: expected={original_vectors.shape[0]} actual={ivf['rows']}"
        )

    flat = faiss.IndexFlatL2(original_vectors.shape[1])
    flat.add(original_vectors)

    sample_count = min(10, original_vectors.shape[0])
    query_ids = random.sample(range(original_vectors.shape[0]), sample_count)
    top1_hits = 0
    overlap_total = 0.0

    for qid in query_ids:
        query = np.ascontiguousarray(original_vectors[qid : qid + 1], dtype=np.float32)
        flat_d, flat_i = flat.search(query, 8)
        ivf_d, ivf_i = native_ivf_search(ivf, query[0], top_k=8, nprobe=nprobe)

        if flat_i.size and ivf_i.size:
            flat_top_vec = original_vectors[int(flat_i[0][0])]
            ivf_top_vec = sorted_vectors[int(ivf_i[0])]
            if np.allclose(flat_top_vec, ivf_top_vec, atol=1e-4):
                top1_hits += 1

        flat_set = {
            tuple(np.round(original_vectors[int(idx)], 5))
            for idx in flat_i[0]
            if int(idx) >= 0
        }
        ivf_set = {
            tuple(np.round(sorted_vectors[int(idx)], 5))
            for idx in ivf_i
            if int(idx) >= 0
        }
        if flat_set:
            overlap_total += len(flat_set & ivf_set) / len(flat_set)

    print(
        "quality: top1_match={}/{} recall@8={:.2%}".format(
            top1_hits,
            sample_count,
            overlap_total / max(sample_count, 1),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model_vectors.bin (raw float32) to native IVF binary index"
    )
    parser.add_argument("--input", type=Path, help="input binary path")
    parser.add_argument("--output", type=Path, help="output IVF binary path")
    parser.add_argument("--dims", type=int, help="vector dims, e.g. 768")
    parser.add_argument("--nlist", type=int, default=DEFAULT_NLIST, help="IVF cluster count")
    parser.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE, help="recommended probe count")
    parser.add_argument(
        "--non-interactive",
        "--no-interactive",
        dest="non_interactive",
        action="store_true",
        help="disable prompts and use args/defaults",
    )
    args = parser.parse_args()

    interactive = not args.non_interactive and sys.stdin.isatty()
    input_path = args.input or DEFAULT_INPUT
    output_path = args.output or DEFAULT_OUTPUT
    nlist = max(1, args.nlist)
    nprobe = max(1, args.nprobe)

    if interactive:
        input_path = Path(prompt_str("Input file", str(input_path)))
        output_path = Path(prompt_str("Output file", str(output_path)))
        nlist = prompt_int("nlist", nlist, minimum=1)
        nprobe = prompt_int("default nprobe", nprobe, minimum=1)

    if not input_path.exists():
        raise RuntimeError(f"input file not found: {input_path}")

    dims = resolve_dims(input_path, args.dims, interactive)
    t0 = time.perf_counter()
    vectors = load_vectors(input_path, dims)
    t_load = time.perf_counter()
    rows = vectors.shape[0]
    print(
        f"loaded: rows={rows} dims={dims} input={input_path} nlist={nlist} nprobe={nprobe}"
    )

    centroids, offsets, sorted_vectors = build_ivf(vectors, dims, nlist, nprobe)
    t_build = time.perf_counter()
    write_ivf_bin(output_path, centroids, offsets, sorted_vectors, nprobe)
    t_write = time.perf_counter()
    quality_check(output_path, vectors, sorted_vectors, nprobe)
    t_check = time.perf_counter()

    print(f"done: {output_path}")
    print(
        "time: load={:.2f}s build={:.2f}s write={:.2f}s check={:.2f}s total={:.2f}s".format(
            t_load - t0,
            t_build - t_load,
            t_write - t_build,
            t_check - t_write,
            t_check - t0,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(1)
    except Exception as exc:
        print(f"\nerror: {exc}")
        traceback.print_exc()
        sys.exit(1)
