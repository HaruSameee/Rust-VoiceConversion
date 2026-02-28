import argparse
import json
from pathlib import Path
import sys

import numpy as np


def convert_index_to_bin(index_path: str, out_path: str) -> tuple[int, int]:
    import faiss

    index = faiss.read_index(index_path)
    if index.ntotal <= 0:
        raise RuntimeError("added.index にベクトルがありません")

    try:
        vectors = index.reconstruct_n(0, index.ntotal)
    except Exception as e_reconstruct_n:
        if hasattr(index, "quantizer"):
            try:
                vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
                for i in range(index.ntotal):
                    vectors[i] = index.reconstruct(i)
            except Exception as e:
                raise RuntimeError(f"added.index からベクトル復元に失敗しました: {e}") from e
        else:
            raise RuntimeError(f"added.index からベクトル復元に失敗しました: {e_reconstruct_n}") from e_reconstruct_n

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise RuntimeError("added.index のベクトル形状が不正です")

    with open(out_path, "wb") as f:
        f.write(vectors.tobytes())

    return int(vectors.shape[0]), int(vectors.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path")
    parser.add_argument("out_path")
    parser.add_argument("--meta-out")
    args = parser.parse_args()

    try:
        rows, dims = convert_index_to_bin(args.index_path, args.out_path)
        payload = json.dumps({"rows": rows, "dims": dims})
        if args.meta_out:
            Path(args.meta_out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
    except Exception as e:
        print(f"エラー: added.index の変換に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
