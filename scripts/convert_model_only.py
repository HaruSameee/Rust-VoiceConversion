from __future__ import annotations

import argparse
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Tuple

import numpy as np


if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _patch_torch_load() -> None:
    import torch

    if getattr(torch.load, "_rustvc_safe_patch", False):
        return

    original_load = torch.load

    def _safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    _safe_load._rustvc_safe_patch = True  # type: ignore[attr-defined]
    torch.load = _safe_load


def _configure_script_module_paths() -> None:
    candidates = [
        BASE_DIR,
        BASE_DIR / "scripts",
        Path.cwd(),
        Path.cwd() / "scripts",
    ]

    for p in candidates:
        if p.exists():
            p_str = str(p.resolve())
            if p_str not in sys.path:
                sys.path.insert(0, p_str)

    infer_pack_ok = any((p / "infer_pack").is_dir() for p in candidates if p.exists())
    if not infer_pack_ok:
        raise RuntimeError(
            "infer_pack フォルダが見つかりません。scripts/infer_pack を配置してください。"
        )

    import infer_pack

    lib_mod = types.ModuleType("lib")
    lib_mod.infer_pack = infer_pack
    sys.modules["lib"] = lib_mod
    sys.modules["lib.infer_pack"] = infer_pack


def export_generator(pth_path: str, out_path: str) -> None:
    import torch

    _patch_torch_load()
    _configure_script_module_paths()

    from infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

    print(f"\n[generator] loading {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    sd = ckpt.get("weight", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    if sd is None:
        raise RuntimeError("checkpoint から state_dict を取得できませんでした。")

    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("checkpoint に config が見つかりません。")

    version = str(ckpt.get("version", "v2"))
    model = SynthesizerTrnMsNSFsidM(*cfg, version=version, is_half=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[generator] 警告: 不足キーあり（先頭5件）: {missing[:5]}")
    if unexpected:
        print(f"[generator] 警告: 余剰キーあり（先頭5件）: {unexpected[:5]}")
    model.eval().remove_weight_norm()

    n_frames = 100
    if "enc_p.emb_phone.weight" in sd:
        hidden_dim = int(sd["enc_p.emb_phone.weight"].shape[1])
    else:
        hidden_dim = 768 if version == "v2" else 256

    phone = torch.zeros(1, n_frames, hidden_dim)
    phone_l = torch.LongTensor([n_frames])
    pitch = torch.zeros(1, n_frames, dtype=torch.long)
    pitchf = torch.zeros(1, n_frames)
    sid = torch.zeros(1, dtype=torch.long)
    rnd = torch.zeros(1, 192, n_frames)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            args=(phone, phone_l, pitch, pitchf, sid, rnd),
            f=out_path,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"],
            output_names=["audio"],
            dynamic_axes={
                "phone": {0: "batch", 1: "n_frames"},
                "phone_lengths": {0: "batch"},
                "pitch": {0: "batch", 1: "n_frames"},
                "pitchf": {0: "batch", 1: "n_frames"},
                "sid": {0: "batch"},
                "rnd": {0: "batch", 2: "n_frames"},
                "audio": {0: "batch", 2: "n_samples"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"[generator] -> {out_path}")


def convert_index_to_bin(index_path: str, out_path: str) -> Tuple[int, int]:
    try:
        import faiss
    except ImportError:
        raise RuntimeError("faiss が見つかりません。`pip install faiss-cpu` を実行してください。")

    print("[index] faissインデックスを読み込み中...")
    index = faiss.read_index(index_path)
    if index.ntotal <= 0:
        raise RuntimeError("added.index にベクトルがありません。")

    try:
        vectors = index.reconstruct_n(0, index.ntotal)
    except Exception:
        vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
        for i in range(index.ntotal):
            vectors[i] = index.reconstruct(i)

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise RuntimeError("added.index のベクトル形状が不正です。")

    rows = int(vectors.shape[0])
    dims = int(vectors.shape[1])
    if dims != 768:
        print(f"[index] 警告: 次元数が {dims} です（通常は768）。")

    with open(out_path, "wb") as f:
        f.write(vectors.tobytes())
    return rows, dims


def resolve_input_path(cli_value: str | None, prompt: str, required_ext: str | None = None) -> str:
    if cli_value:
        path = cli_value.strip()
    else:
        path = input(prompt).strip()

    if not path:
        return ""
    if not os.path.isfile(path):
        raise RuntimeError(f"ファイルが見つかりません: {path}")
    if required_ext and not path.lower().endswith(required_ext):
        raise RuntimeError(f"拡張子が不正です: {path} (必要: {required_ext})")
    return path


def main() -> None:
    _ensure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="RustVC: モデル変換専用スクリプト (.pth -> onnx, added.index -> bin)"
    )
    parser.add_argument("--pth", help="学習済み .pth のパス")
    parser.add_argument("--index", help="added.index のパス（省略可）")
    parser.add_argument("--out-dir", default="model", help="出力先ディレクトリ（既定: model）")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="index変換をスキップする",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== モデル変換（setup不要）===")
    pth_path = resolve_input_path(
        args.pth,
        "学習済みモデル(.pth)のパスを入力してください: ",
        required_ext=".pth",
    )
    if not pth_path:
        raise RuntimeError(".pth のパスが未入力です。")

    onnx_out = str(out_dir / "model_dynamic.onnx")
    export_generator(pth_path, onnx_out)
    print(f"変換完了: {onnx_out}")

    if args.skip_index:
        print("index変換はスキップしました。")
        print("完了しました。")
        return

    index_path = resolve_input_path(
        args.index,
        "added.index のパスを入力してください（不要ならEnterでスキップ）: ",
        required_ext=".index",
    )
    if not index_path:
        print("added.index はスキップしました。")
        print("完了しました。")
        return

    bin_out = str(out_dir / "model_vectors.bin")
    rows, dims = convert_index_to_bin(index_path, bin_out)
    print(f"変換完了: {bin_out} ({rows}行, {dims}次元)")
    print("完了しました。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラー: {e}")
        traceback.print_exc()
        sys.exit(1)
