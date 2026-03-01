# pyinstaller setup.spec
#
# 必要パッケージ:
#   requests, tqdm, numpy, faiss-cpu, onnx, onnxscript

import json
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import traceback
import types
import ctypes
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from tqdm import tqdm

# PyInstaller frozen exe: add _MEIPASS to sys.path
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

STEP_LABEL_TOTAL = 6

ORT_VERSION = "1.23.0"
ORT_PACKAGE_URL = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    "v1.23.0/onnxruntime-win-x64-gpu-1.23.0.zip"
)
ORT_REQUIRED_DLLS = [
    "onnxruntime.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_tensorrt.dll",
]
CUDA_PACKAGE_URLS = {
    "cuda_cudart": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip",
    "libcublas": "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip",
    "libcufft": "https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.2.1.3-archive.zip",
    "libcurand": "https://developer.download.nvidia.com/compute/cuda/redist/libcurand/windows-x86_64/libcurand-windows-x86_64-10.3.5.147-archive.zip",
    "libcusolver": "https://developer.download.nvidia.com/compute/cuda/redist/libcusolver/windows-x86_64/libcusolver-windows-x86_64-11.6.1.9-archive.zip",
    "libcusparse": "https://developer.download.nvidia.com/compute/cuda/redist/libcusparse/windows-x86_64/libcusparse-windows-x86_64-12.3.1.170-archive.zip",
}
CUDA_REQUIRED_DLLS = [
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_11.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_12.dll",
]
CUDNN_PACKAGE_URL = (
    "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/"
    "windows-x86_64/cudnn-windows-x86_64-9.1.1.17_cuda12-archive.zip"
)
CUDNN_REQUIRED_DLLS = [
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
]

MODE_TABLE = {
    "ultra_low": 12000,
    "low": 24000,
    "mid": 48000,
}


def print_step(step_no: int, title: str) -> None:
    print(f"\n=== Step {step_no}/{STEP_LABEL_TOTAL}: {title} ===")


class TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def open_run_log_file() -> Tuple[Path, object]:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"setup_{stamp}.log"
    fp = log_path.open("w", encoding="utf-8", newline="\n")
    return log_path, fp


def _discover_python_command() -> str:
    candidates = []
    env_python = os.environ.get("PYTHON")
    if env_python:
        candidates.append(env_python)
    candidates.extend(["python", "py -3.10", "py"])

    for cmd in candidates:
        try:
            subprocess.run(
                cmd.split() + ["--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            return cmd
        except Exception:
            continue
    return ""


def _resolve_helper_script(script_name: str) -> Path:
    candidates = [
        Path(BASE_DIR) / script_name,
        Path(BASE_DIR) / "scripts" / script_name,
        Path.cwd() / script_name,
        Path.cwd() / "scripts" / script_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(p) for p in candidates)
    raise RuntimeError(
        f"{script_name} が見つかりません。\n探索先:\n{searched}\nBASE_DIR={BASE_DIR}"
    )


def _load_script_module(script_name: str, module_name: str):
    script_path = _resolve_helper_script(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{script_name} のロードに失敗しました: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _OrtApiBase(ctypes.Structure):
    _fields_ = [("_get_api", ctypes.c_void_p), ("get_version_string", ctypes.c_void_p)]


def _attach_site_packages_from_python(python_cmd: str) -> None:
    probe = (
        "import json,site;"
        "paths=[];"
        "g=getattr(site,'getsitepackages',None);"
        "paths.extend(g() if callable(g) else []);"
        "u=site.getusersitepackages();"
        "paths.extend(u if isinstance(u,list) else [u]);"
        "print(json.dumps(paths))"
    )
    proc = subprocess.run(
        python_cmd.split() + ["-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return

    try:
        paths = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return

    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        if not p.exists():
            continue

        p_str = str(p.resolve())
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

        torch_lib = p / "torch" / "lib"
        if torch_lib.exists():
            torch_lib_str = str(torch_lib.resolve())
            os.environ["PATH"] = f"{torch_lib_str};{os.environ.get('PATH', '')}"
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib_str)
                except OSError:
                    pass


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


def ensure_torch_available() -> None:
    print_step(1, "torch 確認")
    try:
        import torch

        if getattr(sys, "frozen", False):
            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.exists():
                torch_lib_str = str(torch_lib)
                os.environ["PATH"] = f"{torch_lib_str};{os.environ.get('PATH', '')}"
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(torch_lib_str)
                    except OSError:
                        pass
        print(f"torch {torch.__version__} 検出済み")
    except ImportError as exc:
        raise RuntimeError(
            "torch の import に失敗しました。setup.exe は torch 同梱版を前提としています。"
        ) from exc

    _patch_torch_load()


def _configure_script_module_paths() -> None:
    base = Path(BASE_DIR)
    candidates = [
        base,
        base / "scripts",
        Path.cwd(),
        Path.cwd() / "scripts",
    ]

    for p in candidates:
        if p.exists():
            p_str = str(p.resolve())
            if p_str not in sys.path:
                sys.path.insert(0, p_str)

    infer_pack_ok = any((p / "infer_pack").is_dir() for p in candidates if p.exists())
    rmvpe_ok = any(
        ((p / "rmvpe").is_dir() or (p / "rmvpe.py").is_file())
        for p in candidates
        if p.exists()
    )
    if not infer_pack_ok:
        raise RuntimeError(
            "infer_pack フォルダが見つかりません。setup.exe と一緒に infer_pack を同梱してください。"
        )
    if not rmvpe_ok:
        raise RuntimeError(
            "rmvpe フォルダ（または rmvpe.py）が見つかりません。setup.exe と一緒に rmvpe を同梱してください。"
        )

    import infer_pack

    lib_mod = types.ModuleType("lib")
    lib_mod.infer_pack = infer_pack
    sys.modules["lib"] = lib_mod
    sys.modules["lib.infer_pack"] = infer_pack


def classify_gpu_mode(gpu_name: str) -> str:
    upper = gpu_name.upper()
    if ("RTX 20" in upper) or ("RTX 30" in upper) or ("RTX 40" in upper):
        return "ultra_low"
    if ("GTX 1080 TI" in upper) or ("GTX 1070" in upper) or ("GTX 1080" in upper):
        return "low"
    return "mid"


def detect_gpu_name() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "nvidia-smi が見つかりません。NVIDIAドライバをインストールしてください。"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi の実行に失敗しました: {e.stderr.strip() or e}") from e

    gpu_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not gpu_lines:
        raise RuntimeError("GPU名を検出できませんでした。")
    return gpu_lines[0]


def select_mode_interactive(suggested_mode: str) -> Tuple[str, int]:
    suggested_block = MODE_TABLE[suggested_mode]
    options = [
        ("1", "ultra_low", "超低遅延 (RTX3060以上推奨, block_size=12000)"),
        ("2", "low", "低遅延   (GTX1080以上推奨, block_size=24000)"),
        ("3", "mid", "中遅延   (GTX1080以下推奨, block_size=48000)"),
    ]
    for key, mode, label in options:
        mark = " ← 推奨" if mode == suggested_mode else ""
        print(f"{key}. {label}{mark}")

    raw = input("モードを選択してください [1-3] (Enterで推奨を使用): ").strip()
    if raw == "":
        return suggested_mode, suggested_block
    if raw not in {"1", "2", "3"}:
        raise RuntimeError("モード選択が不正です。1-3 のいずれかを入力してください。")

    selected_mode = {"1": "ultra_low", "2": "low", "3": "mid"}[raw]
    return selected_mode, MODE_TABLE[selected_mode]


def parse_version_tuple(raw: str) -> Tuple[int, int, int]:
    parts = (raw or "").strip().split(".")
    values = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        values.append(int(digits or "0"))
    while len(values) < 3:
        values.append(0)
    return tuple(values[:3])


def get_onnxruntime_version_from_dll(path: Path) -> str:
    handle = None
    try:
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(str(path.parent))
        dll = ctypes.WinDLL(str(path))
        get_api_base = dll.OrtGetApiBase
        get_api_base.restype = ctypes.POINTER(_OrtApiBase)
        api_base = get_api_base()
        if not api_base:
            raise RuntimeError("OrtGetApiBase returned NULL")
        version_fn = ctypes.CFUNCTYPE(ctypes.c_char_p)(api_base.contents.get_version_string)
        version_ptr = version_fn()
        if not version_ptr:
            raise RuntimeError("GetVersionString returned NULL")
        return version_ptr.decode("utf-8", errors="replace").strip()
    finally:
        if handle is not None:
            handle.close()


def is_supported_ort_version(raw: str) -> bool:
    return parse_version_tuple(raw) >= (1, 23, 0)


def collect_missing_dlls(model_dir: Path, names) -> list[str]:
    return [name for name in names if not (model_dir / name).exists()]


def remove_existing_files(model_dir: Path, names) -> None:
    for name in names:
        path = model_dir / name
        if path.exists():
            path.unlink()


def extract_dlls_from_zip(zip_path: Path, dest_dir: Path, required_names) -> None:
    required_lower = {name.lower() for name in required_names}
    extracted = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_name = Path(info.filename).name
            if not member_name.lower().endswith(".dll"):
                continue
            target = dest_dir / member_name
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.add(member_name.lower())

    missing = [name for name in required_names if name.lower() not in extracted and not (dest_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"required DLLs were not found in zip: {', ' .join(missing)} ({zip_path.name})"
        )


def download_package(url: str, temp_root: Path, label: str) -> Path:
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or f"{label}.zip"
    dest = temp_root / filename
    print(f"  downloading {label}: {url}")
    download_file(url, str(dest))
    return dest


def ensure_ort_dlls(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    ort_dll = model_dir / "onnxruntime.dll"
    ort_needs_refresh = not ort_dll.exists()
    if ort_dll.exists():
        try:
            version = get_onnxruntime_version_from_dll(ort_dll)
            print(f"  existing onnxruntime.dll: {version}")
            if not is_supported_ort_version(version):
                print(f"  replacing old onnxruntime.dll: {version}")
                ort_needs_refresh = True
        except Exception as exc:
            print(f"  warning: failed to inspect onnxruntime.dll, refreshing bundle: {exc}")
            ort_needs_refresh = True

    if ort_needs_refresh:
        remove_existing_files(model_dir, ORT_REQUIRED_DLLS)

    ort_missing = collect_missing_dlls(model_dir, ORT_REQUIRED_DLLS)
    cuda_package_map = {
        "cuda_cudart": ["cudart64_12.dll"],
        "libcublas": ["cublas64_12.dll", "cublasLt64_12.dll"],
        "libcufft": ["cufft64_11.dll"],
        "libcurand": ["curand64_10.dll"],
        "libcusolver": ["cusolver64_11.dll"],
        "libcusparse": ["cusparse64_12.dll"],
    }
    cuda_missing_packages = [
        key
        for key, names in cuda_package_map.items()
        if collect_missing_dlls(model_dir, names)
    ]
    cudnn_missing = collect_missing_dlls(model_dir, CUDNN_REQUIRED_DLLS)

    if not ort_missing and not cuda_missing_packages and not cudnn_missing:
        print("  all required DLLs are already present in model/")
        return

    with tempfile.TemporaryDirectory(prefix="rustvc_ort_") as temp_dir:
        temp_root = Path(temp_dir)

        if ort_missing:
            zip_path = download_package(ORT_PACKAGE_URL, temp_root, "ONNX Runtime 1.23.0")
            print("  extracting onnxruntime DLLs...")
            extract_dlls_from_zip(zip_path, model_dir, ORT_REQUIRED_DLLS)

        for package_key in cuda_missing_packages:
            zip_path = download_package(CUDA_PACKAGE_URLS[package_key], temp_root, package_key)
            print(f"  extracting DLLs from {package_key}...")
            extract_dlls_from_zip(zip_path, model_dir, cuda_package_map[package_key])

        if cudnn_missing:
            zip_path = download_package(CUDNN_PACKAGE_URL, temp_root, "cuDNN 9.1.1")
            print("  extracting cuDNN DLLs...")
            extract_dlls_from_zip(zip_path, model_dir, CUDNN_REQUIRED_DLLS)

    all_required = ORT_REQUIRED_DLLS + CUDA_REQUIRED_DLLS + CUDNN_REQUIRED_DLLS
    missing = collect_missing_dlls(model_dir, all_required)
    if missing:
        raise RuntimeError(f"failed to place required DLLs into model/: {', ' .join(missing)}")

    version = get_onnxruntime_version_from_dll(model_dir / 'onnxruntime.dll')
    if not is_supported_ort_version(version):
        raise RuntimeError(f"onnxruntime.dll version is still unsupported after refresh: {version}")

    print(f"  onnxruntime {version} and all required DLLs were placed in model/")


def export_generator(pth_path: str, out_path: str) -> None:
    import torch

    _patch_torch_load()
    _configure_script_module_paths()

    try:
        from infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
    except Exception as e:
        raise RuntimeError(f"infer_pack.models_onnx の読み込みに失敗しました: {e}") from e

    print(f"\n[generator] loading {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    sd = ckpt.get("weight", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    if sd is None:
        raise RuntimeError("checkpoint から state_dict を取得できませんでした。")

    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("checkpointにconfigが見つかりません。")

    version = str(ckpt.get("version", "v2"))
    model = SynthesizerTrnMsNSFsidM(*cfg, version=version, is_half=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[generator] 警告: 不足キーがあります（先頭5件）: {missing[:5]}")
    if unexpected:
        print(f"[generator] 警告: 余剰キーがあります（先頭5件）: {unexpected[:5]}")
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
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "faiss の import に失敗しました。setup.exe の同梱内容が不足している可能性があります。"
            )
        print("faissが見つかりません。python -m pip install faiss-cpu を実行してください")
        sys.exit(1)

    print("[index] faissインデックスを読み込み中...")
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        raise RuntimeError(f"faissインデックスの読み込みに失敗しました: {e}") from e

    if index.ntotal <= 0:
        raise RuntimeError("added.index にベクトルが存在しません。")

    try:
        vectors = index.reconstruct_n(0, index.ntotal)
    except Exception as e_reconstruct_n:
        if hasattr(index, "quantizer"):
            try:
                vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
                for i in range(index.ntotal):
                    vectors[i] = index.reconstruct(i)
            except Exception as e:
                raise RuntimeError(f"added.index からベクトル抽出に失敗しました: {e}") from e
        else:
            raise RuntimeError(
                f"added.index からベクトル抽出に失敗しました: {e_reconstruct_n}"
            ) from e_reconstruct_n

    if not isinstance(vectors, np.ndarray):
        vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise RuntimeError("added.index のベクトル形状が不正です。")

    vectors = vectors.astype(np.float32, copy=False)
    rows = int(vectors.shape[0])
    dims = int(vectors.shape[1])
    print(f"[index] {rows}行, {dims}次元 を検出")
    if dims != 768:
        print(f"警告: 次元数が{dims}です。768次元モデル以外は動作しない可能性があります")

    try:
        with open(out_path, "wb") as f:
            f.write(vectors.tobytes())
    except Exception as e:
        raise RuntimeError(f"model_vectors.bin の書き出しに失敗しました: {e}") from e

    return rows, dims


def ensure_exists(path: str, message: str) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(message)


def convert_ivf_index(base_dir: str) -> None:
    base_path = Path(base_dir)
    vectors_bin = base_path / "model" / "model_vectors.bin"
    ivf_bin = base_path / "model" / "model_vectors_ivf.bin"

    if not vectors_bin.exists():
        return

    print("\n--- IVF インデックス生成 ---")
    print(f"  入力: {vectors_bin}")
    print(f"  出力: {ivf_bin}")

    script_path = base_path / "scripts" / "convert_to_ivf.py"
    if not script_path.exists():
        print(f"  警告: convert_to_ivf.py が見つかりません: {script_path}")
        return

    try:
        helper = _load_script_module("convert_to_ivf.py", "_rustvc_convert_to_ivf")
        dims = helper.resolve_dims(vectors_bin, None, False)
        t0 = datetime.now()
        vectors = helper.load_vectors(vectors_bin, dims)
        centroids, offsets, sorted_vectors = helper.build_ivf(
            vectors, dims, 3812, 32
        )
        helper.write_ivf_bin(ivf_bin, centroids, offsets, sorted_vectors, 32)
        try:
            helper.quality_check(ivf_bin, vectors, sorted_vectors, 32)
        except Exception as quality_exc:
            print(f"  警告: IVF quality check をスキップしました: {quality_exc}")
        elapsed = (datetime.now() - t0).total_seconds()
        size_mb = ivf_bin.stat().st_size / 1024 / 1024
        print(f"  ✅ IVF インデックス生成完了: {size_mb:.1f} MB ({elapsed:.2f}s)")
    except Exception as exc:
        print(f"  ⚠️  IVF変換に失敗しました: {exc}")
        print("     model_vectors.bin は引き続き使用できます。")


def run_setup() -> None:
    ensure_torch_available()

    print_step(2, "GPU検出")
    gpu_name = detect_gpu_name()
    suggested_mode = classify_gpu_mode(gpu_name)
    suggested_block = MODE_TABLE[suggested_mode]
    print(f"検出されたGPU: {gpu_name}")
    print(f"推奨モード: {suggested_mode} (block_size={suggested_block})")

    print_step(3, "モード選択")
    selected_mode, selected_block = select_mode_interactive(suggested_mode)
    os.makedirs("./model", exist_ok=True)
    with open("./model/mode.txt", "w", encoding="utf-8") as f:
        f.write(str(selected_block))
    print(f"選択モード: {selected_mode} (block_size={selected_block})")
    print("mode.txt に保存しました: ./model/mode.txt")

    print_step(4, "Required DLLs")
    ensure_ort_dlls(Path("./model"))


    print_step(5, ".pth → model_dynamic.onnx 変換")
    pth_path = input("学習済みモデル(.pth)のパスを入力してください: ").strip().strip('"')
    if not pth_path:
        raise RuntimeError(".pth のパスが未入力です。")
    if not pth_path.lower().endswith(".pth"):
        raise RuntimeError("拡張子が .pth のファイルを指定してください。")
    ensure_exists(pth_path, f".pth ファイルが見つかりません: {pth_path}")
    os.makedirs("./model", exist_ok=True)
    out_onnx = "./model/model_dynamic.onnx"
    export_generator(pth_path, out_onnx)
    print("変換完了: model/model_dynamic.onnx")

    print_step(6, "added.index → model_vectors.bin 変換")
    index_path = input("added.index ファイルのパスを入力してください: ").strip().strip('"')
    if not index_path:
        raise RuntimeError("added.index のパスが未入力です。")
    ensure_exists(index_path, f"added.index ファイルが見つかりません: {index_path}")
    out_bin = "./model/model_vectors.bin"
    rows, dims = convert_index_to_bin(index_path, out_bin)
    print(f"変換完了: model/model_vectors.bin ({rows}行, {dims}次元)")
    convert_ivf_index(BASE_DIR)

    print("\nセットアップ完了！RustVC.exeを起動してください。")


def main() -> None:
    log_path, log_fp = open_run_log_file()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_fp)
    sys.stderr = TeeStream(original_stderr, log_fp)

    exit_code = 0
    try:
        run_setup()
    except Exception as e:
        print(f"\nエラー: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()
        print(f"ログ保存先: {log_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
