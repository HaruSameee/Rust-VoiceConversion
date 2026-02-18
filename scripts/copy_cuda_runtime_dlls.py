#!/usr/bin/env python3
"""Copy CUDA runtime DLLs required by onnxruntime-gpu into a target directory.

This script tries two strategies:
1. Copy all DLLs found under Python wheel bins (site-packages/nvidia/**/bin).
2. Ensure minimum required DLL names exist, searching common roots.
"""

from __future__ import annotations

import argparse
import os
import shutil
import site
import sys
from pathlib import Path


REQUIRED_DLL_PROFILES = {
    # onnxruntime-gpu (default PyPI): CUDA 12 + cuDNN 9
    "cu12": [
        "cublasLt64_12.dll",
        "cublas64_12.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "cudart64_12.dll",
        "cudnn64_9.dll",
    ],
    # onnxruntime-cuda-11 feed: CUDA 11 + cuDNN 8
    "cu11": [
        "cublasLt64_11.dll",
        "cublas64_11.dll",
        "cufft64_10.dll",
        "curand64_10.dll",
        "cudart64_110.dll",
        "cudnn64_8.dll",
    ],
}


def collect_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(p: str | Path | None) -> None:
        if not p:
            return
        try:
            path = Path(p).expanduser().resolve()
        except OSError:
            return
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    for p in site.getsitepackages():
        add(p)
    add(site.getusersitepackages())

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        add(Path(cuda_path) / "bin")

    for p in os.environ.get("PATH", "").split(os.pathsep):
        add(p)

    return roots


def collect_nvidia_bin_dirs(roots: list[Path]) -> list[Path]:
    bins: list[Path] = []
    seen: set[str] = set()

    for root in roots:
        try:
            for p in root.glob("nvidia/**/bin"):
                if not p.is_dir():
                    continue
                key = str(p).lower()
                if key in seen:
                    continue
                seen.add(key)
                bins.append(p)
        except OSError:
            continue

    return bins


def copy_all_from_nvidia_bins(dest: Path, roots: list[Path]) -> int:
    copied = 0
    for bin_dir in collect_nvidia_bin_dirs(roots):
        try:
            for dll in bin_dir.glob("*.dll"):
                if not dll.is_file():
                    continue
                dst = dest / dll.name
                try:
                    shutil.copy2(dll, dst)
                    copied += 1
                except OSError:
                    continue
        except OSError:
            continue
    return copied


def find_dll(name: str, roots: list[Path]) -> Path | None:
    name_lc = name.lower()

    for root in roots:
        direct = root / name
        if direct.is_file():
            return direct

        # Common Python wheel layout: .../site-packages/nvidia/*/bin/*.dll
        try:
            for candidate in root.glob(f"nvidia/**/bin/{name}"):
                if candidate.is_file():
                    return candidate
        except OSError:
            pass

        # Common conda layout
        conda_candidate = root / "Library" / "bin" / name
        if conda_candidate.is_file():
            return conda_candidate

        # Last resort: case-insensitive scan at current depth only.
        try:
            for entry in root.iterdir():
                if entry.is_file() and entry.name.lower() == name_lc:
                    return entry
        except OSError:
            pass

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True, help="Destination directory")
    parser.add_argument(
        "--profile",
        choices=["auto", "cu12", "cu11"],
        default="auto",
        help="CUDA runtime profile to validate",
    )
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    roots = collect_roots()
    if not roots:
        print("[ERROR] No search roots found.")
        return 1

    copied_all = copy_all_from_nvidia_bins(dest, roots)
    if copied_all > 0:
        print(f"[INFO] copied {copied_all} DLL(s) from nvidia wheel bins")

    if args.profile == "auto":
        required_dlls = REQUIRED_DLL_PROFILES["cu12"]
    else:
        required_dlls = REQUIRED_DLL_PROFILES[args.profile]

    copied = 0
    present = 0
    missing: list[str] = []
    for dll in required_dlls:
        dst = dest / dll
        if dst.is_file():
            present += 1
            continue
        src = find_dll(dll, roots)
        if src is None:
            missing.append(dll)
            continue
        try:
            shutil.copy2(src, dst)
            copied += 1
            print(f"[INFO] copied {dll} <- {src}")
        except OSError as e:
            print(f"[WARN] failed to copy {dll} from {src}: {e}")
            missing.append(dll)

    profile_name = args.profile if args.profile != "auto" else "cu12(auto)"
    print(
        "[INFO] CUDA dependency copy summary: "
        f"profile={profile_name} present={present} copied={copied} missing={len(missing)}"
    )
    if missing:
        print("[WARN] Missing CUDA DLLs:")
        for dll in missing:
            print(f"  - {dll}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
