# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata


ROOT = Path(SPECPATH).resolve()
infer_pack_path = ROOT / "infer_pack"
if not infer_pack_path.exists():
    infer_pack_path = ROOT / "scripts" / "infer_pack"

rmvpe_path = ROOT / "rmvpe"
if not rmvpe_path.exists():
    rmvpe_path = ROOT / "scripts" / "rmvpe"

datas = [
    (str(infer_pack_path), "infer_pack"),
    (str(rmvpe_path), "rmvpe"),
    ("scripts/convert_to_ivf.py", "scripts"),
    ("scripts/export_generator_standalone.py", "scripts"),
    ("scripts/convert_index_standalone.py", "scripts"),
]
binaries = []

datas += collect_data_files("torch")
datas += collect_data_files("onnx")
datas += collect_data_files("onnxscript")
datas += collect_data_files("faiss")
datas += copy_metadata("torch")
datas += copy_metadata("onnx")
datas += copy_metadata("onnxscript")
datas += copy_metadata("faiss-cpu")

binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("onnx")
binaries += collect_dynamic_libs("onnxscript")
binaries += collect_dynamic_libs("faiss")

a = Analysis(
    ["setup.py"],
    pathex=[".", "scripts"],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "torch",
        "torch._C",
        "onnx",
        "onnxscript",
        "faiss",
        "tqdm",
        "requests",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torchaudio",
        "torchvision",
        "torchtext",
        "transformers",
        "onnxruntime",
        "numba",
        "scipy",
        "pandas",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="setup",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="setup",
)
