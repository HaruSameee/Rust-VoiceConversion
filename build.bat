@echo off
setlocal

echo Building RustVC setup.exe...
python -m pip install pyinstaller faiss-cpu tqdm requests --quiet
if errorlevel 1 (
  echo Failed to install build dependencies.
  exit /b 1
)

python -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo torch is not installed in the build environment.
  echo Run: python -m pip install torch
  exit /b 1
)

python -c "import onnx, onnxscript" >nul 2>nul
if errorlevel 1 (
  echo onnx and onnxscript are required in the build environment.
  echo Run: python -m pip install onnx onnxscript
  exit /b 1
)

set "INFER_PACK_PATH=infer_pack"
if not exist "%INFER_PACK_PATH%" set "INFER_PACK_PATH=scripts\infer_pack"
set "RMVPE_PATH=rmvpe"
if not exist "%RMVPE_PATH%" set "RMVPE_PATH=scripts\rmvpe"

if exist build rmdir /s /q build
if exist dist\setup.exe del /q dist\setup.exe
if exist setup.spec del /q setup.spec

pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --name setup ^
  --add-data "%INFER_PACK_PATH%;infer_pack" ^
  --add-data "%RMVPE_PATH%;rmvpe" ^
  --add-data "scripts\export_generator_standalone.py;scripts" ^
  --add-data "scripts\convert_index_standalone.py;scripts" ^
  --add-data "scripts\convert_to_ivf.py;scripts" ^
  --paths "scripts" ^
  --hidden-import torch ^
  --hidden-import torch._C ^
  --hidden-import onnx ^
  --hidden-import onnxscript ^
  --hidden-import faiss ^
  --hidden-import tqdm ^
  --hidden-import requests ^
  --collect-binaries torch ^
  --collect-data torch ^
  --copy-metadata torch ^
  --collect-binaries onnx ^
  --collect-data onnx ^
  --copy-metadata onnx ^
  --collect-binaries onnxscript ^
  --collect-data onnxscript ^
  --copy-metadata onnxscript ^
  --collect-binaries faiss ^
  --collect-data faiss ^
  --copy-metadata faiss-cpu ^
  --exclude-module torchaudio ^
  --exclude-module torchvision ^
  --exclude-module torchtext ^
  --exclude-module transformers ^
  --exclude-module onnxruntime ^
  --exclude-module numba ^
  --exclude-module scipy ^
  --exclude-module pandas ^
  setup.py
if errorlevel 1 (
  echo PyInstaller build failed.
  exit /b 1
)

echo.
echo Build complete: dist\setup.exe
exit /b 0
