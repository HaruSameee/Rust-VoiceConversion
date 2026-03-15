@echo off
setlocal

echo Building RustVC setup.exe...
python -m pip install pyinstaller faiss-cpu tqdm requests onnx onnxscript scipy --quiet
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

python -c "import scipy" >nul 2>nul
if errorlevel 1 (
  echo scipy is required in the build environment.
  echo Run: python -m pip install scipy
  exit /b 1
)

set "INFER_PACK_PATH=infer_pack"
if not exist "%INFER_PACK_PATH%" set "INFER_PACK_PATH=scripts\infer_pack"
set "RMVPE_PATH=rmvpe"
if not exist "%RMVPE_PATH%" set "RMVPE_PATH=scripts\rmvpe"
if not exist "%INFER_PACK_PATH%" (
  echo infer_pack directory not found.
  exit /b 1
)
if not exist "%RMVPE_PATH%" (
  echo rmvpe directory not found.
  exit /b 1
)

if exist build rmdir /s /q build
if exist dist\setup rmdir /s /q dist\setup

pyinstaller --noconfirm --clean setup.spec
if errorlevel 1 (
  echo PyInstaller build failed.
  exit /b 1
)

echo.
echo Build complete: dist\setup\setup.exe
exit /b 0
