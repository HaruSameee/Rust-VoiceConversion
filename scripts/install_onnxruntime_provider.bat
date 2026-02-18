@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=cuda"

if /I "%TARGET%"=="gpu" set "TARGET=cuda"
if /I "%TARGET%"=="cuda-legacy" set "TARGET=cuda11"
if /I "%TARGET%"=="legacy" set "TARGET=cuda11"
set "CUDA_RUNTIME="
set "PKG_FEED="
if /I "%TARGET%"=="cuda" (
  set "PKG=onnxruntime-gpu[cuda,cudnn]"
  set "EXPECT=CUDAExecutionProvider"
  set "CUDA_RUNTIME=12"
  set "PKG_FEED=default"
) else if /I "%TARGET%"=="cuda11" (
  set "PKG=onnxruntime-gpu"
  set "EXPECT=CUDAExecutionProvider"
  set "CUDA_RUNTIME=11"
  set "PKG_FEED=onnxruntime-cuda-11"
) else if /I "%TARGET%"=="directml" (
  set "PKG=onnxruntime-directml"
  set "EXPECT=DmlExecutionProvider"
) else if /I "%TARGET%"=="cpu" (
  set "PKG=onnxruntime"
  set "EXPECT=CPUExecutionProvider"
) else (
  echo [ERROR] Unknown target: %TARGET%
  echo Usage:
  echo   install_onnxruntime_provider.bat cuda
  echo   install_onnxruntime_provider.bat cuda11
  echo   install_onnxruntime_provider.bat cuda-legacy
  echo   install_onnxruntime_provider.bat directml
  echo   install_onnxruntime_provider.bat cpu
  exit /b 1
)

set "ROOT=%~dp0.."
pushd "%ROOT%" >nul

call :resolve_python
if errorlevel 1 (
  popd >nul
  exit /b 1
)

echo.
echo [INFO] Target            : %TARGET%
echo [INFO] Python            : !PY_CMD!
echo [INFO] Package           : %PKG%
if defined CUDA_RUNTIME echo [INFO] CUDA runtime profile: %CUDA_RUNTIME%
if defined PKG_FEED if not "%PKG_FEED%"=="default" echo [INFO] Package feed      : %PKG_FEED%
if /I "%TARGET%"=="cuda11" (
  echo [WARN] This app build currently links ort 2.x and expects ONNX Runtime 1.23.x+.
  echo [WARN] cuda11 installs ONNX Runtime 1.20.x and may cause startup panic.
)
echo [INFO] Workspace         : %ROOT%

echo.
echo [STEP] Upgrade pip
call !PY_CMD! -m pip install --upgrade pip
if errorlevel 1 goto :pip_failed

echo.
echo [STEP] Remove conflicting onnxruntime packages
call !PY_CMD! -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul

echo.
if defined CUDA_RUNTIME (
  if /I "%CUDA_RUNTIME%"=="11" (
    echo [STEP] Install ONNX Runtime GPU - CUDA 11 feed
    call !PY_CMD! -m pip install --upgrade coloredlogs flatbuffers numpy packaging protobuf sympy >nul 2>nul
    call !PY_CMD! -m pip install --upgrade onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
    if errorlevel 1 goto :pip_failed
  ) else (
    echo [STEP] Install ONNX Runtime GPU - CUDA 12 default
    call !PY_CMD! -m pip install --upgrade onnxruntime-gpu[cuda,cudnn]
    if errorlevel 1 goto :pip_failed
  )
) else (
  echo [STEP] Install %PKG%
  call !PY_CMD! -m pip install --upgrade %PKG%
  if errorlevel 1 goto :pip_failed
)

if defined CUDA_RUNTIME (
  echo.
  echo [STEP] Install CUDA runtime wheels - recommended
  if /I "%CUDA_RUNTIME%"=="11" (
    call !PY_CMD! -m pip install --upgrade nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cudnn-cu11==8.9.5.29 nvidia-cufft-cu11 nvidia-curand-cu11 >nul 2>nul
  ) else (
    call !PY_CMD! -m pip install --upgrade nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 >nul 2>nul
  )
  if errorlevel 1 (
    echo [WARN] Failed to install some CUDA runtime wheels.
    if /I "%CUDA_RUNTIME%"=="11" (
      echo [WARN] If startup fails with missing CUDA DLLs, install CUDA 11.8 + cuDNN8 runtime manually.
    ) else (
      echo [WARN] If startup fails with missing CUDA DLLs, install CUDA 12 + cuDNN9 runtime manually.
    )
  )
)

echo.
echo [STEP] Verify providers and locate onnxruntime.dll
set "ORT_DLL="
set "PROVIDERS="
set "ORT_VERSION="
for /f "usebackq delims=" %%I in (`!PY_CMD! -c "import onnxruntime as ort; print(ort.__version__)"`) do set "ORT_VERSION=%%I"
for /f "usebackq delims=" %%I in (`!PY_CMD! -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"`) do set "PROVIDERS=%%I"
for /f "usebackq delims=" %%I in (`!PY_CMD! -c "import os, onnxruntime as ort; print(os.path.join(os.path.dirname(ort.__file__), 'capi', 'onnxruntime.dll'))"`) do set "ORT_DLL=%%I"

echo [INFO] onnxruntime version   : !ORT_VERSION!
echo [INFO] available providers   : !PROVIDERS!
echo [INFO] python ORT dll path   : !ORT_DLL!

if not defined ORT_DLL (
  echo [ERROR] Failed to detect onnxruntime.dll path from Python.
  popd >nul
  exit /b 1
)
if not exist "!ORT_DLL!" (
  echo [ERROR] onnxruntime.dll not found: !ORT_DLL!
  popd >nul
  exit /b 1
)

if not exist "model" mkdir "model" >nul 2>nul
for %%I in ("!ORT_DLL!") do set "ORT_DIR=%%~dpI"
if not defined ORT_DIR (
  echo [ERROR] Failed to resolve ORT directory from: !ORT_DLL!
  popd >nul
  exit /b 1
)

echo.
echo [STEP] Clean previous ORT/CUDA DLLs in model\
for %%P in (onnxruntime*.dll cublas*.dll cudnn*.dll cudart*.dll cufft*.dll curand*.dll) do (
  del /Q "model\%%P" >nul 2>nul
)

echo.
echo [STEP] Copy ORT runtime bundle to model\
set "COPIED_COUNT=0"
for %%F in ("!ORT_DIR!onnxruntime*.dll") do (
  copy /Y "%%~fF" "model\" >nul
  if not errorlevel 1 (
    set /a COPIED_COUNT+=1
  )
)
if "!COPIED_COUNT!"=="0" (
  echo [ERROR] Failed to copy ORT runtime DLLs from: !ORT_DIR!
  popd >nul
  exit /b 1
)
echo [INFO] Copied !COPIED_COUNT! file^(s^) to model\ (onnxruntime*.dll)

if not exist "model\onnxruntime_providers_shared.dll" (
  echo [WARN] model\onnxruntime_providers_shared.dll is missing.
  echo [WARN] CUDA/DirectML provider initialization will fail without it.
)

if defined CUDA_RUNTIME (
  echo.
  echo [STEP] Copy CUDA dependency DLLs to model\
  if /I "%CUDA_RUNTIME%"=="11" (
    call !PY_CMD! "%ROOT%\scripts\copy_cuda_runtime_dlls.py" --dest "%ROOT%\model" --profile cu11
  ) else (
    call !PY_CMD! "%ROOT%\scripts\copy_cuda_runtime_dlls.py" --dest "%ROOT%\model" --profile cu12
  )
  if errorlevel 1 (
    echo [WARN] CUDA dependency DLL setup is incomplete.
    if /I "%CUDA_RUNTIME%"=="11" (
      echo [WARN] If startup fails with missing CUDA DLLs, install CUDA 11.8 + cuDNN8 runtime and rerun this script.
    ) else (
      echo [WARN] If startup fails with missing CUDA DLLs, install CUDA 12 + cuDNN9 runtime and rerun this script.
    )
  )
)

echo.
echo [STEP] Check expected provider: %EXPECT%
echo !PROVIDERS! | findstr /I /C:"%EXPECT%" >nul
if errorlevel 1 (
  echo [WARN] Expected provider "%EXPECT%" is not available.
  echo [WARN] App may still run on CPU only.
) else (
  echo [OK] Expected provider is available.
)

echo.
echo [DONE] Installation complete.
echo        Restart the app and check log:
echo        [vc-inference] ort providers available: ...
echo.
popd >nul
exit /b 0

:resolve_python
set "PY_CMD="
where python >nul 2>nul
if not errorlevel 1 set "PY_CMD=python"
if not defined PY_CMD (
  where py >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3"
)
if not defined PY_CMD (
  echo [ERROR] Python not found. Install Python 3 and retry.
  exit /b 1
)
call !PY_CMD! -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python command failed: !PY_CMD!
  exit /b 1
)
exit /b 0

:pip_failed
echo [ERROR] pip installation failed.
echo         Check network / Python environment and retry.
popd >nul
exit /b 1
