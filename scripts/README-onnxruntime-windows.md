# ONNX Runtime Setup (Windows)

This project loads `model/onnxruntime.dll` and related `onnxruntime*.dll` files.
The available inference provider depends on which runtime bundle you install.

- Recommended: CUDA (`onnxruntime-gpu`)
- Supported: DirectML (`onnxruntime-directml`)
- Discouraged: CPU (`onnxruntime`)

## Quick install

Run from project root:

```bat
scripts\install_onnxruntime_provider.bat cuda
```

Other options:

```bat
scripts\install_onnxruntime_provider.bat cuda11
scripts\install_onnxruntime_provider.bat directml
scripts\install_onnxruntime_provider.bat cpu
```

Wrapper scripts:

```bat
scripts\install_onnxruntime_gpu.bat
scripts\install_onnxruntime_gpu_legacy.bat
scripts\install_onnxruntime_directml.bat
```

## CUDA profile selection

- `cuda`:
  - Default PyPI GPU build (`onnxruntime-gpu[cuda,cudnn]`)
  - CUDA 12 + cuDNN 9 runtime family
  - Preferred when it works on your GPU
- `cuda11` / `cuda-legacy`:
  - Legacy CUDA 11 feed (`onnxruntime-cuda-11`)
  - Useful for older NVIDIA GPUs (for example GTX 10xx class)
  - Use this when CUDA 12 build shows errors like:
    - `no kernel image is available for execution on the device`
    - `CUDNN_FE failure`
  - Note: current app build may require ORT `1.23.x+`; `cuda11` installs `1.20.x` and can fail at startup.

## What the installer does

1. Reinstalls the selected Python `onnxruntime*` package.
2. Detects `onnxruntime.dll` from Python site-packages.
3. Copies `onnxruntime*.dll` into `model\`.
4. For CUDA targets, copies CUDA dependency DLLs into `model\`.
5. Prints detected providers.

## Required runtime DLL notes

- CUDA providers require ORT side DLLs in the same folder:
  - `onnxruntime.dll`
  - `onnxruntime_providers_shared.dll`
  - `onnxruntime_providers_cuda.dll` (CUDA) or `onnxruntime_providers_dml.dll` (DirectML)
- CUDA also requires NVIDIA runtime DLLs (varies by profile).

## Verify installation

```bat
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

App log should show:

```text
[vc-inference] ort providers available: cuda=... dml=... cpu=...
```

## Troubleshooting

- Missing `onnxruntime_providers_shared.dll`:
  - ORT DLL bundle is incomplete.
  - Re-run installer and confirm `model\onnxruntime*.dll` exists.
- Missing CUDA DLL (for example `cublasLt64_12.dll`):
  - Re-run CUDA installer.
  - Install matching CUDA runtime manually if needed.
- CUDA provider is visible but inference fails with `CUDNN_FE` or `no kernel image`:
  - Switch to legacy CUDA profile:
    - `scripts\install_onnxruntime_provider.bat cuda11`
  - If CUDA is still unstable on this machine, use DirectML as fallback.
