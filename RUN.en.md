# Rust-VC Run Guide (Windows)

## 1. Prerequisites

- Rust (stable)
- Node.js + npm
- ONNX Runtime DLL bundle compatible with this project
- Optional NVIDIA GPU (CUDA runtime bundle)

## 2. Install UI Dependencies

```powershell
npm --prefix apps/tauri/ui install
```

## 3. Prepare Model Files

Put these files in `model/`:

- `model.onnx`
- `hubert.onnx` or `hubert_strict.onnx`
- `rmvpe.onnx` or `rmvpe_strict.onnx`
- `model_vectors.bin` (optional)

## 4. Install ONNX Runtime Provider Bundle

```powershell
scripts\install_onnxruntime_provider.bat cuda
```

Alternatives:

- `cuda11` for older GPUs
- `directml` for non-NVIDIA GPUs
- `cpu` for CPU-only execution

## 5. Start the App

```powershell
cargo tauri dev
```

## 6. Useful Runtime Settings

- `sample_rate`: `48000`
- `block_size`: start from `8192`, tune if underrun/latency is high
- `ort_provider`: `cuda` / `directml` / `cpu`
- `ort_threads`: try `0/1` (auto/1) or `1/1` for CUDA experiments
- `rmvpe_th`: common value `0.01`

## 7. Build Checks

```powershell
cargo check --workspace
```

```powershell
npm --prefix apps/tauri/ui run build
```

## 8. Troubleshooting Quick Tips

- Provider DLL errors: reinstall with `scripts\install_onnxruntime_provider.bat ...`
- CUDA issues: verify GPU/runtime compatibility, then compare with `ort_provider=cpu`
- Audio underrun/noise: reduce load, lower extra processing, retune block/thread settings
