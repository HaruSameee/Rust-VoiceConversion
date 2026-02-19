# Rust-VC

Low-latency RVC voice conversion engine in Rust with a Tauri desktop UI.

## Overview

This repository contains:

- Real-time audio I/O (`cpal`)
- ONNX Runtime inference for RVC, HuBERT, and RMVPE
- Optional binary vector index lookup (`.bin`)
- Tauri + TypeScript UI with runtime preset save/apply

Current focus is stability and real-time performance on Windows + CUDA.

## Repository Layout

- `crates/vc-core`: shared config, errors, inference trait, pipeline
- `crates/vc-signal`: DSP helpers (resample, pitch helpers, RMS mix, postprocess)
- `crates/vc-inference`: ONNX inference engine (`RvcOrtEngine`, zero-copy path)
- `crates/vc-audio`: real-time audio worker and stream control
- `apps/tauri/src-tauri`: Tauri backend commands and app state
- `apps/tauri/ui`: Tauri frontend (Vite + TypeScript)
- `scripts`: export/install helper scripts

## Requirements

- Windows 10/11
- Rust toolchain (`stable`)
- Node.js + npm
- ONNX Runtime DLL bundle matching the `ort` crate version
- Optional NVIDIA GPU for CUDA execution

## Quick Start (Windows)

1. Install UI dependencies:

```powershell
npm --prefix apps/tauri/ui install
```

2. Place model files in `model/`:

- `model.onnx` (RVC decoder model)
- `hubert.onnx` or `hubert_strict.onnx`
- `rmvpe.onnx` or `rmvpe_strict.onnx`
- `model_vectors.bin` (optional feature index)

3. Install ONNX Runtime provider bundle:

```powershell
scripts\install_onnxruntime_provider.bat cuda
```

Use `cuda11` for older GPUs, or `directml` / `cpu` if needed.

4. Run app:

```powershell
cargo tauri dev
```

## Runtime Notes

- Default sample rate: `48000`
- Strict HuBERT input contract: `16000` samples
- Strict frame target to RVC decoder: `50` frames
- RMVPE default threshold: `0.01`
- Presets are stored in browser local storage from the UI

## Zero-Copy Path (Experimental)

`vc-inference` includes `ZeroCopyInferenceEngine`:

- ONNX IoBinding-based execution path
- HuBERT/RMVPE/decoder tensors preallocated
- Frame contract shaping before decoder input
- Explicit shutdown/drop synchronization hooks

If zero-copy init fails, engine falls back to the standard path.

## Troubleshooting

- Missing provider DLLs:
  - verify `onnxruntime.dll` and `onnxruntime_providers_shared.dll` are present
  - rerun `scripts\install_onnxruntime_provider.bat ...`
- CUDA errors:
  - check GPU compatibility with installed runtime bundle
  - try `ort_provider=cpu` to confirm model correctness
- Real-time underrun / crackle:
  - lower model load
  - tune block size / thread settings
  - reduce extra processing options

## Development

Rust checks:

```powershell
cargo check --workspace
```

Targeted checks:

```powershell
cargo check -p vc-inference
cargo check -p vc-audio
cargo check -p rust-vc-tauri
```

UI build:

```powershell
npm --prefix apps/tauri/ui run build
```

## Related Docs

- `TODO.md`
- `scripts/README-onnxruntime-windows.md`
