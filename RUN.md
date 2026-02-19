# Rust-VC Run Guide

Single-file bilingual run guide (日本語 / English).

## 1. Prerequisites / 事前準備

- Rust (stable)
- Node.js + npm
- ONNX Runtime DLL bundle compatible with this project
- Optional NVIDIA GPU (for CUDA)

## 2. Install UI dependencies / UI 依存関係の導入

```powershell
npm --prefix apps/tauri/ui install
```

## 3. Prepare model files / モデル配置

Put these files under `model/`:

- `model.onnx`
- `hubert.onnx` or `hubert_strict.onnx`
- `rmvpe.onnx` or `rmvpe_strict.onnx`
- `model_vectors.bin` (optional)

## 4. Install ONNX Runtime provider / Provider 導入

```powershell
scripts\install_onnxruntime_provider.bat cuda
```

Alternatives: `cuda11`, `directml`, `cpu`.

## 5. Start app / 起動

```powershell
cargo tauri dev
```

## 6. Common runtime settings / よく使う設定

- `sample_rate`: `48000`
- `block_size`: start from `8192`
- `ort_provider`: `cuda` / `directml` / `cpu`
- `ort_threads`: try `0/1` or `1/1` for tuning
- `rmvpe_th`: `0.01`

## 7. Build checks / ビルド確認

```powershell
cargo check --workspace
npm --prefix apps/tauri/ui run build
```
