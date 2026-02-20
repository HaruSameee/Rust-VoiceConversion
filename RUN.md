# Run Guide

## Japanese

### 1. 前提
- Rust (stable)
- Node.js + npm
- ONNX Runtime DLL
- (任意) NVIDIA GPU + CUDA

### 2. UI依存のインストール
```powershell
npm --prefix apps/tauri/ui install
```

### 3. モデル配置
`model/` に以下を配置:
- `model.onnx`
- `hubert_pad80.onnx` または `hubert_strict.onnx`
- `rmvpe_strict.onnx`
- `model_vectors.bin` (任意)

### 4. Providerインストール
```powershell
scripts\install_onnxruntime_provider.bat cuda
```
代替: `cuda11`, `directml`, `cpu`

### 5. 起動
```powershell
cargo tauri dev
```

### 6. 推奨初期値
- `sample_rate=48000`
- `block_size=8192`
- `ort_provider=cuda` (環境に応じて変更)
- `ort_threads=0/1`
- `rmvpe_th=0.01`

## English

### 1. Prerequisites
- Rust (stable)
- Node.js + npm
- ONNX Runtime DLL
- (Optional) NVIDIA GPU + CUDA

### 2. Install UI dependencies
```powershell
npm --prefix apps/tauri/ui install
```

### 3. Place model files
Put these files under `model/`:
- `model.onnx`
- `hubert_pad80.onnx` or `hubert_strict.onnx`
- `rmvpe_strict.onnx`
- `model_vectors.bin` (optional)

### 4. Install provider
```powershell
scripts\install_onnxruntime_provider.bat cuda
```
Alternatives: `cuda11`, `directml`, `cpu`

### 5. Start app
```powershell
cargo tauri dev
```

### 6. Recommended initial settings
- `sample_rate=48000`
- `block_size=8192`
- `ort_provider=cuda` (change per environment)
- `ort_threads=0/1`
- `rmvpe_th=0.01`
