# Rust-VC 実行ガイド（Windows）

## 1. 事前準備

- Rust（stable）
- Node.js + npm
- このプロジェクトと互換の ONNX Runtime DLL 一式
- 任意: NVIDIA GPU（CUDA Runtime Bundle）

## 2. UI 依存関係のインストール

```powershell
npm --prefix apps/tauri/ui install
```

## 3. モデルファイルの配置

`model/` に次を配置します。

- `model.onnx`
- `hubert.onnx` または `hubert_strict.onnx`
- `rmvpe.onnx` または `rmvpe_strict.onnx`
- `model_vectors.bin`（任意）

## 4. ONNX Runtime Provider の導入

```powershell
scripts\install_onnxruntime_provider.bat cuda
```

代替:

- 古い GPU: `cuda11`
- 非 NVIDIA GPU: `directml`
- CPU 実行: `cpu`

## 5. アプリ起動

```powershell
cargo tauri dev
```

## 6. 主要ランタイム設定

- `sample_rate`: `48000`
- `block_size`: まず `8192` から開始し、途切れ/遅延に応じて調整
- `ort_provider`: `cuda` / `directml` / `cpu`
- `ort_threads`: CUDA 検証では `0/1`（auto/1）や `1/1` を試す
- `rmvpe_th`: 目安 `0.01`

## 7. ビルド確認

```powershell
cargo check --workspace
```

```powershell
npm --prefix apps/tauri/ui run build
```

## 8. トラブル時の切り分け

- Provider DLL エラー: `scripts\install_onnxruntime_provider.bat ...` を再実行
- CUDA エラー: GPU/Runtime の互換性確認後に `ort_provider=cpu` と比較
- 音切れ/ノイズ: 負荷低減、追加処理の削減、block/thread 設定の再調整
