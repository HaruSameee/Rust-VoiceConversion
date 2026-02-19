# Rust-VC

Single-file bilingual docs (日本語 / English).

- 日本語: [日本語セクション](#ja)
- English: [English section](#en)
- TODO: `TODO.md`
- Run Guide: `RUN.md`

<a id="ja"></a>

## 日本語

### 概要

Rust + Tauri で構成した、低レイテンシな RVC リアルタイム音声変換エンジンです。

### 主な構成

- `crates/vc-core`: 共通設定、エラー、推論トレイト
- `crates/vc-signal`: DSP 補助（リサンプル、ピッチ補助、後処理）
- `crates/vc-inference`: ONNX 推論（標準経路 + ゼロコピー経路）
- `crates/vc-audio`: リアルタイム音声ワーカー
- `apps/tauri/src-tauri`: バックエンド
- `apps/tauri/ui`: フロントエンド

### クイックスタート（Windows）

```powershell
npm --prefix apps/tauri/ui install
scripts\install_onnxruntime_provider.bat cuda
cargo tauri dev
```

モデルは `model/` に配置してください（`model.onnx`, `hubert*.onnx`, `rmvpe*.onnx`, optional `model_vectors.bin`）。

詳細な実行手順は `RUN.md`、開発タスクは `TODO.md` を参照してください。

<a id="en"></a>

## English

### Overview

Low-latency RVC voice conversion engine in Rust with a Tauri desktop UI.

### Main Layout

- `crates/vc-core`: shared config, errors, inference trait
- `crates/vc-signal`: DSP helpers (resample, pitch helpers, postprocess)
- `crates/vc-inference`: ONNX inference (standard + zero-copy paths)
- `crates/vc-audio`: real-time audio worker
- `apps/tauri/src-tauri`: backend
- `apps/tauri/ui`: frontend

### Quick Start (Windows)

```powershell
npm --prefix apps/tauri/ui install
scripts\install_onnxruntime_provider.bat cuda
cargo tauri dev
```

Place models in `model/` (`model.onnx`, `hubert*.onnx`, `rmvpe*.onnx`, optional `model_vectors.bin`).

See `RUN.md` for run details and `TODO.md` for development tasks.
