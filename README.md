# Rust-VC

Rust + Tauri で構成されたリアルタイム RVC エンジンです。

## 日本語

### 概要
- ONNX Runtime で `HuBERT / RMVPE / RVC Decoder` を実行します。
- Tauri UI からランタイム設定を変更できます。
- リアルタイム処理（`vc-audio`）と推論（`vc-inference`）を分離しています。

### 主要ディレクトリ
- `crates/vc-core`: 共有設定・トレイト・パイプライン
- `crates/vc-signal`: リサンプル/DSP
- `crates/vc-inference`: ONNX 推論エンジン
- `crates/vc-audio`: リアルタイム音声 I/O
- `apps/tauri/src-tauri`: Tauri バックエンド
- `apps/tauri/ui`: フロントエンド
- `scripts`: モデル変換・検証スクリプト

### ドキュメント
- 実行手順: `RUN.md`
- 開発タスク: `TODO.md`

## English

### Overview
- Real-time RVC engine implemented with Rust + Tauri.
- Runs `HuBERT / RMVPE / RVC Decoder` via ONNX Runtime.
- Runtime parameters are controlled from the Tauri UI.

### Main directories
- `crates/vc-core`: shared config/traits/pipeline
- `crates/vc-signal`: resampling and DSP utilities
- `crates/vc-inference`: ONNX inference engine
- `crates/vc-audio`: real-time audio I/O worker
- `apps/tauri/src-tauri`: Tauri backend
- `apps/tauri/ui`: frontend
- `scripts`: export/patch/verification tools

### Docs
- Run guide: `RUN.md`
- Development tasks: `TODO.md`
