# Rust-VC

Rust + Tauri based real-time RVC engine.

## Japanese

### 概要
- Rust製のリアルタイム音声変換エンジンです。
- ONNX Runtime を使って HuBERT / RMVPE / RVC Decoder を実行します。
- Tauri UI からランタイム設定を変更できます。

### 主要ディレクトリ
- `crates/vc-core`: 共通設定・トレイト・エラー
- `crates/vc-signal`: リサンプルや信号処理ユーティリティ
- `crates/vc-inference`: ONNX 推論エンジン
- `crates/vc-audio`: リアルタイム音声I/Oワーカー
- `apps/tauri/src-tauri`: Tauriバックエンド
- `apps/tauri/ui`: フロントエンド

### ドキュメント
- 実行手順: `RUN.md`
- 開発タスク: `TODO.md`

## English

### Overview
- Real-time voice conversion engine implemented in Rust.
- Runs HuBERT / RMVPE / RVC Decoder via ONNX Runtime.
- Runtime settings are controlled from the Tauri UI.

### Main directories
- `crates/vc-core`: shared config, traits, errors
- `crates/vc-signal`: resample and DSP utilities
- `crates/vc-inference`: ONNX inference engine
- `crates/vc-audio`: real-time audio I/O worker
- `apps/tauri/src-tauri`: Tauri backend
- `apps/tauri/ui`: frontend

### Docs
- Run guide: `RUN.md`
- Development tasks: `TODO.md`
