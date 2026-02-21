# Rust-VC

## 日本語
Rust + Tauri で構成した、リアルタイム RVC 推論エンジンです。  
ONNX Runtime で `HuBERT / RMVPE / RVC Decoder` を実行します。

### 主な構成
- `crates/vc-core`: 共通設定・トレイト・パイプライン定義
- `crates/vc-signal`: リサンプル/DSP
- `crates/vc-inference`: ONNX 推論エンジン
- `crates/vc-audio`: リアルタイム音声 I/O とバッファ制御
- `apps/tauri/src-tauri`: Tauri バックエンド
- `apps/tauri/ui`: フロントエンド
- `scripts`: エクスポート/検証スクリプト

### ドキュメント
- 実行手順: `RUN.md`
- 開発タスク: `TODO.md`

### 補足
- デバッグ WAV ダンプ（`debug_input.wav` / `debug_output.wav`）は UI の `record_dump` で ON/OFF できます。

## English
Rust-VC is a real-time RVC inference engine built with Rust + Tauri.  
It runs `HuBERT / RMVPE / RVC Decoder` on ONNX Runtime.

### Project layout
- `crates/vc-core`: shared config/traits/pipeline
- `crates/vc-signal`: resampling and DSP helpers
- `crates/vc-inference`: ONNX inference engine
- `crates/vc-audio`: real-time audio I/O and buffering
- `apps/tauri/src-tauri`: Tauri backend
- `apps/tauri/ui`: frontend
- `scripts`: export/patch/verification scripts

### Docs
- Run guide: `RUN.md`
- Development tasks: `TODO.md`

### Note
- Debug WAV dump (`debug_input.wav` / `debug_output.wav`) is toggled by the UI `record_dump` option.
