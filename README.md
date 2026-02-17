# Rust-VC

Python依存を最小化し、RustのみでリアルタイムVC（RVC系）を組むためのワークスペースです。

## 目的

- 推論エンジンの脱Python（`ort` による ONNX Runtime 連携）
- 音声処理を型安全なクレートで実装
- Tauri フロントエンドとの低遅延連携

## 構成

- `crates/vc-core`: 設定型・エラー型・推論パイプライン抽象
- `crates/vc-audio`: `cpal` ベースの Audio I/O 補助（デバイス列挙・レベルメータ）
- `crates/vc-signal`: `rustfft` / `ndarray` ベースの STFT・正規化
- `crates/vc-inference`: `ort` 依存を閉じ込めた推論層（現状はスタブ推論）
- `apps/tauri/src-tauri`: Tauri Commands で UI <-> Rust コアを橋渡し

## Tauri Command

- `list_audio_devices_cmd`
- `get_runtime_config_cmd`
- `set_runtime_config_cmd`
- `start_engine_cmd`
- `stop_engine_cmd`
- `get_engine_status_cmd`

## 次段階（実運用に必要）

1. `vc-inference` に RMVPE + RVC ONNX 入出力テンソル処理を実装
2. `vc-audio` に入出力ストリーム + リングバッファ + 遅延制御を実装
3. Tauri 側で状態イベント配信（meter/underrun/latency）を追加
4. `apps/tauri/ui` に React/Vue/Svelte の制御UIを実装
