# Rust-VC

Rustだけでリアルタイムのボイスチェンジャー（RVC系）を動かすためのプロジェクトです。
Python依存を減らし、推論・音声処理・UI連携をRust側に集約することを目的にしています。

## 目的

- 推論エンジンの脱Python（`ort` + ONNX Runtime）
- 音声処理を型安全なRustクレートで実装
- Tauri経由でUIからリアルタイム制御

## 構成

- `crates/vc-core`
  - 設定型、エラー型、`VoiceChanger` の共通層
- `crates/vc-inference`
  - ONNXモデルのロードと推論（`RvcOrtEngine`）
- `crates/vc-signal`
  - STFT、正規化、リサンプリングなどの信号処理
- `crates/vc-audio`
  - `cpal` を使った入出力ストリームとレベル監視
- `apps/tauri/src-tauri`
  - Tauriコマンド（デバイス列挙、起動/停止、状態取得）

## Run

1. ONNXモデルを `model/` に配置する（またはUIで絶対パスを指定）
2. `cargo run -p rust-vc-tauri` を実行
3. 起動したUIで `Save Config` -> `Start` を押す
4. マイク入力が推論を通ってスピーカーへ出力される

補足:
- 環境変数未設定時は `model/model.onnx` などを既定値として使います
- 現状は「動くプロトタイプ」を優先した実装です

## Todo

1. RVC本体仕様に合わせた前後処理の精緻化
2. レイテンシ最適化（バッファ戦略、ブロックサイズ調整）
3. UIからのモデル切り替えと詳細パラメータ制御
4. 異常系（デバイス切断、推論失敗時）のリカバリ強化
