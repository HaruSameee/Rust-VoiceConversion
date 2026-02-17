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

1. `apps/tauri/ui` で `npm install`
2. ONNXモデルを `model/` に配置する（またはUIで絶対パスを指定）
   - `model.onnx`（RVC本体）
   - `hubert.onnx`（特徴量抽出）
   - `rmvpe.onnx`（ピッチ抽出）
3. `apps/tauri/src-tauri` で `cargo tauri dev` を実行
4. 起動したUIで `Save` -> `Start` を押す
5. マイク入力が推論を通ってスピーカーへ出力される

補足:
- 環境変数未設定時は `model/model.onnx` などを既定値として使います
- 現状のRVC推論は `hubert_path` と `pitch_extractor_path` が必須です
- `*.index`（FAISSバイナリ）は対応済みです
  - 初回だけ Python + `faiss` で展開し、`*.rustvc.cache` を生成します
  - 2回目以降はキャッシュを読むため高速です
- `.bin`（float32生配列）も対応済みです
  - 既定では 768 次元として読み込みます
  - 必要なら `RUST_VC_INDEX_BIN_DIM` で次元指定できます
- 推論デバイス設定（UI）
  - `ort_provider`: `auto` / `cuda` / `directml` / `cpu`
  - `ort_device_id`: GPUデバイス番号
  - `ort_gpu_mem_limit_mb`: CUDAメモリアリーナ上限（MB, 0で無制限）
- 音質調整（UI）
  - `index_smooth_alpha`: Index由来の声質変動を時間方向に平滑化（0.0〜0.98）
  - `pitch_smooth_alpha`: F0平滑化量（0.0〜0.98）
  - `f0_median_filter_radius`: F0メディアン平滑化の半径（0〜9）
  - `rms_mix_rate`: 入力音量エンベロープ継承率（0.0〜1.0）
  - 小さめ（0.05〜0.20）にすると機械感が減りやすい
- 現状は「動くプロトタイプ」を優先した実装です

## Index変換スクリプト

`scripts/export_faiss_index_bin.py` で `.index` から生の `float32` バイナリ `.bin` を作れます。

- 最小実行（自動検出）:
  - `python scripts/export_faiss_index_bin.py`
  - `model/model.index` があればそれを使います
- 明示指定:
  - `python scripts/export_faiss_index_bin.py --index model/model.index`
  - `python scripts/export_faiss_index_bin.py --index model/model.index --out model/model_vectors.bin`

出力時に `vector count` と `dimension(期待値768)` を表示し、次元不一致ならエラー終了します。
この `.bin` は UI の `index_path` にそのまま指定できます。

## Todo

1. RVC本体仕様に合わせた前後処理の精緻化
2. レイテンシ最適化（バッファ戦略、ブロックサイズ調整）
3. UIからのモデル切り替えと詳細パラメータ制御
4. 異常系（デバイス切断、推論失敗時）のリカバリ強化
