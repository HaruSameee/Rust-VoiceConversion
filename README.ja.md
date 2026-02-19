# Rust-VC

Rust + Tauri で構成した、低レイテンシな RVC リアルタイム音声変換エンジンです。

## 概要

このリポジトリには次が含まれます。

- リアルタイム音声 I/O（`cpal`）
- RVC / HuBERT / RMVPE の ONNX Runtime 推論
- 任意のバイナリベクトルインデックス（`.bin`）参照
- ランタイムプリセット保存/適用付きの Tauri + TypeScript UI

現在の主眼は、Windows + CUDA での安定性とリアルタイム性能です。

## リポジトリ構成

- `crates/vc-core`: 共通設定、エラー、推論トレイト、パイプライン
- `crates/vc-signal`: DSP ヘルパー（リサンプル、ピッチ補助、RMS ミックス、後処理）
- `crates/vc-inference`: ONNX 推論エンジン（`RvcOrtEngine`、ゼロコピー経路）
- `crates/vc-audio`: リアルタイム音声ワーカーとストリーム制御
- `apps/tauri/src-tauri`: Tauri バックエンドコマンドと状態管理
- `apps/tauri/ui`: Tauri フロントエンド（Vite + TypeScript）
- `scripts`: 変換/導入補助スクリプト

## 動作要件

- Windows 10/11
- Rust ツールチェーン（`stable`）
- Node.js + npm
- `ort` クレートのバージョンに一致する ONNX Runtime DLL 一式
- CUDA 実行時は NVIDIA GPU（任意）

## クイックスタート（Windows）

1. UI 依存関係をインストール:

```powershell
npm --prefix apps/tauri/ui install
```

2. `model/` にモデルを配置:

- `model.onnx`（RVC デコーダーモデル）
- `hubert.onnx` または `hubert_strict.onnx`
- `rmvpe.onnx` または `rmvpe_strict.onnx`
- `model_vectors.bin`（任意、特徴量インデックス）

3. ONNX Runtime Provider を導入:

```powershell
scripts\install_onnxruntime_provider.bat cuda
```

古い GPU の場合は `cuda11`、環境に応じて `directml` / `cpu` を利用してください。

4. アプリ起動:

```powershell
cargo tauri dev
```

## ランタイムメモ

- デフォルトサンプルレート: `48000`
- Strict HuBERT 入力契約: `16000` サンプル
- RVC デコーダーへ渡すフレーム契約: `50` フレーム
- RMVPE 既定しきい値: `0.01`
- プリセットは UI のブラウザローカルストレージに保存

## ゼロコピー経路（実験的）

`vc-inference` は `ZeroCopyInferenceEngine` を含みます。

- ONNX IoBinding ベース実行
- HuBERT / RMVPE / Decoder テンソルを事前確保
- Decoder 前でフレーム契約に整形
- シャットダウン時の明示的な同期フック

ゼロコピー初期化に失敗した場合は、標準経路へフォールバックします。

## トラブルシューティング

- Provider DLL が見つからない:
  - `onnxruntime.dll` と `onnxruntime_providers_shared.dll` の配置を確認
  - `scripts\install_onnxruntime_provider.bat ...` を再実行
- CUDA エラー:
  - 導入済み Runtime Bundle と GPU の互換性を確認
  - `ort_provider=cpu` でモデル自体が正しいか切り分け
- リアルタイム途切れ/ノイズ:
  - モデル負荷を下げる
  - block size / thread 設定を調整
  - 追加処理オプションを減らす

## 開発

Rust チェック:

```powershell
cargo check --workspace
```

対象別チェック:

```powershell
cargo check -p vc-inference
cargo check -p vc-audio
cargo check -p rust-vc-tauri
```

UI ビルド:

```powershell
npm --prefix apps/tauri/ui run build
```

## 関連ドキュメント

- `README.md`（言語選択）
- `TODO.ja.md`
- `RUN.ja.md`
- `scripts/README-onnxruntime-windows.md`
