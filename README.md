# Rust-VC

Rust でリアルタイムのボイスチェンジャー（RVC 系）を動かすプロジェクトです。  
推論・音声処理・UI 連携を Rust 側に寄せ、Python 依存を最小化しています。

## 特徴

- Tauri UI + Rust backend
- ONNX Runtime 実行プロバイダ切り替え対応
- `.bin` 特徴量インデックス対応（FAISS `.index` 変換）
- リアルタイム向けのバッファ制御と SOLA 連結
- UI プリセット保存（ローカル保存）

## ディレクトリ構成

- `crates/vc-core`: 設定型、共通 API、エラー定義
- `crates/vc-inference`: ONNX 推論エンジン（RVC / HuBERT / RMVPE）
- `crates/vc-signal`: 信号処理（リサンプル、F0 後処理、RMS ミックス）
- `crates/vc-audio`: `cpal` ベースの入出力ストリーム
- `apps/tauri/src-tauri`: Tauri コマンド、設定保存、実行制御
- `apps/tauri/ui`: UI（Vite + TypeScript）

## Quick Start (Windows)

1. UI 依存をインストール

```powershell
npm --prefix apps/tauri/ui install
```

2. `model/` にモデルを配置

- `model.onnx` (RVC 本体)
- `hubert.onnx` (内容特徴抽出)
- `rmvpe.onnx` (F0 推定)
- `model_vectors.bin` (任意、インデックス)

3. ONNX Runtime プロバイダを導入

```powershell
# 推奨: CUDA
scripts\install_onnxruntime_provider.bat cuda

# 旧世代 GPU 向け（必要時のみ）
scripts\install_onnxruntime_provider.bat cuda11

# 代替: DirectML
scripts\install_onnxruntime_provider.bat directml
```

4. 起動

```powershell
cargo tauri dev
```

5. UI で `Save` -> `Start`

## 実行プロバイダ方針

- 推奨: `cuda`
- 対応: `directml`
- 非推奨: `cpu`（リアルタイム用途では遅延・途切れが出やすい）

UI の `ort_provider` で `auto / cuda / directml / cpu` を選択できます。

## 直近の既定値・挙動（更新）

- HuBERT 入力は 16kHz 側で固定長 `16000` サンプル
- HuBERT 入力不足分は反射パディングで補完
- `rmvpe_threshold` の既定値は `0.01`
- `hubert_context_samples_16k` の既定値は `16000`

これにより、可変長入力由来の形状不整合を避け、RMVPE の有声検出を通しやすくしています。

## UI プリセット保存

Engine タブで以下が利用できます。

- `Preset Name` 入力
- `Save Preset`
- `Apply Preset`
- `Delete Preset`

プリセットはブラウザ側ローカルストレージ（キー: `rust_vc_presets_v1`）に保存されます。  
`Save` ボタンは従来どおりアプリ設定保存です。

## 主要パラメータのメモ

- `pitch_shift_semitones`: 大きすぎると機械的な高音になりやすい
- `rmvpe_threshold`: 高すぎると unvoiced 増加、低すぎるとノイズ拾い増加
- `index_rate`: 高すぎると変換キャラは強くなるが誤変換リスク増
- `rms_mix_rate`: `0.0` で入力エンベロープ追従が強い、`1.0` でほぼ無効

## よくあるエラーと対処

- `onnxruntime_providers_shared.dll is missing`
  - プロバイダ DLL セットが不足。`install_onnxruntime_provider.bat` を再実行
- `cublasLt64_12.dll` / `cufft64_11.dll` が missing
  - CUDA ランタイム DLL が不足。`cuda` または `cuda11` を再導入
- `no kernel image is available for execution on the device`
  - GPU と CUDA ビルドの不整合。旧世代 GPU は `cuda11` を試行
- `ort ... expected >= 1.23.x, but got 1.20.1`
  - `ort` クレートと `onnxruntime.dll` のバージョン不一致。対応版へ更新

## FAISS index 変換

`.index` から `.bin` を作る場合:

```powershell
python scripts/export_faiss_index_bin.py --index model/model.index --out model/model_vectors.bin
```

## 開発コマンド

```powershell
cargo check --workspace
cargo test -p vc-signal
cargo test -p vc-inference
npm --prefix apps/tauri/ui run build
```

## TODO（優先度順）

- P0: HuBERT の `Where/ReduceSum` 形状エラー再発ケースをログ採取し、固定長入力経路で再現しないことを確認
- P0: RMVPE の all-unvoiced 発生条件を詰め、しきい値・フォールバックの既定値を安定化
- P1: プリセットの Import/Export（JSON）対応
- P1: 起動前チェック（CUDA DLL / onnxruntime.dll バージョン整合）の UI ガイド強化
- P1: リアルタイム向けプリセット（低遅延 / 高品質 / 安定重視）を同梱
- P2: `vc-inference` と `vc-signal` の回帰テスト拡充（サンプル波形ベース）
- P2: パフォーマンス計測コマンド（block elapsed / queue headroom）を開発用に整備

詳細な進捗管理は `TODO.md` を参照してください。

## 補足

Windows 向け ONNX Runtime 導入詳細は `scripts/README-onnxruntime-windows.md` も参照してください。
