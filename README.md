# Rust-VC

Rustでリアルタイムのボイスチェンジャー（RVC系）を動かすプロジェクトです。  
推論・音声処理・UI連携をRust側に寄せ、Python依存はモデル準備と実行環境セットアップに限定しています。

## 構成

- `crates/vc-core`: 設定型、エラー型、共通API
- `crates/vc-inference`: ONNXモデルのロードと推論（`RvcOrtEngine`）
- `crates/vc-signal`: 正規化、リサンプリング、RMS/F0処理
- `crates/vc-audio`: `cpal` 入出力ストリームとバッファ制御
- `apps/tauri/src-tauri`: Tauriコマンド（起動/停止、デバイス列挙、状態取得）

## Quick Start

1. UI依存のインストール
   - `npm --prefix apps/tauri/ui install`
2. モデルを `model/` に配置
   - `model.onnx`（RVC本体）
   - `hubert.onnx`（特徴抽出）
   - `rmvpe.onnx`（F0抽出）
3. ONNX Runtime の導入（Windows）
   - 推奨: `scripts\install_onnxruntime_provider.bat cuda`
   - 代替: `scripts\install_onnxruntime_provider.bat directml`
4. 開発起動
   - `cargo tauri dev`
5. UIで `Save` -> `Start`

## モデル/Index 補足

- 環境変数未設定時は `model/model.onnx` など既定パスを使用
- 現状のRVC推論は `hubert_path` と `pitch_extractor_path` が必須
- `*.index` は初回のみ Python + `faiss` で `*.rustvc.cache` を生成し、2回目以降はキャッシュ読込
- `.bin`（float32生配列）も対応済み。既定次元は 768（`RUST_VC_INDEX_BIN_DIM` で変更可）

## 推論デバイス方針

- 推奨順: `cuda` > `directml` > `cpu`
- UI設定:
  - `ort_provider`: `auto` / `cuda` / `directml` / `cpu`
  - `ort_device_id`: GPUデバイス番号
  - `ort_gpu_mem_limit_mb`: CUDAメモリアリーナ上限（0は無制限）
- Windows導入手順: `scripts/README-onnxruntime-windows.md`

CUDAで `no kernel image is available for execution on the device` / `CUDNN_FE failure` が出る場合:

- まず以下で再確認
  - `RUST_VC_CUDA_CONV_ALGO=default`
  - `RUST_VC_CUDA_CONV_MAX_WORKSPACE=0`
- それでも不可なら `cuda11` を検討
  - `scripts\install_onnxruntime_provider.bat cuda11`
  - 注: 現在の `ort` バージョン制約により、`onnxruntime.dll 1.20.x` では起動非互換になる場合があります

## 音質調整の要点

- `response_threshold`: 反応しきい値（`-50` 付近推奨、dBFS）
- `index_smooth_alpha`: Index参照の時間方向平滑化
- `pitch_smooth_alpha`: F0平滑化
- `f0_median_filter_radius`: F0メディアン平滑化半径
- `rms_mix_rate`: 入力音量エンベロープ継承率（0.0〜1.0）
  - `0.0` が最も入力に追従
  - `1.0` はほぼ無効（出力そのまま）

## FAISS Index 変換

`scripts/export_faiss_index_bin.py` で `.index` から `.bin` を作成できます。

- 自動検出:
  - `python scripts/export_faiss_index_bin.py`
- 明示指定:
  - `python scripts/export_faiss_index_bin.py --index model/model.index`
  - `python scripts/export_faiss_index_bin.py --index model/model.index --out model/model_vectors.bin`

## TODO

1. RVC本体仕様に合わせた前後処理の精緻化
2. レイテンシ最適化（バッファ戦略、ブロックサイズ調整）
3. UIの詳細パラメータ説明拡充
4. 異常系（デバイス切断、推論失敗時）の回復強化
