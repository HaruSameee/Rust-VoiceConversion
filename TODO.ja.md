# Rust-VC TODO

## P0（最優先）

- [ ] 長時間の start/stop ループで残っている `STATUS_ACCESS_VIOLATION` を解消
- [ ] ゼロコピー終了経路（`prepare_for_shutdown` -> drop）の統合テストを追加
- [ ] Strict モデル + CUDA でのリアルタイム安定性を検証（`slow block`、`underrun`、queue headroom）
- [ ] HuBERT 49->50 フレーム整形方針を音声 A/B 比較で最終確定

## P1（重要）

- [ ] Tauri UI にプリセットの import/export（JSON ファイル）を実装
- [ ] UI にランタイム診断パネルを追加（レイテンシ、キュー、provider、フレーム統計）
- [ ] 起動時に ONNX Runtime DLL / Provider の不一致を自動検出・検証
- [ ] 低遅延 / バランス / 高音質のプロファイルプリセットを追加

## P2（余力）

- [ ] フレーム整形ヘルパーと RMVPE デコード経路のユニットテストを追加
- [ ] 推論経路比較（標準 vs ゼロコピー）のベンチコマンドを追加
- [ ] モデル変換ワークフロー（`scripts/export_strict_onnx.py`）のドキュメント拡充
- [ ] 不具合報告向けに任意のテレメトリログ出力を追加
