# Rust-VC TODO

## P0 (最優先)

- [ ] HuBERT の `Where/ReduceSum` 形状エラー再発ケースを収集し、固定長入力経路で再現しないことを確認
- [ ] RMVPE の all-unvoiced 発生条件を詰め、`rmvpe_threshold` と fallback 既定値を安定化
- [ ] 実機ログで `slow block` / `queue headroom` の閾値を確定し、途切れ対策の既定値を決定

## P1 (重要)

- [ ] プリセット Import/Export（JSON）を実装
- [ ] 起動前チェック（CUDA DLL / onnxruntime.dll バージョン整合）の UI ガイドを追加
- [ ] 低遅延 / 高品質 / 安定重視のランタイムプリセットを同梱

## P2 (改善)

- [ ] `vc-inference` と `vc-signal` の回帰テストを拡充（サンプル波形ベース）
- [ ] パフォーマンス計測用コマンド（block elapsed / queue headroom）を開発向けに整備
- [ ] ドキュメントを運用手順（GPUセットアップ、トラブルシュート）中心に整理
