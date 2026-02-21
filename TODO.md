# TODO

## 日本語

### P0（最優先）
- [ ] start/stop の長時間連続試験で `STATUS_ACCESS_VIOLATION` 再発を潰す
- [ ] 終了順序（`prepare_shutdown -> worker join`）の統合テスト追加
- [ ] `silence_skips` / `queue_ms` / `slow block` の回帰テストを自動化
- [ ] 出力スライスとモデル遅延補正値の最終キャリブレーション

### P1（重要）
- [ ] UI の `record_dump` をプロファイル保存対象にする
- [ ] 起動時の Provider/DLL ミスマッチ診断をさらに明確化
- [ ] ランタイム診断パネル（queue, budget, inference time）を追加
- [ ] SOLA / tail / slice の推奨プリセット（低遅延/安定）を用意

### P2（改善）
- [ ] `vc-inference` の F0 デコード/平滑化テスト強化
- [ ] `scripts/` のエクスポート手順を最短フローで整理
- [ ] 標準コピー経路（zero-copy無効）の性能ベンチを追加

## English

### P0 (Critical)
- [ ] Re-verify long-run start/stop stability and prevent `STATUS_ACCESS_VIOLATION` regressions
- [ ] Add integration tests for shutdown ordering (`prepare_shutdown -> worker join`)
- [ ] Automate regression checks for `silence_skips`, `queue_ms`, and `slow block`
- [ ] Finalize decoder slice and model delay calibration values

### P1 (Important)
- [ ] Include UI `record_dump` in profile save/load
- [ ] Improve startup diagnostics for provider/DLL mismatches
- [ ] Add runtime diagnostics panel (queue, budget, inference time)
- [ ] Provide tuning presets for SOLA/tail/slice (low-latency vs stable)

### P2 (Nice to have)
- [ ] Expand F0 decode/smoothing tests in `vc-inference`
- [ ] Simplify and document minimal export flow in `scripts/`
- [ ] Add performance benchmark for standard-copy path (zero-copy disabled)
