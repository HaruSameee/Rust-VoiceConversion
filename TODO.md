# TODO

## 日本語

### P0（優先）
- [ ] start/stop 連打時の安定性を再検証し、`STATUS_ACCESS_VIOLATION` を再発させない
- [ ] 推論終了シーケンス（`prepare_shutdown` -> thread join）の統合テスト追加
- [ ] 長時間稼働時の `slow block / underrun / queue` の回帰テスト整備
- [ ] HuBERT 49/50 フレーム契約の最終方針を固定（モデル側/ホスト側）

### P1（重要）
- [ ] UI でランタイムプリセットの保存/読込（JSON）
- [ ] 起動時の Provider/DLL ミスマッチ診断を強化
- [ ] ランタイム診断パネル（遅延/キュー/推論時間）を UI に追加
- [ ] 音質チューニング項目（slice/tail/fade）を UI から調整可能にする

### P2（改善）
- [ ] `vc-inference` の F0 デコード/補間テストを拡充
- [ ] `scripts/` のモデル変換手順を整理し、最低限の実行順を明文化
- [ ] 標準コピー経路の性能ベンチ（CPU/CUDA）を追加

## English

### P0 (Critical)
- [ ] Re-verify start/stop stability and prevent `STATUS_ACCESS_VIOLATION` regressions
- [ ] Add integration tests for shutdown ordering (`prepare_shutdown` -> thread join)
- [ ] Add long-run regression tests for `slow block / underrun / queue` behavior
- [ ] Finalize HuBERT 49/50 frame contract policy (model-side vs host-side)

### P1 (Important)
- [ ] Add runtime preset import/export (JSON) in UI
- [ ] Improve provider/DLL mismatch diagnostics on startup
- [ ] Add runtime diagnostics panel (latency/queue/inference timing)
- [ ] Expose audio tuning parameters (slice/tail/fade) in UI

### P2 (Nice to have)
- [ ] Expand F0 decode/resampling tests in `vc-inference`
- [ ] Clean up model export flow in `scripts/` with minimal documented sequence
- [ ] Add performance benchmark for standard-copy path (CPU/CUDA)
