# TODO

## 日本語

### P0（最優先）
- [ ] start/stop の長時間連続試験で `STATUS_ACCESS_VIOLATION` 再発を潰す
- [ ] 終了順序（`prepare_shutdown -> worker join`）の統合テスト追加
- [ ] `silence_skips` / `queue_ms` / `slow block` の回帰テストを自動化
- [ ] 出力スライスとモデル遅延補正値の最終キャリブレーション
- [ ] `setup.exe` の DLL 同梱フローをクリーン環境で再検証（DL/展開/起動まで）

### P1（重要）
- [ ] UI の `record_dump` をプロファイル保存対象にする
- [ ] 起動時の Provider/DLL ミスマッチ診断をさらに明確化
- [ ] ランタイム診断パネル（queue, budget, inference time）を追加
- [ ] SOLA / tail / slice の推奨プリセット（低遅延/安定）を用意
- [ ] `setup.py` の DLL 取得失敗時メッセージを日本語/英語で整理する
- [ ] `dist/setup/` 配布手順をスクリーンショット付きで文書化する

### P2（改善）
- [ ] `vc-inference` の F0 デコード/平滑化テスト強化
- [ ] `scripts/` のエクスポート手順を最短フローで整理
- [ ] 標準コピー経路（zero-copy無効）の性能ベンチを追加
- [ ] setup のダウンロード対象 URL / バージョンを設定ファイル化する

## English

### P0 (Critical)
- [ ] Re-verify long-run start/stop stability and prevent `STATUS_ACCESS_VIOLATION` regressions
- [ ] Add integration tests for shutdown ordering (`prepare_shutdown -> worker join`)
- [ ] Automate regression checks for `silence_skips`, `queue_ms`, and `slow block`
- [ ] Finalize decoder slice and model delay calibration values
- [ ] Re-verify the packaged `setup.exe` DLL bundle flow on a clean machine end-to-end

### P1 (Important)
- [ ] Include UI `record_dump` in profile save/load
- [ ] Improve startup diagnostics for provider/DLL mismatches
- [ ] Add runtime diagnostics panel (queue, budget, inference time)
- [ ] Provide tuning presets for SOLA/tail/slice (low-latency vs stable)
- [ ] Clean up `setup.py` download failure messaging in both Japanese and English
- [ ] Document `dist/setup/` distribution steps with screenshots

### P2 (Nice to have)
- [ ] Expand F0 decode/smoothing tests in `vc-inference`
- [ ] Simplify and document minimal export flow in `scripts/`
- [ ] Add performance benchmark for standard-copy path (zero-copy disabled)
- [ ] Move setup download URLs/version pins into a dedicated config
