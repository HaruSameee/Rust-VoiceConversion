# TODO

## P0 Critical
- [ ] Eliminate remaining `STATUS_ACCESS_VIOLATION` cases in repeated start/stop loops.
- [ ] Add integration test for inference shutdown ordering (`prepare_shutdown` to thread join).
- [ ] Stabilize long-run real-time behavior on CUDA (slow block, underrun, queue headroom).
- [ ] Finalize HuBERT frame policy (49/50) with reproducible A/B audio checks.

## P1 Important
- [ ] Add preset import/export (JSON) in Tauri UI.
- [ ] Add runtime diagnostics panel (latency, queue, provider, frame stats).
- [ ] Improve provider/DLL mismatch detection on startup.
- [ ] Add profile presets: low-latency / balanced / quality.

## P2 Nice to have
- [ ] Expand unit tests for frame shaping and RMVPE decode paths.
- [ ] Add benchmark command for standard vs zero-copy inference comparison.
- [ ] Expand `scripts/export_strict_onnx.py` documentation.
- [ ] Add optional telemetry log export for issue reports.

## 日本語メモ
- [ ] start/stop ループ時のクラッシュを完全解消する。
- [ ] シャットダウン順序の統合テストを追加する。
- [ ] CUDA実行時の長時間安定性を確認する。
- [ ] HuBERT 49/50フレーム方針を最終確定する。
