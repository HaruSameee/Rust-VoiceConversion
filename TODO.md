# Rust-VC TODO

Single-file bilingual task list (日本語 / English).

<a id="ja-en-shared"></a>

## P0 (Critical / 最優先)

- [ ] 長時間 start/stop で残る `STATUS_ACCESS_VIOLATION` の解消 / Eliminate remaining `STATUS_ACCESS_VIOLATION` in long-run start/stop loops
- [ ] ゼロコピー終了経路の統合テスト追加 / Add integration test for zero-copy shutdown path (`prepare_for_shutdown` -> drop)
- [ ] Strict モデル + CUDA の安定性検証 / Verify stable real-time behavior with strict models on CUDA (`slow block`, `underrun`, queue headroom)
- [ ] HuBERT 49->50 フレーム整形方針の最終確定 / Finalize HuBERT 49->50 shaping policy with audio A/B validation

## P1 (Important / 重要)

- [ ] UI でプリセット import/export（JSON） / Preset import/export (JSON file) in Tauri UI
- [ ] ランタイム診断パネル追加 / Better runtime diagnostics panel (latency, queue, provider, frame stats)
- [ ] DLL/Provider 不一致の起動時検証 / Auto-detect and validate ONNX runtime DLL/provider mismatch
- [ ] 低遅延/バランス/高音質プリセット / Add profile presets for low-latency / balanced / quality modes

## P2 (Nice to Have / 余力)

- [ ] フレーム整形/RMVPE 経路のユニットテスト / Unit tests for frame shaping helpers and RMVPE decode paths
- [ ] 標準 vs ゼロコピー比較ベンチ / Benchmark command for inference path comparison (standard vs zero-copy)
- [ ] `scripts/export_strict_onnx.py` のドキュメント拡充 / Expand docs for model conversion workflow
- [ ] 任意のテレメトリログ出力 / Optional telemetry log file export for issue reports
