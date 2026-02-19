# Rust-VC TODO

## P0 (Critical)

- [ ] Eliminate remaining `STATUS_ACCESS_VIOLATION` cases in long-run start/stop loops
- [ ] Add integration test for zero-copy shutdown path (`prepare_for_shutdown` -> drop)
- [ ] Verify stable real-time behavior with strict models on CUDA (`slow block`, `underrun`, queue headroom)
- [ ] Finalize HuBERT 49->50 shaping policy with audio A/B validation

## P1 (Important)

- [ ] Preset import/export (JSON file) in Tauri UI
- [ ] Better runtime diagnostics panel in UI (latency, queue, provider, frame stats)
- [ ] Auto-detect and validate ONNX runtime DLL/provider mismatch at startup
- [ ] Add profile presets for low-latency / balanced / quality modes

## P2 (Nice to Have)

- [ ] Unit tests for frame shaping helpers and RMVPE decode paths
- [ ] Add benchmark command for inference path comparison (standard vs zero-copy)
- [ ] Expand docs for model conversion workflow (`scripts/export_strict_onnx.py`)
- [ ] Add optional telemetry log file export for issue reports
