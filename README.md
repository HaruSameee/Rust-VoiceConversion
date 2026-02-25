# Rust-VC

Rust + Tauri real-time RVC inference engine (ONNX Runtime).

## Repository Layout
- `crates/vc-core`: runtime/model config and shared types
- `crates/vc-signal`: DSP utilities (resample, mel, f0 helpers)
- `crates/vc-inference`: HuBERT / RMVPE / Generator inference
- `crates/vc-audio`: realtime IO, queueing, SOLA/OLA, VAD flow
- `apps/tauri/src-tauri`: backend commands and app lifecycle
- `apps/tauri/ui`: frontend settings UI
- `scripts`: export / verification tooling

## Getting Started
Run guide is in `RUN.md`.

Quick start:
```powershell
npm --prefix apps/tauri/ui install
cargo tauri dev
```

## Runtime Notes
- `process_window` is computed in `vc-audio` from block geometry.
- Playback queue size is controlled by `target_buffer_ms` (independent from process window).
- Decoder slice tuning is controlled by:
  - `output_slice_offset_samples`
  - `sola_search_ms`
  - `output_tail_offset_ms`
- Debug dump can be enabled with `record_dump=true` (writes `debug_input.wav` / `debug_output.wav`).

## Alignment / Parity Workflow
1. Enable `record_dump` in UI, speak a few seconds, stop engine.
2. Measure lag:
```powershell
python scripts/verify_alignment.py --input debug_input.wav --output debug_output.wav --sample-rate 48000 --current-slice-offset 6054
```
3. Compare RMVPE mel parity:
```powershell
python scripts/verify_rmvpe_mel_parity.py --input apps/tauri/src-tauri/debug_input.wav --run-rust-dump --rust-csv rust_mel.csv --save-prefix mel_parity
```

## Git Hygiene
- Model artifacts and large local assets under `model/` are ignored by `.gitignore`.
- Local debug dumps (`debug_*.wav`, `mel_parity_*.npy`, logs) are ignored.
- Before commit, run:
```powershell
cargo check -p vc-inference
cargo check -p vc-audio
cargo check -p rust-vc-tauri
```

## Dev Build Performance
Workspace enables higher `dev` optimization for hot crates to keep `cargo tauri dev` usable:
- `vc-inference`
- `vc-audio`
- `vc-signal`
- `ort`
- `ndarray`
