# Changelog

## world-rs v0.1.0 — 2026-09-12

Initial release, ported from the vocoder workspace.

- Pure-Rust WORLD vocoder: DIO, Harvest, StoneMask, CheapTrick, D4C, Synthesis.
- Polyphase resampler (`audiojs/resample` port, f32 I/O).
- Validated against the C++ reference (pinned `d625e76`) via the
  `test/world-equivalence` harness; see `docs/` for per-algorithm notes.
