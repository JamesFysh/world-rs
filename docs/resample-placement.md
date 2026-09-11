# Resample placement

**Resample lives in `world-rs`, NOT in the core WORLD crate.** C++ WORLD
has no `Resample.cpp` — resampling is a custom Rust module with a JS
reference, and keeping it out of core preserves the purity of the WORLD
analysis primitives (DIO, Harvest, StoneMask, CheapTrick, D4C, Synthesis).

**Rates are integer `u32` only; zero is rejected** (JS silently rounds
non-integer rates; Rust strictness prevents silent errors).

**`get_y_length(f0_length, frame_period_ms, fs)` is public:**
`(f0_length-1)*frame_period/1000*fs + 1` (mirrors the C++ caller-side
computation; aids JS interop).

**Rate set:** any integer pair is supported (polyphase covers all rational
rates at constant cost); test vectors are pinned for 16k↔48k (primary use).

**Reference:** `https://github.com/audiojs/resample`
(`@audio/resample-polyphase`, MIT). Vector generation pins the Node
version in `generate-vectors.sh` for JS-reference determinism.
