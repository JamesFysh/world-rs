# Precision contract

C++ WORLD is authoritative and uses `double`. The Rust port therefore uses
**f64 for all WORLD core math** (DIO, Harvest, StoneMask, CheapTrick, D4C,
Synthesis, support primitives). Near-bit parity with C++ is the goal;
tolerance-based comparison (F0 RMSE < 1 Hz, SP correlation > 0.99,
audio PSNR > 40 dB) is the gate — not bit-exactness (FFT backend Route A:
`rustfft`/`realfft`, see `docs/world-divergences.md`).

**Resample is f32 I/O.** The resampler is a port of the `audiojs/resample`
`@audio/resample-polyphase` atom (32-tap Kaiser, β=8.6): prototype designed
in f64, coefficients stored f32, accumulation f64, input/output f32 —
matching the JS `Float32Array` reference.

**WASM boundary:** WORLD APIs cross as f64/`Float64Array`; Resample I/O
crosses as f32/`Float32Array`. Spectrograms cross as flat row-major
`Float64Array` (no jagged arrays, no extra copies).

**Determinism:** PRNG is a deterministic xorshift (`x=123456789,
y=362436069, z=521288629, w=88675123`), reseeded once per
CheapTrick/Synthesis call. Consumption order is part of the algorithm —
draw counts per pulse/frame are pinned by tests.
