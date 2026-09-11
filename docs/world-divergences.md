# WORLD divergences from C++

Rule: **C++ behaviour is authoritative.** Where Python/Java disagree with
C++, the port follows C++. All deviations below are intentional and locked.

## Intended API deviations
- `dio()` returns intermediate `f0Candidates`/`f0Scores` (C++ does not;
  accepted for diagnostics).
- `get_y_length()` public helper (C++ callers compute inline).
- Constant-AP convenience default 0.5 (full AP matrix retained).

## Intended safety deviations (typed `Result` where C++ is UB/silent)
- DIO zero-fills `f0` when `f0_length <= voice_range_minimum` (C++ leaves
  the buffer unwritten); rejects `f0_floor < 1 Hz`, bad
  `channels_in_octave`, NaN `allowed_range`, too-short input.
- CheapTrick rejects bad `fft_size` / non-finite `f0` (C++ would OOM/OOB).
- StoneMask validates `fs`/`f0` finiteness and length agreement.
- Synthesis validates `fs`/`frame_period_ms`, bounds `y_length`.
- D4C LoveTrain: `power_spectrum` zero-initialized, `boundary2` clamped
  to `fft_size-1` (C++ reads uninitialized memory / OOB-writes for
  fs ≤ 15800 / ≤ 7900).
- `SynthesisRealtime` out of scope (prototype: six behavioural
  divergences, latent UB, no consumer).

## Accepted tolerance divergences (not bugs)
- FFT backend Route A (`rustfft`/`realfft`); comparison is F0 RMSE < 1 Hz
  + V/U pattern match, SP correlation > 0.99, PSNR > 40 dB.
- c2c conjugation convention and unnormalized inverses reproduced exactly;
  explicit `÷ fft_size` at call sites.
- Resample output-length `round(n·to/from)` vs `round(n·L/M)`: proven
  unreachable divergence for real audio (fuzz to n=1e7, incl. 160/147).
- Unvoiced-noise synthesis on silence fixtures: bounded non-zero max-abs
  (≈4.9e-2, PSNR ≈ 53 dB; gate PSNR ≥ 35 dB, max-abs ≤ 0.1) from
  floating-point pulse-boundary shifts — inherent to Route A tolerance.
- Harvest fixtures use harmonic content (`harm_100/200/500/700`,
  `harm_chirp`, `speech_like`, `noise`, `silence`): pure tones are
  untrackable by C++ Harvest, so parity RMSE < 1 Hz / voicing ≥ 99% is
  measured on this set.

## Visibility
- Support primitives (`decimate`, `interp1`, `interp1q`) and FFT wrappers
  are crate-private (public in C++ headers, internal-only consumers).
- `no-panic` applies to top-level public entry points; internal helpers
  keep `debug_assert!` preconditions.
