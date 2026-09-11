# StoneMask

StoneMask refines a rough F0 contour (typically from DIO or Harvest) into a
precise one. In plain English, pitch trackers get the *neighbourhood* of the
pitch right but are often off by a fraction of a period — and those small
errors smear CheapTrick's spectral envelope and add roughness to synthesis.
StoneMask fixes each voiced frame's estimate to the exact local period.

It achieves this by looking at the waveform itself around each frame: using
the candidate period implied by the rough F0, it measures the instantaneous
frequency of the signal in that neighbourhood (via zero-crossing intervals
of the low-passed waveform) and nudges the estimate to match. Unvoiced
frames (F0 = 0) pass through untouched. It is cheap, has no tunable options,
and should be treated as a mandatory polish step between pitch tracking and
spectral analysis.

## Signature

```rust
pub fn stone_mask(
    x: &[f64],
    x_length: usize,
    fs: f64,
    temporal_positions: &[f64],
    f0: &[f64],
    f0_length: usize,
) -> Result<Vec<f64>, StoneMaskError>
```

There is no options struct: the only inputs are the signal, the sample rate,
and the rough contour with its time grid. `x_length` / `f0_length` are passed
explicitly (rather than taken from the slices) as a wasm-boundary
convention — pass `x.len()` and `f0.len()`. The returned `Vec<f64>` is the
refined contour on the same grid, same length, with unvoiced frames still 0.

## Cookbook

Standard placement — immediately after pitch tracking, before analysis:

```rust
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::stonemask::stone_mask;

let dio_option = initialize_dio_option();
let pitch = dio(&x, fs, &dio_option)?;
let f0 = stone_mask(
    &x, x.len(), fs,
    &pitch.temporal_positions,
    &pitch.f0, pitch.f0.len(),
)?;
// `f0` now feeds cheaptrick() and d4c().
```

Sanity check: refinement should move most voiced frames by only a few
percent. A quick diagnostic:

```rust
let rel_change: Vec<f64> = pitch.f0.iter().zip(f0.iter())
    .filter(|(&old, _)| old > 0.0)
    .map(|(&old, &new)| (new - old).abs() / old)
    .collect();
let max = rel_change.iter().cloned().fold(0.0_f64, f64::max);
assert!(max < 0.2, "refinement moved F0 by >20%: check fs and input");
```

StoneMask cannot rescue a bad contour — if DIO/Harvest reports voiced where
there is silence (or vice versa), fix the tracker's floor/ceiling first.
Refinement only sharpens frames that are already approximately right.
