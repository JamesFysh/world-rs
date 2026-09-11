# D4C

The D4C algorithm estimates *band aperiodicity*: how much of the signal in
each frequency band is noise-like versus periodic, frame by frame. In plain
English, voiced speech is never a pure buzz — breathiness and turbulence add
noise, and they do so unevenly across the spectrum (breathy voice is noisier
at high frequencies, for example). Synthesis needs that noise map to sound
natural instead of robotic.

D4C builds it by windowing the waveform around each frame and measuring how
much the energy fluctuates *within* a pitch period: a perfectly periodic
signal repeats exactly each period, so any deviation from that repetition
quantifies the aperiodic component. The result is a time–frequency map (same
shape as the spectrogram) with values near 0 for strongly periodic bands and
near 1 for noise-dominated ones.

## Signature

```rust
pub fn d4c(
    x: &[f64],
    fs: f64,
    temporal_positions: &[f64],
    f0: &[f64],
    fft_size: i32,
    option: &D4COption,
) -> Result<Vec<Vec<f64>>, D4CError>
pub fn initialize_d4c_option() -> D4COption
pub struct D4COption {
    /// Threshold for aperiodicity estimation (default `kThreshold = 0.85`).
    pub threshold: f64,
}
```

Returns one aperiodicity vector per frame (`aperiodicity[frame][bin]`),
sized `fft_size / 2 + 1` like the CheapTrick output it pairs with. Feed it
directly to `synthesis()` alongside the spectrogram.

## Options

D4C has a single knob:

```rust
let mut option = initialize_d4c_option();
assert_eq!(option.threshold, 0.85);
```

- `threshold` (default `0.85`) — the voicing-decision threshold applied to
  the internal periodicity measure. Frames scoring above it count as
  periodic (aperiodicity pushed toward 0); below it, noisy (toward 1).
  Raise it if breathy/unvoiced regions are being reported as too periodic
  (buzzy synthesis); lower it if voiced regions sound too noisy or hoarse.
  Small adjustments (±0.05) are usually enough — the default matches the
  C++ reference.

`fft_size` is a function argument rather than an option: use the same value
CheapTrick used (from `CheapTrickOption::fft_size`) so the two maps align
bin-for-bin.

## Cookbook

Aperiodicity in a standard pipeline:

```rust
use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::d4c::{d4c, initialize_d4c_option};
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::stonemask::stone_mask;

let pitch = dio(&x, fs, &initialize_dio_option())?;
let f0 = stone_mask(&x, x.len(), fs, &pitch.temporal_positions, &pitch.f0, pitch.f0.len())?;
let ct_option = initialize_cheaptrick_option(fs);
let spectrogram = cheaptrick(&x, fs, &pitch.temporal_positions, &f0, &ct_option)?;
let d4c_option = initialize_d4c_option();
let aperiodicity = d4c(&x, fs, &pitch.temporal_positions, &f0, ct_option.fft_size, &d4c_option)?;

assert_eq!(aperiodicity.len(), spectrogram.len());
assert!(aperiodicity.iter().flatten().all(|&v| (0.0..=1.0).contains(&v)));
```

Reading the map: average each frame's vector to get overall noisiness over
time — sustained vowels should sit low (~0.0–0.3 in most bands), fricatives
and silence near 1. If *everything* reads near 0, the F0 track is probably
marking unvoiced regions as voiced (widen the tracker's view, or raise
`threshold`).

Constant-aperiodicity shortcut for drafts: synthesis accepts any map, so
`vec![vec![0.5; bins]; frames]` stands in for D4C while debugging the rest
of the pipeline — it sounds buzzy, but it isolates pitch/envelope problems
from aperiodicity ones.
