# DIO

DIO estimates the pitch contour (F0 over time) of a speech signal. In plain
English, it answers "how high is the voice at each moment, and is it voiced
at all?" — the foundation every later stage (StoneMask refinement,
CheapTrick, D4C, synthesis) builds on. Unvoiced frames get F0 = 0.

It achieves this with speed in mind: a cascade of simple event detectors
runs the signal through a series of low-pass filters with decreasing cutoff
frequencies and counts zero-crossings in each band, producing raw F0
candidates per frame. A post-processing pass then fixes octave jumps,
smooths the contour, and decides voiced vs. unvoiced. The result is a fast,
rough pitch track — accurate enough to drive the pipeline, and cheap enough
for real-time use. When accuracy matters more than speed, Harvest is the
slower, more careful alternative.

## Signature

```rust
pub fn dio(x: &[f64], fs: f64, option: &DioOption) -> Result<DioResult, DioError>
pub fn initialize_dio_option() -> DioOption
pub struct DioOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub channels_in_octave: f64,
    /// Frame period in msec.
    pub frame_period: f64,
    /// Speed control, valid range (1, 2, ..., 12).
    pub speed: i32,
    /// Threshold used for fixing the F0 contour.
    pub allowed_range: f64,
}
pub struct DioResult {
    pub f0: Vec<f64>,
    pub temporal_positions: Vec<f64>,
    // ... plus per-frame voicing metadata; see rustdoc.
}
```

## Options

Defaults from `initialize_dio_option()` match the C++ reference:

```rust
DioOption {
    f0_floor: 71.0,          // ignore pitch below 71 Hz
    f0_ceil: 800.0,          // ignore pitch above 800 Hz
    channels_in_octave: 2.0, // filter-bank density: detectors per octave
    frame_period: 5.0,       // one F0 estimate every 5 ms
    speed: 1,                // 1 = most accurate; up to 12 = fastest
    allowed_range: 0.1,      // octave-jump fixing threshold (relative)
}
```

What each control does:

- `f0_floor` / `f0_ceil` — the plausible pitch band. Anything outside is
  treated as unvoiced. Widen for singing or children's voices; narrow to
  reject low rumble or high whistle interference.
- `channels_in_octave` — how many detector channels cover each octave.
  More channels = finer candidate resolution at higher cost. Rarely changed.
- `frame_period` — time between F0 estimates in milliseconds. Smaller
  values give finer temporal resolution and proportionally more frames for
  downstream stages.
- `speed` — downsampling factor for the detector cascade (1–12). Higher
  values skip more samples: much faster, slightly coarser. Use 1 when F0
  feeds StoneMask + re-synthesis; raise it for live/low-power scenarios.
- `allowed_range` — how far a frame's F0 may jump from its neighbours
  before post-processing calls it an octave error and corrects it. Lower
  values smooth more aggressively.

## Cookbook

Basic pitch tracking with defaults:

```rust
use world_rs::dio::{dio, initialize_dio_option};

let option = initialize_dio_option();
let result = dio(&x, fs, &option)?;
let f0 = result.f0;                         // Hz per frame, 0.0 = unvoiced
let tpos = result.temporal_positions;       // seconds per frame

let voiced_frames = f0.iter().filter(|&&v| v > 0.0).count();
println!("voiced: {voiced_frames}/{} frames", f0.len());
```

F0 out of range usually means the signal, not the tracker: check `fs` is
correct and the audio isn't silent/DC-shifted before widening the floor/ceil.

Feeding the rest of the pipeline (refine, then analyse):

```rust
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::stonemask::stone_mask;

let option = initialize_dio_option();
let dio_result = dio(&x, fs, &option)?;
// StoneMask needs lengths explicitly (wasm-boundary convention).
let refined = stone_mask(
    &x, x.len(), fs,
    &dio_result.temporal_positions,
    &dio_result.f0, dio_result.f0.len(),
)?;
```

Fast draft for interactive use (coarser, ~4–8× quicker):

```rust
use world_rs::dio::{dio, initialize_dio_option};

let mut option = initialize_dio_option();
option.speed = 4;
let result = dio(&x, fs, &option)?;
```
