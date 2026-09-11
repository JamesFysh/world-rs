# Harvest

Harvest estimates the pitch contour (F0 over time) — the same job as DIO,
done more carefully. In plain English, where DIO takes a quick vote across a
few filtered bands, Harvest auditions a large crowd of F0 candidates: it
runs the signal through a dense bank of band-pass filters, scores every
candidate in every frame for periodicity, and then picks the most consistent
path through time, complete with explicit voiced/unvoiced decisions. The
result is a smoother, more reliable contour, especially on creaky voice,
rapid pitch movement and noisy recordings.

It achieves this in two stages. First, candidate generation: overlapping
band-pass filters produce many F0 hypotheses per frame (each filter "hears"
best near its own centre frequency). Second, refinement: candidates are
scored by periodicity strength and continuity with neighbouring frames, and
a final pass stitches the winners into one contour while marking frames
with no convincing candidate as unvoiced (F0 = 0). The price is compute —
Harvest is several times slower than DIO, so use DIO for drafts and
real-time work, Harvest for final-quality analysis.

## Signature

```rust
pub fn harvest(x: &[f64], fs: f64, option: &HarvestOption) -> Result<HarvestResult, HarvestError>
pub fn initialize_harvest_option() -> HarvestOption
pub struct HarvestOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub frame_period: f64,
}
pub struct HarvestResult {
    pub f0: Vec<f64>,
    pub temporal_positions: Vec<f64>,
}
```

## Options

Defaults from `initialize_harvest_option()` match the C++ reference:

```rust
HarvestOption {
    f0_floor: 71.0,    // ignore pitch below 71 Hz
    f0_ceil: 800.0,    // ignore pitch above 800 Hz
    frame_period: 5.0, // one F0 estimate every 5 ms
}
```

What each control does:

- `f0_floor` / `f0_ceil` — the plausible pitch band, which also bounds the
  candidate search. Anything outside is unvoiced by definition. Widen for
  singing or children's voices; keep tight for telephone speech to reject
  out-of-band noise.
- `frame_period` — time between F0 estimates in milliseconds. Finer values
  track fast pitch movement better and give downstream stages more frames,
  at linear cost.

Harvest has no speed knob (unlike DIO): quality is the point. If it is too
slow, switch to DIO rather than coarsening the frame period past ~5 ms.

## Cookbook

Drop-in upgrade over DIO — same result shape, same downstream code:

```rust
use world_rs::harvest::{harvest, initialize_harvest_option};
use world_rs::stonemask::stone_mask;

let option = initialize_harvest_option();
let pitch = harvest(&x, fs, &option)?;
// StoneMask refinement still applies: Harvest is smoother, not perfect.
let f0 = stone_mask(&x, x.len(), fs, &pitch.temporal_positions, &pitch.f0, pitch.f0.len())?;
```

Comparing trackers on the same audio (useful when deciding which to ship):

```rust
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::harvest::{harvest, initialize_harvest_option};

let dio_f0 = dio(&x, fs, &initialize_dio_option())?.f0;
let harvest_f0 = harvest(&x, fs, &initialize_harvest_option())?.f0;
let agree = dio_f0.iter().zip(&harvest_f0)
    .filter(|(&d, &h)| d > 0.0 && h > 0.0)
    .filter(|(&d, &h)| (d - h).abs() / h < 0.05)
    .count();
println!("trackers agree within 5% on {agree} voiced frames");
```

Errors: `EmptyInput`, `NonPositiveSampleRate`, `NonPositiveFramePeriod`,
`InvalidF0Range`, `NonFiniteInput`, `TooShortInput`, `TooManyFrames`.

Allocation budget enforced: `MAX_HARVEST_CELLS = 64_000_000`,
`MAX_HARVEST_FRAMES = 8_000_000` — very long inputs at fine frame periods
hit `TooManyFrames`; raise `frame_period` or process in chunks.
