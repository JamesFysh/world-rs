# Harvest

Pitch detection via Harvest algorithm.

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

## Defaults

`initialize_harvest_option()` returns `f0_floor = 71.0`, `f0_ceil = 800.0`, `frame_period = 5.0`.

## Cookbook

```rust
use world_rs::harvest::{harvest, initialize_harvest_option};

let option = initialize_harvest_option();
let res = harvest(&x, fs, &option)?;
let f0 = res.f0;
let tpos = res.temporal_positions;
```

Errors: `EmptyInput`, `NonPositiveSampleRate`, `NonPositiveFramePeriod`, `InvalidF0Range`, `NonFiniteInput`, `TooShortInput`, `TooManyFrames`.

Allocation budget enforced: `MAX_HARVEST_CELLS = 64_000_000`, `MAX_HARVEST_FRAMES = 8_000_000`.
