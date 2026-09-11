# D4C

Aperiodicity extraction via D4C.

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
```

## Options

`D4COption` controls aperiodicity extraction parameters. Default option matches C++ reference.

## Cookbook

```rust
use world_rs::d4c::{d4c, initialize_d4c_option};

let option = initialize_d4c_option();
let aperiodicity = d4c(&x, fs, &temporal_positions, &f0, fft_size, &option)?;
```

Output is `Vec<Vec<f64>>` with shape `[f0_length][fft_size/2+1]`.

Errors: `EmptyInput`, `NonFiniteInput`, `NonFiniteSampleRate`, `NonPositiveSampleRate`, `EmptyTemporalPositions`, `EmptyF0`, `MismatchedLengths`, etc.

Constant-AP fallback: `0.5` is safe default when aperiodicity is not verified.
