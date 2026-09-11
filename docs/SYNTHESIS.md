# Synthesis

Waveform synthesis from F0, spectrogram, and aperiodicity.

## Signature

```rust
pub fn synthesis(
    f0: &F0Contour,
    f0_length: usize,
    spectrogram: &Spectrogram,
    aperiodicity: &Aperiodicity,
    fft_size: usize,
    frame_period_ms: f64,
    fs: f64,
    y_length: usize,
) -> Result<Vec<f64>, SynthesisError>
```

## Helper

```rust
pub fn get_y_length(f0_length: usize, frame_period_ms: f64, fs: f64) -> usize
// = (f0_length-1)*frame_period/1000*fs + 1
```

## Cookbook

```rust
use world_rs::synthesis::synthesis;

let y = synthesis(&f0, f0_len, &spectrogram, &aperiodicity, fft_size, frame_period_ms, fs, y_len)?;
```

Notes:
- `fft_size` must be power of two.
- `frame_period_ms` is in milliseconds.
- PRNG is reseeded per call for deterministic output.
- Last pulse contributes zero.
