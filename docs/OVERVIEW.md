# Overview

`world-rs` is a pure-Rust implementation of the WORLD vocoder algorithms plus a polyphase resampler.

## Crate layout

```
world-rs-core modules merged into world-rs/src/
  cheaptrick/
  common/
  constants/
  d4c/
  dio/
  fft/
  harvest/
  matlab/
  resample/
  stonemask/
  synthesis/
```

## Public API

- `dio::dio(x: &[f64], fs: f64, option: &DioOption) -> Result<DioResult, DioError>`
- `dio::initialize_dio_option() -> DioOption`
- `harvest::harvest(x: &[f64], fs: f64, option: &HarvestOption) -> Result<HarvestResult, HarvestError>`
- `harvest::initialize_harvest_option() -> HarvestOption`
- `stonemask::stone_mask(x: &[f64], x_length: usize, fs: f64, temporal_positions: &[f64], f0: &[f64], f0_length: usize) -> Result<Vec<f64>, StoneMaskError>`
- `cheaptrick::cheaptrick(x: &[f64], fs: f64, temporal_positions: &[f64], f0: &[f64], option: &CheapTrickOption) -> Result<Vec<Vec<f64>>, CheapTrickError>`
- `d4c::d4c(x: &[f64], fs: f64, temporal_positions: &[f64], f0: &[f64], fft_size: i32, option: &D4COption) -> Result<Vec<Vec<f64>>, D4CError>`
- `synthesis::synthesis(f0: &F0Contour, f0_length: usize, spectrogram: &Spectrogram, aperiodicity: &Aperiodicity, fft_size: usize, frame_period_ms: f64, fs: f64, y_length: usize) -> Result<Vec<f64>, SynthesisError>`
- `resample::resample(x: &MonoSignal, from: u32, to: u32) -> Vec<f32>`

## Precision

Core math `f64`. Resample I/O `f32`. See `docs/precision-contract.md`.

## Usage cookbook

Typical pipeline:

1. Pitch: `dio` or `harvest`
2. Refine: `stone_mask`
3. Spectral envelope: `cheaptrick`
4. Aperiodicity: `d4c` or constant 0.5
5. Synthesis: `synthesis`
6. Resample: `resample` for sample-rate conversion
