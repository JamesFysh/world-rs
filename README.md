# world-rs

A pure-Rust port of the [WORLD](https://github.com/mmorise/World) speech vocoder — DIO, Harvest, StoneMask, CheapTrick, D4C and Synthesis — plus a polyphase resampler.

## Algorithms

- [DIO](docs/DIO.md) — pitch detection
- [Harvest](docs/HARVEST.md) — pitch detection
- [StoneMask](docs/STONEMASK.md) — F0 refinement
- [CheapTrick](docs/CHEAPTRICK.md) — spectral envelope
- [D4C](docs/D4C.md) — aperiodicity
- [Synthesis](docs/SYNTHESIS.md) — waveform synthesis
- [Resample](docs/RESAMPLE.md) — polyphase resampler

## Usage

```rust
use world_rs::dio::{dio, initialize_dio_option, DioOption};
use world_rs::stonemask::stone_mask;
use world_rs::cheaptrick::{cheaptrick, CheapTrickOption};
use world_rs::synthesis::synthesis;
use world_rs::resample::resample;

// DIO
let option = initialize_dio_option();
let DioResult { f0, temporal_positions, ... } = dio(&x, fs, &option)?;

// StoneMask
// stone_mask(x: &[f64], x_length: usize, fs: f64, temporal_positions: &[f64], f0: &[f64], f0_length: usize)
let refined_f0 = stone_mask(&x, x.len(), fs, &temporal_positions, &f0, f0.len())?;

// CheapTrick
let spectrogram = cheaptrick(&x, fs, &temporal_positions, &f0, &option)?;

// Synthesis
// synthesis(f0: &F0Contour, f0_length: usize, spectrogram: &Spectrogram, aperiodicity: &Aperiodicity, fft_size: usize, frame_period_ms: f64, fs: f64, y_length: usize)
let y = synthesis(&f0_contour, f0_len, &spectrogram, &aperiodicity, fft_size, frame_period_ms, fs, y_len)?;

// Resample
// resample(x: &MonoSignal, from: u32, to: u32) -> Vec<f32>
let y_resampled = resample(&x_f32, 16000, 48000);
```

WASM wrapper:

```ts
// stone_mask(x: Float64Array, fs: number, temporal_positions: Float64Array, f0: Float64Array) => Float64Array
const refined = stone_mask(x, fs, tpos, f0);
```

## Precision contract

WORLD APIs cross the WASM boundary as `f64`/`Float64Array`; Resample I/O is `f32`/`Float32Array`.

See [`docs/precision-contract.md`](docs/precision-contract.md).

## Equivalence harness

Test vectors and golden data:

- `test-vector-data/` — generators and golden vectors
- `test/world-equivalence/` — `compare.py`, `metrics.py`, fixtures

Run the harness via the Python entry points in `test/world-equivalence/`.

## Wasm

`world-rs-wasm` is built with `wasm-pack`. npm publishing is deferred; build locally:

```bash
wasm-pack build world-rs-wasm --target web
wasm-pack test --node world-rs-wasm
```

## License

BSD-3-Clause + NOTICE. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for upstream WORLD and audiojs/resample attributions.
