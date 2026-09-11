# Synthesis

Synthesis turns analysis parameters back into audible speech. In plain
English, it is the reverse of the rest of the pipeline: given a pitch
contour (F0 over time), a spectral envelope (the vocal-tract filter shape
per frame), and the D4C aperiodicity map (how noisy each band is), it
rebuilds a waveform.

It does this frame by frame: it generates an excitation signal — a pulse
train buzzing at the local F0 mixed with white noise in the proportions the
aperiodicity map dictates — shapes it through the frame's spectral envelope
via FFT filtering, and overlap-adds the frames into one continuous signal at
the requested sample rate. Unvoiced frames get pure shaped noise; voiced
frames get the buzz/noise mix. The output is deterministic: the internal
PRNG is reseeded on every call, so the same inputs always give the same
waveform.

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

There is no options struct — behaviour is fully determined by the parameter
maps. Like the other frame-based APIs, lengths ride alongside the data
(`f0_length`, `y_length`) as a wasm-boundary convention.

## Helper

```rust
pub fn get_y_length(f0_length: usize, frame_period_ms: f64, fs: f64) -> usize
// = (f0_length-1)*frame_period/1000*fs + 1
```

Always size the output with this helper rather than hand-computing: it is
the exact inverse of the tracker's frame grid, so the waveform lines up
sample-accurately with the input.

## Cookbook

Full analysis → re-synthesis round trip (copy-and-resynthesize):

```rust
use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::d4c::{d4c, initialize_d4c_option};
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::stonemask::stone_mask;
use world_rs::synthesis::{get_y_length, synthesis};

let frame_period_ms = 5.0;
let pitch = dio(&x, fs, &initialize_dio_option())?;
let f0 = stone_mask(&x, x.len(), fs, &pitch.temporal_positions, &pitch.f0, pitch.f0.len())?;
let ct_option = initialize_cheaptrick_option(fs);
let spectrogram = cheaptrick(&x, fs, &pitch.temporal_positions, &f0, &ct_option)?;
let aperiodicity = d4c(&x, fs, &pitch.temporal_positions, &f0, ct_option.fft_size, &initialize_d4c_option())?;

let y_len = get_y_length(f0.len(), frame_period_ms, fs);
let y = synthesis(&f0, f0.len(), &spectrogram, &aperiodicity, ct_option.fft_size as usize, frame_period_ms, fs, y_len)?;
assert_eq!(y.len(), y_len);
```

Pitch-shifting (the classic vocoder trick): the envelope and aperiodicity
stay fixed while only F0 changes, so identity is preserved:

```rust
// One octave up, same voice:
let shifted: Vec<f64> = f0.iter().map(|&v| if v > 0.0 { v * 2.0 } else { 0.0 }).collect();
let y_high = synthesis(&shifted, f0.len(), &spectrogram, &aperiodicity, fft_size, frame_period_ms, fs, y_len)?;
```

Robot voice (diagnostic): a flat F0 with the real envelope isolates
envelope problems from pitch problems:

```rust
let flat: Vec<f64> = f0.iter().map(|&v| if v > 0.0 { 120.0 } else { 0.0 }).collect();
let y_robot = synthesis(&flat, f0.len(), &spectrogram, &aperiodicity, fft_size, frame_period_ms, fs, y_len)?;
```

Notes:
- `fft_size` must be power of two — reuse `CheapTrickOption::fft_size` so
  the maps align bin-for-bin.
- `frame_period_ms` is in milliseconds and must match the tracker's grid.
- PRNG is reseeded per call for deterministic output.
- Last pulse contributes zero.
