# CheapTrick

CheapTrick estimates the *spectral envelope*: the smooth shape of the
spectrum with the pitch harmonics removed. In plain English, speech is a
buzzy pitch source filtered by the throat, mouth and nose — the envelope is
that filter shape, i.e. what makes an "ah" different from an "ee" at the same
pitch. Synthesis replays speech by pushing a new pitch source through this
stored shape, so the envelope must capture resonances (formants) while
ignoring the fine harmonic comb.

It achieves this with pitch-adaptive analysis: for each frame it windows the
signal with a length tied to the local pitch period (about three periods, so
the window always spans the same number of harmonics regardless of pitch),
takes the power spectrum, smooths it in the cepstral domain (liftering) to
erase harmonic ripples, and then applies a spectral recovery step that
undoes the systematic bias liftering introduces. The output is one smooth
spectrum per frame.

CheapTrick needs a *refined* F0 contour — run StoneMask first. A jittery
pitch track mis-sizes the adaptive windows and the envelope inherits the
error.

## Signature

```rust
pub fn cheaptrick(
    x: &[f64],
    fs: f64,
    temporal_positions: &[f64],
    f0: &[f64],
    option: &CheapTrickOption,
) -> Result<Vec<Vec<f64>>, CheapTrickError>
pub fn initialize_cheaptrick_option(fs: f64) -> CheapTrickOption
pub struct CheapTrickOption {
    /// Cepstral smoothing/recovery parameter (default `-0.15`).
    pub q1: f64,
    /// Lower F0 limit (Hz) used to determine `fft_size` (default `kFloorF0`).
    pub f0_floor: f64,
    /// FFT size; `get_fft_size_for_cheaptrick(fs, self)`.
    pub fft_size: i32,
}
```

Returns one power spectrum per frame (`spectrogram[frame][bin]`), with
`fft_size / 2 + 1` bins each. Feed it directly to `synthesis()` as the
spectrogram argument.

## Options

Unlike DIO/Harvest, the initializer takes the sample rate, because the FFT
size is derived from it:

```rust
// fs = 16000 -> fft_size sized for 3*fs/f0_floor, rounded up to a power of two.
let option = initialize_cheaptrick_option(16000.0);
```

What each control does:

- `q1` (default `-0.15`) — the liftering/smoothing strength in the cepstral
  domain, plus the matching recovery compensation. More negative = smoother
  envelope (formants preserved, detail lost); closer to zero = more detail
  but harmonic ripple leaks back in. The default matches the C++ reference
  and is right for nearly all speech.
- `f0_floor` (default `71.0`) — the lowest pitch the window sizing assumes.
  It sets the longest analysis window; lower values mean larger FFTs. Keep
  it consistent with the tracker's floor.
- `fft_size` — computed by the initializer as
  `2^(1 + floor(ln(3*fs/f0_floor + 1) / ln 2))`; override only if you need a
  fixed size (it must stay a power of two ≥ the default, or low pitches get
  truncated windows).

## Cookbook

Envelope extraction in a standard pipeline:

```rust
use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::stonemask::stone_mask;

let pitch = dio(&x, fs, &initialize_dio_option())?;
let f0 = stone_mask(&x, x.len(), fs, &pitch.temporal_positions, &pitch.f0, pitch.f0.len())?;
let option = initialize_cheaptrick_option(fs);
let spectrogram = cheaptrick(&x, fs, &pitch.temporal_positions, &f0, &option)?;

assert_eq!(spectrogram.len(), f0.len()); // one spectrum per frame
```

Inspecting the result: each row is a power spectrum, so formants show up as
broad humps. A quick check that smoothing worked — neighbouring bins should
vary gradually, with no sharp harmonic teeth:

```rust
let frame = &spectrogram[spectrogram.len() / 2];
let peak = frame.iter().cloned().fold(0.0_f64, f64::max);
assert!(peak.is_finite() && peak > 0.0);
```

If the envelope looks harmonic (comb-like) rather than smooth, the usual
cause is an unrefined F0 track — make sure StoneMask ran before this step,
not raw DIO output.
