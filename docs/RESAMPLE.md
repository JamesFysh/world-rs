# Resample

Resample converts audio between sample rates (e.g. 16 kHz → 48 kHz). In
plain English, the WORLD pipeline analyses speech at whatever rate the
microphone gave it, but playback devices, models and files each want their
own rate — resampling bridges that gap without changing pitch or duration.
It is also the step that moves audio across the wasm boundary at the rate
the web side expects.

It achieves this with polyphase filtering: a single 32-tap Kaiser-windowed
low-pass prototype (β = 8.6, designed in f64, stored as f32) is decomposed
into phases, and each output sample is computed by stepping through the
input at the exact rational ratio with f64 accumulation. Anti-aliasing is
built into the prototype, so downsampling does not fold high frequencies
back into the audible band. I/O is `f32` throughout, matching the
`Float32Array` convention on the JS side (see `docs/precision-contract.md`).

This is a port of the `audiojs/resample` `@audio/resample-polyphase` atom
(MIT); see `NOTICE` for attribution.

## Signature

```rust
pub fn resample(x: &MonoSignal, from: u32, to: u32) -> Vec<f32>
```

`MonoSignal` is `[f32]` (pass `&Vec<f32>` or any f32 slice). There is no options struct and no `Result`:
`from == to` returns the input unchanged, and a zero rate returns an empty
buffer. The output length is `round(x.len() * to / from)`.

## Cookbook

Upsampling pipeline output for playback:

```rust
use world_rs::resample::resample;

let y_16k: Vec<f32> = analysis_pipeline(); // whatever rate analysis used
let y_48k = resample(&y_16k, 16000, 48000);
assert_eq!(y_48k.len(), (y_16k.len() * 48000 + 16000 / 2) / 16000);
```

Preparing microphone input for a 16 kHz pipeline:

```rust
use world_rs::resample::resample;

let mic_48k: Vec<f32> = read_microphone(); // device rate
let mic_16k = resample(&mic_48k, 48000, 16000);
// `mic_16k` is f32 mono: cast to f64 only at the WORLD boundary.
let x: Vec<f64> = mic_16k.iter().map(|&s| s as f64).collect();
```

Round-trip check (down then up should stay close to the original, modulo
the filter's edge transient — trim a few dozen samples at each end before
comparing):

```rust
let back = resample(&resample(&x, 48000, 16000), 16000, 48000);
```
