// Polyphase resample reference vectors (Phase 4-8).
//
// Generates the deterministic resample test cases as binary files for the Rust
// accuracy tests (`crates/world-rs/tests/resample_accuracy.rs`). Each case
// writes a `<name>.in` (the f32 input PCM) and a `<name>.out` (the JS
// polyphase reference output, f32). The Rust test reads the input, runs
// `resample`, and compares the output to the reference bit-for-bit and by PSNR.
//
// The reference is the `@audio/resample-polyphase` atom
// (`ext_src/resample/packages/resample-polyphase/polyphase.js`), the algorithmic
// reference for the Rust port. The input is written to `<name>.in` so the Rust
// test resamples the *exact* samples the reference saw (no JS/Rust sine
// re-derivation drift).
//
// File layout (all little-endian):
//   <name>.in : u32 n, n * f32 input samples
//   <name>.out: u32 from, u32 to, u32 m, m * f32 reference output samples

import { writeFileSync } from 'fs';
import polyphase from '../../ext_src/resample/packages/resample-polyphase/polyphase.js';

const OUT = './vectors/resample';

// A pure sine, `freq` Hz at sample rate `sr`, `n` samples (f32). The argument
// and sine are computed in f64 (JS numbers are doubles) then downcast to f32.
function sine(freq, n, sr) {
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return d;
}

// Max-amplitude edge case: full-scale alternating +/-1.0 (a Nyquist square),
// the worst-case |x| <= 1 stress for the polyphase accumulation.
function maxamp(n) {
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = i % 2 === 0 ? 1.0 : -1.0;
  return d;
}

// Linear chirp sweeping f0 -> f1 Hz over `n` samples at rate `sr` (f32):
// x[i] = sin(2*pi*(f0*i + (f1-f0)*i^2/(2*n)) / sr).
function chirp(f0, f1, n, sr) {
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = f0 * i + ((f1 - f0) * i * i) / (2 * n);
    d[i] = Math.sin(2 * Math.PI * t / sr);
  }
  return d;
}

const cases = [
  // PSNR / bit-for-bit: 16k -> 48k (3:1 upsample) and 48k -> 16k (1:3
  // downsample) of a 440 Hz sine.
  { name: 'sine440_16k_to_48k', from: 16000, to: 48000, sig: () => sine(440, 4096, 16000) },
  { name: 'sine440_48k_to_16k', from: 48000, to: 16000, sig: () => sine(440, 4096, 48000) },
  // Non-integer rational ratio: 44100 -> 48000 (reduced by gcd to L/M = 160/147).
  { name: 'sine440_44100_to_48000', from: 44100, to: 48000, sig: () => sine(440, 4410, 44100) },
  // Max-amplitude edge case: full-scale Nyquist square, 16k -> 48k.
  { name: 'maxamp_16k_to_48k', from: 16000, to: 48000, sig: () => maxamp(1024) },
  // Swept sine: 100 -> 15000 Hz chirp at 48k, downsampled to 16k (anti-alias).
  { name: 'swept_48k_to_16k', from: 48000, to: 16000, sig: () => chirp(100, 15000, 4096, 48000) },
];

for (const c of cases) {
  const x = c.sig();
  const y = polyphase(x, { from: c.from, to: c.to });

  const inBuf = Buffer.alloc(4 + x.length * 4);
  inBuf.writeUInt32LE(x.length, 0);
  for (let i = 0; i < x.length; i++) inBuf.writeFloatLE(x[i], 4 + i * 4);
  writeFileSync(`${OUT}/${c.name}.in`, inBuf);

  const outBuf = Buffer.alloc(12 + y.length * 4);
  outBuf.writeUInt32LE(c.from, 0);
  outBuf.writeUInt32LE(c.to, 4);
  outBuf.writeUInt32LE(y.length, 8);
  for (let i = 0; i < y.length; i++) outBuf.writeFloatLE(y[i], 12 + i * 4);
  writeFileSync(`${OUT}/${c.name}.out`, outBuf);

  console.log(`${c.name}: in=${x.length} out=${y.length} from=${c.from} to=${c.to}`);
}
