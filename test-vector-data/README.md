# WORLD Test Vectors — Pre-Phase 0

Reproducible reference vectors for DIO, CheapTrick, Synthesis and Resample.

## Scope

Vectors are generated from authoritative references:
- DIO / CheapTrick / Synthesis: C++ WORLD reference `ext_src/world-cpp` directly. C++ is authoritative.
- Resample: JS reference library `ext_src/resample` polyphase atom.

## Tolerances

- Audio PSNR >40 dB vs C++ reference for DIO/CheapTrick/Synthesis and vs JS reference for Resample
- F0 RMSE <1 Hz
- SP correlation >0.99

## Files

```
test-vector-data/
  README.md
  generate-vectors.sh          # pre-Phase 0 vectors (WAV/CSV/NPY)
  generate-dio-vectors.sh      # DIO Phase 1 accuracy vectors
  generate-cheaptrick-vectors.sh  # CheapTrick Phase 2 accuracy vectors
  generate-synthesis-vectors.sh   # Synthesis Phase 3 accuracy vectors
  dio_reference.cpp            # C++ DIO driver
  cheaptrick_reference.cpp     # C++ CheapTrick driver
  synthesis_reference.cpp      # C++ Synthesis driver
  stonemask_vectors.cpp        # StoneMask vectors
  resample_vectors.mjs         # JS resample driver (pre-Phase 0 WAVs)
  resample_reference.mjs       # JS resample accuracy driver (Phase 4-8)
  vectors/
    test_sine.wav
    dio/                       # .in/.out binary pairs (Phase 1)
    cheaptrick/                # .in/.out binary pairs (Phase 2)
    synthesis/                 # .in/.out binary pairs (Phase 3)
      sine_220_16k.{in,out}    # 220 Hz sine, 16 kHz, FFT 1024
      chirp_16k.{in,out}       # 100→600 Hz chirp, 16 kHz, FFT 1024
      jump_16k.{in,out}        # 200→500 Hz F0 discontinuity, 16 kHz
      transition_16k.{in,out}  # 440 Hz → unvoiced, 16 kHz
      unvoiced_16k.{in,out}    # all-unvoiced contour, 16 kHz
      silence_16k.{in,out}     # all-unvoiced, 1e-6 envelope
      sine_440_48k.{in,out}    # 440 Hz sine, 48 kHz, FFT 2048
      test_sine_synth.wav      # pre-Phase 0
      test_sine_y_length.txt
      vaiueo2d_synth_cpp.wav
      vaiueo2d_synth_cpp2.wav
    resample/
      16k_to_48k.wav             # pre-Phase 0 (16-bit WAV)
      48k_to_16k.wav             # pre-Phase 0 (16-bit WAV)
      sine440_16k_to_48k.{in,out}    # 440 Hz sine, 16k→48k (Phase 4-8)
      sine440_48k_to_16k.{in,out}    # 440 Hz sine, 48k→16k (Phase 4-8)
      sine440_44100_to_48000.{in,out} # 440 Hz sine, 44.1k→48k (Phase 4-8)
      maxamp_16k_to_48k.{in,out}     # full-scale Nyquist square (Phase 4-8)
      swept_48k_to_16k.{in,out}      # 100→15000 Hz chirp (Phase 4-8)
```

Synthesis `.in` layout: u32 f0_length, u32 fft_size, f64 frame_period_ms,
f64 fs, then f0[f0_length], sp[f0_length][fft_size/2+1],
ap[f0_length][fft_size/2+1] (all little-endian f64). `.out` layout: u32
y_length, then y[y_length].

Resample `.in` layout (Phase 4-8): u32 n, then n * f32 input samples
(little-endian). `.out` layout: u32 from, u32 to, u32 m, then m * f32 JS
polyphase reference output samples (little-endian). Generated from
`ext_src/resample` by `resample_reference.mjs`; read by
`crates/world-rs/tests/resample_accuracy.rs`.

## Reproducibility

Run:
```bash
bash generate-vectors.sh
bash generate-dio-vectors.sh
bash generate-cheaptrick-vectors.sh
bash generate-synthesis-vectors.sh
bash generate-resample-vectors.sh
```

All outputs are deterministic. PRNG seeded per spec. No network access required.
The C++ drivers build against `ext_src/world-cpp/build/libworld.a` (rebuild
the C++ reference first if it is missing).

## Locked decisions applied

- f64 throughout
- StoneMask included
- FFT Route A
- Resample integer u32 rates, in world-rs
- get_y_length public
