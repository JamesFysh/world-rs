"""
Fixture generation for WORLD-rs equivalence testing.

Provenance:
- speech_clean.wav, speech_silence.wav, long.wav, speech_noise.wav
  Source: ext_src/world-cpp/test/vaiueo2d.wav (22050 Hz mono, 0.79 s, ~17500 frames)
  License: vendored with world-cpp, acceptable for test fixtures.
  Processing: resampled 22050 -> 16000 Hz via scipy.signal.resample_poly, tiled to target length.
  speech_clean: 5 s tiled, no crossfade, tile count ~6
  speech_silence: 5 s tiled with 100 ms silence inserted between tiles to guarantee >60 ms pauses
  long.wav: 10 s tiled
  speech_noise: speech_clean + pink noise at SNR 10 dB

- music_vocal.wav
  Synthesized deterministically: 5 s, 220 Hz sine with 5 Hz vibrato (+/-5 Hz), accompaniment = low-pass filtered white noise at -20 dBFS relative to vocal.
  No external source.

All other fixtures are synthesized deterministically with fixed seed.
"""

import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, chirp, butter, filtfilt

SR = 16000
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SRC_SPEECH = Path(__file__).parent.parent.parent / "ext_src/world-cpp/test/vaiueo2d.wav"

np.random.seed(0)

def ensure_dir():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

def write_wav(path, data, sr=SR, subtype='PCM_16'):
    # Ensure mono float64 -> soundfile handles conversion
    data = np.asarray(data, dtype=np.float64)
    sf.write(path, data, sr, subtype=subtype)

def generate_sine(freq, duration, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    y = np.sin(2 * np.pi * freq * t)
    # 0 dBFS peak
    return y

def generate_chirp(f0, f1, duration, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    y = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    # Normalize to 0 dBFS peak
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y

def generate_silence(duration, sr=SR):
    return np.zeros(int(sr * duration))

def generate_short_tone(freq, duration, sr=SR):
    return generate_sine(freq, duration, sr)

def generate_clipping(duration, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    # 0 dBFS sine
    y = np.sin(2 * np.pi * 440 * t)
    # Add quantization noise ~ -60 dBFS
    noise = np.random.uniform(-0.5, 0.5, size=y.shape) / 1024
    y = y + noise
    # Hard clip to [-1,1]
    y = np.clip(y, -1.0, 1.0)
    return y

def generate_extreme_f0(sr=SR):
    duration = 2.0
    # 30 Hz sine, below floor 71 Hz
    return generate_sine(30.0, duration, sr)

def generate_music_vocal(duration, sr=SR):
    n = int(sr * duration)
    t = np.arange(n) / sr
    # Vibrato: 220 Hz +/-5 Hz at 5 Hz rate
    freq_mod = 220.0 + 5.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    vocal = np.sin(phase)
    # Normalize vocal to -3 dBFS peak to leave headroom
    vocal = vocal / np.max(np.abs(vocal)) * 0.707

    # Accompaniment: low-pass filtered white noise at -20 dBFS relative
    rng = np.random.RandomState(0)
    noise = rng.randn(n)
    # Simple 1st order low-pass ~ 2 kHz
    b, a = butter(2, 2000 / (sr / 2), btype='low')
    noise = filtfilt(b, a, noise)
    noise = noise / np.max(np.abs(noise)) * 0.1  # -20 dBFS approx
    y = vocal + noise
    # Normalize overall peak to 0.9
    y = y / np.max(np.abs(y)) * 0.9
    return y

def generate_nan_input(duration, sr=SR):
    n = int(sr * duration)
    y = np.full(n, np.nan, dtype=np.float32)
    return y

def write_nan_wav(path, duration, sr=SR):
    import struct
    n = int(sr * duration)
    data_size = n * 4
    # WAV header for IEEE float
    # RIFF header
    riff_header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    fmt_chunk = struct.pack('<4sIHHIIHH', b'fmt ', 16, 3, 1, sr, sr*4, 4, 32)
    data_header = struct.pack('<4sI', b'data', data_size)
    # NaN float32 canonical quiet NaN: 0x7fc00000 little endian = 00 00 c0 7f
    nan_bytes = bytes([0x00, 0x00, 0xc0, 0x7f])
    data = nan_bytes * n
    with open(path, 'wb') as f:
        f.write(riff_header)
        f.write(fmt_chunk)
        f.write(data_header)
        f.write(data)

def resample_audio(data, orig_sr, target_sr):
    # Use polyphase resampling
    # Find rational approximation
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    # resample_poly expects up/down
    y = resample_poly(data, up, down)
    return y

def derive_speech_fixtures(src_path):
    # Load source
    if not src_path.exists():
        raise FileNotFoundError(f"Source speech not found: {src_path}")
    data, sr_src = sf.read(src_path, always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    # Resample to 16 kHz
    data_16 = resample_audio(data, sr_src, SR)
    # Normalize to peak ~0.9
    peak = np.max(np.abs(data_16))
    if peak > 0:
        data_16 = data_16 / peak * 0.9

    # Helper to tile
    def tile_to_length(sig, length_sec):
        n_needed = int(SR * length_sec)
        repeats = int(np.ceil(n_needed / len(sig)))
        tiled = np.tile(sig, repeats)[:n_needed]
        return tiled

    # speech_clean 5 s
    clean = tile_to_length(data_16, 5.0)
    write_wav(FIXTURES_DIR / "speech_clean.wav", clean)

    # speech_silence 5 s with pauses
    pause = np.zeros(int(SR * 0.1))  # 100 ms pause >60 ms
    # Build by alternating speech chunk and pause
    chunk_len = int(len(data_16) * 0.5)  # use half source per chunk
    out = []
    pos = 0
    target_n = int(SR * 5.0)
    while pos < target_n:
        remaining = target_n - pos
        take = min(chunk_len, remaining)
        out.append(data_16[:take])
        pos += take
        if pos < target_n:
            # add pause if space
            pause_take = min(len(pause), target_n - pos)
            out.append(pause[:pause_take])
            pos += pause_take
    silence = np.concatenate(out)
    write_wav(FIXTURES_DIR / "speech_silence.wav", silence)

    # long.wav 10 s
    long_sig = tile_to_length(data_16, 10.0)
    write_wav(FIXTURES_DIR / "long.wav", long_sig)

    # speech_noise: clean + pink noise SNR 10 dB
    clean_full = tile_to_length(data_16, 5.0)
    # Generate pink-ish noise via filtered white
    rng = np.random.RandomState(1)
    noise = rng.randn(len(clean_full))
    b, a = butter(2, 4000 / (SR / 2), btype='low')
    noise = filtfilt(b, a, noise)
    # Scale to achieve SNR 10 dB
    signal_power = np.mean(clean_full ** 2)
    noise_power_target = signal_power / (10 ** (10 / 10))
    noise = noise / np.sqrt(np.mean(noise ** 2)) * np.sqrt(noise_power_target)
    noisy = clean_full + noise
    # Normalize peak
    noisy = noisy / np.max(np.abs(noisy)) * 0.9
    write_wav(FIXTURES_DIR / "speech_noise.wav", noisy)

def main():
    ensure_dir()
    # Deterministic sines
    write_wav(FIXTURES_DIR / "sine_71.wav", generate_sine(71, 2.0))
    write_wav(FIXTURES_DIR / "sine_200.wav", generate_sine(200, 2.0))
    write_wav(FIXTURES_DIR / "sine_500.wav", generate_sine(500, 2.0))
    write_wav(FIXTURES_DIR / "sine_800.wav", generate_sine(800, 2.0))

    write_wav(FIXTURES_DIR / "chirp_71_800.wav", generate_chirp(71, 800, 2.0))
    write_wav(FIXTURES_DIR / "silence.wav", generate_silence(2.0))
    write_wav(FIXTURES_DIR / "short.wav", generate_short_tone(800, 0.05))
    write_wav(FIXTURES_DIR / "clipping.wav", generate_clipping(2.0))
    write_wav(FIXTURES_DIR / "extreme_f0.wav", generate_extreme_f0())
    write_wav(FIXTURES_DIR / "music_vocal.wav", generate_music_vocal(5.0))

    # NaN input: use FLOAT subtype, deterministic manual write
    write_nan_wav(FIXTURES_DIR / "nan_input.wav", 2.0, SR)

    # Speech derived fixtures
    derive_speech_fixtures(SRC_SPEECH)

    # Verify count
    files = sorted([p.name for p in FIXTURES_DIR.glob("*.wav")])
    print(f"Generated {len(files)} fixtures:")
    for f in files:
        print(" ", f)

if __name__ == "__main__":
    main()
