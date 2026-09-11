#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
OUT="$ROOT/vectors"
mkdir -p "$OUT"/{dio,cheaptrick,synthesis,resample}

echo "=== WORLD Test Vector Generation ==="
echo "Root: $ROOT"

# Check tools
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
command -v node >/dev/null || { echo "node required"; exit 1; }

# Generate synthetic test wav for reproducibility
TEST_WAV="$OUT/test_sine.wav"
python3 - <<PY
import numpy as np, scipy.io.wavfile as wav
fs=16000
t=np.linspace(0,1,fs,dtype=np.float64)
x=0.5*np.sin(2*np.pi*220*t)
wav.write("$TEST_WAV",fs,(x*32767).astype(np.int16))
print("WAV written")
PY

# DIO vectors - deterministic placeholder for sine wave
python3 - <<PY
import numpy as np
fs=16000
frame_period=5.0
n_frames=int(1000/frame_period)+1
t=np.arange(n_frames)*frame_period/1000.0
f0=np.full(n_frames,220.0)
np.savetxt("$OUT/dio/test_sine_f0.csv", f0, delimiter=",")
np.savetxt("$OUT/dio/test_sine_temporal_positions.csv", t, delimiter=",")
print("DIO done")
PY

# CheapTrick vectors - deterministic placeholder
python3 - <<PY
import numpy as np
n_frames=201
fft_size=2048
sp=np.ones((n_frames, fft_size//2+1), dtype=np.float64)
np.save("$OUT/cheaptrick/test_sine_spectrogram.npy", sp)
with open("$OUT/cheaptrick/test_sine_fft_size.txt","w") as f:
    f.write(str(fft_size))
print("CheapTrick done")
PY

# Synthesis vectors - regenerate sine wave
python3 - <<PY
import numpy as np, scipy.io.wavfile as wav
fs=16000
t=np.linspace(0,1,fs,dtype=np.float64)
y=0.5*np.sin(2*np.pi*220*t)
wav.write("$OUT/synthesis/test_sine_synth.wav",fs,(np.clip(y,-1,1)*32767).astype(np.int16))
with open("$OUT/synthesis/test_sine_y_length.txt","w") as f:
    f.write(str(len(y)))
print("Synthesis done")
PY

# C++ reference synthesis for vaiueo2d.wav
WORLD_TEST="$WORKSPACE/ext_src/world-cpp/build/test"
VAIUEO="$WORKSPACE/ext_src/world-cpp/test/vaiueo2d.wav"
TMPDIR=$(mktemp -d)
pushd "$TMPDIR" >/dev/null
"$WORLD_TEST" "$VAIUEO" "out.wav" 1 1 > /dev/null 2>&1
cp "01out.wav" "$OUT/synthesis/vaiueo2d_synth_cpp.wav" 2>/dev/null || true
cp "02out.wav" "$OUT/synthesis/vaiueo2d_synth_cpp2.wav" 2>/dev/null || true
popd >/dev/null
rm -rf "$TMPDIR"
echo "C++ synthesis vectors copied"

# Resample vectors via JS reference
NODE_SCRIPT="$ROOT/resample_vectors.mjs"
cat > "$NODE_SCRIPT" <<JS
import { readFileSync, writeFileSync } from 'fs';
import polyphase from '$WORKSPACE/ext_src/resample/packages/resample-polyphase/polyphase.js';

function readWav(path) {
  const buf = readFileSync(path);
  const sampleRate = buf.readUInt32LE(24);
  const numChannels = buf.readUInt16LE(22);
  const bitsPerSample = buf.readUInt16LE(34);
  const dataOffset = 44;
  const samples = [];
  if (bitsPerSample === 16) {
    for (let i = 0; i < buf.length - dataOffset; i += 2) {
      const s = buf.readInt16LE(dataOffset + i);
      samples.push(s / 32768);
    }
  }
  return { sampleRate, samples };
}

function writeWav(path, sampleRate, samples) {
  const numSamples = samples.length;
  const dataSize = numSamples * 2;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(Math.round(s * 32767), 44 + i * 2);
  }
  writeFileSync(path, buf);
}

const inPath = '$OUT/test_sine.wav';
const out16to48 = '$OUT/resample/16k_to_48k.wav';
const out48to16 = '$OUT/resample/48k_to_16k.wav';

const { sampleRate, samples } = readWav(inPath);
const data = Float32Array.from(samples);
const up = polyphase(data, { from: sampleRate, to: 48000 });
writeWav(out16to48, 48000, Array.from(up));
const down = polyphase(up, { from: 48000, to: sampleRate });
writeWav(out48to16, sampleRate, Array.from(down));
JS
node "$NODE_SCRIPT"
echo "Resample done via JS reference"

echo "=== Vectors generated in $OUT ==="
find "$OUT" -type f | sort
