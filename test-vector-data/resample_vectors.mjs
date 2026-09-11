import { readFileSync, writeFileSync } from 'fs';
import polyphase from '../../ext_src/resample/packages/resample-polyphase/polyphase.js';

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

const inPath = './vectors/test_sine.wav';
const out16to48 = './vectors/resample/16k_to_48k.wav';
const out48to16 = './vectors/resample/48k_to_16k.wav';

const { sampleRate, samples } = readWav(inPath);
const data = Float32Array.from(samples);
const up = polyphase(data, { from: sampleRate, to: 48000 });
writeWav(out16to48, 48000, Array.from(up));
const down = polyphase(up, { from: 48000, to: sampleRate });
writeWav(out48to16, sampleRate, Array.from(down));
