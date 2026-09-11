#!/usr/bin/env python3
"""
Template script for MOS blind discrimination stimulus preparation.

Usage:
  python prepare_stimuli.py --cpp-dir ... --rs-dir ... --fixtures ... --out-dir ... --manifest ... --target-lufs -23

This script is scaffolding. It expects synthesis outputs as .npy files
produced by run_cpp.py / run_rs.py. It extracts excerpts, loudness normalizes
pairs to target LUFS using pyloudnorm, writes WAV files, and updates manifest.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

def load_synthesis(npy_path):
    arr = np.load(npy_path)
    return arr.astype(np.float64)

def loudness_normalize(signal, fs, target_lufs):
    if pyln is None:
        raise RuntimeError("pyloudnorm not installed")
    meter = pyln.Meter(fs)
    loudness = meter.integrated_loudness(signal)
    normalize = pyln.normalize.loudness(signal, loudness, target_lufs)
    return normalize

def extract_excerpt(signal, fs, start_s, end_s):
    start = int(start_s * fs)
    end = int(end_s * fs)
    return signal[start:end]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpp-dir', required=True)
    parser.add_argument('--rs-dir', required=True)
    parser.add_argument('--fixtures', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--target-lufs', type=float, default=-23.0)
    parser.add_argument('--excerpt-duration', type=float, default=6.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir = out_dir / 'stimuli'
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    fs = 16000

    for fixture in args.fixtures:
        # Find synthesis y.npy files
        cpp_pattern = f"{fixture}.synthesis.y.npy"
        rs_pattern = f"{fixture}.synthesis.y.npy"
        # Simple search
        cpp_path = None
        rs_path = None
        for root, _, files in os.walk(args.cpp_dir):
            if cpp_pattern in files:
                cpp_path = os.path.join(root, cpp_pattern)
                break
        for root, _, files in os.walk(args.rs_dir):
            if rs_pattern in files:
                rs_path = os.path.join(root, rs_pattern)
                break
        if not cpp_path or not rs_path:
            print(f"Warning: missing outputs for {fixture}")
            continue

        cpp_sig = load_synthesis(cpp_path)
        rs_sig = load_synthesis(rs_path)

        # Pick excerpt start: simple heuristic, avoid start
        duration = len(cpp_sig) / fs
        start_s = max(0.5, (duration - args.excerpt_duration) / 2)
        end_s = start_s + args.excerpt_duration
        if end_s > duration:
            end_s = duration
            start_s = end_s - args.excerpt_duration

        cpp_excerpt = extract_excerpt(cpp_sig, fs, start_s, end_s)
        rs_excerpt = extract_excerpt(rs_sig, fs, start_s, end_s)

        # Loudness normalize
        if pyln is not None:
            cpp_norm = loudness_normalize(cpp_excerpt, fs, args.target_lufs)
            rs_norm = loudness_normalize(rs_excerpt, fs, args.target_lufs)
        else:
            cpp_norm = cpp_excerpt
            rs_norm = rs_excerpt

        pair_id = f"{fixture}_01"
        cpp_wav = stimuli_dir / f"{pair_id}_cpp.wav"
        rs_wav = stimuli_dir / f"{pair_id}_rs.wav"
        sf.write(cpp_wav, cpp_norm, fs)
        sf.write(rs_wav, rs_norm, fs)

        manifest.append({
            "pair_id": pair_id,
            "fixture": fixture,
            "excerpt_start_s": start_s,
            "excerpt_end_s": end_s,
            "cpp_path": cpp_path,
            "rs_path": rs_path,
            "stimulus_cpp_wav": str(cpp_wav),
            "stimulus_rs_wav": str(rs_wav),
            "loudness_lufs_cpp": args.target_lufs,
            "loudness_lufs_rs": args.target_lufs,
            "duration_s": args.excerpt_duration
        })

    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} pairs to {args.manifest}")

if __name__ == '__main__':
    main()
