import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DRIVER_SRC = Path(__file__).parent / "cpp_driver.cpp"
DRIVER_BIN = Path(__file__).parent / "cpp_driver"
WORLD_SRC = Path(__file__).parent.parent.parent / "ext_src" / "world-cpp" / "src"
WORLD_LIB = Path(__file__).parent.parent.parent / "ext_src" / "world-cpp" / "build" / "libworld.a"

PHASES = ["dio", "stonemask", "cheaptrick", "synthesis"]
TIMEOUT = 30

def build_driver():
    if DRIVER_BIN.exists():
        src_mtime = DRIVER_SRC.stat().st_mtime
        lib_mtime = WORLD_LIB.stat().st_mtime
        bin_mtime = DRIVER_BIN.stat().st_mtime
        if bin_mtime >= src_mtime and bin_mtime >= lib_mtime:
            return
    cmd = [
        "g++", "-O2", f"-I{WORLD_SRC}", "-o", str(DRIVER_BIN),
        str(DRIVER_SRC), str(WORLD_LIB), "-lm"
    ]
    print("Building C++ driver...", file=sys.stderr)
    subprocess.check_call(cmd)

def decode_wav(path):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data[:,0]
    data = data.astype(np.float64)
    return float(sr), data

def write_raw(path, fs, samples):
    with open(path, "wb") as f:
        n = np.uint32(len(samples))
        f.write(n.tobytes())
        f.write(np.float64(fs).tobytes())
        f.write(samples.tobytes())

def run_phase(driver_bin, phase, raw_path, out_path, timeout):
    try:
        start = time.time()
        subprocess.run([str(driver_bin), phase, str(raw_path), str(out_path)],
                       check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "ok", time.time()-start
    except subprocess.TimeoutExpired:
        return "timeout", None
    except subprocess.CalledProcessError as e:
        return "crashed", None

def convert_to_npy(fixture_name, phase, out_raw_path, output_dir):
    if not os.path.exists(out_raw_path):
        return
    if phase == "dio":
        with open(out_raw_path, "rb") as f:
            f0_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            frame_period = np.frombuffer(f.read(8), dtype=np.float64)[0]
            f0 = np.frombuffer(f.read(f0_len*8), dtype=np.float64).reshape(-1)
            t = np.frombuffer(f.read(f0_len*8), dtype=np.float64).reshape(-1)
        np.save(os.path.join(output_dir, f"{fixture_name}.dio.f0.npy"), f0)
        np.save(os.path.join(output_dir, f"{fixture_name}.dio.t.npy"), t)
    elif phase == "stonemask":
        with open(out_raw_path, "rb") as f:
            f0_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            f0 = np.frombuffer(f.read(f0_len*8), dtype=np.float64).reshape(-1)
        np.save(os.path.join(output_dir, f"{fixture_name}.stonemask.f0.npy"), f0)
    elif phase == "cheaptrick":
        with open(out_raw_path, "rb") as f:
            f0_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            fft_size = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            f0 = np.frombuffer(f.read(f0_len*8), dtype=np.float64)
            t = np.frombuffer(f.read(f0_len*8), dtype=np.float64)
            bins = fft_size//2 + 1
            sp = np.frombuffer(f.read(f0_len*bins*8), dtype=np.float64).reshape(f0_len, bins)
        np.save(os.path.join(output_dir, f"{fixture_name}.cheaptrick.sp.npy"), sp)
    elif phase == "synthesis":
        with open(out_raw_path, "rb") as f:
            y_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            y = np.frombuffer(f.read(y_len*8), dtype=np.float64).reshape(-1)
        np.save(os.path.join(output_dir, f"{fixture_name}.synthesis.y.npy"), y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fixtures_dir = Path(args.fixtures)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_driver()
    print("D4C/AP outputs deferred to EQ-TASK-15", file=sys.stderr)
    results = {}
    for wav_path in sorted(fixtures_dir.glob("*.wav")):
        fixture_name = wav_path.stem
        fs, samples = decode_wav(str(wav_path))
        with tempfile.TemporaryDirectory() as td:
            raw_path = Path(td) / "in.raw"
            write_raw(raw_path, fs, samples)
            for phase in PHASES:
                out_raw = Path(td) / f"out_{phase}.raw"
                status, _ = run_phase(DRIVER_BIN, phase, raw_path, out_raw, TIMEOUT)
                key = f"{fixture_name}.{phase}"
                results[key] = status
                if status == "ok":
                    convert_to_npy(fixture_name, phase, out_raw, output_dir)
                else:
                    print(f"Warning: {key} {status}", file=sys.stderr)
    sidecar = output_dir / "results.json"
    with open(sidecar, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Results written to {output_dir}")

if __name__ == "__main__":
    main()
