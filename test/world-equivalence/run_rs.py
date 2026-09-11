import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Build profile: release (cargo build --release) — matches accuracy tests for FP parity

PHASES = ["dio", "stonemask", "cheaptrick", "synthesis"]
BINARY_NAME = "equiv_dump"
CARGO_PROJECT = Path(__file__).resolve().parents[2]
EXAMPLE_BIN = CARGO_PROJECT / "target" / "release" / "examples" / BINARY_NAME

def build_helper():
    print("Building world-rs equiv_dump...", file=sys.stderr)
    cmd = [
        "cargo", "build", "--release",
        "-p", "world-rs",
        "--example", BINARY_NAME
    ]
    subprocess.check_call(cmd, cwd=str(CARGO_PROJECT))

def run_phase(wav_path, phase):
    try:
        result = subprocess.run(
            [str(EXAMPLE_BIN), str(wav_path), phase],
            capture_output=True, text=True, timeout=60
        )
    except Exception as e:
        return {"error": f"run_error: {e}"}
    if result.returncode != 0:
        return {"error": f"nonzero_exit: {result.stderr}"}
    try:
        data = json.loads(result.stdout)
    except Exception as e:
        return {"error": f"json_parse_error: {e}"}
    return data

def write_npy(output_dir, fixture_name, phase, data):
    os.makedirs(output_dir, exist_ok=True)
    if "error" in data:
        return False
    if phase == "dio":
        f0 = np.array(data["f0"], dtype=np.float64)
        t = np.array(data["t"], dtype=np.float64)
        np.save(os.path.join(output_dir, f"{fixture_name}.dio.f0.npy"), f0)
        np.save(os.path.join(output_dir, f"{fixture_name}.dio.t.npy"), t)
    elif phase == "stonemask":
        f0 = np.array(data["f0"], dtype=np.float64)
        np.save(os.path.join(output_dir, f"{fixture_name}.stonemask.f0.npy"), f0)
    elif phase == "cheaptrick":
        sp = np.array(data["sp"], dtype=np.float64)
        np.save(os.path.join(output_dir, f"{fixture_name}.cheaptrick.sp.npy"), sp)
    elif phase == "synthesis":
        y = np.array(data["y"], dtype=np.float64)
        np.save(os.path.join(output_dir, f"{fixture_name}.synthesis.y.npy"), y)
    else:
        return False
    return True

def write_sidecar(output_dir, results):
    sidecar_path = os.path.join(output_dir, "results.json")
    with open(sidecar_path, "w") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_helper()

    results = {}
    for wav_path in sorted(fixtures_dir.glob("*.wav")):
        fixture_name = wav_path.stem
        for phase in PHASES:
            key = f"{fixture_name}.{phase}"
            data = run_phase(wav_path, phase)
            if "error" in data:
                results[key] = data["error"]
                # never abort, continue
                continue
            ok = write_npy(str(output_dir), fixture_name, phase, data)
            if ok:
                results[key] = "ok"
            else:
                results[key] = "write_failed"
    write_sidecar(str(output_dir), results)
    print(f"Done. Results written to {output_dir}")

if __name__ == "__main__":
    main()
