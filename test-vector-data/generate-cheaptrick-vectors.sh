#!/usr/bin/env bash
# Generate CheapTrick reference vectors (Phase 2-7).
#
# Builds the C++ reference harness (test-vector-data/cheaptrick_reference.cpp)
# against the prebuilt libworld.a and runs it to emit, for each deterministic
# test signal + F0 contour, the input signal and the C++ CheapTrick()
# spectrogram as binary files. The Rust accuracy tests read these files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
WORLD="$WORKSPACE/ext_src/world-cpp"
WORLD_SRC="$WORLD/src"
WORLD_BUILD="$WORLD/build"
OUT="$ROOT/vectors/cheaptrick"

mkdir -p "$OUT"

# Build the C++ library if it is missing.
if [ ! -f "$WORLD_BUILD/libworld.a" ]; then
  echo "libworld.a not found; building C++ reference library..."
  (cd "$WORLD" && make)
fi

echo "=== CheapTrick reference vector generation ==="
# Compile the harness into the git-ignored scratch dir so the committed
# vectors dir only holds the reference data.
BIN="$WORKSPACE/scratch/cheaptrick_reference"
g++ -O2 -I"$WORLD_SRC" -o "$BIN" "$ROOT/cheaptrick_reference.cpp" \
  "$WORLD_BUILD/libworld.a" -lm
"$BIN" "$OUT"

echo "=== CheapTrick reference vectors written to $OUT ==="
ls -1 "$OUT" | sort
