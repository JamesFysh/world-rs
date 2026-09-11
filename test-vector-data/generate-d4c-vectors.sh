#!/usr/bin/env bash
# Generate D4C reference vectors
# Per EQ-TASK-15 §3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
WORLD="$WORKSPACE/ext_src/world-cpp"
WORLD_SRC="$WORLD/src"
WORLD_BUILD="$WORLD/build"
OUT="$ROOT/vectors/d4c"

mkdir -p "$OUT"

# Build the C++ library if it is missing.
if [ ! -f "$WORLD_BUILD/libworld.a" ]; then
  echo "libworld.a not found; building C++ reference library..."
  (cd "$WORLD" && make)
fi

echo "=== D4C reference vector generation ==="
BIN="$WORKSPACE/scratch/d4c_reference"
g++ -O2 -I"$WORLD" -I"$WORLD_SRC" -o "$BIN" "$ROOT/d4c_reference.cpp" \
  "$WORLD_BUILD/libworld.a" "$WORLD/tools/audioio.cpp" -lm
"$BIN" "$OUT"

echo "=== D4C reference vectors written to $OUT ==="
ls -lh "$OUT" | sort

