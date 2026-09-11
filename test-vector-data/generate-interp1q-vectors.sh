#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
WORLD="$WORKSPACE/ext_src/world-cpp"
WORLD_SRC="$WORLD/src"
WORLD_BUILD="$WORLD/build"
OUT="$ROOT/vectors/matlab"
mkdir -p "$OUT"
if [ ! -f "$WORLD_BUILD/libworld.a" ]; then
  echo "libworld.a not found; building C++ reference library..."
  (cd "$WORLD" && make)
fi
BIN="$WORKSPACE/scratch/interp1q_reference"
g++ -O2 -I"$WORLD_SRC" -o "$BIN" "$ROOT/interp1q_reference.cpp" "$WORLD_BUILD/libworld.a" -lm
"$BIN" "$OUT"
echo "=== interp1q vectors written to $OUT ==="
ls -1 "$OUT"
