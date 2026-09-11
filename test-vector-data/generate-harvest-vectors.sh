#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
WORLD="$WORKSPACE/ext_src/world-cpp"
WORLD_SRC="$WORLD/src"
WORLD_BUILD="$WORLD/build"
OUT="$ROOT/vectors/harvest"
mkdir -p "$OUT"

if [ ! -f "$WORLD_BUILD/libworld.a" ]; then
  echo "libworld.a not found; building C++ reference library..."
  (cd "$WORLD" && make)
fi

BIN="$WORKSPACE/scratch/harvest_reference"
g++ -O2 -I"$WORLD_SRC" -o "$BIN" "$ROOT/harvest_reference.cpp" \
  "$WORLD_BUILD/libworld.a" -lm
"$BIN" "$OUT"

echo "=== Harvest reference vectors written to $OUT ==="
ls -1 "$OUT" | sort
