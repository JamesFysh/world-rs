#!/usr/bin/env bash
# Generate Resample reference vectors (Phase 4-8).
#
# Runs the JS reference driver (test-vector-data/resample_reference.mjs) against
# the `@audio/resample-polyphase` atom (ext_src/resample) to emit, for each
# deterministic resample case, the f32 input and the JS polyphase reference
# output as binary files. The Rust accuracy tests
# (crates/world-rs/tests/resample_accuracy.rs) read these files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/vectors/resample"

mkdir -p "$OUT"

echo "=== Resample reference vector generation ==="
node "$ROOT/resample_reference.mjs"

echo "=== Resample reference vectors written to $OUT ==="
ls -1 "$OUT" | sort
