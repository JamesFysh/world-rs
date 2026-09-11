# AGENTS.md — world-rs

## Build / Test commands

- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --check`
- `wasm-pack test --node world-rs-wasm`
- `cargo bench --workspace -- --test` (smoke)
- Python harness entry: `test/world-equivalence/` with `pyproject.toml`

All commands are verified to run successfully.

## Precision contract

- WORLD core math: `f64` throughout
- Resample I/O: `f32` at WASM boundary
- WASM boundary: WORLD APIs as `Float64Array`, Resample as `Float32Array`

See [`docs/precision-contract.md`](docs/precision-contract.md).

## Linting policy

Workspace lints:

- `unsafe_code = "forbid"`
- `panic = "deny"`
- `unwrap_used = "deny"`

Test-only allow pattern in lib files:

```rust
#![cfg_attr(test, allow(clippy::panic, clippy::unwrap_used))]
```

## Vector workflow

- Generators in `test-vector-data/`
- Golden set committed
- Regeneration instructions in `test/world-equivalence/`
- C++ reference via URL, no submodule
- Pinned rev `d625e76` from `equivalence.yml`

## Doc-comment hygiene

- No PHASE tags
- No `docs/impl` links
- Normative content lives in `docs/precision-contract.md`, `docs/resample-placement.md`, `docs/world-divergences.md`, `docs/benchmarks/`. Link, do not duplicate.

## Key docs

- [`docs/precision-contract.md`](docs/precision-contract.md)
- [`docs/resample-placement.md`](docs/resample-placement.md)
- [`docs/world-divergences.md`](docs/world-divergences.md)
- [`docs/benchmarks/README.md`](docs/benchmarks/README.md)
