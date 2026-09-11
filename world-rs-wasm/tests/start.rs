//! WASM-runtime test for the module-init `start` binding (console panic hook).
//!
//! Run with `wasm-pack test --node`. The `start` function is auto-invoked by
//! `wasm-bindgen` at WASM module instantiation (before any test runs); this
//! test additionally calls it directly to confirm it is idempotent and safe to
//! re-invoke (it installs the console panic hook via `set_once`).

use wasm_bindgen_test::*;

use world_rs_wasm::start;

/// `start` installs the console panic hook once; re-invoking it is a no-op and
/// must not panic.
#[wasm_bindgen_test]
fn start_is_idempotent() {
    start();
    start();
}
