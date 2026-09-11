//! WASM bindings for the WORLD pitch extraction and synthesis pipeline.
//!
//! This crate wraps the pure-Rust `world-rs` (synthesis, resample) and
//! `world-rs-core` (DIO, CheapTrick, StoneMask) crates with `wasm-bindgen`
//! bindings so the TypeScript layer can call the WORLD pipeline directly.
//!
//! Precision contract (locked, `WORLD-decisions.md`): the WORLD core APIs
//! (DIO, CheapTrick, StoneMask, Synthesis) cross the WASM boundary as
//! `Float64Array` (f64), while the Resample I/O crosses as `Float32Array`
//! (f32), matching the JS reference.

#![cfg_attr(test, allow(clippy::panic, clippy::unwrap_used))]
use wasm_bindgen::prelude::*;

pub mod cheaptrick;
pub mod d4c;
pub mod dio;
pub mod harvest;
pub mod resample;
pub mod stonemask;
pub mod synthesis;

/// Install the console panic hook once, at module start, so any Rust panic is
/// surfaced to the JS console instead of silently trapping the WASM instance.
#[wasm_bindgen(start, skip_jsdoc)]
pub fn start() {
    console_error_panic_hook::set_once();
}
