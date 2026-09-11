//! WASM bindings for StoneMask F0 refinement (Phase 5-4).
//!
//! Wraps `world_rs_core::stonemask` for JavaScript. Per the locked precision
//! contract (`WORLD-decisions.md`), the WORLD core APIs cross the WASM boundary
//! as `Float64Array` (f64); only Resample I/O uses `Float32Array` (f32).
//!
//! JS-facing API (Rust names are preserved by `wasm-bindgen`):
//! - `stone_mask(x, fs, temporal_positions, f0)` — refines
//!   a DIO F0 contour and returns it as a `Float64Array`.

use wasm_bindgen::prelude::*;

use js_sys::Float64Array;
use world_rs::stonemask as core_stonemask;

/// StoneMask F0 refinement.
///
/// JS signature: `stone_mask(x: Float64Array, fs: number,
/// temporal_positions: Float64Array, f0: Float64Array) => Float64Array`.
///
/// `x` is the input signal (f64), `fs` the sampling frequency in Hz,
/// `temporal_positions` the frame time grid (seconds), `f0` the DIO F0 contour
/// in Hz. Returns the refined F0 contour as a `Float64Array`; frames outside the
/// guard (`f0 <= 40` or `f0 > fs/12`) are returned as `0.0`.
#[wasm_bindgen(skip_jsdoc)]
pub fn stone_mask(
    x: Float64Array,
    fs: f64,
    temporal_positions: Float64Array,
    f0: Float64Array,
) -> Result<Float64Array, JsValue> {
    let x_vec: Vec<f64> = x.to_vec();
    let temporal_positions_vec: Vec<f64> = temporal_positions.to_vec();
    let f0_vec: Vec<f64> = f0.to_vec();
    let x_len = x_vec.len();
    let refined = core_stonemask::stone_mask(
        &x_vec,
        x_len,
        fs,
        &temporal_positions_vec,
        &f0_vec,
        f0_vec.len(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(Float64Array::from(refined.as_slice()))
}
