//! WASM bindings for polyphase sample-rate conversion (Phase 5-4).
//!
//! Wraps `world_rs::resample::resample_polyphase` for JavaScript. Per the
//! locked precision contract (`WORLD-decisions.md`), the Resample I/O crosses
//! the WASM boundary as `Float32Array` (f32) — the documented deviation from
//! the f64 WORLD core, matching the JS reference.
//!
//! JS-facing API (Rust names are preserved by `wasm-bindgen`):
//! - `resample(x, from, to)` — converts a mono PCM signal between two positive
//!   integer sample rates and returns a `Float32Array`.

use wasm_bindgen::prelude::*;

use js_sys::Float32Array;
use world_rs::resample as core_resample;

/// Polyphase sample-rate conversion.
///
/// JS signature: `resample(x: Float32Array, from: number, to: number) =>
/// Float32Array`.
///
/// `x` is the mono PCM input (f32 samples), `from` the input sample rate in Hz,
/// and `to` the output sample rate in Hz. Both rates are positive integers; a
/// zero rate is rejected by returning `Err(JsValue)` with a plain string message
/// (catchable, not a `RangeError` instance). Returns the resampled
/// buffer as a `Float32Array` of length `round(x.length * to / from)`; when
/// `from == to` the input is returned unchanged.
#[wasm_bindgen(skip_jsdoc)]
pub fn resample(x: Float32Array, from: u32, to: u32) -> Result<Float32Array, JsValue> {
    if from == 0 || to == 0 {
        return Err(JsValue::from_str(
            "resample: from/to must be positive sample rates",
        ));
    }
    let x: Vec<f32> = x.to_vec();
    let y = core_resample::resample_polyphase(&x, from, to);
    Ok(Float32Array::from(y.as_slice()))
}

#[cfg(test)]
mod tests {
    // The binding crosses the WASM boundary with a `js_sys::Float32Array`,
    // whose element accessors (`length`/`to_vec`) only exist for the wasm
    // target, so these tests run under `wasm-pack test` (wasm32) and are
    // excluded from native `cargo test`.
    #[cfg(target_arch = "wasm32")]
    mod wasm {
        use crate::resample::resample;
        use js_sys::Float32Array;

        /// Identity case (`from == to`): the input is returned unchanged.
        #[test]
        fn test_resample_identity() {
            let x: Vec<f32> = vec![0.25, -0.5, 1.0, 0.0, -0.75];
            let out = resample(Float32Array::from(x.as_slice()), 16000, 16000).unwrap();
            let got: Vec<f32> = out.to_vec();
            assert_eq!(got, x);
        }

        /// 16k→48k is a 3:1 upsample: the output length is `input * 3`.
        #[test]
        fn test_resample_output_length() {
            let x: Vec<f32> = vec![0.0; 1000];
            let up = resample(Float32Array::from(x.as_slice()), 16000, 48000).unwrap();
            assert_eq!(up.length() as usize, 3000);
            // 48k→16k is a 1:3 downsample: the output length is `round(input / 3)`.
            let down = resample(Float32Array::from(x.as_slice()), 48000, 16000).unwrap();
            assert_eq!(down.length() as usize, (1000.0_f64 / 3.0).round() as usize);
        }

        /// Zero-rate contract: a zero `from` rate returns an error.
        #[test]
        fn test_resample_zero_from_err() {
            let x: Vec<f32> = vec![1.0, 2.0];
            let res = resample(Float32Array::from(x.as_slice()), 0, 16000);
            assert!(res.is_err());
            assert_eq!(
                res.unwrap_err().as_string().unwrap(),
                "resample: from/to must be positive sample rates"
            );
        }

        /// Zero-rate contract: a zero `to` rate returns an error.
        #[test]
        fn test_resample_zero_to_err() {
            let x: Vec<f32> = vec![1.0, 2.0];
            let res = resample(Float32Array::from(x.as_slice()), 16000, 0);
            assert!(res.is_err());
            assert_eq!(
                res.unwrap_err().as_string().unwrap(),
                "resample: from/to must be positive sample rates"
            );
        }
    }
}
