//! WASM bindings for WORLD synthesis (Phase 5-4).
//!
//! Wraps `world_rs::synthesis` for JavaScript. Per the locked precision
//! contract (`WORLD-decisions.md`), the WORLD core APIs cross the WASM boundary
//! as `Float64Array` (f64); only Resample I/O uses `Float32Array` (f32).
//!
//! JS-facing API (Rust names are preserved by `wasm-bindgen`):
//! - `synthesis(f0, f0_length, spectrogram, aperiodicity, fft_size,
//!   frame_period_ms, fs, y_length)` — reconstructs the waveform and returns a
//!   `Float64Array` PCM buffer. `spectrogram`/`aperiodicity` are flat row-major
//!   `Float64Array`s.
//! - `get_y_length(f0_length, frame_period_ms, fs)` — the output length to
//!   pre-allocate before calling `synthesis`.
//! - `constant_aperiodicity(f0_length, fft_size)` — the constant-AP (0.5)
//!   aperiodicity matrix used while D4C is deferred.

use wasm_bindgen::prelude::*;

use js_sys::Float64Array;
use world_rs::synthesis as core_synthesis;

/// WORLD synthesis: reconstruct a waveform from an F0 contour, a CheapTrick
/// spectral envelope, and an aperiodicity spectrogram.
///
/// JS signature: `synthesis(f0: Float64Array, f0_length: number,
/// spectrogram: Float64Array, aperiodicity: Float64Array, fft_size: number,
/// frame_period_ms: number, fs: number, y_length: number) => Float64Array`.
///
/// `f0` is the F0 contour in Hz (`0.0` marks unvoiced frames), `f0_length` the
/// number of frames (must equal `f0.length`), `spectrogram` the CheapTrick
/// spectral envelope as a flat row-major `Float64Array` of length
/// `f0_length * (fft_size/2+1)`, `aperiodicity` the aperiodicity spectrogram of
/// the same flat shape (use [`constant_aperiodicity`] while D4C is deferred),
/// `fft_size` the FFT size (a power of two), `frame_period_ms` the temporal
/// period in msec, `fs` the sampling frequency in Hz, and `y_length` the output
/// length (see [`get_y_length`]). Returns the reconstructed waveform as a
/// `Float64Array`.
///
/// # Errors
///
/// Returns a JS exception (a `JsValue` string) when the input is invalid
/// (empty F0, non-positive sample rate or frame period, a non-power-of-two
/// `fft_size`, or a dimension mismatch), mirroring
/// `world_rs::synthesis::SynthesisError`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(skip_jsdoc)]
pub fn synthesis(
    f0: Float64Array,
    f0_length: u32,
    spectrogram: Float64Array,
    aperiodicity: Float64Array,
    fft_size: u32,
    frame_period_ms: f64,
    fs: f64,
    y_length: u32,
) -> Result<Float64Array, JsValue> {
    let f0_vec: Vec<f64> = f0.to_vec();
    let sp_flat: Vec<f64> = spectrogram.to_vec();
    let ap_flat: Vec<f64> = aperiodicity.to_vec();
    let rows = f0_length as usize;
    let cols = (fft_size as usize) / 2 + 1;
    if sp_flat.len() != rows * cols || ap_flat.len() != rows * cols {
        return Err(JsValue::from_str(
            "synthesis: spectrogram/aperiodicity length mismatch",
        ));
    }
    let spectrogram: Vec<Vec<f64>> = sp_flat.chunks(cols).map(|c| c.to_vec()).collect();
    let aperiodicity: Vec<Vec<f64>> = ap_flat.chunks(cols).map(|c| c.to_vec()).collect();
    core_synthesis::synthesis(
        &f0_vec,
        rows,
        &spectrogram,
        &aperiodicity,
        fft_size as usize,
        frame_period_ms,
        fs,
        y_length as usize,
    )
    .map(|y| Float64Array::from(y.as_slice()))
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Number of samples in the synthesised output for a given F0 contour length,
/// frame period (msec), and sample rate.
///
/// JS signature: `get_y_length(f0_length: number, frame_period_ms: number,
/// fs: number) => number`.
///
/// Convenience helper for pre-allocating the output buffer before calling
/// [`synthesis`]: `(f0_length - 1) * frame_period_ms / 1000 * fs + 1`.
#[wasm_bindgen(skip_jsdoc)]
pub fn get_y_length(f0_length: u32, frame_period_ms: f64, fs: f64) -> u32 {
    core_synthesis::get_y_length(f0_length as usize, frame_period_ms, fs) as u32
}

/// Constant-AP convenience: an `f0_length x (fft_size / 2 + 1)` aperiodicity
/// matrix filled with the default aperiodicity (0.5), returned as a flat
/// row-major `Float64Array`.
///
/// JS signature: `constant_aperiodicity(f0_length: number, fft_size: number)
/// => Float64Array`.
///
/// Used as the `aperiodicity` argument to [`synthesis`] while D4C is deferred
/// (locked decision "Constant-AP convention": full AP matrix retained,
/// convenience default constant AP = 0.5).
#[wasm_bindgen(skip_jsdoc)]
pub fn constant_aperiodicity(f0_length: u32, fft_size: u32) -> Float64Array {
    let rows = core_synthesis::constant_aperiodicity(f0_length as usize, fft_size as usize);
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Float64Array::from(flat.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_y_length_matches_core() {
        assert_eq!(get_y_length(10, 5.0, 16000.0), 721);
        assert_eq!(get_y_length(1, 5.0, 16000.0), 1);
        assert_eq!(get_y_length(201, 5.0, 16000.0), 16001);
    }

    // The bindings cross the WASM boundary with `js_sys` typed arrays, whose
    // element accessors (`length`/`to_vec`) only exist for the wasm target, so
    // these tests run under `wasm-pack test` (wasm32) and are excluded from
    // native `cargo test`.
    #[cfg(target_arch = "wasm32")]
    mod wasm {
        use super::*;

        /// Builds a valid `(f0, spectrogram, aperiodicity)` triple for
        /// `f0_length` frames and the given `fft_size`, with every spectral bin
        /// set to `1.0` and the constant-AP aperiodicity.
        fn valid_inputs(f0_length: usize, fft_size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let bins = fft_size / 2 + 1;
            let f0 = vec![100.0; f0_length];
            let spectrogram = vec![1.0; f0_length * bins];
            let aperiodicity = vec![core_synthesis::DEFAULT_APERIODICITY; f0_length * bins];
            (f0, spectrogram, aperiodicity)
        }

        #[test]
        fn test_constant_aperiodicity_shape_and_value() {
            let ap = constant_aperiodicity(5, 128);
            assert_eq!(ap.length() as usize, 5 * (128 / 2 + 1));
            for v in ap.to_vec() {
                assert_eq!(v, core_synthesis::DEFAULT_APERIODICITY);
            }
        }

        #[test]
        fn test_synthesis_returns_expected_length() {
            let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
            let fft_size = 128usize;
            let frame_period_ms = 5.0;
            let fs = 16000.0;
            let y_length = core_synthesis::get_y_length(10, frame_period_ms, fs);
            let y = synthesis(
                Float64Array::from(f0.as_slice()),
                10,
                Float64Array::from(spectrogram.as_slice()),
                Float64Array::from(aperiodicity.as_slice()),
                fft_size as u32,
                frame_period_ms,
                fs,
                y_length as u32,
            )
            .expect("synthesis should succeed");
            assert_eq!(y.length() as usize, y_length);
        }

        #[test]
        fn test_synthesis_invalid_fft_size_errors() {
            let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
            let result = synthesis(
                Float64Array::from(f0.as_slice()),
                10,
                Float64Array::from(spectrogram.as_slice()),
                Float64Array::from(aperiodicity.as_slice()),
                100, // not a power of two
                5.0,
                16000.0,
                81,
            );
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .as_string()
                .unwrap()
                .contains("power of two"));
        }
    }
}
