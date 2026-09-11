//! WASM bindings for CheapTrick spectral envelope extraction (Phase 5-3).
//!
//! Wraps `world_rs::cheaptrick` for JavaScript. Per the locked precision
//! contract (`WORLD-decisions.md`), the WORLD core APIs cross the WASM boundary
//! as `Float64Array` (f64); only Resample I/O uses `Float32Array` (f32).
//!
//! JS-facing API (Rust names are preserved by `wasm-bindgen`):
//! - `CheapTrickOption` — options struct (constructor + getters/setters).
//! - `initialize_cheaptrick_option(fs)` — returns the default options.
//! - `cheaptrick(x, fs, temporal_positions, f0, option)` — runs CheapTrick and
//!   returns the spectral envelope as a flat row-major `Float64Array`.

use wasm_bindgen::prelude::*;

use js_sys::Float64Array;
use world_rs::cheaptrick as core_cheaptrick;
use world_rs::cheaptrick::CheapTrickOption as CoreCheapTrickOption;

/// WASM-facing CheapTrick options, mirroring
/// `world_rs::cheaptrick::CheapTrickOption` (C++ `CheapTrickOption`,
/// `world/cheaptrick.h:16-20`).
#[wasm_bindgen]
pub struct CheapTrickOption {
    q1: f64,
    f0_floor: f64,
    fft_size: i32,
}

#[wasm_bindgen]
impl CheapTrickOption {
    /// Construct options with explicit values. Prefer
    /// [`initialize_cheaptrick_option`] for the C++ reference defaults.
    #[wasm_bindgen(constructor, skip_jsdoc)]
    pub fn new(q1: f64, f0_floor: f64, fft_size: i32) -> Self {
        CheapTrickOption {
            q1,
            f0_floor,
            fft_size,
        }
    }

    /// Cepstral smoothing/recovery parameter (default `-0.15`).
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn q1(&self) -> f64 {
        self.q1
    }
    #[wasm_bindgen(setter)]
    pub fn set_q1(&mut self, value: f64) {
        self.q1 = value;
    }

    /// Lower F0 limit (Hz) used to determine `fft_size`.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_floor(&self) -> f64 {
        self.f0_floor
    }
    #[wasm_bindgen(setter)]
    pub fn set_f0_floor(&mut self, value: f64) {
        self.f0_floor = value;
    }

    /// FFT size; each spectral envelope row has `fft_size / 2 + 1` bins.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn fft_size(&self) -> i32 {
        self.fft_size
    }
    #[wasm_bindgen(setter)]
    pub fn set_fft_size(&mut self, value: i32) {
        self.fft_size = value;
    }
}

/// Default CheapTrick options matching the C++ reference
/// (`world_rs::cheaptrick::initialize_cheaptrick_option`).
fn default_option(fs: f64) -> CheapTrickOption {
    let core = core_cheaptrick::initialize_cheaptrick_option(fs);
    CheapTrickOption {
        q1: core.q1,
        f0_floor: core.f0_floor,
        fft_size: core.fft_size,
    }
}

/// Convert a WASM [`CheapTrickOption`] into the core [`CoreCheapTrickOption`]
/// consumed by `world_rs::cheaptrick::cheaptrick`.
fn to_core_option(option: &CheapTrickOption) -> CoreCheapTrickOption {
    CoreCheapTrickOption {
        q1: option.q1,
        f0_floor: option.f0_floor,
        fft_size: option.fft_size,
    }
}

/// CheapTrick spectral envelope extraction.
///
/// JS signature: `cheaptrick(x: Float64Array, fs: number,
/// temporal_positions: Float64Array, f0: Float64Array,
/// option: CheapTrickOption) => Float64Array`.
///
/// `x` is the input signal (f64), `fs` the sampling frequency in Hz,
/// `temporal_positions` the frame time grid (seconds), `f0` the F0 contour in
/// Hz, and `option` the CheapTrick options. Returns the spectral envelope as a
/// flat row-major `Float64Array` of length `frames * (fft_size/2+1)`, where
/// `frames = f0.length`. The caller can reconstruct rows with the known
/// `fft_size`.
///
/// # Errors
///
/// Returns a JS exception (a `JsValue` string) when the input is invalid
/// (empty signal, non-positive sample rate, or a `f0`/`temporal_positions`
/// length mismatch), mirroring `world_rs::cheaptrick::CheapTrickError`.
#[wasm_bindgen(skip_jsdoc)]
pub fn cheaptrick(
    x: Float64Array,
    fs: f64,
    temporal_positions: Float64Array,
    f0: Float64Array,
    option: &CheapTrickOption,
) -> Result<Float64Array, JsValue> {
    let x: Vec<f64> = x.to_vec();
    let temporal_positions: Vec<f64> = temporal_positions.to_vec();
    let f0: Vec<f64> = f0.to_vec();
    let core_option = to_core_option(option);
    core_cheaptrick::cheaptrick(&x, fs, &temporal_positions, &f0, &core_option)
        .map(|spectrogram| {
            let flat: Vec<f64> = spectrogram.into_iter().flatten().collect();
            Float64Array::from(flat.as_slice())
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Returns the default CheapTrick options
/// (`world_rs::cheaptrick::initialize_cheaptrick_option`).
///
/// JS signature: `initialize_cheaptrick_option(fs: number) => CheapTrickOption`.
#[wasm_bindgen(skip_jsdoc)]
pub fn initialize_cheaptrick_option(fs: f64) -> CheapTrickOption {
    default_option(fs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_option_matches_cpp_reference() {
        let option = default_option(16000.0);
        assert_eq!(option.q1, -0.15);
        assert_eq!(option.f0_floor, 71.0);
        assert_eq!(option.fft_size, 1024);
    }

    #[test]
    fn test_to_core_option_maps_all_fields() {
        let option = CheapTrickOption::new(-0.2, 80.0, 2048);
        let core = to_core_option(&option);
        assert_eq!(core.q1, -0.2);
        assert_eq!(core.f0_floor, 80.0);
        assert_eq!(core.fft_size, 2048);
    }
}
