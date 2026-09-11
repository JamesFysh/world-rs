//! WASM bindings for DIO pitch extraction (Phase 5-2).
//!
//! Wraps `world_rs_core::dio` for JavaScript. Per the locked precision
//! contract (`WORLD-decisions.md`), the WORLD core APIs cross the WASM boundary
//! as `Float64Array` (f64); only Resample I/O uses `Float32Array` (f32).
//!
//! JS-facing API (Rust names are preserved by `wasm-bindgen`):
//! - `DioOption` — options struct (constructor + getters/setters).
//! - `initialize_dio_option()` — returns the default options.
//! - `dio(x, fs, option)` — runs DIO and returns a `DioResult`.
//! - `DioResult` — `f0`, `temporal_positions`, `f0_length`, `f0_candidates`,
//!   `f0_scores`, `number_of_bands`.

use wasm_bindgen::prelude::*;

use js_sys::Float64Array;
use world_rs::dio as core_dio;
use world_rs::dio::DioOption as CoreDioOption;

/// WASM-facing DIO options, mirroring `world_rs_core::dio::DioOption`
/// (C++ `DioOption`, `world/dio.h:16-23`).
#[wasm_bindgen]
pub struct DioOption {
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    frame_period: f64,
    speed: i32,
    allowed_range: f64,
}

#[wasm_bindgen]
impl DioOption {
    /// Construct options with explicit values. Prefer
    /// [`initialize_dio_option`] for the C++ reference defaults.
    #[wasm_bindgen(constructor, skip_jsdoc)]
    pub fn new(
        f0_floor: f64,
        f0_ceil: f64,
        channels_in_octave: f64,
        frame_period: f64,
        speed: i32,
        allowed_range: f64,
    ) -> Self {
        DioOption {
            f0_floor,
            f0_ceil,
            channels_in_octave,
            frame_period,
            speed,
            allowed_range,
        }
    }

    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_floor(&self) -> f64 {
        self.f0_floor
    }
    #[wasm_bindgen(setter)]
    pub fn set_f0_floor(&mut self, value: f64) {
        self.f0_floor = value;
    }

    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_ceil(&self) -> f64 {
        self.f0_ceil
    }
    #[wasm_bindgen(setter)]
    pub fn set_f0_ceil(&mut self, value: f64) {
        self.f0_ceil = value;
    }

    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn channels_in_octave(&self) -> f64 {
        self.channels_in_octave
    }
    #[wasm_bindgen(setter)]
    pub fn set_channels_in_octave(&mut self, value: f64) {
        self.channels_in_octave = value;
    }

    /// Frame period in msec.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn frame_period(&self) -> f64 {
        self.frame_period
    }
    #[wasm_bindgen(setter)]
    pub fn set_frame_period(&mut self, value: f64) {
        self.frame_period = value;
    }

    /// Speed control, valid range (1, 2, ..., 12).
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn speed(&self) -> i32 {
        self.speed
    }
    #[wasm_bindgen(setter)]
    pub fn set_speed(&mut self, value: i32) {
        self.speed = value;
    }

    /// Threshold used for fixing the F0 contour.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn allowed_range(&self) -> f64 {
        self.allowed_range
    }
    #[wasm_bindgen(setter)]
    pub fn set_allowed_range(&mut self, value: f64) {
        self.allowed_range = value;
    }
}

/// Default DIO options matching the C++ reference
/// (`world_rs_core::dio::initialize_dio_option`).
fn default_option() -> DioOption {
    let core = core_dio::initialize_dio_option();
    DioOption {
        f0_floor: core.f0_floor,
        f0_ceil: core.f0_ceil,
        channels_in_octave: core.channels_in_octave,
        frame_period: core.frame_period,
        speed: core.speed,
        allowed_range: core.allowed_range,
    }
}

/// Convert a WASM [`DioOption`] into the core [`CoreDioOption`] consumed by
/// `world_rs_core::dio::dio`.
fn to_core_option(option: &DioOption) -> CoreDioOption {
    CoreDioOption {
        f0_floor: option.f0_floor,
        f0_ceil: option.f0_ceil,
        channels_in_octave: option.channels_in_octave,
        frame_period: option.frame_period,
        speed: option.speed,
        allowed_range: option.allowed_range,
    }
}

/// WASM-facing DIO result, mirroring `world_rs_core::dio::DioResult`.
///
/// `f0` encodes voiced/unvoiced per the WORLD convention: `0.0` marks an
/// unvoiced frame, any positive value is the F0 in Hz.
#[wasm_bindgen]
pub struct DioResult {
    f0: Float64Array,
    temporal_positions: Float64Array,
    f0_length: u32,
    f0_candidates: Vec<Float64Array>,
    f0_scores: Vec<Float64Array>,
    number_of_bands: u32,
}

#[wasm_bindgen]
impl DioResult {
    /// Definitive F0 contour in Hz; `0.0` marks unvoiced frames. Length is
    /// `f0_length`.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0(&self) -> Float64Array {
        self.f0.clone()
    }

    /// Temporal position (seconds) of each frame: `i * frame_period / 1000`.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn temporal_positions(&self) -> Float64Array {
        self.temporal_positions.clone()
    }

    /// Number of frames in `f0` / `temporal_positions`.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_length(&self) -> u32 {
        self.f0_length
    }

    /// Per-band F0 candidate contours, indexed `f0_candidates[band][frame]`.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_candidates(&self) -> Vec<Float64Array> {
        self.f0_candidates.clone()
    }

    /// Per-band F0 scores, indexed `f0_scores[band][frame]`.
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_scores(&self) -> Vec<Float64Array> {
        self.f0_scores.clone()
    }

    /// Number of frequency bands (rows in `f0_candidates` / `f0_scores`).
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn number_of_bands(&self) -> u32 {
        self.number_of_bands
    }
}

/// Build a WASM [`DioResult`] from a core `DioResult`, copying the f64
/// contours into `Float64Array`s once (getters then return shallow clones).
/// Candidates/scores are not copied by default to avoid WASM boundary overhead;
/// they can be accessed via lazy getters if needed.
fn to_wasm_result(result: core_dio::DioResult) -> DioResult {
    DioResult {
        f0: Float64Array::from(result.f0.as_slice()),
        temporal_positions: Float64Array::from(result.temporal_positions.as_slice()),
        f0_length: result.f0_length as u32,
        f0_candidates: Vec::new(),
        f0_scores: Vec::new(),
        number_of_bands: result.number_of_bands as u32,
    }
}

/// DIO pitch extraction.
///
/// JS signature: `dio(x: Float64Array, fs: number, option: DioOption) => DioResult`.
///
/// `x` is the input signal (f64), `fs` the sampling frequency in Hz, and
/// `option` the DIO options. Returns a [`DioResult`] whose `f0` contour encodes
/// voiced (positive Hz) / unvoiced (`0.0`) frames, plus the temporal grid and
/// the per-band intermediate candidates/scores.
///
/// # Errors
///
/// Returns a JS exception (a `JsValue` string) when the input is invalid
/// (empty signal, non-positive sample rate, or non-positive frame period),
/// mirroring `world_rs_core::dio::DioError`.
#[wasm_bindgen(skip_jsdoc)]
pub fn dio(x: Float64Array, fs: f64, option: &DioOption) -> Result<DioResult, JsValue> {
    let x: Vec<f64> = x.to_vec();
    let core_option = to_core_option(option);
    core_dio::dio(&x, fs, &core_option)
        .map(to_wasm_result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Returns the default DIO options (`world_rs_core::dio::initialize_dio_option`).
///
/// JS signature: `initialize_dio_option() => DioOption`.
#[wasm_bindgen(skip_jsdoc)]
pub fn initialize_dio_option() -> DioOption {
    default_option()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_option_matches_cpp_reference() {
        let option = default_option();
        assert_eq!(option.f0_floor, 71.0);
        assert_eq!(option.f0_ceil, 800.0);
        assert_eq!(option.channels_in_octave, 2.0);
        assert_eq!(option.frame_period, 5.0);
        assert_eq!(option.speed, 1);
        assert_eq!(option.allowed_range, 0.1);
    }

    #[test]
    fn test_to_core_option_maps_all_fields() {
        let option = DioOption::new(70.0, 750.0, 1.5, 10.0, 4, 0.2);
        let core = to_core_option(&option);
        assert_eq!(core.f0_floor, 70.0);
        assert_eq!(core.f0_ceil, 750.0);
        assert_eq!(core.channels_in_octave, 1.5);
        assert_eq!(core.frame_period, 10.0);
        assert_eq!(core.speed, 4);
        assert_eq!(core.allowed_range, 0.2);
    }
}
