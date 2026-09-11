//! WASM bindings for Harvest F0 extraction.
//!
//! Wraps `world_rs_core::harvest` for JavaScript. Per the locked precision
//! contract (`WORLD-decisions.md`), the WORLD core APIs cross the WASM boundary
//! as `Float64Array` (f64).
//!
//! JS-facing API (Rust names are preserved by `wasm-bindgen`):
//! - `HarvestOption` — options struct (constructor + getters/setters).
//! - `initialize_harvest_option()` — returns the default options.
//! - `harvest(x, fs, option)` — runs Harvest and returns a `HarvestResult`.
//! - `HarvestResult` — `f0`, `temporal_positions`.

use wasm_bindgen::prelude::*;

use js_sys::Float64Array;
use world_rs::harvest as core_harvest;
use world_rs::harvest::HarvestOption as CoreHarvestOption;

/// WASM-facing Harvest options, mirroring `world_rs_core::harvest::HarvestOption`.
#[wasm_bindgen]
pub struct HarvestOption {
    f0_floor: f64,
    f0_ceil: f64,
    frame_period: f64,
}

#[wasm_bindgen]
impl HarvestOption {
    #[wasm_bindgen(constructor, skip_jsdoc)]
    pub fn new(f0_floor: f64, f0_ceil: f64, frame_period: f64) -> Self {
        Self {
            f0_floor,
            f0_ceil,
            frame_period,
        }
    }
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_floor(&self) -> f64 {
        self.f0_floor
    }
    #[wasm_bindgen(setter)]
    pub fn set_f0_floor(&mut self, v: f64) {
        self.f0_floor = v;
    }
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn f0_ceil(&self) -> f64 {
        self.f0_ceil
    }
    #[wasm_bindgen(setter)]
    pub fn set_f0_ceil(&mut self, v: f64) {
        self.f0_ceil = v;
    }
    #[wasm_bindgen(getter, skip_jsdoc)]
    pub fn frame_period(&self) -> f64 {
        self.frame_period
    }
    #[wasm_bindgen(setter)]
    pub fn set_frame_period(&mut self, v: f64) {
        self.frame_period = v;
    }
}

/// WASM-facing Harvest result, mirroring `world_rs_core::harvest::HarvestResult`.
#[wasm_bindgen]
pub struct HarvestResult {
    f0: Float64Array,
    temporal_positions: Float64Array,
}

#[wasm_bindgen]
impl HarvestResult {
    #[wasm_bindgen(getter)]
    pub fn f0(&self) -> Float64Array {
        self.f0.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn temporal_positions(&self) -> Float64Array {
        self.temporal_positions.clone()
    }
}

/// Returns the default Harvest options
/// (`world_rs_core::harvest::initialize_harvest_option`).
#[wasm_bindgen]
pub fn initialize_harvest_option() -> HarvestOption {
    let o = core_harvest::initialize_harvest_option();
    HarvestOption {
        f0_floor: o.f0_floor,
        f0_ceil: o.f0_ceil,
        frame_period: o.frame_period,
    }
}

/// Harvest F0 extraction.
///
/// JS signature: `harvest(x: Float64Array, fs: number, option: HarvestOption)
/// => HarvestResult`.
///
/// # Errors
///
/// Returns a JS exception (a `JsValue` string) when the input is invalid,
/// mirroring `world_rs_core::harvest::HarvestError`.
#[wasm_bindgen]
pub fn harvest(x: Float64Array, fs: f64, option: &HarvestOption) -> Result<HarvestResult, JsValue> {
    let slice: Vec<f64> = x.to_vec();
    let core_opt = CoreHarvestOption {
        f0_floor: option.f0_floor,
        f0_ceil: option.f0_ceil,
        frame_period: option.frame_period,
    };
    match core_harvest::harvest(&slice, fs, &core_opt) {
        Ok(res) => {
            let f0_arr = Float64Array::from(&res.f0[..]);
            let tpos_arr = Float64Array::from(&res.temporal_positions[..]);
            Ok(HarvestResult {
                f0: f0_arr,
                temporal_positions: tpos_arr,
            })
        }
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}
