use js_sys::Float64Array;
use wasm_bindgen::prelude::*;
use world_rs::d4c as core_d4c;
use world_rs::d4c::D4COption as CoreD4COption;

#[wasm_bindgen]
pub struct D4COption {
    threshold: f64,
}

#[wasm_bindgen]
impl D4COption {
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f64) -> Self {
        D4COption { threshold }
    }

    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    #[wasm_bindgen(setter)]
    pub fn set_threshold(&mut self, v: f64) {
        self.threshold = v;
    }
}

fn to_core_option(opt: &D4COption) -> CoreD4COption {
    CoreD4COption {
        threshold: opt.threshold,
    }
}

#[wasm_bindgen]
pub fn initialize_d4c_option() -> D4COption {
    let core = core_d4c::initialize_d4c_option();
    D4COption {
        threshold: core.threshold,
    }
}

#[wasm_bindgen]
pub fn d4c(
    x: Float64Array,
    fs: f64,
    temporal_positions: Float64Array,
    f0: Float64Array,
    fft_size: i32,
    option: &D4COption,
) -> Result<Float64Array, JsValue> {
    let x_vec: Vec<f64> = x.to_vec();
    let tp_vec: Vec<f64> = temporal_positions.to_vec();
    let f0_vec: Vec<f64> = f0.to_vec();
    let core_opt = to_core_option(option);
    core_d4c::d4c(&x_vec, fs, &tp_vec, &f0_vec, fft_size, &core_opt)
        .map(|mat| {
            let flat: Vec<f64> = mat.into_iter().flatten().collect();
            Float64Array::from(flat.as_slice())
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
