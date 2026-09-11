use js_sys::Float64Array;
use wasm_bindgen_test::*;
use world_rs_wasm::harvest::{harvest, initialize_harvest_option};

fn sine(n: usize, hz: f64, fs: f64, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * hz * i as f64 / fs).sin())
        .collect()
}

#[wasm_bindgen_test]
fn harvest_smoke_short_buffer() {
    let fs = 16000.0;
    let x = sine(16000, 200.0, fs, 0.5);
    let option = initialize_harvest_option();
    let result = harvest(Float64Array::from(x.as_slice()), fs, &option)
        .expect("harvest should succeed on short buffer");
    let f0 = result.f0();
    let tpos = result.temporal_positions();
    assert!(f0.length() > 0);
    assert_eq!(f0.length(), tpos.length());
}
