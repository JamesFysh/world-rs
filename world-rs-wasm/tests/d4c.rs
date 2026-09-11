use js_sys::Float64Array;
use wasm_bindgen_test::*;
use world_rs_wasm::d4c::{d4c, initialize_d4c_option};

#[wasm_bindgen_test]
fn d4c_smoke_short_buffer() {
    let x = Float64Array::from(&[0.0; 1024][..]);
    let fs = 16000.0;
    let tpos = Float64Array::from(&[0.0, 0.005, 0.01][..]);
    let f0 = Float64Array::from(&[200.0, 200.0, 0.0][..]);
    let fft_size = 1024;
    let opt = initialize_d4c_option();
    let ap = d4c(x, fs, tpos, f0, fft_size, &opt).expect("d4c should succeed");
    let len = ap.length() as usize;
    assert_eq!(len, 3 * (fft_size as usize / 2 + 1));
    // Ensure values are finite
    for i in 0..ap.length() {
        let v = ap.get_index(i);
        assert!(v.is_finite());
    }
}
