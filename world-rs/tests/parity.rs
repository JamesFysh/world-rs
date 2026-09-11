// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use world_rs::resample::{resample, MonoSignal};
use world_rs::synthesis::{
    constant_aperiodicity, get_y_length, synthesis, Aperiodicity, F0Contour, Spectrogram,
    SynthesisError,
};

#[test]
#[allow(clippy::type_complexity)]
fn test_synthesis_api() {
    fn _assert_synthesis_api(
        _syn: fn(
            &F0Contour,
            usize,
            &Spectrogram,
            &Aperiodicity,
            usize,
            f64,
            f64,
            usize,
        ) -> Result<Vec<f64>, SynthesisError>,
        _get_y_len: fn(usize, f64, f64) -> usize,
        _const_ap: fn(usize, usize) -> Vec<Vec<f64>>,
    ) {
    }
    _assert_synthesis_api(synthesis, get_y_length, constant_aperiodicity);
}

#[test]
fn test_resample_api() {
    fn _assert_resample_api(_resample: fn(&MonoSignal, u32, u32) -> Vec<f32>) {}
    _assert_resample_api(resample);
}

#[test]
fn test_synthesis_defaults_parity() {
    // Synthesis has no SynthesisOption in this vendored version; frame_period, fs, y_length are plain parameters.
    // Verify get_y_length matches C++ formula.
    let y_len = get_y_length(201, 5.0, 16000.0);
    assert_eq!(y_len, 16001);
    let ap = constant_aperiodicity(10, 1024);
    assert_eq!(ap.len(), 10);
    assert_eq!(ap[0].len(), 513);
}
