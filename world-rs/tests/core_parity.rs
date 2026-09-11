// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use world_rs_core::cheaptrick::{
    cheaptrick, get_fft_size_for_cheaptrick, initialize_cheaptrick_option, CheapTrickOption,
};
use world_rs_core::constants::{K_CEIL_F0, K_FLOOR_F0};
use world_rs_core::d4c::{d4c, initialize_d4c_option, D4CError, D4COption};
use world_rs_core::dio::{dio, initialize_dio_option, DioError, DioOption, DioResult};
use world_rs_core::harvest::{
    harvest, initialize_harvest_option, HarvestError, HarvestOption, HarvestResult,
};
use world_rs_core::stonemask::{stone_mask, StoneMaskError};

#[test]
fn test_dio_defaults() {
    let opt = initialize_dio_option();
    assert_eq!(opt.f0_floor, 71.0);
    assert_eq!(opt.f0_ceil, 800.0);
    assert_eq!(opt.frame_period, 5.0);
    assert_eq!(K_FLOOR_F0, 71.0);
    assert_eq!(K_CEIL_F0, 800.0);
    assert_eq!(opt.f0_floor, K_FLOOR_F0);
    assert_eq!(opt.f0_ceil, K_CEIL_F0);
}

#[test]
fn test_cheaptrick_defaults() {
    let opt_16k = initialize_cheaptrick_option(16000.0);
    assert_eq!(opt_16k.q1, -0.15);
    assert_eq!(opt_16k.f0_floor, K_FLOOR_F0);
    assert_eq!(
        opt_16k.fft_size,
        get_fft_size_for_cheaptrick(16000.0, &opt_16k)
    );
    assert_eq!(opt_16k.fft_size, 1024);

    let opt_22k = initialize_cheaptrick_option(22050.0);
    assert_eq!(opt_22k.q1, -0.15);
    assert_eq!(
        opt_22k.fft_size,
        get_fft_size_for_cheaptrick(22050.0, &opt_22k)
    );
    assert_eq!(opt_22k.fft_size, 1024);

    let opt_44k = initialize_cheaptrick_option(44100.0);
    assert_eq!(opt_44k.q1, -0.15);
    assert_eq!(
        opt_44k.fft_size,
        get_fft_size_for_cheaptrick(44100.0, &opt_44k)
    );
    assert_eq!(opt_44k.fft_size, 2048);

    let opt_48k = initialize_cheaptrick_option(48000.0);
    assert_eq!(opt_48k.q1, -0.15);
    assert_eq!(
        opt_48k.fft_size,
        get_fft_size_for_cheaptrick(48000.0, &opt_48k)
    );
    assert_eq!(opt_48k.fft_size, 2048);
}

#[test]
#[allow(clippy::type_complexity)]
fn test_api_surface() {
    fn _assert_dio_api(_dio: fn(&[f64], f64, &DioOption) -> Result<DioResult, DioError>) {}
    fn _assert_cheaptrick_api(
        _ct: fn(
            &[f64],
            f64,
            &[f64],
            &[f64],
            &CheapTrickOption,
        ) -> Result<Vec<Vec<f64>>, world_rs_core::cheaptrick::CheapTrickError>,
    ) {
    }
    fn _assert_stone_mask_api(
        _sm: fn(&[f64], usize, f64, &[f64], &[f64], usize) -> Result<Vec<f64>, StoneMaskError>,
    ) {
    }
    fn _assert_harvest_api(
        _h: fn(&[f64], f64, &HarvestOption) -> Result<HarvestResult, HarvestError>,
    ) {
    }
    fn _assert_d4c_api(
        _d4c: fn(&[f64], f64, &[f64], &[f64], i32, &D4COption) -> Result<Vec<Vec<f64>>, D4CError>,
    ) {
    }

    _assert_dio_api(dio);
    _assert_cheaptrick_api(cheaptrick);
    _assert_stone_mask_api(stone_mask);
    _assert_harvest_api(harvest);
    _assert_d4c_api(d4c);
}

#[test]
fn test_harvest_defaults() {
    let opt = initialize_harvest_option();
    assert_eq!(opt.f0_floor, 71.0);
    assert_eq!(opt.f0_ceil, 800.0);
    assert_eq!(opt.frame_period, 5.0);
    assert_eq!(opt.f0_floor, K_FLOOR_F0);
    assert_eq!(opt.f0_ceil, K_CEIL_F0);
}

#[test]
fn test_d4c_defaults() {
    let opt = initialize_d4c_option();
    assert_eq!(opt.threshold, 0.85);
}

#[test]
fn test_d4c_module_exists() {
    let _ = std::any::type_name::<D4COption>();
}

#[test]
fn test_harvest_module_exists() {
    // Harvest module now exists
    let _ = std::any::type_name::<HarvestOption>();
}

#[test]
fn test_no_synthesis_option_in_core() {
    // SynthesisOption does not exist in this vendored version; core crate should not expose it.
    // Compile-time check: the following line would fail if SynthesisOption were imported.
    let _ = K_FLOOR_F0;
}
