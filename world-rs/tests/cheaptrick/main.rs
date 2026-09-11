//! CheapTrick module integration tests (Phase 2-1: scaffolding & type
//! definitions).

use world_rs_core::cheaptrick::{
    get_f0_floor_for_cheaptrick, get_fft_size_for_cheaptrick, initialize_cheaptrick_option,
    CheapTrickOption,
};

#[test]
fn cheaptrick_option_defaults_match_cpp_reference() {
    // C++: InitializeCheapTrickOption (cheaptrick.cpp:231-240)
    let option = initialize_cheaptrick_option(16000.0);
    assert_eq!(option.q1, -0.15);
    assert_eq!(option.f0_floor, 71.0);
    assert_eq!(option.fft_size, 1024);
}

#[test]
fn cheaptrick_option_fields_match_cpp_order() {
    let option = CheapTrickOption {
        q1: 1.0,
        f0_floor: 2.0,
        fft_size: 3,
    };
    assert_eq!(
        option,
        CheapTrickOption {
            q1: 1.0,
            f0_floor: 2.0,
            fft_size: 3,
        }
    );
}

#[test]
fn get_fft_size_for_cheaptrick_matches_cpp_formula() {
    // C++: GetFFTSizeForCheapTrick (cheaptrick.cpp:191-194)
    let option = initialize_cheaptrick_option(16000.0);
    assert_eq!(get_fft_size_for_cheaptrick(16000.0, &option), 1024);
    assert_eq!(get_fft_size_for_cheaptrick(48000.0, &option), 2048);
    let size = get_fft_size_for_cheaptrick(16000.0, &option);
    assert!(size > 0 && (size & (size - 1)) == 0);
}

#[test]
fn get_f0_floor_for_cheaptrick_matches_cpp_formula() {
    // C++: GetF0FloorForCheapTrick (cheaptrick.cpp:196-198)
    let f0 = get_f0_floor_for_cheaptrick(16000.0, 1024);
    assert!((f0 - 3.0 * 16000.0 / (1024.0 - 3.0)).abs() < 1e-9);
    assert!(f0 > 0.0);
}
