//! DIO module integration tests (Phase 1-1: scaffolding & type definitions).

use world_rs_core::dio::{dio, get_samples_for_dio, initialize_dio_option, DioError, DioOption};

#[test]
fn dio_option_defaults_match_cpp_reference() {
    let option = initialize_dio_option();
    assert_eq!(option.f0_floor, 71.0);
    assert_eq!(option.f0_ceil, 800.0);
    assert_eq!(option.channels_in_octave, 2.0);
    assert_eq!(option.frame_period, 5.0);
    assert_eq!(option.speed, 1);
    assert_eq!(option.allowed_range, 0.1);
}

#[test]
fn dio_option_fields_match_cpp_order() {
    let option = DioOption {
        f0_floor: 1.0,
        f0_ceil: 2.0,
        channels_in_octave: 3.0,
        frame_period: 4.0,
        speed: 5,
        allowed_range: 6.0,
    };
    assert_eq!(
        option,
        DioOption {
            f0_floor: 1.0,
            f0_ceil: 2.0,
            channels_in_octave: 3.0,
            frame_period: 4.0,
            speed: 5,
            allowed_range: 6.0,
        }
    );
}

#[test]
fn get_samples_for_dio_matches_cpp_formula() {
    // C++: static_cast<int>(1000.0 * x_length / fs / frame_period) + 1
    assert_eq!(get_samples_for_dio(16000.0, 16000, 5.0), 201);
    assert_eq!(get_samples_for_dio(44100.0, 44100, 5.0), 201);
    assert_eq!(get_samples_for_dio(48000.0, 960000, 10.0), 2001);
    assert_eq!(get_samples_for_dio(16000.0, 16001, 5.0), 201);
    assert_eq!(get_samples_for_dio(16000.0, 79, 5.0), 1);
    assert_eq!(get_samples_for_dio(16000.0, 0, 5.0), 1);
}

// Invalid inputs are reported via the public `DioError` type (graceful, no
// panic) and the type is constructible/observable from outside the crate.
#[test]
fn dio_invalid_input_returns_public_error() {
    let option = initialize_dio_option();
    assert_eq!(dio(&[], 16000.0, &option), Err(DioError::EmptyInput));
    assert_eq!(
        dio(&[0.0; 100], 0.0, &option),
        Err(DioError::NonPositiveSampleRate { fs: 0.0 })
    );
    assert_eq!(
        dio(&[0.0; 100], -44100.0, &option),
        Err(DioError::NonPositiveSampleRate { fs: -44100.0 })
    );
    let mut opt = initialize_dio_option();
    opt.frame_period = -1.0;
    assert_eq!(
        dio(&[0.0; 100], 16000.0, &opt),
        Err(DioError::NonPositiveFramePeriod { frame_period: -1.0 })
    );
}
