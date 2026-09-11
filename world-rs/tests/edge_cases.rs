// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use world_rs_core::d4c::{d4c, initialize_d4c_option};
use world_rs_core::dio::{dio, initialize_dio_option, DioError, DioOption};
use world_rs_core::harvest::{harvest, initialize_harvest_option, HarvestError};

fn default_option() -> DioOption {
    initialize_dio_option()
}

#[test]
fn test_silence() {
    let fs = 16000.0;
    let n = 16000;
    let x = vec![0.0; n];
    let opt = default_option();
    let res = dio(&x, fs, &opt);
    assert!(res.is_ok(), "silence should succeed");
    let result = res.unwrap();
    // All frames should be unvoiced (f0 = 0)
    assert!(result.f0.iter().all(|&v| v == 0.0));
}

#[test]
fn test_nan_samples() {
    let fs = 16000.0;
    let mut x = vec![0.0; 16000];
    x[100] = f64::NAN;
    let opt = default_option();
    let err = dio(&x, fs, &opt).unwrap_err();
    assert_eq!(err, DioError::NonFiniteInput);
}

#[test]
fn test_empty() {
    let fs = 16000.0;
    let x: Vec<f64> = vec![];
    let opt = default_option();
    let err = dio(&x, fs, &opt).unwrap_err();
    assert_eq!(err, DioError::EmptyInput);
}

#[test]
fn test_f0_floor_zero() {
    let fs = 16000.0;
    let x = vec![0.0; 16000];
    let mut opt = default_option();
    opt.f0_floor = 0.0;
    let err = dio(&x, fs, &opt).unwrap_err();
    assert_eq!(err, DioError::InvalidF0Range);
}

#[test]
fn test_negative_fs() {
    let fs = -16000.0;
    let x = vec![0.0; 16000];
    let opt = default_option();
    let err = dio(&x, fs, &opt).unwrap_err();
    assert_eq!(err, DioError::NonPositiveSampleRate { fs: -16000.0 });
}

#[test]
fn test_nan_fs() {
    let fs = f64::NAN;
    let x = vec![0.0; 16000];
    let opt = default_option();
    let err = dio(&x, fs, &opt).unwrap_err();
    match err {
        DioError::NonPositiveSampleRate { fs: v } => assert!(v.is_nan()),
        _ => panic!("expected NonPositiveSampleRate"),
    }
}

#[test]
fn test_nan_frame_period() {
    let fs = 16000.0;
    let x = vec![0.0; 16000];
    let mut opt = default_option();
    opt.frame_period = f64::NAN;
    let err = dio(&x, fs, &opt).unwrap_err();
    match err {
        DioError::NonPositiveFramePeriod { frame_period: v } => assert!(v.is_nan()),
        _ => panic!("expected NonPositiveFramePeriod"),
    }
}

#[test]
fn test_single_sample() {
    let fs = 16000.0;
    let x = vec![0.0];
    let opt = default_option();
    let err = dio(&x, fs, &opt).unwrap_err();
    assert_eq!(err, DioError::TooShortInput);
}

#[test]
fn test_clipping() {
    let fs = 16000.0;
    let n = 16000;
    let x = vec![1.0; n];
    let opt = default_option();
    let res = dio(&x, fs, &opt);
    assert!(res.is_ok(), "clipping input should not error");
}

#[test]
fn test_d4c_all_unvoiced() {
    let fs = 16000.0;
    let x = vec![0.0; 1024];
    let f0 = vec![0.0; 10];
    let tpos: Vec<f64> = (0..10).map(|i| i as f64 * 0.005).collect();
    let opt = initialize_d4c_option();
    let ap = d4c(&x, fs, &tpos, &f0, 1024, &opt).expect("d4c should succeed");
    let safe_guard = 1e-12;
    for frame in ap {
        for v in frame {
            assert!(
                (v - (1.0 - safe_guard)).abs() < 1e-12,
                "unvoiced frame should be ~1"
            );
        }
    }
}

#[test]
fn test_d4c_single_frame() {
    let fs = 16000.0;
    let x = vec![0.0; 1024];
    let f0 = vec![200.0];
    let tpos = vec![0.0];
    let opt = initialize_d4c_option();
    let ap = d4c(&x, fs, &tpos, &f0, 1024, &opt).expect("d4c should succeed");
    assert_eq!(ap.len(), 1);
    assert_eq!(ap[0].len(), 1024 / 2 + 1);
}

#[test]
fn test_d4c_low_fs_zero_aperiodicities() {
    // fs <= 6000 => number_of_aperiodicities == 0
    let fs = 6000.0;
    let x = vec![0.0; 1024];
    let f0 = vec![200.0; 5];
    let tpos: Vec<f64> = (0..5).map(|i| i as f64 * 0.005).collect();
    let opt = initialize_d4c_option();
    let ap = d4c(&x, fs, &tpos, &f0, 1024, &opt).expect("d4c should not panic for low fs");
    assert_eq!(ap.len(), 5);
    for frame in ap {
        assert_eq!(frame.len(), 1024 / 2 + 1);
    }
}

#[test]
fn test_d4c_lovetrain_low_fs_hazard() {
    // fs <= 15800, especially <= 7900, tests clamped semantics
    for &fs in &[15800.0, 12000.0, 8000.0, 7900.0] {
        let x = vec![0.0; 1024];
        let f0 = vec![200.0; 5];
        let tpos: Vec<f64> = (0..5).map(|i| i as f64 * 0.005).collect();
        let opt = initialize_d4c_option();
        let ap = d4c(&x, fs, &tpos, &f0, 1024, &opt).expect("d4c should not panic for low fs");
        assert_eq!(ap.len(), 5);
        for frame in &ap {
            assert_eq!(frame.len(), 1024 / 2 + 1);
            for &v in frame {
                assert!(v.is_finite(), "aperiodicity must be finite");
            }
        }
    }
}

#[test]
fn test_d4c_very_high_f0() {
    let fs = 16000.0;
    let x = vec![0.0; 1024];
    let f0 = vec![8000.0; 5];
    let tpos: Vec<f64> = (0..5).map(|i| i as f64 * 0.005).collect();
    let opt = initialize_d4c_option();
    let ap = d4c(&x, fs, &tpos, &f0, 1024, &opt).expect("d4c should handle high f0");
    assert_eq!(ap.len(), 5);
    for frame in ap {
        assert_eq!(frame.len(), 1024 / 2 + 1);
    }
}

#[test]
fn test_harvest_empty() {
    let fs = 16000.0;
    let x: Vec<f64> = vec![];
    let opt = initialize_harvest_option();
    let err = harvest(&x, fs, &opt).unwrap_err();
    assert_eq!(err, HarvestError::EmptyInput);
}

#[test]
fn test_harvest_nan_input() {
    let fs = 16000.0;
    let mut x = vec![0.0; 16000];
    x[100] = f64::NAN;
    let opt = initialize_harvest_option();
    let err = harvest(&x, fs, &opt).unwrap_err();
    assert_eq!(err, HarvestError::NonFiniteInput);
}

#[test]
fn test_harvest_inf_input() {
    let fs = 16000.0;
    let mut x = vec![0.0; 16000];
    x[100] = f64::INFINITY;
    let opt = initialize_harvest_option();
    let err = harvest(&x, fs, &opt).unwrap_err();
    assert_eq!(err, HarvestError::NonFiniteInput);
}

#[test]
fn test_harvest_fs_zero() {
    let fs = 0.0;
    let x = vec![0.0; 16000];
    let opt = initialize_harvest_option();
    let err = harvest(&x, fs, &opt).unwrap_err();
    assert_eq!(err, HarvestError::NonPositiveSampleRate { fs: 0.0 });
}

#[test]
fn test_harvest_frame_period_zero() {
    let fs = 16000.0;
    let x = vec![0.0; 16000];
    let mut opt = initialize_harvest_option();
    opt.frame_period = 0.0;
    let err = harvest(&x, fs, &opt).unwrap_err();
    assert_eq!(
        err,
        HarvestError::NonPositiveFramePeriod { frame_period: 0.0 }
    );
}

#[test]
fn test_harvest_f0_ceil_le_floor() {
    let fs = 16000.0;
    let x = vec![0.0; 16000];
    let mut opt = initialize_harvest_option();
    opt.f0_floor = 500.0;
    opt.f0_ceil = 400.0;
    let err = harvest(&x, fs, &opt).unwrap_err();
    assert_eq!(err, HarvestError::InvalidF0Range);
}

#[test]
fn test_harvest_too_short() {
    let fs = 16000.0;
    let x = vec![0.0; 10];
    let opt = initialize_harvest_option();
    let err = harvest(&x, fs, &opt).unwrap_err();
    assert_eq!(err, HarvestError::TooShortInput);
}

#[test]
fn test_harvest_frame_period_one_fast_path() {
    let fs = 16000.0;
    let x = vec![0.0; 16000];
    let mut opt = initialize_harvest_option();
    opt.frame_period = 1.0;
    let res = harvest(&x, fs, &opt);
    assert!(res.is_ok(), "frame_period=1.0 should succeed");
}

#[test]
fn test_harvest_fs_not_divisible_by_8000() {
    let fs = 11025.0;
    let x = vec![0.0; 22050];
    let opt = initialize_harvest_option();
    let res = harvest(&x, fs, &opt);
    assert!(res.is_ok(), "fs not divisible by 8000 should succeed");
}
