// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use world_rs_core::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs_core::d4c::{d4c, initialize_d4c_option};
use world_rs_core::dio::{dio, initialize_dio_option};
use world_rs_core::harvest::{harvest, initialize_harvest_option};
use world_rs_core::stonemask::stone_mask;

#[test]
fn dio_f0_floor_too_low_no_panic() {
    let x = vec![0.0; 16000];
    let mut opt = initialize_dio_option();
    opt.f0_floor = 1e-20;
    let res = dio(&x, 16000.0, &opt);
    assert!(res.is_err(), "dio should error on tiny f0_floor");
}

#[test]
fn dio_huge_channels_in_octave_no_panic() {
    let x = vec![0.0; 16000];
    let mut opt = initialize_dio_option();
    opt.channels_in_octave = 1e18;
    let res = dio(&x, 16000.0, &opt);
    assert!(res.is_err(), "dio should error on huge channels_in_octave");
}

#[test]
fn dio_small_waveform_speed_gt_one_no_panic() {
    let x = vec![0.0; 8];
    let mut opt = initialize_dio_option();
    opt.speed = 2;
    let res = dio(&x, 16000.0, &opt);
    assert!(
        res.is_err() || res.is_ok(),
        "dio must not panic on tiny input"
    );
}

#[test]
fn cheaptrick_invalid_fft_size_no_panic() {
    let x = vec![0.0; 16000];
    let mut opt = initialize_cheaptrick_option(16000.0);
    opt.fft_size = 0;
    let res = cheaptrick(&x, 16000.0, &[0.0], &[0.0], &opt);
    assert!(res.is_err(), "cheaptrick should error on fft_size 0");
}

#[test]
fn cheaptrick_nonfinite_f0_no_panic() {
    let x = vec![0.0; 16000];
    let opt = initialize_cheaptrick_option(16000.0);
    let f0 = vec![f64::NAN];
    let res = cheaptrick(&x, 16000.0, &[0.0], &f0, &opt);
    assert!(res.is_err(), "cheaptrick should error on NaN f0");
}

#[test]
fn stone_mask_nonfinite_inputs_no_panic() {
    let x = vec![0.0; 16000];
    let f0 = vec![f64::NAN];
    let positions = vec![0.0];
    let res =
        std::panic::catch_unwind(|| stone_mask(&x, x.len(), f64::NAN, &positions, &f0, 1)).unwrap();
    assert!(
        res.is_err() || res.unwrap().iter().all(|v| v.is_finite()),
        "stone_mask must not panic on NaN"
    );
}

#[test]
fn cheaptrick_inf_f0_no_panic() {
    let x = vec![0.0; 16000];
    let opt = initialize_cheaptrick_option(16000.0);
    let f0 = vec![f64::INFINITY];
    let res = cheaptrick(&x, 16000.0, &[0.0], &f0, &opt);
    assert!(res.is_err(), "cheaptrick should error on inf f0");
}

#[test]
fn nan_property_no_nan_output() {
    let x: Vec<f64> = (0..16000).map(|i| (i as f64 * 0.001).sin()).collect();
    let opt = initialize_dio_option();
    let dio_res = dio(&x, 16000.0, &opt).expect("valid dio");
    let f0 = stone_mask(
        &x,
        x.len(),
        16000.0,
        &dio_res.temporal_positions,
        &dio_res.f0,
        dio_res.f0_length,
    )
    .unwrap();
    let ct_opt = initialize_cheaptrick_option(16000.0);
    let sp = cheaptrick(&x, 16000.0, &dio_res.temporal_positions, &f0, &ct_opt)
        .expect("valid cheaptrick");
    for row in sp {
        for v in row {
            assert!(v.is_finite(), "spectrogram contains non-finite");
        }
    }
}

#[test]
fn dio_degenerate_frame_period_rejected_before_alloc() {
    // OOM regression: `frame_period -> 0` plans ~1e15 frames; must fail with
    // TooManyFrames instead of attempting a multi-GB allocation.
    let x = vec![0.0; 16000];
    let mut opt = initialize_dio_option();
    opt.frame_period = 1e-12;
    let res = dio(&x, 16000.0, &opt);
    assert!(
        matches!(res, Err(world_rs_core::dio::DioError::TooManyFrames { .. })),
        "dio should reject degenerate frame plan, got {res:?}"
    );
}

#[test]
fn harvest_degenerate_frame_period_rejected_before_alloc() {
    let x = vec![0.0; 16000];
    let mut opt = initialize_harvest_option();
    opt.frame_period = 1e-12;
    let res = harvest(&x, 16000.0, &opt);
    assert!(
        matches!(
            res,
            Err(world_rs_core::harvest::HarvestError::TooManyFrames { .. })
        ),
        "harvest should reject degenerate frame plan, got {res:?}"
    );
}

#[test]
fn cheaptrick_giant_fft_size_rejected_before_alloc() {
    // OOM regression: uncapped `fft_size` scales FFT scratch + lifters.
    let x = vec![0.0; 16000];
    let mut opt = initialize_cheaptrick_option(16000.0);
    opt.fft_size = 1 << 21;
    let res = cheaptrick(&x, 16000.0, &[0.0], &[100.0], &opt);
    assert!(
        res.is_err(),
        "cheaptrick should reject giant fft_size, got {res:?}"
    );
}

#[test]
fn d4c_giant_fft_size_rejected_before_alloc() {
    let x = vec![0.0; 16000];
    let option = initialize_d4c_option();
    let res = d4c(&x, 16000.0, &[0.0], &[100.0], 1 << 21, &option);
    assert!(
        res.is_err(),
        "d4c should reject giant fft_size, got {res:?}"
    );
}

#[test]
fn d4c_degenerate_sample_rate_rejected_before_alloc() {
    // OOM regression (property-4ba1ff class): `fs = 1e308` saturates the
    // fs-derived `fft_size_d4c` cast to `usize::MAX`; `ForwardRealFFT::new`
    // would then attempt a tens-of-GB plan (39GB anon-rss observed).
    let x = vec![0.0; 256];
    let option = initialize_d4c_option();
    let res = d4c(&x, 1e308, &[0.0], &[100.0], 1024, &option);
    assert!(
        res.is_err(),
        "d4c should reject degenerate sample rate, got {res:?}"
    );
}

#[test]
fn stonemask_degenerate_fs_f0_ratio_returns_unvoiced() {
    // Review finding #1: `fs` near 1e9 with a low f0 planned ~600MB vecs
    // plus a 2^28 FFT per frame. Must return 0.0 (unvoiced) without a
    // multi-GB attempt.
    let x = vec![0.0; 256];
    let res = world_rs_core::stonemask::stone_mask(&x, x.len(), 1e9, &[0.0], &[40.0], 1);
    let refined = res.expect("stonemask should not error on degenerate ratio");
    assert_eq!(
        refined,
        vec![0.0],
        "degenerate plan should refine to unvoiced"
    );
}

#[test]
fn dc_correction_huge_f0_clamped_to_nyquist() {
    // Review finding #3: huge-but-finite f0 saturated the float->int cast
    // to `usize::MAX` and sized the fallback buffer accordingly.
    let input = vec![1.0; 513];
    let mut output = vec![0.0; 513];
    world_rs_core::common::dc_correction(&input, 1e308, 16000, 1024, &mut output);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn linear_smoothing_huge_width_clamped_to_nyquist() {
    let input = vec![1.0; 513];
    let mut output = vec![0.0; 513];
    world_rs_core::common::linear_smoothing(&input, 1e308, 16000, 1024, &mut output);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn harvest_degenerate_sample_rate_skips_refinement() {
    // Review finding: `harvest()` has no `fs` upper bound, so `fs ≈ 1e8`
    // planned ~160MB time vecs plus a 2^26 FFT per refine frame. The
    // per-frame budget must bail to unvoiced instead.
    let x = vec![0.1; 500_000];
    let mut opt = initialize_harvest_option();
    opt.frame_period = 5.0;
    let res = harvest(&x, 1e8, &opt);
    let r = res.expect("harvest should not error on degenerate rate");
    assert!(
        r.f0.iter().all(|&v| v == 0.0),
        "degenerate-rate frames should refine to unvoiced"
    );
}
