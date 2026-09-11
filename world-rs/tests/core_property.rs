use proptest::prelude::*;
use world_rs_core::cheaptrick::{cheaptrick, initialize_cheaptrick_option, CheapTrickOption};
use world_rs_core::d4c::{d4c, initialize_d4c_option, D4COption};
use world_rs_core::dio::{dio, DioOption, DioResult};
use world_rs_core::stonemask::stone_mask;

fn finite_f64_strategy() -> impl Strategy<Value = f64> {
    0.01f64..1.0f64
}

fn finite_vec_f64(len: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(finite_f64_strategy(), len)
}

fn dio_option_strategy() -> impl Strategy<Value = DioOption> {
    (
        71.0f64..800.0,
        100.0f64..1000.0,
        1.0f64..64.0,
        1.0f64..20.0,
        1i32..12,
        0.01f64..1.0,
    )
        .prop_map(
            |(f0_floor, f0_ceil, channels, frame_period, speed, allowed_range)| DioOption {
                f0_floor,
                f0_ceil: f0_ceil.max(f0_floor + 1.0),
                channels_in_octave: channels,
                frame_period,
                speed,
                allowed_range,
            },
        )
}

proptest! {
    #[test]
    fn dio_no_nan_output(x in finite_vec_f64(1024), fs in 8000.0..48000.0, opt in dio_option_strategy()) {
        let res = dio(&x, fs, &opt);
        if let Ok(DioResult { f0, temporal_positions, .. }) = res {
            for &v in &f0 {
                prop_assert!(v.is_finite());
            }
            for &v in &temporal_positions {
                prop_assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn dio_no_panic_arbitrary(
        x in proptest::collection::vec(any::<f64>(), 0..1024),
        fs in any::<f64>(),
        f0_floor in any::<f64>(),
        f0_ceil in any::<f64>(),
        channels_in_octave in any::<f64>(),
        frame_period in any::<f64>(),
        speed in -1024i32..1024,
        allowed_range in any::<f64>(),
    ) {
        let opt = DioOption { f0_floor, f0_ceil, channels_in_octave, frame_period, speed, allowed_range };
        let result = std::panic::catch_unwind(|| dio(&x, fs, &opt));
        prop_assert!(result.is_ok(), "dio panicked");
    }
}

proptest! {
    #[test]
    fn cheaptrick_no_nan_output(
        x in finite_vec_f64(8192),
        fs in 8000.0..48000.0,
        f0_len in 1usize..32usize,
        power in 10u32..13,
    ) {
        let fft_size = 1i32 << power;
        let opt = CheapTrickOption { fft_size, ..initialize_cheaptrick_option(fs) };
        let temporal_positions: Vec<f64> = (0..f0_len).map(|i| i as f64 * 0.005).collect();
        let f0_vals = vec![200.0; f0_len];
        let res = cheaptrick(&x, fs, &temporal_positions, &f0_vals, &opt);
        if let Ok(sp) = res {
            for row in sp {
                for v in row {
                    prop_assert!(v.is_finite());
                }
            }
        }
    }

    #[test]
    fn cheaptrick_no_panic_arbitrary(
        x in proptest::collection::vec(any::<f64>(), 0..256),
        fs in any::<f64>(),
        temporal_positions in proptest::collection::vec(any::<f64>(), 0..16),
        f0 in proptest::collection::vec(any::<f64>(), 0..16),
        q1 in any::<f64>(),
        f0_floor in any::<f64>(),
        fft_size in -1024i32..1024,
    ) {
        let opt = CheapTrickOption { q1, f0_floor, fft_size };
        let result = std::panic::catch_unwind(|| cheaptrick(&x, fs, &temporal_positions, &f0, &opt));
        prop_assert!(result.is_ok(), "cheaptrick panicked");
    }
}

proptest! {
    #[test]
    fn stonemask_no_nan_output(
        x in finite_vec_f64(2048),
        fs in 8000.0..48000.0,
        f0_len in 1usize..32usize,
    ) {
        let x_len = x.len();
        let temporal_positions: Vec<f64> = (0..f0_len).map(|i| i as f64 * 0.005).collect();
        let f0: Vec<f64> = vec![200.0; f0_len];
        let res = stone_mask(&x, x_len, fs, &temporal_positions, &f0, f0_len);
        if let Ok(refined) = res {
            for v in refined {
                prop_assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn stonemask_no_panic_arbitrary(
        x in proptest::collection::vec(any::<f64>(), 0..256),
        x_len in 0usize..256,
        fs in any::<f64>(),
        temporal_positions in proptest::collection::vec(any::<f64>(), 0..16),
        f0 in proptest::collection::vec(any::<f64>(), 0..16),
        f0_len in 0usize..16,
    ) {
        let result = std::panic::catch_unwind(|| stone_mask(&x, x_len, fs, &temporal_positions, &f0, f0_len));
        prop_assert!(result.is_ok(), "stonemask panicked");
    }
}

proptest! {
    #[test]
    fn d4c_no_nan_output(
        x in finite_vec_f64(8192),
        fs in 8000.0..48000.0,
        f0_len in 1usize..32usize,
        power in 8u32..13,
    ) {
        let fft_size = 1i32 << power;
        let option = initialize_d4c_option();
        let temporal_positions: Vec<f64> = (0..f0_len).map(|i| i as f64 * 0.005).collect();
        let f0_vals = vec![200.0; f0_len];
        let res = d4c(&x, fs, &temporal_positions, &f0_vals, fft_size, &option);
        if let Ok(ap) = res {
            for row in ap {
                for v in row {
                    prop_assert!(v.is_finite());
                }
            }
        }
    }

    #[test]
    fn d4c_no_panic_arbitrary(
        x in proptest::collection::vec(any::<f64>(), 0..256),
        fs in any::<f64>(),
        temporal_positions in proptest::collection::vec(any::<f64>(), 0..16),
        f0 in proptest::collection::vec(any::<f64>(), 0..16),
        fft_size in -1024i32..1024,
        threshold in any::<f64>(),
    ) {
        let option = D4COption { threshold };
        let result = std::panic::catch_unwind(|| d4c(&x, fs, &temporal_positions, &f0, fft_size, &option));
        prop_assert!(result.is_ok(), "d4c panicked");
    }
}
