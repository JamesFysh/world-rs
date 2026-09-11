#![allow(clippy::approx_constant, clippy::excessive_precision)]

/// C++: const double kCutOff = 50.0; // for Dio()
pub const K_CUT_OFF: f64 = 50.0;

/// C++: const double kFloorF0StoneMask = 40.0; // for StoneMask()
pub const K_FLOOR_F0_STONE_MASK: f64 = 40.0;

/// C++: const double kPi = 3.1415926535897932384;
pub const K_PI: f64 = 3.1415926535897932384;

/// C++: const double kMySafeGuardMinimum = 0.000000000001;
pub const K_MY_SAFE_GUARD_MINIMUM: f64 = 0.000000000001;

/// C++: const double kEps = 0.00000000000000022204460492503131;
pub const K_EPS: f64 = 0.00000000000000022204460492503131;

/// C++: const double kFloorF0 = 71.0;
pub const K_FLOOR_F0: f64 = 71.0;

/// C++: const double kCeilF0 = 800.0;
pub const K_CEIL_F0: f64 = 800.0;

/// C++: const double kDefaultF0 = 500.0;
pub const K_DEFAULT_F0: f64 = 500.0;

/// C++: const double kLog2 = 0.69314718055994529;
pub const K_LOG2: f64 = 0.69314718055994529;

/// C++: const double kMaximumValue = 100000.0; // Maximum standard deviation not to be selected as a best f0.
pub const K_MAXIMUM_VALUE: f64 = 100000.0;

/// C++: const int kHanning = 1; // for D4C()
pub const K_HANNING: f64 = 1.0;

/// C++: const int kBlackman = 2; // for D4C()
pub const K_BLACKMAN: f64 = 2.0;

/// C++: const double kFrequencyInterval = 3000.0; // for D4C()
pub const K_FREQUENCY_INTERVAL: f64 = 3000.0;

/// C++: const double kUpperLimit = 15000.0; // for D4C()
pub const K_UPPER_LIMIT: f64 = 15000.0;

/// C++: const double kThreshold = 0.85; // for D4C()
pub const K_THRESHOLD: f64 = 0.85;

/// C++: const double kFloorF0D4C = 47.0; // for D4C()
pub const K_FLOOR_F0_D4C: f64 = 47.0;

/// C++: const double kSafeGuardD4C = 0.000001; // for D4C()
pub const K_SAFE_GUARD_D4C: f64 = 0.000001;

/// C++: const double kM0 = 1127.01048; // for Codec (Mel scale)
pub const K_M0: f64 = 1127.01048;

/// C++: const double kF0 = 700.0; // for Codec (Mel scale)
pub const K_F0: f64 = 700.0;

/// C++: const double kFloorFrequency = 40.0; // for Codec (Mel scale)
pub const K_FLOOR_FREQUENCY: f64 = 40.0;

/// C++: const double kCeilFrequency = 20000.0; // for Codec (Mel scale)
pub const K_CEIL_FREQUENCY: f64 = 20000.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_match_cpp_reference() {
        assert_eq!(K_CUT_OFF, 50.0);
        assert_eq!(K_FLOOR_F0_STONE_MASK, 40.0);
        assert_eq!(K_PI, 3.1415926535897932384);
        assert_eq!(K_MY_SAFE_GUARD_MINIMUM, 0.000000000001);
        assert_eq!(K_EPS, 0.00000000000000022204460492503131);
        assert_eq!(K_FLOOR_F0, 71.0);
        assert_eq!(K_CEIL_F0, 800.0);
        assert_eq!(K_DEFAULT_F0, 500.0);
        assert_eq!(K_LOG2, 0.69314718055994529);
        assert_eq!(K_MAXIMUM_VALUE, 100000.0);
        assert_eq!(K_HANNING, 1.0);
        assert_eq!(K_BLACKMAN, 2.0);
        assert_eq!(K_FREQUENCY_INTERVAL, 3000.0);
        assert_eq!(K_UPPER_LIMIT, 15000.0);
        assert_eq!(K_THRESHOLD, 0.85);
        assert_eq!(K_FLOOR_F0_D4C, 47.0);
        assert_eq!(K_SAFE_GUARD_D4C, 0.000001);
        assert_eq!(K_M0, 1127.01048);
        assert_eq!(K_F0, 700.0);
        assert_eq!(K_FLOOR_FREQUENCY, 40.0);
        assert_eq!(K_CEIL_FREQUENCY, 20000.0);
    }
}
