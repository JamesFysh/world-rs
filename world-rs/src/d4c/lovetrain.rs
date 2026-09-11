use crate::constants::{K_BLACKMAN, K_LOG2};
use crate::d4c::window::get_windowed_waveform;
use crate::fft::ForwardRealFFT;
use crate::matlab::{matlab_round, RandnState};

pub(crate) fn d4c_love_train_sub(
    x: &[f64],
    fs: i32,
    x_length: usize,
    current_f0: f64,
    current_position: f64,
    fft_size: usize,
    boundary0: usize,
    boundary1: usize,
    boundary2: usize,
    forward_real_fft: &mut ForwardRealFFT,
    randn_state: &mut RandnState,
) -> f64 {
    let window_length = matlab_round(1.5 * fs as f64 / current_f0) as usize * 2 + 1;
    for i in 0..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    get_windowed_waveform(
        x,
        x_length,
        fs,
        current_f0,
        current_position,
        K_BLACKMAN,
        3.0,
        &mut forward_real_fft.waveform,
        randn_state,
    );
    for i in window_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    forward_real_fft.forward();
    let half = fft_size / 2 + 1;
    let mut power_spectrum = vec![0.0f64; fft_size];
    for slot in power_spectrum.iter_mut().take(boundary0.min(half - 1) + 1) {
        *slot = 0.0;
    }
    for (i, slot) in power_spectrum
        .iter_mut()
        .enumerate()
        .take(half)
        .skip(boundary0 + 1)
    {
        let re = forward_real_fft.spectrum[i].re;
        let im = forward_real_fft.spectrum[i].im;
        *slot = re * re + im * im;
    }
    let max_idx = boundary2.min(power_spectrum.len() - 1);
    for i in boundary0..=max_idx {
        if i > 0 {
            power_spectrum[i] += power_spectrum[i - 1];
        }
    }
    let b1 = boundary1.min(power_spectrum.len() - 1);
    let b2 = boundary2.min(power_spectrum.len() - 1);
    if power_spectrum[b2] != 0.0 {
        power_spectrum[b1] / power_spectrum[b2]
    } else {
        0.0
    }
}

pub(crate) fn d4c_love_train(
    x: &[f64],
    fs: i32,
    x_length: usize,
    f0: &[f64],
    f0_length: usize,
    temporal_positions: &[f64],
    aperiodicity0: &mut [f64],
    randn_state: &mut RandnState,
) {
    let lowest_f0 = 40.0;
    let fft_size_f = 2.0_f64.powi(1 + ((3.0 * fs as f64 / lowest_f0 + 1.0).ln() / K_LOG2) as i32);
    let fft_size = fft_size_f as usize;
    let mut forward_real_fft = ForwardRealFFT::new(fft_size);
    let boundary0 = ((100.0 * fft_size as f64 / fs as f64).ceil()) as usize;
    let boundary1 = ((4000.0 * fft_size as f64 / fs as f64).ceil()) as usize;
    let boundary2 = ((7900.0 * fft_size as f64 / fs as f64).ceil()) as usize;
    for i in 0..f0_length {
        if f0[i] == 0.0 {
            aperiodicity0[i] = 0.0;
            continue;
        }
        let cur_f0 = f0[i].max(lowest_f0);
        aperiodicity0[i] = d4c_love_train_sub(
            x,
            fs,
            x_length,
            cur_f0,
            temporal_positions[i],
            fft_size,
            boundary0,
            boundary1,
            boundary2,
            &mut forward_real_fft,
            randn_state,
        );
    }
}
