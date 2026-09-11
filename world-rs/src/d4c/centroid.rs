use crate::common::{dc_correction_into, linear_smoothing_into};
use crate::constants::{K_BLACKMAN, K_HANNING};
use crate::d4c::window::get_windowed_waveform;
use crate::fft::ForwardRealFFT;
use crate::matlab::{matlab_round, RandnState};

pub(crate) fn get_centroid(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    forward_real_fft: &mut ForwardRealFFT,
    centroid: &mut [f64],
    randn_state: &mut RandnState,
) {
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
        4.0,
        &mut forward_real_fft.waveform,
        randn_state,
    );
    let limit = matlab_round(2.0 * fs as f64 / current_f0) as usize * 2;
    let limit = limit.min(forward_real_fft.waveform.len());
    let mut power = 0.0;
    for i in 0..=limit {
        let v = forward_real_fft.waveform[i];
        power += v * v;
    }
    let norm = if power > 0.0 { power.sqrt() } else { 1.0 };
    for i in 0..=limit {
        forward_real_fft.waveform[i] /= norm;
    }
    // Snapshot the normalized time waveform: `forward()` below overwrites
    // its input buffer (realfft), and the ramped second transform needs the
    // intact signal (C++/fftw preserves it).
    let saved_waveform = forward_real_fft.waveform.clone();
    forward_real_fft.forward();
    let half = fft_size / 2 + 1;
    let mut tmp_real = vec![0.0f64; half];
    let mut tmp_imag = vec![0.0f64; half];
    for i in 0..half {
        tmp_real[i] = forward_real_fft.spectrum[i].re;
        tmp_imag[i] = forward_real_fft.spectrum[i].im;
    }
    // C++ re-reads the (normalized, windowed) time waveform here for the
    // ramped second transform — fftw preserves its input buffer, but realfft
    // `forward()` overwrites it (see fft/mod.rs `forward()` docs). Restore
    // from a snapshot so the ramp applies to the signal, not FFT scratch.
    forward_real_fft.waveform.copy_from_slice(&saved_waveform);
    for i in 0..fft_size {
        forward_real_fft.waveform[i] *= i as f64 + 1.0;
    }
    forward_real_fft.forward();
    for i in 0..half {
        centroid[i] = forward_real_fft.spectrum[i].re * tmp_real[i]
            + tmp_imag[i] * forward_real_fft.spectrum[i].im;
    }
}

pub(crate) fn get_static_centroid(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    forward_real_fft: &mut ForwardRealFFT,
    static_centroid: &mut [f64],
    randn_state: &mut RandnState,
) {
    let half = fft_size / 2 + 1;
    let mut centroid1 = vec![0.0f64; half];
    let mut centroid2 = vec![0.0f64; half];
    get_centroid(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position - 0.25 / current_f0,
        forward_real_fft,
        &mut centroid1,
        randn_state,
    );
    get_centroid(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position + 0.25 / current_f0,
        forward_real_fft,
        &mut centroid2,
        randn_state,
    );
    for i in 0..half {
        static_centroid[i] = centroid1[i] + centroid2[i];
    }
    let tmp = static_centroid.to_vec();
    let mut scratch = vec![0.0f64; crate::common::dc_correction_scratch_capacity(fft_size, half)];
    dc_correction_into(
        &tmp,
        current_f0,
        fs,
        fft_size,
        static_centroid,
        &mut scratch,
    );
}

pub(crate) fn get_smoothed_power_spectrum(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    forward_real_fft: &mut ForwardRealFFT,
    smoothed_power_spectrum: &mut [f64],
    randn_state: &mut RandnState,
) {
    for i in 0..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    get_windowed_waveform(
        x,
        x_length,
        fs,
        current_f0,
        current_position,
        K_HANNING,
        4.0,
        &mut forward_real_fft.waveform,
        randn_state,
    );
    forward_real_fft.forward();
    let half = fft_size / 2 + 1;
    for (i, slot) in smoothed_power_spectrum.iter_mut().enumerate().take(half) {
        let re = forward_real_fft.spectrum[i].re;
        let im = forward_real_fft.spectrum[i].im;
        *slot = re * re + im * im;
    }
    let tmp = smoothed_power_spectrum.to_vec();
    let mut scratch = vec![0.0f64; crate::common::dc_correction_scratch_capacity(fft_size, half)];
    dc_correction_into(
        &tmp,
        current_f0,
        fs,
        fft_size,
        smoothed_power_spectrum,
        &mut scratch,
    );
    let tmp2 = smoothed_power_spectrum.to_vec();
    let mut scratch2 = vec![0.0f64; crate::common::linear_smoothing_scratch_capacity(fft_size)];
    linear_smoothing_into(
        &tmp2,
        current_f0,
        fs,
        fft_size,
        smoothed_power_spectrum,
        &mut scratch2,
    );
}

pub(crate) fn get_static_group_delay(
    static_centroid: &[f64],
    smoothed_power_spectrum: &[f64],
    fs: i32,
    f0: f64,
    fft_size: usize,
    static_group_delay: &mut [f64],
) {
    let half = fft_size / 2 + 1;
    for i in 0..half {
        let denom = smoothed_power_spectrum[i];
        static_group_delay[i] = if denom != 0.0 {
            static_centroid[i] / denom
        } else {
            0.0
        };
    }
    let tmp = static_group_delay.to_vec();
    let mut scratch = vec![0.0f64; crate::common::linear_smoothing_scratch_capacity(fft_size)];
    linear_smoothing_into(
        &tmp,
        f0 / 2.0,
        fs,
        fft_size,
        static_group_delay,
        &mut scratch,
    );
    let tmp2 = static_group_delay.to_vec();
    let mut smoothed_group_delay = vec![0.0f64; half];
    let mut scratch2 = vec![0.0f64; crate::common::linear_smoothing_scratch_capacity(fft_size)];
    linear_smoothing_into(
        &tmp2,
        f0,
        fs,
        fft_size,
        &mut smoothed_group_delay,
        &mut scratch2,
    );
    for i in 0..half {
        static_group_delay[i] -= smoothed_group_delay[i];
    }
}

pub(crate) fn get_coarse_aperiodicity(
    static_group_delay: &[f64],
    fs: i32,
    fft_size: usize,
    number_of_aperiodicities: usize,
    window: &[f64],
    window_length: usize,
    forward_real_fft: &mut ForwardRealFFT,
    coarse_aperiodicity: &mut [f64],
) {
    let boundary = matlab_round(fft_size as f64 * 8.0 / window_length as f64) as usize;
    let half_window_length = window_length / 2;
    let mut power_spectrum = vec![0.0f64; fft_size / 2 + 1];
    for (i, slot) in coarse_aperiodicity
        .iter_mut()
        .enumerate()
        .take(number_of_aperiodicities)
    {
        let center = ((crate::constants::K_FREQUENCY_INTERVAL * (i as f64 + 1.0) * fft_size as f64)
            / fs as f64) as usize;
        for j in 0..=half_window_length * 2 {
            let idx = center.saturating_sub(half_window_length).saturating_add(j);
            if idx < static_group_delay.len() && j < window.len() {
                forward_real_fft.waveform[j] = static_group_delay[idx] * window[j];
            } else {
                forward_real_fft.waveform[j] = 0.0;
            }
        }
        for i2 in half_window_length * 2 + 1..fft_size {
            forward_real_fft.waveform[i2] = 0.0;
        }
        forward_real_fft.forward();
        let half = fft_size / 2 + 1;
        for (j, slot) in power_spectrum.iter_mut().enumerate().take(half) {
            let re = forward_real_fft.spectrum[j].re;
            let im = forward_real_fft.spectrum[j].im;
            *slot = re * re + im * im;
        }
        // Only two prefix sums of the sorted spectrum are read: the total
        // (`idx2`, the last bin) and the sum of the smallest `idx1 + 1`
        // bins. Partial selection + partitioned sums replace the full sort
        // (sums are order-independent up to float rounding).
        let idx1 = fft_size / 2 - boundary - 1;
        let idx2 = fft_size / 2;
        if idx1 < half && idx2 < half {
            let total: f64 = power_spectrum[..half].iter().sum();
            if total != 0.0 {
                power_spectrum[..half].select_nth_unstable_by(idx1, |a, b| a.total_cmp(b));
                let small: f64 = power_spectrum[..=idx1].iter().sum();
                *slot = 10.0 * (small / total).log10();
            } else {
                *slot = -60.0;
            }
        } else {
            *slot = -60.0;
        }
    }
}
