use crate::d4c::centroid::{
    get_coarse_aperiodicity, get_smoothed_power_spectrum, get_static_centroid,
    get_static_group_delay,
};
use crate::fft::ForwardRealFFT;
use crate::matlab::interp1;

pub(crate) fn d4c_general_body(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    number_of_aperiodicities: usize,
    window: &[f64],
    window_length: usize,
    forward_real_fft: &mut ForwardRealFFT,
    coarse_aperiodicity: &mut [f64],
    randn_state: &mut crate::matlab::RandnState,
) {
    let half = fft_size / 2 + 1;
    let mut static_centroid = vec![0.0f64; half];
    let mut smoothed_power_spectrum = vec![0.0f64; half];
    let mut static_group_delay = vec![0.0f64; half];
    get_static_centroid(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position,
        forward_real_fft,
        &mut static_centroid,
        randn_state,
    );
    get_smoothed_power_spectrum(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position,
        forward_real_fft,
        &mut smoothed_power_spectrum,
        randn_state,
    );
    get_static_group_delay(
        &static_centroid,
        &smoothed_power_spectrum,
        fs,
        current_f0,
        fft_size,
        &mut static_group_delay,
    );
    get_coarse_aperiodicity(
        &static_group_delay,
        fs,
        fft_size,
        number_of_aperiodicities,
        window,
        window_length,
        forward_real_fft,
        coarse_aperiodicity,
    );
    for slot in coarse_aperiodicity.iter_mut() {
        *slot = (*slot + (current_f0 - 100.0) / 50.0).min(0.0);
    }
}

pub(crate) fn get_aperiodicity(
    coarse_frequency_axis: &[f64],
    coarse_aperiodicity: &[f64],
    number_of_aperiodicities: usize,
    frequency_axis: &[f64],
    fft_size: usize,
    aperiodicity: &mut [f64],
) {
    let _n_coarse = number_of_aperiodicities + 2;
    let mut yi = vec![0.0f64; fft_size / 2 + 1];
    interp1(
        coarse_frequency_axis,
        coarse_aperiodicity,
        frequency_axis,
        &mut yi,
    );
    for i in 0..=fft_size / 2 {
        aperiodicity[i] = 10.0_f64.powf(yi[i] / 20.0);
    }
}
