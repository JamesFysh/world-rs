use crate::constants::{K_CEIL_F0, K_FLOOR_F0};

/// Allocation budget for one `harvest()` call: at most `MAX_HARVEST_CELLS`
/// `width × frames` f64 cells (≈512MB for the candidate/score matrices)
/// and at most `MAX_HARVEST_FRAMES` frames. Plans beyond this fail with
/// [`HarvestError::TooManyFrames`] before allocating.
const MAX_HARVEST_CELLS: u64 = 64_000_000;
const MAX_HARVEST_FRAMES: usize = 8_000_000;

mod candidates;
mod general;
mod postprocess;
mod refine;

#[derive(Debug, Clone, PartialEq)]
pub struct HarvestOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub frame_period: f64,
}

pub fn initialize_harvest_option() -> HarvestOption {
    HarvestOption {
        f0_floor: K_FLOOR_F0,
        f0_ceil: K_CEIL_F0,
        frame_period: 5.0,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HarvestError {
    EmptyInput,
    NonPositiveSampleRate {
        fs: f64,
    },
    NonPositiveFramePeriod {
        frame_period: f64,
    },
    InvalidF0Range,
    NonFiniteInput,
    TooShortInput,
    /// The frame plan (`width × frames`) exceeds the allocation budget
    /// (`MAX_HARVEST_CELLS`) or `frames` exceeds `MAX_HARVEST_FRAMES`.
    /// Rejects degenerate plans before any frame-major buffer is allocated.
    TooManyFrames {
        width: usize,
        frames: usize,
    },
}

impl std::fmt::Display for HarvestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HarvestError::EmptyInput => write!(f, "input signal is empty"),
            HarvestError::NonPositiveSampleRate { fs } => {
                write!(f, "sample rate must be positive, got {fs}")
            }
            HarvestError::NonPositiveFramePeriod { frame_period } => {
                write!(f, "frame period must be positive, got {frame_period}")
            }
            HarvestError::InvalidF0Range => write!(f, "invalid f0 range"),
            HarvestError::NonFiniteInput => write!(f, "input contains non-finite samples"),
            HarvestError::TooShortInput => write!(f, "input too short for one frame"),
            HarvestError::TooManyFrames { width, frames } => write!(
                f,
                "frame plan too large: {width} candidates × {frames} frames exceeds budget"
            ),
        }
    }
}

impl std::error::Error for HarvestError {}

#[derive(Debug, Clone, PartialEq)]
pub struct HarvestResult {
    pub f0: Vec<f64>,
    pub temporal_positions: Vec<f64>,
}

pub fn get_samples_for_harvest(fs: f64, x_length: usize, frame_period: f64) -> usize {
    if !fs.is_finite() || fs <= 0.0 || !frame_period.is_finite() || frame_period <= 0.0 {
        return 0;
    }
    let v = 1000.0 * x_length as f64 / fs / frame_period;
    if !v.is_finite() {
        return 0;
    }
    let n = v as usize;
    n.saturating_add(1)
}

pub fn harvest(x: &[f64], fs: f64, option: &HarvestOption) -> Result<HarvestResult, HarvestError> {
    if x.is_empty() {
        return Err(HarvestError::EmptyInput);
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(HarvestError::NonPositiveSampleRate { fs });
    }
    if !option.frame_period.is_finite() || option.frame_period <= 0.0 {
        return Err(HarvestError::NonPositiveFramePeriod {
            frame_period: option.frame_period,
        });
    }
    if !option.f0_floor.is_finite()
        || !option.f0_ceil.is_finite()
        || option.f0_floor <= 0.0
        || option.f0_ceil <= option.f0_floor
    {
        return Err(HarvestError::InvalidF0Range);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(HarvestError::NonFiniteInput);
    }
    let min_samples = (fs * option.frame_period / 1000.0).ceil() as usize;
    if x.len() < min_samples {
        return Err(HarvestError::TooShortInput);
    }

    let f0_length = get_samples_for_harvest(fs, x.len(), option.frame_period);
    // Allocation budget: mirror the `max_candidates` width computation in
    // `general::harvest_general_body` (channels from the f0 range,
    // `round(channels/10) * 7` candidates) and reject degenerate plans
    // (e.g. `frame_period → 0`, `f0_floor → 0`) before allocating.
    let log_ratio =
        ((option.f0_ceil * 1.1) / (option.f0_floor * 0.9)).ln() / std::f64::consts::LN_2;
    let channels = 1_usize.saturating_add((log_ratio * 40.0).floor() as usize);
    let width = ((channels as f64 / 10.0).round() as usize).saturating_mul(7);
    if f0_length > MAX_HARVEST_FRAMES
        || (width as u128) * (f0_length as u128) > MAX_HARVEST_CELLS as u128
    {
        return Err(HarvestError::TooManyFrames {
            width,
            frames: f0_length,
        });
    }
    let mut f0 = vec![0.0f64; f0_length];
    let mut temporal_positions = vec![0.0f64; f0_length];

    if (option.frame_period - 1.0).abs() < 1e-12 {
        general::harvest_general_body(
            x,
            fs,
            option.frame_period,
            option.f0_floor,
            option.f0_ceil,
            &mut temporal_positions,
            &mut f0,
        );
    } else {
        let basic_len = get_samples_for_harvest(fs, x.len(), 1.0);
        let mut basic_f0 = vec![0.0f64; basic_len];
        let mut basic_tpos = vec![0.0f64; basic_len];
        general::harvest_general_body(
            x,
            fs,
            1.0,
            option.f0_floor,
            option.f0_ceil,
            &mut basic_tpos,
            &mut basic_f0,
        );
        for i in 0..f0_length {
            temporal_positions[i] = i as f64 * option.frame_period / 1000.0;
            let tpos_ms = temporal_positions[i] * 1000.0;
            let idx = crate::matlab::matlab_round(tpos_ms) as usize;
            let idx_clamped = idx.min(basic_len.saturating_sub(1));
            f0[i] = basic_f0[idx_clamped];
        }
    }

    Ok(HarvestResult {
        f0,
        temporal_positions,
    })
}
