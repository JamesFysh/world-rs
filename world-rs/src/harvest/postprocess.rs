#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::map_clone)]
#![allow(clippy::cast_abs_to_unsigned)]
#![allow(clippy::excessive_precision)]

pub(crate) fn filtering_f0(
    a: &[f64],
    b: &[f64],
    x: &[f64],
    x_length: usize,
    st: usize,
    ed: usize,
    y: &mut [f64],
) {
    if x_length == 0 || y.len() < x_length || a.len() < 2 || b.len() < 2 {
        return;
    }
    let mut x_mut = x.to_vec();
    let st_i = st.min(x_length.saturating_sub(1));
    let ed_i = ed.min(x_length.saturating_sub(1));
    for i in 0..st_i {
        x_mut[i] = x_mut[st_i];
    }
    for i in ed_i + 1..x_length {
        x_mut[i] = x_mut[ed_i];
    }
    let mut w = [0.0f64, 0.0f64];
    let mut tmp_x = vec![0.0f64; x_length];
    for i in 0..x_length {
        let wt = x_mut[i] + a[0] * w[0] + a[1] * w[1];
        let idx = x_length - i - 1;
        tmp_x[idx] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
        w[1] = w[0];
        w[0] = wt;
    }
    w = [0.0, 0.0];
    for i in 0..x_length {
        let wt = tmp_x[i] + a[0] * w[0] + a[1] * w[1];
        let idx = x_length - i - 1;
        if idx < y.len() {
            y[idx] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
        }
        w[1] = w[0];
        w[0] = wt;
    }
}

pub(crate) fn fix_f0_contour(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
    best_f0_contour: &mut [f64],
) {
    let mut tmp1 = vec![0.0f64; f0_length];
    let mut tmp2 = vec![0.0f64; f0_length];
    search_f0_base(
        f0_candidates,
        f0_scores,
        f0_length,
        number_of_candidates,
        &mut tmp1,
    );
    fix_step1(&tmp1, f0_length, 0.008, &mut tmp2);
    fix_step2(&tmp2, f0_length, 6, &mut tmp1);
    fix_step3(
        &tmp1,
        f0_length,
        number_of_candidates,
        f0_candidates,
        0.18,
        f0_scores,
        &mut tmp2,
    );
    fix_step4(&tmp2, f0_length, 9, best_f0_contour);
}

pub(crate) fn smooth_f0_contour(f0: &[f64], f0_length: usize, smoothed_f0: &mut [f64]) {
    let b = [0.0078202080334971724, 0.015640416066994345];
    let a = [1.7347257688092754, -0.76600660094326412];
    let lag = 300usize;
    let new_f0_length = f0_length + lag * 2;
    let mut f0_contour = vec![0.0f64; new_f0_length];
    for i in lag..lag + f0_length {
        f0_contour[i] = f0[i - lag];
    }
    let boundary_list = {
        let mut bl = vec![0i32; new_f0_length];
        let n = get_boundary_list(&f0_contour, new_f0_length, &mut bl);
        bl.truncate(n);
        bl
    };
    let number_of_boundaries = boundary_list.len();
    let num_sections = number_of_boundaries / 2;
    if num_sections == 0 {
        for i in 0..f0_length.min(smoothed_f0.len()) {
            smoothed_f0[i] = 0.0;
        }
        return;
    }
    let multi_channel_f0 = get_multi_channel_f0(
        &f0_contour,
        new_f0_length,
        &boundary_list,
        number_of_boundaries,
    );
    let mut filtered = vec![0.0f64; new_f0_length];
    for i in 0..num_sections {
        let st = boundary_list[i * 2] as usize;
        let ed = boundary_list[i * 2 + 1] as usize;
        let ch = &multi_channel_f0[i];
        filtering_f0(&a, &b, ch, new_f0_length, st, ed, &mut filtered);
        for j in st..=ed {
            if j < new_f0_length && j >= lag && (j - lag) < smoothed_f0.len() {
                smoothed_f0[j - lag] = filtered[j];
            }
        }
    }
}

pub(crate) fn select_best_f0(
    reference_f0: f64,
    f0_candidates: &[f64],
    number_of_candidates: usize,
    allowed_range: f64,
    best_error: &mut f64,
) -> f64 {
    let mut best_f0 = 0.0;
    *best_error = allowed_range;
    for i in 0..number_of_candidates {
        if let Some(&cand) = f0_candidates.get(i) {
            let tmp = (reference_f0 - cand).abs() / reference_f0;
            if tmp > *best_error {
                continue;
            }
            best_f0 = cand;
            *best_error = tmp;
        }
    }
    best_f0
}

pub(crate) fn remove_unreliable_candidates_sub(
    i: usize,
    j: usize,
    tmp_f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    if i >= f0_candidates.len() || j >= f0_candidates[i].len() {
        return;
    }
    let reference_f0 = f0_candidates[i][j];
    if reference_f0 == 0.0 {
        return;
    }
    let threshold = 0.05;
    let mut error1 = 1.0;
    let mut error2 = 1.0;
    if i + 1 < tmp_f0_candidates.len() {
        let next = &tmp_f0_candidates[i + 1];
        select_best_f0(reference_f0, next, number_of_candidates, 1.0, &mut error1);
    }
    if i >= 1 {
        let prev = &tmp_f0_candidates[i - 1];
        select_best_f0(reference_f0, prev, number_of_candidates, 1.0, &mut error2);
    }
    let min_error = error1.min(error2);
    if min_error <= threshold {
        return;
    }
    f0_candidates[i][j] = 0.0;
    if j < f0_scores[i].len() {
        f0_scores[i][j] = 0.0;
    }
}

pub(crate) fn remove_unreliable_candidates(
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    debug_assert!(number_of_candidates <= f0_candidates.first().map_or(0, |v| v.len()));
    let tmp_f0_candidates: Vec<Vec<f64>> = f0_candidates.to_vec();
    for i in 1..f0_length.saturating_sub(1) {
        if i >= f0_candidates.len() {
            continue;
        }
        let row_len = f0_candidates[i].len().min(number_of_candidates);
        for j in 0..row_len {
            remove_unreliable_candidates_sub(
                i,
                j,
                &tmp_f0_candidates,
                number_of_candidates,
                f0_candidates,
                f0_scores,
            );
        }
    }
}

pub(crate) fn search_f0_base(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
    base_f0_contour: &mut [f64],
) {
    for i in 0..f0_length.min(base_f0_contour.len()) {
        let mut best_score = 0.0;
        let mut best_f0 = 0.0;
        let cand_row = f0_candidates.get(i);
        let score_row = f0_scores.get(i);
        if let (Some(cands), Some(scores)) = (cand_row, score_row) {
            let n = number_of_candidates.min(cands.len()).min(scores.len());
            for j in 0..n {
                let s = scores[j];
                if s > best_score {
                    best_score = s;
                    best_f0 = cands[j];
                }
            }
        }
        base_f0_contour[i] = best_f0;
    }
}

pub(crate) fn fix_step1(
    f0_base: &[f64],
    f0_length: usize,
    allowed_range: f64,
    f0_step1: &mut [f64],
) {
    for i in 0..f0_length.min(f0_step1.len()) {
        f0_step1[i] = 0.0;
    }
    for i in 2..f0_length {
        if i >= f0_base.len() || i >= f0_step1.len() {
            continue;
        }
        let f0_i = f0_base[i];
        if f0_i == 0.0 {
            continue;
        }
        let f0_im1 = f0_base[i - 1];
        let f0_im2 = f0_base[i - 2];
        let reference_f0 = f0_im1 * 2.0 - f0_im2;
        let cond1 = ((f0_i - reference_f0).abs() / reference_f0) > allowed_range;
        let cond2 = ((f0_i - f0_im1).abs() / f0_im1) > allowed_range;
        if cond1 && cond2 {
            f0_step1[i] = 0.0;
        } else {
            f0_step1[i] = f0_i;
        }
    }
}

pub(crate) fn get_boundary_list(f0: &[f64], f0_length: usize, boundary_list: &mut [i32]) -> usize {
    let mut vuv = vec![0i32; f0_length];
    for i in 0..f0_length.min(f0.len()) {
        vuv[i] = if f0[i] > 0.0 { 1 } else { 0 };
    }
    if f0_length > 0 {
        vuv[0] = 0;
        vuv[f0_length - 1] = 0;
    }
    let mut number_of_boundaries = 0usize;
    for i in 1..f0_length {
        if i >= vuv.len() {
            break;
        }
        if vuv[i] - vuv[i - 1] != 0 {
            if number_of_boundaries < boundary_list.len() {
                boundary_list[number_of_boundaries] =
                    (i as i32) - (number_of_boundaries % 2) as i32;
                number_of_boundaries += 1;
            } else {
                break;
            }
        }
    }
    number_of_boundaries
}

pub(crate) fn fix_step2(
    f0_step1: &[f64],
    f0_length: usize,
    voice_range_minimum: i32,
    f0_step2: &mut [f64],
) {
    for i in 0..f0_length.min(f0_step2.len()) {
        f0_step2[i] = f0_step1.get(i).copied().unwrap_or(0.0);
    }
    let mut boundary_list = vec![0i32; f0_length];
    let number_of_boundaries = get_boundary_list(f0_step1, f0_length, &mut boundary_list);
    for i in 0..number_of_boundaries / 2 {
        let start = boundary_list[i * 2] as usize;
        let end = boundary_list[i * 2 + 1] as usize;
        if end >= start && (end as i32 - start as i32) >= voice_range_minimum {
            continue;
        }
        let s = start.min(f0_length);
        let e = end.min(f0_length.saturating_sub(1));
        for j in s..=e {
            if j < f0_step2.len() {
                f0_step2[j] = 0.0;
            }
        }
    }
}

pub(crate) fn get_multi_channel_f0(
    f0: &[f64],
    f0_length: usize,
    boundary_list: &[i32],
    number_of_boundaries: usize,
) -> Vec<Vec<f64>> {
    let num_sections = number_of_boundaries / 2;
    let mut multi = vec![vec![0.0; f0_length]; num_sections];
    for i in 0..num_sections {
        let st = boundary_list[i * 2] as usize;
        let ed = boundary_list[i * 2 + 1] as usize;
        let row = &mut multi[i];
        for j in st..=ed.min(f0_length - 1) {
            if j < f0.len() {
                row[j] = f0[j];
            }
        }
    }
    multi
}

pub(crate) fn search_score(
    f0: f64,
    f0_candidates: &[f64],
    f0_scores: &[f64],
    number_of_candidates: usize,
) -> f64 {
    let mut score = 0.0;
    for i in 0..number_of_candidates {
        if let (Some(&cand), Some(&sc)) = (f0_candidates.get(i), f0_scores.get(i)) {
            if f0 == cand && score < sc {
                score = sc;
            }
        }
    }
    score
}

pub(crate) fn make_sorted_order(
    boundary_list: &[i32],
    number_of_sections: usize,
    order: &mut [usize],
) {
    for i in 0..number_of_sections {
        order[i] = i;
    }
    for i in 1..number_of_sections {
        for j in (0..i).rev() {
            let bi = boundary_list[order[j] * 2];
            let bj = boundary_list[order[i] * 2];
            if bi > bj {
                order.swap(i, j);
            } else {
                break;
            }
        }
    }
}

pub(crate) fn extend_f0(
    _f0: &[f64],
    f0_length: usize,
    origin: i32,
    last_point: i32,
    shift: i32,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    extended_f0: &mut [f64],
) -> i32 {
    let threshold = 4;
    if origin < 0 || (origin as usize) >= f0_length || (origin as usize) >= extended_f0.len() {
        return origin;
    }
    let mut tmp_f0 = extended_f0[origin as usize];
    let mut shifted_origin = origin;
    let distance = (last_point - origin).abs() as usize;
    let mut count = 0;
    for i in 0..=distance {
        let idx = origin + shift * i as i32;
        let next_idx = idx + shift;
        if next_idx < 0
            || (next_idx as usize) >= f0_length
            || (next_idx as usize) >= extended_f0.len()
        {
            break;
        }
        let next_idx_usize = next_idx as usize;
        if next_idx_usize >= f0_candidates.len() {
            break;
        }
        let mut dummy = allowed_range;
        let best = select_best_f0(
            tmp_f0,
            &f0_candidates[next_idx_usize],
            number_of_candidates,
            allowed_range,
            &mut dummy,
        );
        extended_f0[next_idx_usize] = best;
        if best == 0.0 {
            count += 1;
        } else {
            tmp_f0 = best;
            count = 0;
            shifted_origin = next_idx;
        }
        if count == threshold {
            break;
        }
    }
    shifted_origin
}

pub(crate) fn swap_rows(
    multi_channel_f0: &mut [Vec<f64>],
    boundary: &mut [i32],
    index1: usize,
    index2: usize,
) {
    multi_channel_f0.swap(index1, index2);
    let i1 = index1 * 2;
    let i2 = index2 * 2;
    if i1 + 1 < boundary.len() && i2 + 1 < boundary.len() {
        boundary.swap(i1, i2);
        boundary.swap(i1 + 1, i2 + 1);
    }
}

pub(crate) fn extend_sub(
    extended_f0: &[Vec<f64>],
    boundary_list: &[i32],
    number_of_sections: usize,
    selected_extended_f0: &mut [Vec<f64>],
    selected_boundary_list: &mut [i32],
) -> usize {
    let threshold = 2200.0;
    let mut count = 0;
    for i in 0..number_of_sections {
        let st = boundary_list[i * 2] as usize;
        let ed = boundary_list[i * 2 + 1] as usize;
        if ed <= st || ed > extended_f0[i].len() {
            continue;
        }
        let sum: f64 = extended_f0[i][st..ed].iter().sum();
        let mean_f0 = sum / (ed - st) as f64;
        if mean_f0 > 0.0 && threshold / mean_f0 < (ed - st) as f64 {
            swap_rows(selected_extended_f0, selected_boundary_list, count, i);
            count += 1;
        }
    }
    count
}

pub(crate) fn extend(
    multi_channel_f0: &[Vec<f64>],
    number_of_sections: usize,
    f0_length: usize,
    boundary_list: &[i32],
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    extended_f0: &mut [Vec<f64>],
    shifted_boundary_list: &mut [i32],
) -> usize {
    let threshold = 100;
    for i in 0..number_of_sections {
        let start = boundary_list[i * 2];
        let end = boundary_list[i * 2 + 1];
        let last_point = (f0_length as i32 - 2).min(end + threshold);
        shifted_boundary_list[i * 2 + 1] = extend_f0(
            &multi_channel_f0[i],
            f0_length,
            end,
            last_point,
            1,
            f0_candidates,
            number_of_candidates,
            allowed_range,
            &mut extended_f0[i],
        );
        let first_point = (1).max(start - threshold);
        shifted_boundary_list[i * 2] = extend_f0(
            &multi_channel_f0[i],
            f0_length,
            start,
            first_point,
            -1,
            f0_candidates,
            number_of_candidates,
            allowed_range,
            &mut extended_f0[i],
        );
    }
    let snapshot: Vec<Vec<f64>> = extended_f0.iter().map(|v| v.clone()).collect();
    let boundary_snapshot = shifted_boundary_list.to_vec();
    extend_sub(
        &snapshot,
        &boundary_snapshot,
        number_of_sections,
        extended_f0,
        shifted_boundary_list,
    )
}

pub(crate) fn merge_f0_sub(
    f0_1: &[f64],
    f0_length: usize,
    st1: usize,
    ed1: usize,
    f0_2: &[f64],
    st2: usize,
    ed2: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_candidates: usize,
    merged_f0: &mut [f64],
) -> usize {
    if st1 <= st2 && ed1 >= ed2 {
        return ed1;
    }
    let mut score1 = 0.0;
    let mut score2 = 0.0;
    if st2 <= ed1 {
        let end_loop = ed1
            .min(f0_length.saturating_sub(1))
            .min(f0_candidates.len().saturating_sub(1))
            .min(f0_scores.len().saturating_sub(1));
        for i in st2..=end_loop {
            let f1 = f0_1.get(i).copied().unwrap_or(0.0);
            let f2 = f0_2.get(i).copied().unwrap_or(0.0);
            score1 += search_score(f1, &f0_candidates[i], &f0_scores[i], number_of_candidates);
            score2 += search_score(f2, &f0_candidates[i], &f0_scores[i], number_of_candidates);
        }
    }
    if score1 > score2 {
        for i in ed1..=ed2.min(f0_length - 1) {
            merged_f0[i] = f0_2[i];
        }
    } else {
        for i in st2..=ed2.min(f0_length - 1) {
            merged_f0[i] = f0_2[i];
        }
    }
    ed2
}

pub(crate) fn merge_f0(
    multi_channel_f0: &[Vec<f64>],
    boundary_list: &mut [i32],
    number_of_channels: usize,
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_candidates: usize,
    merged_f0: &mut [f64],
) {
    let mut order = vec![0usize; number_of_channels];
    make_sorted_order(boundary_list, number_of_channels, &mut order);
    for i in 0..f0_length.min(merged_f0.len()) {
        merged_f0[i] = multi_channel_f0[0][i];
    }
    for i in 1..number_of_channels {
        let idx = order[i];
        let start = boundary_list[idx * 2] as usize;
        let end = boundary_list[idx * 2 + 1] as usize;
        if boundary_list[idx * 2] - boundary_list[1] > 0 {
            for j in start..=end.min(f0_length - 1) {
                merged_f0[j] = multi_channel_f0[idx][j];
            }
            boundary_list[0] = boundary_list[idx * 2];
            boundary_list[1] = boundary_list[idx * 2 + 1];
        } else {
            let f0_1_snapshot = merged_f0.to_vec();
            boundary_list[1] = merge_f0_sub(
                &f0_1_snapshot,
                f0_length,
                boundary_list[0] as usize,
                boundary_list[1] as usize,
                &multi_channel_f0[idx],
                start,
                end,
                f0_candidates,
                f0_scores,
                number_of_candidates,
                merged_f0,
            ) as i32;
        }
    }
}

pub(crate) fn fix_step3(
    f0_step2: &[f64],
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &[Vec<f64>],
    allowed_range: f64,
    f0_scores: &[Vec<f64>],
    f0_step3: &mut [f64],
) {
    for i in 0..f0_length.min(f0_step3.len()) {
        f0_step3[i] = f0_step2[i];
    }
    let mut boundary_list = vec![0i32; f0_length];
    let number_of_boundaries = get_boundary_list(f0_step2, f0_length, &mut boundary_list);
    let num_sections = number_of_boundaries / 2;
    if num_sections == 0 {
        return;
    }
    let multi_channel_f0 =
        get_multi_channel_f0(f0_step2, f0_length, &boundary_list, number_of_boundaries);
    let mut extended_f0 = multi_channel_f0.clone();
    let mut shifted_boundary_list = vec![0i32; number_of_boundaries];
    let number_of_channels = extend(
        &multi_channel_f0,
        num_sections,
        f0_length,
        &boundary_list,
        f0_candidates,
        number_of_candidates,
        allowed_range,
        &mut extended_f0,
        &mut shifted_boundary_list,
    );
    if number_of_channels != 0 {
        merge_f0(
            &extended_f0,
            &mut shifted_boundary_list,
            number_of_channels,
            f0_length,
            f0_candidates,
            f0_scores,
            number_of_candidates,
            f0_step3,
        );
    }
}

pub(crate) fn fix_step4(f0_step3: &[f64], f0_length: usize, threshold: i32, f0_step4: &mut [f64]) {
    for i in 0..f0_length.min(f0_step4.len()) {
        f0_step4[i] = f0_step3[i];
    }
    let mut boundary_list = vec![0i32; f0_length];
    let number_of_boundaries = get_boundary_list(f0_step3, f0_length, &mut boundary_list);
    let sections = number_of_boundaries / 2;
    if sections < 2 {
        return;
    }
    for i in 0..sections - 1 {
        let start_next = boundary_list[(i + 1) * 2] as usize;
        let end_curr = boundary_list[i * 2 + 1] as usize;
        if start_next <= end_curr {
            continue;
        }
        let distance = start_next as i32 - end_curr as i32 - 1;
        if distance >= threshold {
            continue;
        }
        let tmp0 = f0_step3[end_curr] + 1.0;
        let tmp1 = f0_step3[start_next] - 1.0;
        let coeff = (tmp1 - tmp0) / (distance as f64 + 1.0);
        let mut count = 1;
        for j in end_curr + 1..start_next {
            if j < f0_step4.len() {
                f0_step4[j] = tmp0 + coeff * count as f64;
                count += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_sorted_order_tie_stability() {
        let boundary_list = [10, 20, 10, 30, 5, 15];
        let mut order = [0usize, 1, 2];
        make_sorted_order(&boundary_list, 3, &mut order);
        assert_eq!(order, [0, 2, 1]);
        let boundary_list2 = [10, 20, 10, 30];
        let mut order2 = [0usize, 1];
        make_sorted_order(&boundary_list2, 2, &mut order2);
        assert_eq!(order2, [0, 1]);
    }

    #[test]
    fn fix_step4_linear_interpolation() {
        let f0_step3 = [0.0, 100.0, 100.0, 0.0, 0.0, 110.0, 110.0, 0.0];
        let mut f0_step4 = [0.0; 8];
        fix_step4(&f0_step3, 8, 9, &mut f0_step4);
        assert_eq!(f0_step4[0], 0.0);
        assert_eq!(f0_step4[1], 100.0);
        assert_eq!(f0_step4[2], 100.0);
        assert!((f0_step4[3] - 103.66666666666667).abs() < 1e-6);
        assert!((f0_step4[4] - 106.33333333333334).abs() < 1e-6);
        assert_eq!(f0_step4[5], 110.0);
        assert_eq!(f0_step4[6], 110.0);
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::get_boundary_list;

    #[test]
    fn boundary_list_synthetic_alternation() {
        // T5 requirement: known voiced/unvoiced alternation.
        // vuv (ends forced 0): [0,0,1,1,1,0,0,1,1,0,0]
        // transitions at i=2 (up), 5 (down), 7 (up), 9 (down);
        // stored value is i - (number_of_boundaries % 2).
        let f0 = [
            0.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 200.0, 200.0, 0.0, 0.0,
        ];
        let mut bl = [0i32; 11];
        let n = get_boundary_list(&f0, 11, &mut bl);
        assert_eq!(n, 4);
        assert_eq!(&bl[..4], &[2, 4, 7, 8]);
    }
}
