//  Copyright (C) 2026 by Tobias Hoffmann
//  thoffmann-ml@proton.me
//  https://github.com/thfmn/xjtu-sy-bearing
//
//  This work is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
//  and to permit persons to whom the Software is furnished to do so, subject to the condition that the above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  For more information, visit: https://opensource.org/licenses/MIT
//
//  Author:    Tobias Hoffmann
//  Email:     thoffmann-ml@proton.me
//  License:   MIT
//  Date:      2025-2026
//  Package:   xjtu-sy-bearing onset and RUL prediction ML Pipeline

use crate::models::ScalogramData;
use ndarray::Array3;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

/// Load a CWT scalogram `.npy` file of shape (64, 128, 2) into ScalogramData.
///
/// The third axis represents the two vibration channels:
///   - channel 0 = horizontal (H)
///   - channel 1 = vertical (V)
///
/// Each channel is returned as a Vec<Vec<f32>> (scales x time_bins) which
/// serializes directly to a JSON 2D array for the frontend heatmap renderer.
pub fn load_scalogram(path: &Path) -> Result<ScalogramData, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open npy file {}: {e}", path.display()))?;

    let arr: Array3<f32> = Array3::<f32>::read_npy(file)
        .map_err(|e| format!("Failed to parse npy file {}: {e}", path.display()))?;

    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 2 {
        return Err(format!(
            "Unexpected npy shape {:?}, expected (scales, time_bins, 2)",
            shape
        ));
    }

    let num_scales = shape[0] as u32;
    let num_time_bins = shape[1] as u32;

    // Extract each channel as Vec<Vec<f32>>.
    // The CWT Python code produces rows ordered high-freq-first (row 0 = 6000 Hz,
    // row 63 = 10 Hz) because scales are reversed to go small→large.  We reverse
    // the row order here so that row 0 = lowest frequency (freq_min) and
    // row 63 = highest frequency (freq_max).  This matches Plotly's default
    // heatmap orientation where z[0] is rendered at the bottom of the Y-axis.
    let h_channel: Vec<Vec<f32>> = (0..shape[0])
        .rev()
        .map(|s| (0..shape[1]).map(|t| arr[[s, t, 0]]).collect())
        .collect();

    let v_channel: Vec<Vec<f32>> = (0..shape[0])
        .rev()
        .map(|s| (0..shape[1]).map(|t| arr[[s, t, 1]]).collect())
        .collect();

    Ok(ScalogramData {
        h_channel,
        v_channel,
        freq_min: 10.0,
        freq_max: 6000.0,
        num_scales,
        num_time_bins,
    })
}

/// Load a CWT scalogram and return the **mean power per scale** for each channel.
///
/// This is a lightweight alternative to [`load_scalogram`] for building panoramic
/// views: instead of returning the full 64×128 matrix, we average across the 128
/// time bins to produce a single 64-element vector per channel.
///
/// Returns `(h_means, v_means)`, each `Vec<f32>` of length `num_scales`.
/// Row order is reversed (row 0 = lowest frequency) to match Plotly orientation.
pub fn load_scalogram_means(path: &Path) -> Result<(Vec<f32>, Vec<f32>), String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open npy file {}: {e}", path.display()))?;

    let arr: Array3<f32> = Array3::<f32>::read_npy(file)
        .map_err(|e| format!("Failed to parse npy file {}: {e}", path.display()))?;

    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 2 {
        return Err(format!(
            "Unexpected npy shape {:?}, expected (scales, time_bins, 2)",
            shape
        ));
    }

    let num_time_bins = shape[1] as f32;

    // Compute mean across time bins for each scale, reversed so row 0 = lowest freq.
    let h_means: Vec<f32> = (0..shape[0])
        .rev()
        .map(|s| {
            let sum: f32 = (0..shape[1]).map(|t| arr[[s, t, 0]]).sum();
            sum / num_time_bins
        })
        .collect();

    let v_means: Vec<f32> = (0..shape[0])
        .rev()
        .map(|s| {
            let sum: f32 = (0..shape[1]).map(|t| arr[[s, t, 1]]).sum();
            sum / num_time_bins
        })
        .collect();

    Ok((h_means, v_means))
}
