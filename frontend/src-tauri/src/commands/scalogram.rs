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

use crate::data::npy_loader;
use crate::models::{AppData, OnsetMarker, PanoramicScalogramData, ScalogramData};

/// Return the number of CWT scalogram .npy files available for a bearing.
///
/// Files live at `outputs/spectrograms/cwt/condition={cond}/bearing_id={id}/`.
/// We simply count entries ending in `.npy` rather than parsing filenames.
#[tauri::command]
pub fn get_scalogram_file_count(
    condition: String,
    bearing_id: String,
    state: tauri::State<'_, AppData>,
) -> Result<u32, String> {
    let dir = state
        .data_root
        .join("outputs/spectrograms/cwt")
        .join(format!("condition={condition}"))
        .join(format!("bearing_id={bearing_id}"));

    if !dir.exists() {
        return Ok(0);
    }

    let count = std::fs::read_dir(&dir)
        .map_err(|e| format!("Failed to read scalogram directory: {e}"))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .map_or(false, |ext| ext == "npy")
        })
        .count();

    Ok(count as u32)
}

/// Load and return a single CWT scalogram for the given bearing and file index.
///
/// IMPORTANT: NPY filenames are 1-indexed (1.npy, 2.npy, ...) but the
/// `file_idx` from the CSV is 0-indexed. We add 1 when constructing the path.
#[tauri::command]
pub fn get_scalogram(
    condition: String,
    bearing_id: String,
    file_idx: u32,
    state: tauri::State<'_, AppData>,
) -> Result<ScalogramData, String> {
    let npy_filename = format!("{}.npy", file_idx + 1); // 0-indexed → 1-indexed
    let path = state
        .data_root
        .join("outputs/spectrograms/cwt")
        .join(format!("condition={condition}"))
        .join(format!("bearing_id={bearing_id}"))
        .join(&npy_filename);

    if !path.exists() {
        return Err(format!(
            "Scalogram file not found: {}",
            path.display()
        ));
    }

    npy_loader::load_scalogram(&path)
}

/// Build a panoramic CWT scalogram spanning a bearing's entire run-to-failure.
///
/// For each sampled file, we compute the mean CWT power per frequency scale
/// (averaged across 128 time bins). The result is a `num_scales × N` matrix
/// where N is the number of sampled files — essentially a "lifetime spectrogram".
///
/// Large bearings (e.g. Bearing3_1 with 2538 files) are subsampled to keep
/// the response under ~500 columns. The `step` field in the response indicates
/// how many files were skipped between samples.
#[tauri::command]
pub fn get_panoramic_scalogram(
    condition: String,
    bearing_id: String,
    state: tauri::State<'_, AppData>,
) -> Result<PanoramicScalogramData, String> {
    let dir = state
        .data_root
        .join("outputs/spectrograms/cwt")
        .join(format!("condition={condition}"))
        .join(format!("bearing_id={bearing_id}"));

    if !dir.exists() {
        return Err(format!("Scalogram directory not found: {}", dir.display()));
    }

    // Count .npy files to determine total_files
    let total_files = std::fs::read_dir(&dir)
        .map_err(|e| format!("Failed to read scalogram directory: {e}"))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension().map_or(false, |ext| ext == "npy")
        })
        .count() as u32;

    if total_files == 0 {
        return Err("No scalogram files found".to_string());
    }

    // Subsample large bearings: target ~500 columns max
    let max_columns: u32 = 500;
    let step = if total_files > max_columns {
        (total_files + max_columns - 1) / max_columns // ceil division
    } else {
        1
    };

    let num_scales = 64usize;
    let mut h_panorama: Vec<Vec<f32>> = vec![Vec::new(); num_scales];
    let mut v_panorama: Vec<Vec<f32>> = vec![Vec::new(); num_scales];
    let mut file_indices: Vec<u32> = Vec::new();

    // Load every step-th file (0-indexed), always include the last file
    let mut file_idx: u32 = 0;
    while file_idx < total_files {
        let npy_path = dir.join(format!("{}.npy", file_idx + 1)); // 0→1 indexed

        if npy_path.exists() {
            match npy_loader::load_scalogram_means(&npy_path) {
                Ok((h_means, v_means)) => {
                    for (s, (h, v)) in h_means.into_iter().zip(v_means).enumerate() {
                        if s < num_scales {
                            h_panorama[s].push(h);
                            v_panorama[s].push(v);
                        }
                    }
                    file_indices.push(file_idx);
                }
                Err(e) => {
                    log::warn!("Skipping file_idx {file_idx}: {e}");
                }
            }
        }

        file_idx += step;
    }

    // Always include the last file if not already included
    let last_idx = total_files - 1;
    if file_indices.last() != Some(&last_idx) {
        let npy_path = dir.join(format!("{}.npy", last_idx + 1));
        if npy_path.exists() {
            if let Ok((h_means, v_means)) = npy_loader::load_scalogram_means(&npy_path) {
                for (s, (h, v)) in h_means.into_iter().zip(v_means).enumerate() {
                    if s < num_scales {
                        h_panorama[s].push(h);
                        v_panorama[s].push(v);
                    }
                }
                file_indices.push(last_idx);
            }
        }
    }

    // Onset markers (same logic as health.rs)
    let onset_manual = state.onset_manual.get(&bearing_id).map(|entry| OnsetMarker {
        file_idx: entry.onset_file_idx,
        label: "manual".to_string(),
        confidence: Some(entry.confidence.clone()),
        method: Some(entry.detection_method.clone()),
    });

    let onset_auto = state.onset_auto.get(&bearing_id).map(|entry| OnsetMarker {
        file_idx: entry.onset_file_idx,
        label: "auto".to_string(),
        confidence: None,
        method: Some(entry.detector_method.clone()),
    });

    Ok(PanoramicScalogramData {
        h_panorama,
        v_panorama,
        file_indices,
        freq_min: 10.0,
        freq_max: 6000.0,
        num_scales: num_scales as u32,
        total_files,
        step,
        onset_manual,
        onset_auto,
    })
}
