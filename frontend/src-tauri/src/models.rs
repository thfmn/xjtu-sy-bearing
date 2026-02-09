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

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Input structs (deserialized from data files)
// ---------------------------------------------------------------------------

/// A single row from features_v2.csv — we only keep 7 of the 73 columns.
/// Deserialized manually via header-based column selection (see csv_loader).
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub condition: String,
    pub bearing_id: String,
    pub file_idx: u32,
    pub h_kurtosis: f64,
    pub v_kurtosis: f64,
    pub h_rms: f64,
    pub v_rms: f64,
}

/// An entry from configs/onset_labels.yaml (manual labels).
#[derive(Debug, Deserialize)]
pub struct OnsetEntry {
    pub bearing_id: String,
    pub condition: String,
    pub onset_file_idx: u32,
    pub confidence: String,
    pub detection_method: String,
    pub onset_range: Option<Vec<u32>>,
    pub notes: Option<String>,
}

/// Wrapper for the YAML top-level structure: `{ bearings: [...] }`.
#[derive(Debug, Deserialize)]
pub struct OnsetLabelsFile {
    pub bearings: Vec<OnsetEntry>,
}

/// An entry from outputs/onset/onset_labels_auto.csv.
/// Some idx columns are stored as floats in the CSV (e.g. "72.0") so we
/// deserialize onset_file_idx from the string manually.
#[derive(Debug, Clone)]
pub struct AutoOnsetEntry {
    pub bearing_id: String,
    pub condition: String,
    pub onset_file_idx: u32,
    pub detector_method: String,
}

// ---------------------------------------------------------------------------
// Response structs (serialized to JSON for the Svelte frontend via IPC)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ConditionInfo {
    pub condition: String,
    pub bearings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DatasetInfo {
    pub conditions: Vec<ConditionInfo>,
}

#[derive(Debug, Serialize)]
pub struct OnsetMarker {
    pub file_idx: u32,
    pub label: String,
    pub confidence: Option<String>,
    pub method: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HealthIndicatorData {
    pub bearing_id: String,
    pub condition: String,
    pub file_indices: Vec<u32>,
    pub kurtosis_avg: Vec<f64>,
    pub rms_avg: Vec<f64>,
    pub total_files: u32,
    pub onset_manual: Option<OnsetMarker>,
    pub onset_auto: Option<OnsetMarker>,
}

#[derive(Debug, Serialize)]
pub struct ScalogramData {
    pub h_channel: Vec<Vec<f32>>,
    pub v_channel: Vec<Vec<f32>>,
    pub freq_min: f32,
    pub freq_max: f32,
    pub num_scales: u32,
    pub num_time_bins: u32,
}

/// Panoramic CWT scalogram spanning a bearing's full run-to-failure lifetime.
///
/// Each column represents one file's mean CWT power spectrum (averaged across
/// the 128 time bins). This gives a "lifetime spectrogram" where the x-axis
/// is file index (bearing lifetime) and the y-axis is frequency.
#[derive(Debug, Serialize)]
pub struct PanoramicScalogramData {
    /// Mean H-channel CWT power per scale, one column per sampled file.
    /// Shape: num_scales × num_sampled_files (row 0 = lowest frequency).
    pub h_panorama: Vec<Vec<f32>>,
    /// Mean V-channel CWT power per scale, one column per sampled file.
    pub v_panorama: Vec<Vec<f32>>,
    /// The actual 0-indexed file indices that were sampled.
    pub file_indices: Vec<u32>,
    pub freq_min: f32,
    pub freq_max: f32,
    pub num_scales: u32,
    pub total_files: u32,
    /// Subsampling step used (1 = every file, 5 = every 5th file).
    pub step: u32,
    pub onset_manual: Option<OnsetMarker>,
    pub onset_auto: Option<OnsetMarker>,
}

#[derive(Debug, Serialize)]
pub struct AudioStage {
    pub label: String,
    pub stage_key: String,
    pub file_path: String,
    /// The 0-indexed file index corresponding to this lifecycle stage.
    /// Computed from the stage percentage and total file count:
    /// `int((pct / 100) * (total_files - 1))`.
    pub file_idx: u32,
}

#[derive(Debug, Serialize)]
pub struct AudioFilesResponse {
    pub bearing_id: String,
    pub condition: String,
    pub stages: Vec<AudioStage>,
}

#[derive(Debug, Serialize)]
pub struct WaveformData {
    pub time_ms: Vec<f64>,
    pub amplitude: Vec<f64>,
    pub sample_rate: u32,
    pub duration_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct AudioData {
    pub base64: String,
    pub mime_type: String,
}

// ---------------------------------------------------------------------------
// Application state (loaded once at startup, shared via Tauri's managed state)
// ---------------------------------------------------------------------------

pub struct AppData {
    pub data_root: PathBuf,
    pub features: Vec<FeatureRow>,
    pub features_by_bearing: HashMap<String, Vec<usize>>,
    pub onset_manual: HashMap<String, OnsetEntry>,
    pub onset_auto: HashMap<String, AutoOnsetEntry>,
}
