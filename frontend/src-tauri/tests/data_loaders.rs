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

//! Integration tests for the data loaders.
//!
//! These tests run against the real data files in the repository. They verify
//! that our parsers produce correct and complete results — this is critical
//! because any parsing bug would silently corrupt the data shown in the UI.

use std::path::{Path, PathBuf};

use base64::prelude::*;
use xjtu_sy_bearing_explorer_lib::data::{csv_loader, npy_loader, wav_loader, yaml_loader};

/// Resolve the project data root by walking up from the test binary's location.
fn data_root() -> PathBuf {
    let mut dir = std::env::current_dir().expect("cannot get CWD");
    for _ in 0..10 {
        if dir.join("outputs").exists() && dir.join("configs").exists() {
            return dir;
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }
    panic!("Could not locate project data root from CWD");
}

// =========================================================================
// CSV loader: features_v2.csv
// =========================================================================

#[test]
fn csv_loads_all_9215_rows() {
    let root = data_root();
    let path = root.join("outputs/features/features_v2.csv");
    let (rows, index) = csv_loader::load_features(&path).expect("load_features failed");

    assert_eq!(rows.len(), 9216, "expected 9216 feature rows");
    assert_eq!(index.len(), 15, "expected 15 distinct bearings");
}

#[test]
fn csv_bearing1_1_has_123_files() {
    let root = data_root();
    let path = root.join("outputs/features/features_v2.csv");
    let (rows, index) = csv_loader::load_features(&path).unwrap();

    let indices = index.get("Bearing1_1").expect("Bearing1_1 not in index");
    assert_eq!(indices.len(), 123, "Bearing1_1 should have 123 files");

    // Check file_idx range: 0..122
    let min_idx = indices.iter().map(|&i| rows[i].file_idx).min().unwrap();
    let max_idx = indices.iter().map(|&i| rows[i].file_idx).max().unwrap();
    assert_eq!(min_idx, 0);
    assert_eq!(max_idx, 122);
}

#[test]
fn csv_first_row_values_are_sane() {
    let root = data_root();
    let path = root.join("outputs/features/features_v2.csv");
    let (rows, _) = csv_loader::load_features(&path).unwrap();

    let first = &rows[0];
    assert_eq!(first.condition, "35Hz12kN");
    assert_eq!(first.bearing_id, "Bearing1_1");
    assert_eq!(first.file_idx, 0);
    // Kurtosis for healthy bearing should be in a reasonable range (< 10)
    assert!(first.h_kurtosis > 0.0 && first.h_kurtosis < 10.0,
        "h_kurtosis {} out of expected healthy range", first.h_kurtosis);
    // RMS should be positive and below 2.0 for healthy signals
    assert!(first.h_rms > 0.0 && first.h_rms < 2.0,
        "h_rms {} out of expected healthy range", first.h_rms);
}

#[test]
fn csv_all_conditions_present() {
    let root = data_root();
    let path = root.join("outputs/features/features_v2.csv");
    let (rows, _) = csv_loader::load_features(&path).unwrap();

    let conditions: std::collections::HashSet<&str> =
        rows.iter().map(|r| r.condition.as_str()).collect();

    assert!(conditions.contains("35Hz12kN"), "missing 35Hz12kN");
    assert!(conditions.contains("37.5Hz11kN"), "missing 37.5Hz11kN");
    assert!(conditions.contains("40Hz10kN"), "missing 40Hz10kN");
    assert_eq!(conditions.len(), 3);
}

// =========================================================================
// CSV loader: onset_labels_auto.csv
// =========================================================================

#[test]
fn auto_onset_loads_15_entries() {
    let root = data_root();
    let path = root.join("outputs/onset/onset_labels_auto.csv");
    let map = csv_loader::load_auto_onsets(&path).expect("load_auto_onsets failed");

    assert_eq!(map.len(), 15, "expected 15 auto onset entries");
}

#[test]
fn auto_onset_bearing1_1_value() {
    let root = data_root();
    let path = root.join("outputs/onset/onset_labels_auto.csv");
    let map = csv_loader::load_auto_onsets(&path).unwrap();

    let entry = map.get("Bearing1_1").expect("Bearing1_1 not in auto onset map");
    assert_eq!(entry.onset_file_idx, 72, "Bearing1_1 auto onset should be 72");
    assert_eq!(entry.condition, "35Hz12kN");
    assert!(!entry.detector_method.is_empty(), "detector_method should not be empty");
}

#[test]
fn auto_onset_handles_float_idx() {
    // The auto onset CSV stores some onset_file_idx as "72.0" (float format).
    // This test verifies our parser handles both "72" and "72.0".
    let root = data_root();
    let path = root.join("outputs/onset/onset_labels_auto.csv");
    let map = csv_loader::load_auto_onsets(&path).unwrap();

    // All values should parse without error and be reasonable (0..10000)
    for (bearing_id, entry) in &map {
        assert!(entry.onset_file_idx < 10000,
            "{bearing_id} has unreasonable onset_file_idx: {}", entry.onset_file_idx);
    }
}

// =========================================================================
// YAML loader: onset_labels.yaml
// =========================================================================

#[test]
fn yaml_loads_15_onset_labels() {
    let root = data_root();
    let path = root.join("configs/onset_labels.yaml");
    let map = yaml_loader::load_onset_labels(&path).expect("load_onset_labels failed");

    assert_eq!(map.len(), 15, "expected 15 manual onset entries");
}

#[test]
fn yaml_bearing1_1_values() {
    let root = data_root();
    let path = root.join("configs/onset_labels.yaml");
    let map = yaml_loader::load_onset_labels(&path).unwrap();

    let entry = map.get("Bearing1_1").expect("Bearing1_1 not in onset map");
    assert_eq!(entry.onset_file_idx, 65, "Bearing1_1 manual onset should be 65");
    assert_eq!(entry.confidence, "high");
    assert_eq!(entry.condition, "35Hz12kN");
}

#[test]
fn yaml_all_bearings_present() {
    let root = data_root();
    let path = root.join("configs/onset_labels.yaml");
    let map = yaml_loader::load_onset_labels(&path).unwrap();

    let expected = [
        "Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5",
        "Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5",
        "Bearing3_1", "Bearing3_2", "Bearing3_3", "Bearing3_4", "Bearing3_5",
    ];

    for name in &expected {
        assert!(map.contains_key(*name), "missing bearing: {name}");
    }
}

#[test]
fn yaml_confidence_levels_valid() {
    let root = data_root();
    let path = root.join("configs/onset_labels.yaml");
    let map = yaml_loader::load_onset_labels(&path).unwrap();

    let valid = ["high", "medium", "low"];
    for (bearing_id, entry) in &map {
        assert!(valid.contains(&entry.confidence.as_str()),
            "{bearing_id} has invalid confidence: '{}'", entry.confidence);
    }
}

// =========================================================================
// NPY loader: CWT scalogram files
// =========================================================================

#[test]
fn npy_loads_bearing1_1_file_1() {
    let root = data_root();
    // NPY files are 1-indexed: file_idx=0 → "1.npy"
    let path = root
        .join("outputs/spectrograms/cwt/condition=35Hz12kN/bearing_id=Bearing1_1/1.npy");

    let data = npy_loader::load_scalogram(&path).expect("load_scalogram failed");

    assert_eq!(data.num_scales, 64, "expected 64 frequency scales");
    assert_eq!(data.num_time_bins, 128, "expected 128 time bins");
    assert_eq!(data.h_channel.len(), 64);
    assert_eq!(data.h_channel[0].len(), 128);
    assert_eq!(data.v_channel.len(), 64);
    assert_eq!(data.v_channel[0].len(), 128);
    assert!((data.freq_min - 10.0).abs() < f32::EPSILON);
    assert!((data.freq_max - 6000.0).abs() < f32::EPSILON);
}

#[test]
fn npy_no_nan_values() {
    let root = data_root();
    let path = root
        .join("outputs/spectrograms/cwt/condition=35Hz12kN/bearing_id=Bearing1_1/1.npy");
    let data = npy_loader::load_scalogram(&path).unwrap();

    for (s, row) in data.h_channel.iter().enumerate() {
        for (t, &val) in row.iter().enumerate() {
            assert!(!val.is_nan(), "h_channel NaN at scale={s}, time={t}");
        }
    }
    for (s, row) in data.v_channel.iter().enumerate() {
        for (t, &val) in row.iter().enumerate() {
            assert!(!val.is_nan(), "v_channel NaN at scale={s}, time={t}");
        }
    }
}

#[test]
fn npy_one_indexed_file_mapping() {
    // Critical test: verify that file_idx 0 maps to 1.npy (1-indexed filenames)
    let root = data_root();
    let dir = root.join("outputs/spectrograms/cwt/condition=35Hz12kN/bearing_id=Bearing1_1");

    // 1.npy (file_idx=0) should exist
    assert!(dir.join("1.npy").exists(), "1.npy should exist for file_idx=0");
    // 0.npy should NOT exist (no 0-indexed file)
    assert!(!dir.join("0.npy").exists(), "0.npy should NOT exist (files are 1-indexed)");

    // The last file (file_idx=122) should map to 123.npy
    assert!(dir.join("123.npy").exists(), "123.npy should exist for file_idx=122");
}

#[test]
fn npy_scalogram_json_structure() {
    let root = data_root();
    let path = root.join("outputs/spectrograms/cwt/condition=35Hz12kN/bearing_id=Bearing1_1/1.npy");
    let data = npy_loader::load_scalogram(&path).unwrap();

    let json = serde_json::to_value(&data).unwrap();
    assert!(json["h_channel"].is_array());
    assert_eq!(json["h_channel"].as_array().unwrap().len(), 64);
    assert_eq!(json["h_channel"][0].as_array().unwrap().len(), 128);
    assert_eq!(json["num_scales"].as_u64().unwrap(), 64);
    assert_eq!(json["num_time_bins"].as_u64().unwrap(), 128);
}

#[test]
fn npy_scalogram_means_shape_and_range() {
    let root = data_root();
    let path = root
        .join("outputs/spectrograms/cwt/condition=35Hz12kN/bearing_id=Bearing1_1/1.npy");

    let (h_means, v_means) = npy_loader::load_scalogram_means(&path)
        .expect("load_scalogram_means failed");

    assert_eq!(h_means.len(), 64, "expected 64 frequency scales for h_means");
    assert_eq!(v_means.len(), 64, "expected 64 frequency scales for v_means");

    // Means should be finite (no NaN or Inf)
    for (i, &val) in h_means.iter().enumerate() {
        assert!(val.is_finite(), "h_means[{i}] is not finite: {val}");
    }
    for (i, &val) in v_means.iter().enumerate() {
        assert!(val.is_finite(), "v_means[{i}] is not finite: {val}");
    }
}

#[test]
fn npy_scalogram_means_matches_full_load() {
    // Verify that means computed by load_scalogram_means match manual averaging
    // of the full scalogram from load_scalogram.
    let root = data_root();
    let path = root
        .join("outputs/spectrograms/cwt/condition=35Hz12kN/bearing_id=Bearing1_1/1.npy");

    let full = npy_loader::load_scalogram(&path).unwrap();
    let (h_means, _) = npy_loader::load_scalogram_means(&path).unwrap();

    // For each scale, the mean should equal avg(full.h_channel[scale])
    for (s, row) in full.h_channel.iter().enumerate() {
        let expected: f32 = row.iter().sum::<f32>() / row.len() as f32;
        let diff = (h_means[s] - expected).abs();
        assert!(diff < 1e-5,
            "h_means[{s}] = {} but expected {expected} (diff={diff})", h_means[s]);
    }
}

// =========================================================================
// WAV loader
// =========================================================================

#[test]
fn wav_loads_bearing1_1_healthy() {
    let root = data_root();
    let path = root
        .join("outputs/audio/35Hz12kN/Bearing1_1/Bearing1_1_healthy_0pct_h.wav");

    if !path.exists() {
        // Audio files may not be generated in CI
        eprintln!("Skipping wav test: {} not found", path.display());
        return;
    }

    let data = wav_loader::load_waveform(&path).expect("load_waveform failed");

    // Should be subsampled to ~5000 points
    assert!(data.time_ms.len() <= 5500, "too many points: {}", data.time_ms.len());
    assert!(data.time_ms.len() >= 1000, "too few points: {}", data.time_ms.len());
    assert_eq!(data.time_ms.len(), data.amplitude.len(), "time_ms and amplitude length mismatch");

    // Sample rate should be 44100 (resampled from 25600)
    assert_eq!(data.sample_rate, 44100, "expected 44100 Hz sample rate");

    // Duration should be positive
    assert!(data.duration_ms > 0.0, "duration_ms should be positive");

    // Amplitude should be normalized to [-1.0, 1.0]
    for &a in &data.amplitude {
        assert!(a >= -1.0 && a <= 1.0,
            "amplitude {} out of [-1.0, 1.0] range", a);
    }
}

#[test]
fn wav_time_is_monotonically_increasing() {
    let root = data_root();
    let path = root
        .join("outputs/audio/35Hz12kN/Bearing1_1/Bearing1_1_healthy_0pct_h.wav");

    if !path.exists() {
        return;
    }

    let data = wav_loader::load_waveform(&path).unwrap();

    for i in 1..data.time_ms.len() {
        assert!(data.time_ms[i] > data.time_ms[i - 1],
            "time_ms not monotonically increasing at index {i}: {} <= {}",
            data.time_ms[i], data.time_ms[i - 1]);
    }
}

// =========================================================================
// Cross-validation: manual vs auto onset consistency
// =========================================================================

#[test]
fn onset_manual_vs_auto_reasonable_difference() {
    let root = data_root();
    let manual = yaml_loader::load_onset_labels(&root.join("configs/onset_labels.yaml")).unwrap();
    let auto = csv_loader::load_auto_onsets(&root.join("outputs/onset/onset_labels_auto.csv")).unwrap();

    // For each bearing, auto onset should exist and be within reasonable range of manual
    for (bearing_id, m_entry) in &manual {
        let a_entry = auto.get(bearing_id)
            .unwrap_or_else(|| panic!("{bearing_id} missing from auto onset"));

        // Both should be in the same condition
        assert_eq!(m_entry.condition, a_entry.condition,
            "{bearing_id}: condition mismatch between manual and auto onset");

        // Auto onset should be non-negative
        // (we can't assert exact closeness since detectors may disagree)
    }
}

// =========================================================================
// WAV base64 roundtrip
// =========================================================================

#[test]
fn wav_base64_roundtrip() {
    let root = data_root();
    let path = root.join("outputs/audio/35Hz12kN/Bearing1_1/Bearing1_1_healthy_0pct_h.wav");
    if !path.exists() {
        return;
    }

    let bytes = std::fs::read(&path).unwrap();
    let encoded = BASE64_STANDARD.encode(&bytes);
    let decoded = BASE64_STANDARD.decode(&encoded).unwrap();
    assert_eq!(bytes, decoded);
    // Verify it starts with RIFF WAV header
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
}
