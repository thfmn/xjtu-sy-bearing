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

use base64::prelude::*;

use crate::data::wav_loader;
use crate::models::{AppData, AudioData, AudioFilesResponse, AudioStage, WaveformData};

/// The three lifecycle stages we generate audio files for.
/// Each tuple is (human-readable label, stage key, lifecycle percentage).
const STAGES: &[(&str, &str, u32)] = &[
    ("Healthy (0%)", "healthy_0pct", 0),
    ("Degrading (50%)", "degrading_50pct", 50),
    ("Failed (100%)", "failed_100pct", 100),
];

/// List available WAV audio files for a bearing across its lifecycle stages.
///
/// Audio files are pre-generated at `outputs/audio/{condition}/{bearing_id}/`
/// with the naming pattern `{bearing_id}_{stage}_h.wav`. We check which
/// stages actually exist on disk and return their absolute paths.
#[tauri::command]
pub fn get_audio_files(
    condition: String,
    bearing_id: String,
    state: tauri::State<'_, AppData>,
) -> Result<AudioFilesResponse, String> {
    let audio_dir = state
        .data_root
        .join("outputs/audio")
        .join(&condition)
        .join(&bearing_id);

    // Determine total file count for this bearing to compute stage file indices.
    let total_files = state
        .features_by_bearing
        .get(&bearing_id)
        .map(|indices| indices.len() as u32)
        .unwrap_or(1);

    let mut stages = Vec::new();

    for &(label, stage_key, pct) in STAGES {
        let filename = format!("{bearing_id}_{stage_key}_h.wav");
        let path = audio_dir.join(&filename);

        if path.exists() {
            // Mirror Python's get_lifecycle_indices(): int((pct/100) * (N-1))
            let file_idx = if total_files > 1 {
                (pct as u64 * (total_files - 1) as u64 / 100) as u32
            } else {
                0
            };

            stages.push(AudioStage {
                label: label.to_string(),
                stage_key: stage_key.to_string(),
                file_path: path.to_string_lossy().to_string(),
                file_idx,
            });
        }
    }

    Ok(AudioFilesResponse {
        bearing_id,
        condition,
        stages,
    })
}

/// Load a WAV waveform for a specific bearing and lifecycle stage.
///
/// The waveform is subsampled to ~5000 points to keep the IPC payload
/// small enough for smooth frontend rendering.
#[tauri::command]
pub fn get_waveform(
    condition: String,
    bearing_id: String,
    stage_key: String,
    state: tauri::State<'_, AppData>,
) -> Result<WaveformData, String> {
    let filename = format!("{bearing_id}_{stage_key}_h.wav");
    let path = state
        .data_root
        .join("outputs/audio")
        .join(&condition)
        .join(&bearing_id)
        .join(&filename);

    if !path.exists() {
        return Err(format!("WAV file not found: {}", path.display()));
    }

    wav_loader::load_waveform(&path)
}

/// Return the raw WAV bytes as a base64-encoded string.
///
/// The frontend uses this to build a `data:audio/wav;base64,...` URL for the
/// `<audio>` element, bypassing the Tauri asset protocol which doesn't
/// reliably serve local WAV files.
#[tauri::command]
pub fn get_audio_data(
    condition: String,
    bearing_id: String,
    stage_key: String,
    state: tauri::State<'_, AppData>,
) -> Result<AudioData, String> {
    let filename = format!("{bearing_id}_{stage_key}_h.wav");
    let path = state
        .data_root
        .join("outputs/audio")
        .join(&condition)
        .join(&bearing_id)
        .join(&filename);

    if !path.exists() {
        return Err(format!("WAV file not found: {}", path.display()));
    }

    let bytes =
        std::fs::read(&path).map_err(|e| format!("Failed to read WAV file: {e}"))?;

    let encoded = BASE64_STANDARD.encode(&bytes);

    Ok(AudioData {
        base64: encoded,
        mime_type: "audio/wav".to_string(),
    })
}
