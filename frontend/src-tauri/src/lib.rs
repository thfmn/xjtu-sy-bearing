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

pub mod commands;
pub mod data;
pub mod models;

use models::AppData;

/// Initialize and run the Tauri application.
///
/// Data loading happens eagerly at startup: features CSV (~9215 rows),
/// manual onset YAML (15 entries), and auto onset CSV (15 entries) are all
/// parsed into memory and stored in `AppData` as managed Tauri state.
/// This avoids repeated disk I/O during the interactive session.
///
/// # Panics
/// Panics (via `expect`) if any data file is missing or malformed.
/// This is intentional — the app cannot function without its data files.
pub fn run() {
    env_logger::init();

    // Resolve the data root: the repository root is the parent of frontend/
    let data_root = std::env::current_exe()
        .ok()
        .and_then(|exe| {
            // In dev mode, the exe is deep inside target/debug/
            // Walk up to find the repo root by looking for Cargo.toml or outputs/
            let mut dir = exe.parent()?.to_path_buf();
            for _ in 0..10 {
                if dir.join("outputs").exists() && dir.join("configs").exists() {
                    return Some(dir);
                }
                dir = dir.parent()?.to_path_buf();
            }
            None
        })
        .unwrap_or_else(|| {
            // Fallback: try CWD and its ancestors
            let mut dir = std::env::current_dir().expect("Cannot get CWD");
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
            panic!("Could not locate project data root (expected outputs/ and configs/ directories)");
        });

    log::info!("Data root: {}", data_root.display());

    // Load features CSV
    let features_path = data_root.join("outputs/features/features_v2.csv");
    let (features, features_by_bearing) = data::csv_loader::load_features(&features_path)
        .expect("Failed to load features_v2.csv");

    // Load manual onset labels (YAML)
    let onset_yaml_path = data_root.join("configs/onset_labels.yaml");
    let onset_manual = data::yaml_loader::load_onset_labels(&onset_yaml_path)
        .expect("Failed to load onset_labels.yaml");

    // Load auto onset labels (CSV)
    let auto_onset_path = data_root.join("outputs/onset/onset_labels_auto.csv");
    let onset_auto = data::csv_loader::load_auto_onsets(&auto_onset_path)
        .expect("Failed to load onset_labels_auto.csv");

    let app_data = AppData {
        data_root,
        features,
        features_by_bearing,
        onset_manual,
        onset_auto,
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .manage(app_data)
        .invoke_handler(tauri::generate_handler![
            commands::health::get_dataset_info,
            commands::health::get_health_indicators,
            commands::health::get_bearing_overview,
            commands::scalogram::get_scalogram,
            commands::scalogram::get_scalogram_file_count,
            commands::scalogram::get_panoramic_scalogram,
            commands::audio::get_audio_files,
            commands::audio::get_waveform,
            commands::audio::get_audio_data,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
