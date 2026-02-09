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

use crate::models::{
    AppData, ConditionInfo, DatasetInfo, HealthIndicatorData, OnsetMarker,
};

/// Return dataset structure: 3 operating conditions with 5 bearings each.
///
/// This is a static response derived from the XJTU-SY dataset layout.
/// Rather than scanning the filesystem, we hardcode the known structure
/// since the dataset is fixed and well-defined (15 bearings total).
#[tauri::command]
pub fn get_dataset_info(state: tauri::State<'_, AppData>) -> Result<DatasetInfo, String> {
    // Build from the features index to stay in sync with loaded data
    let mut condition_map: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();

    for row in &state.features {
        condition_map
            .entry(row.condition.clone())
            .or_default();
    }

    // Collect unique bearing_ids per condition
    let mut condition_bearings: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();

    for row in &state.features {
        condition_bearings
            .entry(row.condition.clone())
            .or_default()
            .insert(row.bearing_id.clone());
    }

    let conditions = condition_bearings
        .into_iter()
        .map(|(condition, bearings)| ConditionInfo {
            condition,
            bearings: bearings.into_iter().collect(),
        })
        .collect();

    Ok(DatasetInfo { conditions })
}

/// Return health indicator time series for a specific bearing.
///
/// Computes averaged kurtosis and RMS from horizontal + vertical channels,
/// then attaches manual and auto onset markers if they exist.
#[tauri::command]
pub fn get_health_indicators(
    condition: String,
    bearing_id: String,
    state: tauri::State<'_, AppData>,
) -> Result<HealthIndicatorData, String> {
    let indices = state
        .features_by_bearing
        .get(&bearing_id)
        .ok_or_else(|| format!("No data found for bearing '{bearing_id}'"))?;

    // Collect and sort rows by file_idx
    let mut rows: Vec<_> = indices
        .iter()
        .map(|&i| &state.features[i])
        .filter(|r| r.condition == condition)
        .collect();

    if rows.is_empty() {
        return Err(format!(
            "No data for bearing '{bearing_id}' under condition '{condition}'"
        ));
    }

    rows.sort_by_key(|r| r.file_idx);

    let file_indices: Vec<u32> = rows.iter().map(|r| r.file_idx).collect();
    let kurtosis_avg: Vec<f64> = rows
        .iter()
        .map(|r| (r.h_kurtosis + r.v_kurtosis) / 2.0)
        .collect();
    let rms_avg: Vec<f64> = rows
        .iter()
        .map(|r| (r.h_rms + r.v_rms) / 2.0)
        .collect();
    let total_files = rows.len() as u32;

    // Look up manual onset
    let onset_manual = state.onset_manual.get(&bearing_id).map(|entry| OnsetMarker {
        file_idx: entry.onset_file_idx,
        label: "manual".to_string(),
        confidence: Some(entry.confidence.clone()),
        method: Some(entry.detection_method.clone()),
    });

    // Look up auto onset
    let onset_auto = state.onset_auto.get(&bearing_id).map(|entry| OnsetMarker {
        file_idx: entry.onset_file_idx,
        label: "auto".to_string(),
        confidence: None,
        method: Some(entry.detector_method.clone()),
    });

    Ok(HealthIndicatorData {
        bearing_id,
        condition,
        file_indices,
        kurtosis_avg,
        rms_avg,
        total_files,
        onset_manual,
        onset_auto,
    })
}
