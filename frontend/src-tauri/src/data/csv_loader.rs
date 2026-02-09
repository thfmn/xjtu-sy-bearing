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

use crate::models::{AutoOnsetEntry, FeatureRow};
use std::collections::HashMap;
use std::path::Path;

/// Load features_v2.csv and return (rows, bearing_id -> row indices index).
///
/// The CSV has 73 columns but we only need 7 of them. Rather than using serde
/// derive (which would require listing or ignoring all 73 fields), we parse
/// the header row to find column indices by name, then extract values from
/// each `StringRecord` positionally. This is robust to column reordering.
pub fn load_features(
    path: &Path,
) -> Result<(Vec<FeatureRow>, HashMap<String, Vec<usize>>), String> {
    let mut rdr = csv::Reader::from_path(path)
        .map_err(|e| format!("Failed to open features CSV: {e}"))?;

    // Resolve header column positions by name
    let headers = rdr
        .headers()
        .map_err(|e| format!("Failed to read CSV headers: {e}"))?
        .clone();

    let col = |name: &str| -> Result<usize, String> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("Missing column '{name}' in features CSV"))
    };

    let i_condition = col("condition")?;
    let i_bearing = col("bearing_id")?;
    let i_file_idx = col("file_idx")?;
    let i_h_kurtosis = col("h_kurtosis")?;
    let i_v_kurtosis = col("v_kurtosis")?;
    let i_h_rms = col("h_rms")?;
    let i_v_rms = col("v_rms")?;

    let mut rows = Vec::with_capacity(10_000);
    let mut index: HashMap<String, Vec<usize>> = HashMap::new();

    for (row_num, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| format!("CSV row {row_num}: {e}"))?;

        let row = FeatureRow {
            condition: record[i_condition].to_string(),
            bearing_id: record[i_bearing].to_string(),
            file_idx: record[i_file_idx]
                .parse::<u32>()
                .map_err(|e| format!("Row {row_num} file_idx: {e}"))?,
            h_kurtosis: record[i_h_kurtosis]
                .parse::<f64>()
                .map_err(|e| format!("Row {row_num} h_kurtosis: {e}"))?,
            v_kurtosis: record[i_v_kurtosis]
                .parse::<f64>()
                .map_err(|e| format!("Row {row_num} v_kurtosis: {e}"))?,
            h_rms: record[i_h_rms]
                .parse::<f64>()
                .map_err(|e| format!("Row {row_num} h_rms: {e}"))?,
            v_rms: record[i_v_rms]
                .parse::<f64>()
                .map_err(|e| format!("Row {row_num} v_rms: {e}"))?,
        };

        index
            .entry(row.bearing_id.clone())
            .or_default()
            .push(rows.len());
        rows.push(row);
    }

    log::info!("Loaded {} feature rows for {} bearings", rows.len(), index.len());
    Ok((rows, index))
}

/// Load onset_labels_auto.csv and return a HashMap keyed by bearing_id.
///
/// The `onset_file_idx` column may contain float strings like "72.0", so we
/// parse as f64 first and then truncate to u32.
pub fn load_auto_onsets(
    path: &Path,
) -> Result<HashMap<String, AutoOnsetEntry>, String> {
    let mut rdr = csv::Reader::from_path(path)
        .map_err(|e| format!("Failed to open auto onset CSV: {e}"))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("Failed to read auto onset CSV headers: {e}"))?
        .clone();

    let col = |name: &str| -> Result<usize, String> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("Missing column '{name}' in auto onset CSV"))
    };

    let i_bearing = col("bearing_id")?;
    let i_condition = col("condition")?;
    let i_onset = col("onset_file_idx")?;
    let i_method = col("detector_method")?;

    let mut map = HashMap::new();

    for (row_num, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| format!("Auto onset row {row_num}: {e}"))?;

        // onset_file_idx may be "72" or "72.0" — handle both
        let onset_str = &record[i_onset];
        let onset_idx = onset_str
            .parse::<f64>()
            .map(|v| v as u32)
            .or_else(|_| onset_str.parse::<u32>())
            .map_err(|e| format!("Row {row_num} onset_file_idx '{onset_str}': {e}"))?;

        let entry = AutoOnsetEntry {
            bearing_id: record[i_bearing].to_string(),
            condition: record[i_condition].to_string(),
            onset_file_idx: onset_idx,
            detector_method: record[i_method].to_string(),
        };

        map.insert(entry.bearing_id.clone(), entry);
    }

    log::info!("Loaded {} auto onset entries", map.len());
    Ok(map)
}
