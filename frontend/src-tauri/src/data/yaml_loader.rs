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

use crate::models::{OnsetEntry, OnsetLabelsFile};
use std::collections::HashMap;
use std::path::Path;

/// Parse configs/onset_labels.yaml and return a HashMap keyed by bearing_id.
///
/// The YAML has the shape `{ bearings: [ { bearing_id, condition, ... }, ... ] }`.
/// We flatten it into a lookup map for O(1) access by bearing_id.
pub fn load_onset_labels(
    path: &Path,
) -> Result<HashMap<String, OnsetEntry>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read onset YAML: {e}"))?;

    let file: OnsetLabelsFile = serde_yaml::from_str(&contents)
        .map_err(|e| format!("Failed to parse onset YAML: {e}"))?;

    let mut map = HashMap::new();
    for entry in file.bearings {
        map.insert(entry.bearing_id.clone(), entry);
    }

    log::info!("Loaded {} manual onset entries", map.len());
    Ok(map)
}
