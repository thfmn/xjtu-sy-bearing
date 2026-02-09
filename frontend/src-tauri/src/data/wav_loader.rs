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

use crate::models::WaveformData;
use std::path::Path;

/// Target number of points to send to the frontend for waveform display.
/// Sending all 32k+ samples would bloat the JSON payload; ~5000 is plenty
/// for a responsive line chart.
const TARGET_POINTS: usize = 5000;

/// Load a WAV file and subsample it to ~TARGET_POINTS for frontend display.
///
/// Assumes 16-bit PCM mono (the format produced by our audio generation
/// script). Samples are normalized to [-1.0, 1.0] by dividing by i16::MAX.
pub fn load_waveform(path: &Path) -> Result<WaveformData, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open WAV file {}: {e}", path.display()))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let total_samples: usize = reader.len() as usize;

    // Compute decimation stride
    let stride = (total_samples / TARGET_POINTS).max(1);

    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to read WAV samples: {e}"))?;

    let mut time_ms = Vec::with_capacity(total_samples / stride + 1);
    let mut amplitude = Vec::with_capacity(total_samples / stride + 1);
    let scale = f64::from(i16::MAX);

    for (i, &sample) in samples.iter().enumerate() {
        if i % stride == 0 {
            time_ms.push(i as f64 / f64::from(sample_rate) * 1000.0);
            amplitude.push(f64::from(sample) / scale);
        }
    }

    let duration_ms = total_samples as f64 / f64::from(sample_rate) * 1000.0;

    Ok(WaveformData {
        time_ms,
        amplitude,
        sample_rate,
        duration_ms,
    })
}
