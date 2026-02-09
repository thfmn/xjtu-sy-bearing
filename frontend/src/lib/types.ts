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

/** Mirrors Rust ConditionInfo */
export interface ConditionInfo {
  condition: string;
  bearings: string[];
}

/** Mirrors Rust DatasetInfo */
export interface DatasetInfo {
  conditions: ConditionInfo[];
}

/** Mirrors Rust OnsetMarker */
export interface OnsetMarker {
  file_idx: number;
  label: string;
  confidence: string | null;
  method: string | null;
}

/** Mirrors Rust HealthIndicatorData */
export interface HealthIndicatorData {
  bearing_id: string;
  condition: string;
  file_indices: number[];
  kurtosis_avg: number[];
  rms_avg: number[];
  total_files: number;
  onset_manual: OnsetMarker | null;
  onset_auto: OnsetMarker | null;
}

/** Mirrors Rust ScalogramData */
export interface ScalogramData {
  h_channel: number[][];
  v_channel: number[][];
  freq_min: number;
  freq_max: number;
  num_scales: number;
  num_time_bins: number;
}

/** Mirrors Rust PanoramicScalogramData */
export interface PanoramicScalogramData {
  h_panorama: number[][];
  v_panorama: number[][];
  file_indices: number[];
  freq_min: number;
  freq_max: number;
  num_scales: number;
  total_files: number;
  step: number;
  onset_manual: OnsetMarker | null;
  onset_auto: OnsetMarker | null;
}

/** Mirrors Rust AudioStage */
export interface AudioStage {
  label: string;
  stage_key: string;
  file_path: string;
  file_idx: number;
}

/** Mirrors Rust AudioFilesResponse */
export interface AudioFilesResponse {
  bearing_id: string;
  condition: string;
  stages: AudioStage[];
}

/** Mirrors Rust WaveformData */
export interface WaveformData {
  time_ms: number[];
  amplitude: number[];
  sample_rate: number;
  duration_ms: number;
}

/** Mirrors Rust AudioData — base64-encoded WAV bytes */
export interface AudioData {
  base64: string;
  mime_type: string;
}
