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

import { invoke } from "@tauri-apps/api/core";
import type {
  DatasetInfo,
  HealthIndicatorData,
  BearingOverviewResponse,
  ScalogramData,
  PanoramicScalogramData,
  AudioFilesResponse,
  AudioData,
  WaveformData,
} from "./types";

export async function getDatasetInfo(): Promise<DatasetInfo> {
  return invoke<DatasetInfo>("get_dataset_info");
}

export async function getBearingOverview(): Promise<BearingOverviewResponse> {
  return invoke<BearingOverviewResponse>("get_bearing_overview");
}

export async function getHealthIndicators(
  condition: string,
  bearingId: string,
): Promise<HealthIndicatorData> {
  return invoke<HealthIndicatorData>("get_health_indicators", {
    condition,
    bearingId,
  });
}

export async function getScalogram(
  condition: string,
  bearingId: string,
  fileIdx: number,
): Promise<ScalogramData> {
  return invoke<ScalogramData>("get_scalogram", {
    condition,
    bearingId,
    fileIdx,
  });
}

export async function getScalogramFileCount(
  condition: string,
  bearingId: string,
): Promise<number> {
  return invoke<number>("get_scalogram_file_count", { condition, bearingId });
}

export async function getPanoramicScalogram(
  condition: string,
  bearingId: string,
): Promise<PanoramicScalogramData> {
  return invoke<PanoramicScalogramData>("get_panoramic_scalogram", {
    condition,
    bearingId,
  });
}

export async function getAudioFiles(
  condition: string,
  bearingId: string,
): Promise<AudioFilesResponse> {
  return invoke<AudioFilesResponse>("get_audio_files", {
    condition,
    bearingId,
  });
}

export async function getWaveform(
  condition: string,
  bearingId: string,
  stageKey: string,
): Promise<WaveformData> {
  return invoke<WaveformData>("get_waveform", {
    condition,
    bearingId,
    stageKey,
  });
}

export async function getAudioData(
  condition: string,
  bearingId: string,
  stageKey: string,
): Promise<AudioData> {
  return invoke<AudioData>("get_audio_data", {
    condition,
    bearingId,
    stageKey,
  });
}
