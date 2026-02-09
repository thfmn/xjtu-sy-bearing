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

/** Operating conditions and their bearing IDs (fallback if backend unreachable). */
export const CONDITIONS: Record<string, string[]> = {
  "35Hz12kN": [
    "Bearing1_1",
    "Bearing1_2",
    "Bearing1_3",
    "Bearing1_4",
    "Bearing1_5",
  ],
  "37.5Hz11kN": [
    "Bearing2_1",
    "Bearing2_2",
    "Bearing2_3",
    "Bearing2_4",
    "Bearing2_5",
  ],
  "40Hz10kN": [
    "Bearing3_1",
    "Bearing3_2",
    "Bearing3_3",
    "Bearing3_4",
    "Bearing3_5",
  ],
};

/** Human-readable condition labels. */
export const CONDITION_LABELS: Record<string, string> = {
  "35Hz12kN": "35 Hz / 12 kN",
  "37.5Hz11kN": "37.5 Hz / 11 kN",
  "40Hz10kN": "40 Hz / 10 kN",
};

/** Audio lifecycle stage display names. */
export const AUDIO_STAGE_LABELS: Record<string, string> = {
  healthy_0pct: "Healthy (0%)",
  degrading_50pct: "Degrading (50%)",
  failed_100pct: "Failed (100%)",
};

/** Number of bearings per condition. */
export const BEARINGS_PER_CONDITION = 5;

/** Total number of bearings in the dataset. */
export const TOTAL_BEARINGS = 15;
