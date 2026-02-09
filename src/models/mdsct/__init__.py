#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#  https://github.com/thfmn/xjtu-sy-bearing
#
#  This work is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the condition that the above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  For more information, visit: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   xjtu-sy-bearing onset and RUL prediction ML Pipeline

"""MDSCT: Multi-scale Depthwise Separable Convolution Transformer for bearing RUL.

Reproduction of the architecture described in:
    Li et al. (2024) "Remaining useful life prognostics of bearings based on
    convolution attention networks and enhanced transformer", Heliyon.

Architecture: Raw Signal → Initial Conv1D → MixerBlock (MDSC + ECA) × 3
              → ProbSparse Transformer → AdaptiveAvgPool → Dropout → FC → RUL

Note: The paper does not provide complete architectural details for all
components. This implementation fills in missing details with standard
choices and documents all assumptions. See model.py for specifics.
"""

from .model import MDSCTConfig, build_mdsct, create_default_mdsct
