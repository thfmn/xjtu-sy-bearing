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

"""DTA-MLP: Dynamic Temporal Attention with Mixed MLP for bearing RUL prediction.

Reproduction of the architecture described in:
    Jin et al. (2025) "Enhanced bearing RUL prediction based on dynamic temporal
    attention and mixed MLP", Autonomous Intelligent Systems.

Architecture: CWT Input → CNN Frontend → Transformer (DTA) → CT-MLP → RUL

Note: The paper does not provide complete architectural details. This
implementation fills in missing details with standard choices and documents
all assumptions. See model.py for specifics.
"""

from .model import DTAMLPConfig, build_dta_mlp, create_default_dta_mlp
