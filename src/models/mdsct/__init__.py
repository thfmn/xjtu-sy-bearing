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

Faithful reproduction of the architecture described in:
    Sun et al. (2024) "Remaining useful life prognostics of bearings based on
    convolution attention networks and enhanced transformer", Heliyon, e38317.

Architecture (Table 2, Fig. 7):
    Input (32768, 2) → MinMaxNormalize (Eq. 19)
    → Conv1D stem (1 channel) → AAP1(1024) → Dropout
    → MixerBlock × 3:
        ├─ MDSC Attention: MaxPool → Bottleneck(16) → DSC(24×3) → Concat(72ch)
        │   → BN → AdaptH_Swish(δ) → Dropout → ECA → Residual
        └─ PPSformer: AAP2(96) → Conv1D(16) → PatchEmbed → ProbSparse MHA
            → FFN(256→128) → AAP3(1024)
        → Concatenate (200ch)
    → AAP4(64) → Flatten → Dense(1, sigmoid) → RUL
"""

from .model import MDSCTConfig, build_mdsct, create_default_mdsct
