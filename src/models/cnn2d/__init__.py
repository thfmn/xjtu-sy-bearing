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

"""CNN2D: 2D CNN + Temporal architecture for RUL prediction.

This package implements the 2D CNN architecture that processes
spectrograms (STFT or CWT) through a 2D CNN backbone followed by
temporal aggregation for RUL prediction.

Architecture:
    Spectrogram (128, 128, 2)
        ↓
    DualChannelCNN2DBackbone (per-channel or shared weights)
        ↓
    LateFusion (concat, add, or weighted)
        ↓
    TemporalAggregator (LSTM v1 or Transformer v2)
        ↓
    RULHead (optional uncertainty via Gaussian output)

Key components:
- frontend.py: STFT/CWT spectrogram generation
- backbone.py: 2D CNN feature extraction
- aggregator.py: Temporal sequence aggregation
- model.py: Full model assembly

Usage:
    from src.models.cnn2d import (
        create_cnn2d_lstm,
        create_cnn2d_transformer,
        create_cnn2d_with_uncertainty,
    )

    # Create model with LSTM aggregator
    model = create_cnn2d_lstm()

    # Create model with Transformer aggregator
    model = create_cnn2d_transformer()

    # Create model with uncertainty output
    model = create_cnn2d_with_uncertainty()
"""

# Frontend
from .frontend import (
    FrontendType,
    SpectrogramFrontendConfig,
    STFTFrontend,
    PrecomputedFrontend,
    create_frontend,
    extract_spectrogram_numpy,
)

# Backbone
from .backbone import (
    CNN2DBackboneConfig,
    ConvBlock2D,
    CNN2DBackbone,
    DualChannelCNN2DBackbone,
    create_default_backbone,
)

# Aggregator
from .aggregator import (
    SequenceAggregatorConfig,
    SimplePoolingAggregator,
    CNN2DAggregator,
    create_lstm_aggregator,
    create_transformer_aggregator,
    create_simple_aggregator,
    # Re-exports from pattern1
    LSTMAggregator,
    LSTMAggregatorConfig,
    TransformerAggregator,
    TransformerAggregatorConfig,
)

# Model
from .model import (
    CNN2DConfig,
    LateFusion,
    RULHead,
    build_cnn2d_model,
    create_cnn2d_lstm,
    create_cnn2d_transformer,
    create_cnn2d_with_uncertainty,
    create_cnn2d_simple,
    create_cnn2d_bottleneck,
    get_model_summary,
    print_model_summary,
)

__all__ = [
    # Frontend
    "FrontendType",
    "SpectrogramFrontendConfig",
    "STFTFrontend",
    "PrecomputedFrontend",
    "create_frontend",
    "extract_spectrogram_numpy",
    # Backbone
    "CNN2DBackboneConfig",
    "ConvBlock2D",
    "CNN2DBackbone",
    "DualChannelCNN2DBackbone",
    "create_default_backbone",
    # Aggregator
    "SequenceAggregatorConfig",
    "SimplePoolingAggregator",
    "CNN2DAggregator",
    "create_lstm_aggregator",
    "create_transformer_aggregator",
    "create_simple_aggregator",
    "LSTMAggregator",
    "LSTMAggregatorConfig",
    "TransformerAggregator",
    "TransformerAggregatorConfig",
    # Model
    "CNN2DConfig",
    "LateFusion",
    "RULHead",
    "build_cnn2d_model",
    "create_cnn2d_lstm",
    "create_cnn2d_transformer",
    "create_cnn2d_with_uncertainty",
    "create_cnn2d_simple",
    "create_cnn2d_bottleneck",
    "get_model_summary",
    "print_model_summary",
]
