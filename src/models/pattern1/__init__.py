"""Pattern 1: TCN-Transformer architecture for RUL prediction.

This module implements a dual-channel architecture that processes
raw vibration signals through:
1. Per-sensor stem (Conv1D feature extraction)
2. Multi-resolution TCN (dilated convolutions)
3. Bidirectional cross-attention (channel fusion)
4. Temporal aggregator (LSTM or Transformer)
5. RUL prediction head

Example usage:
    from src.models.pattern1 import create_tcn_transformer_lstm

    # Create model with LSTM aggregator
    model = create_tcn_transformer_lstm(input_length=32768)
    model.compile(optimizer='adam', loss='huber')
    model.fit(X_train, y_train, ...)

    # Or with Transformer aggregator
    from src.models.pattern1 import create_tcn_transformer_transformer
    model = create_tcn_transformer_transformer(input_length=32768)
"""

# Stem module
from .stem import (
    StemConfig,
    build_sensor_stem,
    DualChannelStem,
    create_default_stem,
)

# TCN module
from .tcn import (
    TCNConfig,
    DilatedConvBlock,
    TCNEncoder,
    DualChannelTCN,
    create_default_tcn,
)

# Attention module
from .attention import (
    AttentionConfig,
    CrossAttentionBlock,
    BidirectionalCrossAttention,
    ChannelFusion,
    create_default_cross_attention,
)

# Aggregator module
from .aggregator import (
    LSTMAggregatorConfig,
    LSTMAggregator,
    TransformerAggregatorConfig,
    TransformerAggregator,
    PositionalEncoding,
    TransformerEncoderBlock,
    create_lstm_aggregator,
    create_transformer_aggregator,
)

# Full model
from .model import (
    TCNTransformerConfig,
    RULHead,
    TemporalDownsampler,
    build_tcn_transformer_model,
    create_tcn_transformer_lstm,
    create_tcn_transformer_transformer,
    get_model_summary,
    print_model_summary,
)

__all__ = [
    # Stem
    "StemConfig",
    "build_sensor_stem",
    "DualChannelStem",
    "create_default_stem",
    # TCN
    "TCNConfig",
    "DilatedConvBlock",
    "TCNEncoder",
    "DualChannelTCN",
    "create_default_tcn",
    # Attention
    "AttentionConfig",
    "CrossAttentionBlock",
    "BidirectionalCrossAttention",
    "ChannelFusion",
    "create_default_cross_attention",
    # Aggregator
    "LSTMAggregatorConfig",
    "LSTMAggregator",
    "TransformerAggregatorConfig",
    "TransformerAggregator",
    "PositionalEncoding",
    "TransformerEncoderBlock",
    "create_lstm_aggregator",
    "create_transformer_aggregator",
    # Full model
    "TCNTransformerConfig",
    "RULHead",
    "TemporalDownsampler",
    "build_tcn_transformer_model",
    "create_tcn_transformer_lstm",
    "create_tcn_transformer_transformer",
    "get_model_summary",
    "print_model_summary",
]
