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

"""Cross-attention fusion module for dual-channel feature integration.

Implements cross-attention where horizontal features attend to vertical features
and vice versa, enabling the model to learn inter-channel relationships.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class AttentionConfig:
    """Configuration for cross-attention module.

    Attributes:
        num_heads: Number of attention heads.
        key_dim: Dimension of each attention key (per head).
        value_dim: Dimension of each attention value (per head). If None, uses key_dim.
        dropout_rate: Dropout rate for attention weights.
        use_residual: Whether to add residual connections.
        use_layer_norm: Whether to apply layer normalization.
        output_dim: Output dimension. If None, keeps input dimension.
    """
    num_heads: int = 4
    key_dim: int = 64
    value_dim: Optional[int] = None
    dropout_rate: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    output_dim: Optional[int] = None


class CrossAttentionBlock(keras.layers.Layer):
    """Cross-attention block: queries from one stream, keys/values from another.

    Implements: Attention(Q=stream_a, K=stream_b, V=stream_b)
    With optional residual connection and layer normalization.
    """

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else AttentionConfig()

        # Layers
        self.attention = None
        self.dropout = None
        self.layer_norm = None
        self.output_proj = None

    def build(self, input_shape):
        # input_shape is tuple of (query_shape, key_value_shape)
        query_shape, kv_shape = input_shape
        input_dim = query_shape[-1]

        output_dim = self.config.output_dim or input_dim
        value_dim = self.config.value_dim or self.config.key_dim

        self.attention = layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.key_dim,
            value_dim=value_dim,
            dropout=self.config.dropout_rate,
            name=f"{self.name}_mha"
        )

        if self.config.dropout_rate > 0:
            self.dropout = layers.Dropout(
                self.config.dropout_rate,
                name=f"{self.name}_dropout"
            )

        if self.config.use_layer_norm:
            self.layer_norm = layers.LayerNormalization(
                epsilon=1e-6,
                name=f"{self.name}_ln"
            )

        # Output projection if dimensions differ
        if output_dim != input_dim:
            self.output_proj = layers.Dense(
                output_dim,
                name=f"{self.name}_output_proj"
            )

        super().build(input_shape)

    def call(self, inputs, training=None, return_attention_scores=False):
        """Apply cross-attention.

        Args:
            inputs: Tuple of (query_tensor, key_value_tensor).
            training: Training mode flag.
            return_attention_scores: If True, also return attention weights.

        Returns:
            Output tensor, or (output, attention_scores) if return_attention_scores.
        """
        query, key_value = inputs

        # Cross-attention: Q from query stream, K/V from other stream
        attention_output, attention_scores = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            training=training,
            return_attention_scores=True
        )

        if self.dropout is not None:
            attention_output = self.dropout(attention_output, training=training)

        # Residual connection
        if self.config.use_residual:
            output = query + attention_output
        else:
            output = attention_output

        # Layer normalization
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        # Output projection
        if self.output_proj is not None:
            output = self.output_proj(output)

        if return_attention_scores:
            return output, attention_scores
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "num_heads": self.config.num_heads,
                "key_dim": self.config.key_dim,
                "value_dim": self.config.value_dim,
                "dropout_rate": self.config.dropout_rate,
                "use_residual": self.config.use_residual,
                "use_layer_norm": self.config.use_layer_norm,
                "output_dim": self.config.output_dim,
            }
        })
        return config


class BidirectionalCrossAttention(keras.layers.Layer):
    """Bidirectional cross-attention between horizontal and vertical channels.

    Applies cross-attention in both directions:
    1. H attends to V: H' = CrossAttn(Q=H, K=V, V=V)
    2. V attends to H: V' = CrossAttn(Q=V, K=H, V=H)

    This allows both channels to incorporate information from each other.
    """

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else AttentionConfig()
        self.h_to_v_attention = None  # H queries, V keys/values
        self.v_to_h_attention = None  # V queries, H keys/values

    def build(self, input_shape):
        # Input is tuple (h_features, v_features)
        self.h_to_v_attention = CrossAttentionBlock(
            config=self.config,
            name=f"{self.name}_h_attends_v"
        )
        self.v_to_h_attention = CrossAttentionBlock(
            config=self.config,
            name=f"{self.name}_v_attends_h"
        )
        super().build(input_shape)

    def call(self, inputs, training=None, return_attention_scores=False):
        """Apply bidirectional cross-attention.

        Args:
            inputs: Tuple of (h_features, v_features).
            training: Training mode flag.
            return_attention_scores: If True, also return attention weights.

        Returns:
            Tuple of (h_enhanced, v_enhanced), or with attention scores if requested.
        """
        h_features, v_features = inputs

        if return_attention_scores:
            # H attends to V
            h_enhanced, h_scores = self.h_to_v_attention(
                (h_features, v_features),
                training=training,
                return_attention_scores=True
            )
            # V attends to H
            v_enhanced, v_scores = self.v_to_h_attention(
                (v_features, h_features),
                training=training,
                return_attention_scores=True
            )
            return (h_enhanced, v_enhanced), (h_scores, v_scores)
        else:
            h_enhanced = self.h_to_v_attention(
                (h_features, v_features),
                training=training
            )
            v_enhanced = self.v_to_h_attention(
                (v_features, h_features),
                training=training
            )
            return h_enhanced, v_enhanced

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "num_heads": self.config.num_heads,
                "key_dim": self.config.key_dim,
                "value_dim": self.config.value_dim,
                "dropout_rate": self.config.dropout_rate,
                "use_residual": self.config.use_residual,
                "use_layer_norm": self.config.use_layer_norm,
                "output_dim": self.config.output_dim,
            }
        })
        return config


class ChannelFusion(keras.layers.Layer):
    """Fuse dual-channel features after cross-attention.

    Supports multiple fusion strategies:
    - concat: Concatenate along feature dimension
    - add: Element-wise addition
    - avg: Element-wise average
    - weighted: Learnable weighted combination
    """

    def __init__(
        self,
        fusion_mode: str = "concat",
        output_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fusion_mode = fusion_mode
        self.output_dim = output_dim
        self.projection = None
        self.h_weight = None
        self.v_weight = None

    def build(self, input_shape):
        h_shape, v_shape = input_shape
        h_dim = h_shape[-1]
        v_dim = v_shape[-1]

        if self.fusion_mode == "concat":
            fused_dim = h_dim + v_dim
        else:
            fused_dim = h_dim  # assume equal dims for add/avg/weighted

        # Output projection if specified
        if self.output_dim is not None:
            self.projection = layers.Dense(
                self.output_dim,
                name=f"{self.name}_proj"
            )

        # Learnable weights for weighted fusion
        if self.fusion_mode == "weighted":
            self.h_weight = self.add_weight(
                name="h_weight",
                shape=(1,),
                initializer="ones",
                trainable=True
            )
            self.v_weight = self.add_weight(
                name="v_weight",
                shape=(1,),
                initializer="ones",
                trainable=True
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Fuse dual-channel features.

        Args:
            inputs: Tuple of (h_features, v_features).
            training: Training mode flag.

        Returns:
            Fused features tensor.
        """
        h_features, v_features = inputs

        if self.fusion_mode == "concat":
            fused = tf.concat([h_features, v_features], axis=-1)
        elif self.fusion_mode == "add":
            fused = h_features + v_features
        elif self.fusion_mode == "avg":
            fused = (h_features + v_features) / 2.0
        elif self.fusion_mode == "weighted":
            # Softmax normalization of weights
            weights = tf.nn.softmax([self.h_weight, self.v_weight])
            fused = weights[0] * h_features + weights[1] * v_features
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        if self.projection is not None:
            fused = self.projection(fused)

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            "fusion_mode": self.fusion_mode,
            "output_dim": self.output_dim,
        })
        return config


def create_default_cross_attention() -> BidirectionalCrossAttention:
    """Create default bidirectional cross-attention with PRD-specified settings.

    Returns:
        BidirectionalCrossAttention with 4 heads.
    """
    config = AttentionConfig(
        num_heads=4,
        key_dim=64,
        dropout_rate=0.1,
        use_residual=True,
        use_layer_norm=True,
    )
    return BidirectionalCrossAttention(config=config)
