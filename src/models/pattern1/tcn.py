"""Multi-resolution Temporal Convolutional Network (TCN) module.

Implements dilated causal convolutions for capturing multi-scale temporal patterns
in bearing vibration signals. The receptive field grows exponentially with depth.
"""

from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TCNConfig:
    """Configuration for Temporal Convolutional Network.

    Attributes:
        filters: Number of filters per conv layer.
        kernel_size: Kernel size for dilated convolutions.
        dilations: List of dilation rates (e.g., [1, 2, 4, 8, 16, 32]).
        use_batch_norm: Whether to apply batch normalization.
        dropout_rate: Dropout rate between conv layers.
        use_residual: Whether to use residual connections.
        activation: Activation function name.
        causal: Whether to use causal (no future leakage) padding.
    """
    filters: int = 64
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    use_residual: bool = True
    activation: str = "gelu"
    causal: bool = False  # For RUL, non-causal is fine (use full context)


class DilatedConvBlock(keras.layers.Layer):
    """Single dilated convolution block with optional residual connection.

    Architecture: Conv1D -> BN -> Activation -> Dropout (-> Residual Add)
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = True,
        causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.use_residual = use_residual
        self.causal = causal

        # Layers will be built in build()
        self.conv = None
        self.bn = None
        self.activation = None
        self.dropout = None
        self.residual_conv = None

    def build(self, input_shape):
        input_filters = input_shape[-1]

        # Padding for causal/non-causal convolution
        if self.causal:
            # Causal: pad only on left
            padding = "causal"
        else:
            padding = "same"

        self.conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding=padding,
            name=f"{self.name}_conv"
        )

        if self.use_batch_norm:
            self.bn = layers.BatchNormalization(name=f"{self.name}_bn")

        self.activation = layers.Activation(
            self.activation_name,
            name=f"{self.name}_act"
        )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(
                self.dropout_rate,
                name=f"{self.name}_dropout"
            )

        # Residual projection if dimensions don't match
        if self.use_residual and input_filters != self.filters:
            self.residual_conv = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding="same",
                name=f"{self.name}_residual_proj"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv(inputs)

        if self.bn is not None:
            x = self.bn(x, training=training)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Residual connection
        if self.use_residual:
            residual = inputs
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            x = x + residual

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation_name,
            "use_residual": self.use_residual,
            "causal": self.causal,
        })
        return config


class TCNEncoder(keras.layers.Layer):
    """Multi-resolution TCN encoder with exponentially increasing dilations.

    Creates a stack of dilated conv blocks with increasing dilation rates,
    enabling the model to capture patterns at multiple temporal scales.

    Receptive field = sum(dilation * (kernel_size - 1)) + 1
    For dilations=[1,2,4,8,16,32] and kernel_size=3:
    RF = (1+2+4+8+16+32) * 2 + 1 = 127 samples
    """

    def __init__(
        self,
        config: Optional[TCNConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else TCNConfig()
        self.blocks = []

    def build(self, input_shape):
        # Create dilated conv blocks
        for i, dilation in enumerate(self.config.dilations):
            block = DilatedConvBlock(
                filters=self.config.filters,
                kernel_size=self.config.kernel_size,
                dilation_rate=dilation,
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate,
                activation=self.config.activation,
                use_residual=self.config.use_residual,
                causal=self.config.causal,
                name=f"{self.name}_block_d{dilation}"
            )
            self.blocks.append(block)

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def compute_receptive_field(self) -> int:
        """Compute the receptive field size of the TCN.

        Returns:
            Receptive field in number of input samples.
        """
        rf = 1
        for dilation in self.config.dilations:
            rf += dilation * (self.config.kernel_size - 1)
        return rf

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "filters": self.config.filters,
                "kernel_size": self.config.kernel_size,
                "dilations": self.config.dilations,
                "use_batch_norm": self.config.use_batch_norm,
                "dropout_rate": self.config.dropout_rate,
                "use_residual": self.config.use_residual,
                "activation": self.config.activation,
                "causal": self.config.causal,
            }
        })
        return config

    @classmethod
    def from_config(cls, config):
        tcn_config = TCNConfig(**config.pop("config"))
        return cls(config=tcn_config, **config)


class DualChannelTCN(keras.layers.Layer):
    """Apply TCN encoding to both channels (after stem processing).

    Processes horizontal and vertical channel features through separate
    or shared TCN encoders.
    """

    def __init__(
        self,
        config: Optional[TCNConfig] = None,
        share_weights: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else TCNConfig()
        self.share_weights = share_weights
        self.h_tcn = None
        self.v_tcn = None

    def build(self, input_shape):
        # Input is tuple (h_features, v_features) from stem
        # Each has shape (batch, time_steps, stem_filters)
        self.h_tcn = TCNEncoder(
            config=self.config,
            name="h_tcn"
        )

        if self.share_weights:
            self.v_tcn = self.h_tcn
        else:
            self.v_tcn = TCNEncoder(
                config=self.config,
                name="v_tcn"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Process dual-channel features through TCN.

        Args:
            inputs: Tuple of (h_features, v_features).
            training: Training mode flag.

        Returns:
            Tuple of (h_encoded, v_encoded).
        """
        h_features, v_features = inputs

        h_encoded = self.h_tcn(h_features, training=training)
        v_encoded = self.v_tcn(v_features, training=training)

        return h_encoded, v_encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "filters": self.config.filters,
                "kernel_size": self.config.kernel_size,
                "dilations": self.config.dilations,
                "use_batch_norm": self.config.use_batch_norm,
                "dropout_rate": self.config.dropout_rate,
                "use_residual": self.config.use_residual,
                "activation": self.config.activation,
                "causal": self.config.causal,
            },
            "share_weights": self.share_weights,
        })
        return config


def create_default_tcn() -> DualChannelTCN:
    """Create default dual-channel TCN with PRD-specified settings.

    Returns:
        DualChannelTCN with dilations [1, 2, 4, 8, 16, 32].
    """
    config = TCNConfig(
        filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
        use_batch_norm=True,
        dropout_rate=0.1,
        use_residual=True,
        activation="gelu",
        causal=False,
    )
    return DualChannelTCN(config=config, share_weights=False)
