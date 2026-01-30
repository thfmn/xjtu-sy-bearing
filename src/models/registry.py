"""Central model registry mapping model names to build functions and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a registered model architecture."""

    name: str
    build_fn: Callable  # () -> keras.Model
    input_type: Literal["raw_signal", "spectrogram"]
    default_input_shape: tuple[int, ...]


_REGISTRY: dict[str, ModelInfo] = {}


def register_model(
    name: str,
    build_fn: Callable,
    input_type: Literal["raw_signal", "spectrogram"],
    default_input_shape: tuple[int, ...],
) -> None:
    """Register a model architecture in the global registry."""
    if name in _REGISTRY:
        raise ValueError(f"Model '{name}' is already registered")
    _REGISTRY[name] = ModelInfo(
        name=name,
        build_fn=build_fn,
        input_type=input_type,
        default_input_shape=default_input_shape,
    )


def get_model_info(name: str) -> ModelInfo:
    """Look up a registered model by name.

    Raises:
        KeyError: If the model name is not registered. The error message
            lists all available models.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Model '{name}' is not registered. Available models: {available}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(_REGISTRY)


def build_model(name: str):
    """Build and return a Keras model by registry name.

    Convenience wrapper around ``get_model_info(name).build_fn()``.
    """
    info = get_model_info(name)
    return info.build_fn()


# ---------------------------------------------------------------------------
# Model registrations
# ---------------------------------------------------------------------------

def _register_all() -> None:
    """Register all known model architectures."""
    from src.models.baselines.cnn1d_baseline import create_default_cnn1d

    register_model(
        name="cnn1d_baseline",
        build_fn=create_default_cnn1d,
        input_type="raw_signal",
        default_input_shape=(32768, 2),
    )


_register_all()
