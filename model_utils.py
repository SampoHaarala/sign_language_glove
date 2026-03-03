"""
Model utility functions for training sign language classifiers.

This module contains functions to construct classical and deep learning
models. The classical baseline uses a RandomForest classifier over
extracted feature vectors. Deep models include a 1D CNN followed by an
LSTM (or BiLSTM) layer, capable of consuming sequences of normalised
sensor readings directly. These models are configured with reasonable
default hyperparameters but can be adjusted via function arguments.

TensorFlow and scikit‑learn are optional dependencies; install them
before training.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

# Classical ML
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None  # type: ignore

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    tf = None  # type: ignore
    layers = None  # type: ignore
    models = None  # type: ignore

logger = logging.getLogger(__name__)


def build_random_forest(n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42) -> RandomForestClassifier:
    """Create a RandomForest classifier for gesture recognition.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : Optional[int]
        Maximum depth of each tree. None means fully grown trees.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Configured classifier.
    """
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn is required for the RandomForest model")
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    return clf


def build_cnn_lstm(input_shape: Tuple[int, int], num_classes: int, conv_filters: int = 64, kernel_size: int = 3, lstm_units: int = 64, dropout: float = 0.3, bidirectional: bool = False) -> "tf.keras.Model":
    """Construct a 1D CNN with an optional (Bi)LSTM classifier.

    The network consists of convolutional layers to extract local
    temporal patterns followed by an LSTM (unidirectional or
    bidirectional) to capture longer dependencies. A final dense
    layer produces probabilities over the gesture classes.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input sequences (timesteps, sensors).
    num_classes : int
        Number of target gesture classes.
    conv_filters : int
        Number of filters in the convolutional layer.
    kernel_size : int
        Width of the convolutional kernels.
    lstm_units : int
        Number of hidden units in the LSTM layer.
    dropout : float
        Dropout rate applied before the final dense layer.
    bidirectional : bool
        If True, use a Bidirectional LSTM instead of a unidirectional one.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    if tf is None or layers is None or models is None:
        raise ImportError("TensorFlow is required for the CNN-LSTM model")
    model = models.Sequential(name="cnn_lstm" if not bidirectional else "cnn_bilstm")
    # Input layer
    model.add(layers.Input(shape=input_shape))
    # Convolution
    model.add(layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    # Optionally stack more conv layers
    model.add(layers.Conv1D(filters=conv_filters * 2, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    # Recurrent layer
    if bidirectional:
        model.add(layers.Bidirectional(layers.LSTM(lstm_units)))
    else:
        model.add(layers.LSTM(lstm_units))
    # Dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info("Created %s model with input shape %s and %d classes", model.name, input_shape, num_classes)
    return model


def get_model(model_type: str, *args, **kwargs):
    """Factory function to select and build a model by name.

    Supported model types:
      - 'random_forest': classical RandomForest classifier
      - 'cnn_lstm': convolution + LSTM
      - 'cnn_bilstm': convolution + bidirectional LSTM

    Any additional positional or keyword arguments are forwarded to the
    underlying build function.

    Parameters
    ----------
    model_type : str
        Name of the model to create.

    Returns
    -------
    object
        The instantiated model.
    """
    model_type = model_type.lower()
    if model_type == 'random_forest':
        return build_random_forest(*args, **kwargs)
    elif model_type == 'cnn_lstm':
        return build_cnn_lstm(*args, **kwargs, bidirectional=False)
    elif model_type == 'cnn_bilstm':
        return build_cnn_lstm(*args, **kwargs, bidirectional=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

