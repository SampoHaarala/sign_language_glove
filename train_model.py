"""
Command-line training script for sign language gesture recognition.

This script loads training, validation and test datasets from
directories, extracts features or sequences, constructs a chosen
model and trains it. It then evaluates the model on the validation and
(optional) test sets, prints relevant statistics, and saves the trained
weights. Datasets should consist of text files named
``<sampleId>-<label>.txt`` where ``label`` is the gesture class. Each
file contains rows of normalised sensor readings for a single time
step, separated by whitespace or commas.

Usage example::

    python train_model.py \
        --train_dir /path/to/train \
        --val_dir /path/to/val \
        --test_dir /path/to/test \
        --model cnn_bilstm \
        --window_size 32 --step_size 16 \
        --save_weights /path/to/save/model.h5

Note: apply a moving average or low‑pass filter *before* normalisation.
Filtering after normalisation would distort the calibrated range and
should be avoided according to recent studies.
"""

from __future__ import annotations

import argparse
import os
import logging
import numpy as np
from typing import List, Tuple

# Import utility modules defined in this project
import feature_extractor
import model_utils

# Try optional dependencies
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib  # for saving sklearn models
except ImportError:
    classification_report = None  # type: ignore
    confusion_matrix = None  # type: ignore
    joblib = None  # type: ignore

try:
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
except ImportError:
    tf = None  # type: ignore
    to_categorical = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample(file_path: str) -> np.ndarray:
    """Load a single sample file into a 2D NumPy array.

    Each line in the file should contain sensor readings separated by
    whitespace or commas. Lines with inconsistent lengths are skipped.

    Parameters
    ----------
    file_path : str
        Path to the sample file.

    Returns
    -------
    np.ndarray
        Array of shape (timesteps, sensors).
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split on whitespace or comma
            parts = [p for p in line.replace(',', ' ').split() if p]
            try:
                row = [float(x) for x in parts]
                data.append(row)
            except ValueError:
                continue
    return np.array(data, dtype=np.float32)

def load_dataset(directory: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load all samples and labels from a directory.

    Filenames must follow the pattern ``<sampleId>-<label>.txt``.

    Parameters
    ----------
    directory : str
        Directory containing sample files.

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        List of arrays and corresponding labels.
    """
    samples = []
    labels = []
    for fname in os.listdir(directory):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(directory, fname)
        # label is text after last '-'
        try:
            label = fname.rsplit('-', 1)[1].rsplit('.', 1)[0]
        except IndexError:
            continue
        samples.append(load_sample(path))
        labels.append(label)
    return samples, labels

def prepare_data_for_classical(samples: List[np.ndarray], labels: List[str], window_size: int, step_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw samples to feature vectors and encode labels.

    Each sample is segmented into overlapping windows. Features are
    extracted per window and averaged across windows to produce a
    single feature vector per sample. This reduces sequence length
    variability.

    Parameters
    ----------
    samples : List[np.ndarray]
        Raw sample arrays.
    labels : List[str]
        Corresponding class labels.
    window_size : int
        Window length for feature extraction.
    step_size : int
        Step size between windows.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix and integer labels.
    """
    feature_vectors = []
    for sample in samples:
        feats = feature_extractor.extract_features_from_sample(sample, window_size, step_size)
        # Average over windows to get a single vector
        feature_vectors.append(np.mean(feats, axis=0))
    X = np.stack(feature_vectors)
    # Map string labels to integers
    unique_labels = sorted(set(labels))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_to_idx[lab] for lab in labels], dtype=np.int64)
    return X, y

def prepare_data_for_deep(samples: List[np.ndarray], labels: List[str], window_size: int, step_size: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Prepare sequences and labels for deep models.

    Each raw sample is segmented into overlapping windows of equal
    length. All windows inherit the sample's label. Sequences are
    padded/truncated to fixed length and stacked into a single array.

    Parameters
    ----------
    samples : List[np.ndarray]
        Raw sample arrays.
    labels : List[str]
        Corresponding class labels.
    window_size : int
        Window length for segments.
    step_size : int
        Step size between windows.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        Array of shape (num_windows, window_size, sensors), encoded
        labels as integers, and mapping from label string to index.
    """
    sequences = []
    seq_labels = []
    for sample, label in zip(samples, labels):
        for window in feature_extractor.sliding_windows(sample, window_size, step_size):
            sequences.append(window)
            seq_labels.append(label)
    # Determine number of sensors
    max_sensors = max(seq.shape[1] for seq in sequences)
    # Pad sequences if necessary (should be equal length already)
    X = np.stack(sequences)
    unique_labels = sorted(set(seq_labels))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_to_idx[lab] for lab in seq_labels], dtype=np.int64)
    return X, y, label_to_idx

def main(args: argparse.Namespace) -> None:
    logger.info("Loading datasets...")
    train_samples, train_labels = load_dataset(args.train_dir)
    val_samples, val_labels = load_dataset(args.val_dir) if args.val_dir else ([], [])
    test_samples, test_labels = load_dataset(args.test_dir) if args.test_dir else ([], [])
    logger.info("Loaded %d training samples, %d validation samples, %d test samples", len(train_samples), len(val_samples), len(test_samples))

    if args.model == 'random_forest':
        # Prepare data
        X_train, y_train = prepare_data_for_classical(train_samples, train_labels, args.window_size, args.step_size)
        X_val, y_val = prepare_data_for_classical(val_samples, val_labels, args.window_size, args.step_size) if val_samples else (None, None)
        X_test, y_test = prepare_data_for_classical(test_samples, test_labels, args.window_size, args.step_size) if test_samples else (None, None)
        # Build and train
        clf = model_utils.get_model('random_forest', n_estimators=args.n_estimators, max_depth=args.max_depth)
        logger.info("Training RandomForest on %d samples...", X_train.shape[0])
        clf.fit(X_train, y_train)
        # Evaluate
        if classification_report is not None:
            logger.info("Validation results:")
            if X_val is not None:
                y_pred = clf.predict(X_val)
                logger.info("\n" + classification_report(y_val, y_pred))
            if X_test is not None:
                logger.info("Test results:")
                y_pred = clf.predict(X_test)
                logger.info("\n" + classification_report(y_test, y_pred))
        else:
            logger.warning("scikit-learn metrics not available; skipping detailed report")
        # Save model
        if args.save_weights:
            if joblib is None:
                logger.warning("joblib not available; cannot save RandomForest model")
            else:
                joblib.dump({'model': clf, 'label_to_idx': {lab: idx for idx, lab in enumerate(sorted(set(train_labels))) }}, args.save_weights)
                logger.info("Saved RandomForest model to %s", args.save_weights)
    else:
        # Deep learning
        if tf is None:
            raise ImportError("TensorFlow is required for deep models")
        # Prepare sequences and labels
        X_train, y_train, label_to_idx = prepare_data_for_deep(train_samples, train_labels, args.window_size, args.step_size)
        X_val, y_val, _ = prepare_data_for_deep(val_samples, val_labels, args.window_size, args.step_size) if val_samples else (None, None, None)
        X_test, y_test, _ = prepare_data_for_deep(test_samples, test_labels, args.window_size, args.step_size) if test_samples else (None, None, None)
        num_classes = len(label_to_idx)
        # One-hot encode labels
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes) if y_val is not None else None
        y_test_cat = to_categorical(y_test, num_classes) if y_test is not None else None
        # Build model
        input_shape = X_train.shape[1], X_train.shape[2]
        model = model_utils.get_model(args.model, input_shape=input_shape, num_classes=num_classes,
                                       conv_filters=args.conv_filters, kernel_size=args.kernel_size,
                                       lstm_units=args.lstm_units, dropout=args.dropout)
        if args.load_weights and os.path.isfile(args.load_weights):
            logger.info("Loading weights from %s", args.load_weights)
            model.load_weights(args.load_weights)
        # Train
        logger.info("Training %s model on %d sequences...", args.model, X_train.shape[0])
        history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat) if X_val is not None else None,
                            epochs=args.epochs, batch_size=args.batch_size)
        # Evaluate
        if X_val is not None:
            val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
            logger.info("Validation accuracy: %.4f", val_acc)
        if X_test is not None:
            test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
            logger.info("Test accuracy: %.4f", test_acc)
        # Save weights
        if args.save_weights:
            model.save(args.save_weights)
            logger.info("Saved Keras model to %s", args.save_weights)
    logger.info("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train sign language gesture classifiers.")
    parser.add_argument('--train_dir', required=True, help='Directory containing training samples')
    parser.add_argument('--val_dir', help='Directory containing validation samples')
    parser.add_argument('--test_dir', help='Directory containing test samples')
    parser.add_argument('--model', choices=['random_forest', 'cnn_lstm', 'cnn_bilstm'], default='random_forest', help='Model type to train')
    parser.add_argument('--window_size', type=int, default=32, help='Sliding window length')
    parser.add_argument('--step_size', type=int, default=16, help='Step size between windows')
    # RandomForest params
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees for RandomForest')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of trees for RandomForest')
    # Deep model hyperparameters
    parser.add_argument('--conv_filters', type=int, default=64, help='Number of filters for Conv1D layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for Conv1D layer')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of units in the (Bi)LSTM layer')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate before final dense layers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for deep models')
    parser.add_argument('--save_weights', help='Path to save trained model weights')
    parser.add_argument('--load_weights', help='Optional path to load existing weights')

    args = parser.parse_args()
    main(args)

