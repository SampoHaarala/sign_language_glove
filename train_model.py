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
import csv
import os
import logging
from collections import Counter

import numpy as np
from typing import Dict, List, Optional, Tuple

# Import utility modules defined in this project
import feature_extractor
import gesture_subset_abcd_y as gesture_subset
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
    """Proxy loader to feature_extractor.load_sample."""
    return feature_extractor.load_sample(file_path, ignore_time=True)


def load_dataset(directory: str, num_letters: int | None = None, allowed_labels: list[str] | None = None) -> Tuple[List[np.ndarray], List[str]]:
    """Load all samples and labels from a directory.

    Filenames must follow the pattern ``<sampleId>-<label>.txt`` or ``<sampleId>-<label>-<trial>.csv``.
    Recursively scans all subdirectories.

    Parameters
    ----------
    directory : str
        Directory containing sample files.
    num_letters : int | None
        If specified, limit to the first N letters alphabetically.
    allowed_labels : list[str] | None
        If specified, retain only samples with these labels.

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        List of arrays and corresponding labels.
    """
    samples = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if not (fname.endswith('.txt') or fname.endswith('.csv')):
                continue
            path = os.path.join(root, fname)
            # Extract label: for new format <subject>-<label>-<trial>.ext, label is second-to-last
            # For old format <id>-<label>.ext, label is last part
            try:
                parts = fname.rsplit('.', 1)[0].split('-')
                if len(parts) >= 3:
                    # New format: subject-label-trial
                    label = parts[-2]
                elif len(parts) == 2:
                    # Old format: id-label
                    label = parts[-1]
                else:
                    continue
            except IndexError:
                continue
            samples.append(feature_extractor.load_sample(path, ignore_time=True))
            labels.append(label)
    
    # Filter to a selected subset of labels if requested
    if allowed_labels is not None:
        selected_labels = set(allowed_labels)
        filtered_samples = []
        filtered_labels = []
        for sample, label in zip(samples, labels):
            if label in selected_labels:
                filtered_samples.append(sample)
                filtered_labels.append(label)
        samples = filtered_samples
        labels = filtered_labels
        logger.info("Filtered dataset to labels: %s", sorted(selected_labels))
    elif num_letters is not None:
        unique_labels = sorted(set(labels))
        selected_labels = set(unique_labels[:num_letters])
        filtered_samples = []
        filtered_labels = []
        for sample, label in zip(samples, labels):
            if label in selected_labels:
                filtered_samples.append(sample)
                filtered_labels.append(label)
        samples = filtered_samples
        labels = filtered_labels
        logger.info("Limited to %d letters: %s", num_letters, sorted(selected_labels))
    
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


def prepare_window_level_data_for_random_forest(
    samples: List[np.ndarray],
    labels: List[str],
    window_size: int,
    step_size: int,
    label_to_idx: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """Prepare window-level feature data for Random Forest training.

    Each sample is split into overlapping windows, and every window
    inherits the original sample label. The returned sample_ids array
    maps each window back to its parent sample index.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]
        X_windows, y_windows, sample_ids, label_to_idx, idx_to_label
    """
    if label_to_idx is None:
        unique_labels = sorted(set(labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {idx: lab for lab, idx in label_to_idx.items()}

    feature_rows: list[np.ndarray] = []
    y_windows: list[int] = []
    sample_ids: list[int] = []

    for sample_idx, (sample, label) in enumerate(zip(samples, labels)):
        if label not in label_to_idx:
            raise ValueError(f"Label '{label}' not found in label_to_idx mapping")
        windows = list(feature_extractor.sliding_windows(sample, window_size, step_size))
        if not windows:
            logger.warning("Skipping sample %d because it is shorter than window_size (%d)", sample_idx, window_size)
            continue
        for window in windows:
            feature_rows.append(feature_extractor.extract_features(window))
            y_windows.append(label_to_idx[label])
            sample_ids.append(sample_idx)

    if feature_rows:
        X_windows = np.stack(feature_rows)
    else:
        X_windows = np.empty((0, 0), dtype=np.float32)

    return (
        X_windows,
        np.array(y_windows, dtype=np.int64),
        np.array(sample_ids, dtype=np.int64),
        label_to_idx,
        idx_to_label,
    )


def aggregate_window_majority_vote(
    y_window_pred: np.ndarray,
    proba: np.ndarray | None,
    sample_ids: np.ndarray,
    y_windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate window-level predictions into one sample-level prediction per original sample."""
    sample_indices: dict[int, list[int]] = {}
    for window_idx, sample_id in enumerate(sample_ids.tolist()):
        sample_indices.setdefault(sample_id, []).append(window_idx)

    sample_ids_sorted: list[int] = []
    sample_truth: list[int] = []
    sample_preds: list[int] = []
    sample_conf: list[float] = []

    for sample_id in sorted(sample_indices.keys()):
        indices = sample_indices[sample_id]
        votes = [int(y_window_pred[i]) for i in indices]
        vote_counts = Counter(votes)
        highest_count = max(vote_counts.values())
        tied_labels = [lab for lab, count in vote_counts.items() if count == highest_count]

        if len(tied_labels) == 1 or proba is None:
            chosen_label = min(tied_labels)
        else:
            avg_prob = {
                lab: float(np.mean(proba[indices, lab]))
                for lab in tied_labels
            }
            chosen_label = max(avg_prob, key=avg_prob.get)

        confidence = float(np.mean(proba[indices, chosen_label])) if proba is not None else 0.0

        sample_ids_sorted.append(sample_id)
        sample_truth.append(int(y_windows[indices[0]]))
        sample_preds.append(chosen_label)
        sample_conf.append(confidence)

    return (
        np.array(sample_ids_sorted, dtype=np.int64),
        np.array(sample_truth, dtype=np.int64),
        np.array(sample_preds, dtype=np.int64),
        np.array(sample_conf, dtype=np.float32),
    )


def evaluate_random_forest_window_majority_vote(
    clf,
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    sample_ids: np.ndarray,
    idx_to_label: Dict[int, str],
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Evaluate Random Forest by aggregating window votes to sample-level predictions."""
    if X_windows.size == 0:
        logger.warning("No windows available for %s evaluation", dataset_name)
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), 0.0, 0.0)

    y_window_pred = clf.predict(X_windows)
    window_accuracy = float(np.mean(y_window_pred == y_windows))
    proba = clf.predict_proba(X_windows) if hasattr(clf, 'predict_proba') else None
    sample_ids_sorted, sample_truth, sample_preds, sample_conf = aggregate_window_majority_vote(
        y_window_pred, proba, sample_ids, y_windows
    )
    sample_accuracy = float(np.mean(sample_preds == sample_truth)) if sample_truth.size else 0.0
    average_windows = float(len(sample_ids) / len(sample_truth)) if sample_truth.size else 0.0

    logger.info("=== Random Forest Results (%s) ===", dataset_name)
    logger.info("Sample-level accuracy: %.4f", sample_accuracy)
    logger.info("Window-level accuracy: %.4f", window_accuracy)
    logger.info("Average windows per sample: %.2f", average_windows)

    if classification_report is not None and sample_truth.size:
        label_names = [idx_to_label[i] for i in sorted(idx_to_label)]
        logger.info("Sample-level Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(sample_truth, sample_preds)))
        logger.info("Sample-level Classification Report:")
        logger.info("\n" + classification_report(sample_truth, sample_preds, target_names=label_names))
    elif sample_truth.size:
        logger.warning("scikit-learn metrics not available; skipping detailed report")

    return sample_ids_sorted, sample_truth, sample_preds, sample_conf, window_accuracy, average_windows


def evaluate_cnn_sample_majority_vote(
    model,
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    sample_ids: np.ndarray,
    idx_to_label: Dict[int, str],
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate a CNN-based model by aggregating window predictions to sample-level."""
    if X_windows.size == 0:
        logger.warning("No windows available for %s evaluation", dataset_name)
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), 0.0)

    y_window_proba = model.predict(X_windows, verbose=0)
    y_window_pred = np.argmax(y_window_proba, axis=1)
    sample_ids_sorted, sample_truth, sample_preds, sample_conf = aggregate_window_majority_vote(
        y_window_pred, y_window_proba, sample_ids, y_windows
    )
    sample_accuracy = float(np.mean(sample_preds == sample_truth)) if sample_truth.size else 0.0
    window_accuracy = float(np.mean(y_window_pred == y_windows)) if y_windows.size else 0.0

    logger.info("=== %s Results (%s) ===", model.name, dataset_name)
    logger.info("Sample-level accuracy: %.4f", sample_accuracy)
    logger.info("Window-level accuracy: %.4f", window_accuracy)
    logger.info("Average windows per sample: %.2f", float(len(sample_ids) / len(sample_truth)) if sample_truth.size else 0.0)

    if classification_report is not None and sample_truth.size:
        label_names = [idx_to_label[i] for i in sorted(idx_to_label)]
        logger.info("Sample-level Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(sample_truth, sample_preds)))
        logger.info("Sample-level Classification Report:")
        logger.info("\n" + classification_report(sample_truth, sample_preds, target_names=label_names))
    elif sample_truth.size:
        logger.warning("scikit-learn metrics not available; skipping detailed report")

    return sample_ids_sorted, sample_truth, sample_preds, sample_conf, sample_accuracy


def save_ensemble_results_csv(
    output_path: str,
    sample_ids: np.ndarray,
    y_truth: np.ndarray,
    rf_preds: np.ndarray,
    rf_conf: np.ndarray,
    cnn_preds: np.ndarray,
    cnn_conf: np.ndarray,
    ensemble_preds: np.ndarray,
    idx_to_label: Dict[int, str],
) -> None:
    fieldnames = [
        'sample_id',
        'true_label',
        'rf_prediction',
        'rf_confidence',
        'cnn_bilstm_prediction',
        'cnn_bilstm_confidence',
        'ensemble_prediction',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, true_label, rf_pred, rf_confidence, cnn_pred, cnn_confidence, ensemble_pred in zip(
            sample_ids,
            y_truth,
            rf_preds,
            rf_conf,
            cnn_preds,
            cnn_conf,
            ensemble_preds,
        ):
            writer.writerow({
                'sample_id': int(sample_id),
                'true_label': idx_to_label[int(true_label)],
                'rf_prediction': idx_to_label[int(rf_pred)],
                'rf_confidence': float(rf_confidence),
                'cnn_bilstm_prediction': idx_to_label[int(cnn_pred)],
                'cnn_bilstm_confidence': float(cnn_confidence),
                'ensemble_prediction': idx_to_label[int(ensemble_pred)],
            })
    logger.info("Saved ensemble predictions to %s", output_path)


def prepare_data_for_deep(
    samples: List[np.ndarray],
    labels: List[str],
    window_size: int,
    step_size: int,
    label_to_idx: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """Prepare sequences, labels and sample IDs for deep models.

    The CNN-BiLSTM pipeline consumes windowed time-series sequences
    directly. Each window inherits the original sample label and is
    associated with a sample index so that evaluation can aggregate
    predictions back to one label per original sample.
    """
    sequences = []
    seq_labels = []
    sample_ids: list[int] = []

    for sample_idx, (sample, label) in enumerate(zip(samples, labels)):
        windows = list(feature_extractor.sliding_windows(sample, window_size, step_size))
        if not windows:
            logger.warning("Skipping sample %d because it is shorter than window_size (%d)", sample_idx, window_size)
            continue
        for window in windows:
            sequences.append(window)
            seq_labels.append(label)
            sample_ids.append(sample_idx)

    if not sequences:
        return (
            np.empty((0, window_size, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            label_to_idx or {},
            {},
        )

    X = np.stack(sequences)
    if label_to_idx is None:
        unique_labels = sorted(set(seq_labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {idx: lab for lab, idx in label_to_idx.items()}
    y = np.array([label_to_idx[lab] for lab in seq_labels], dtype=np.int64)
    return X, y, np.array(sample_ids, dtype=np.int64), label_to_idx, idx_to_label

def main(args: argparse.Namespace) -> None:
    logger.info("Loading datasets...")
    allowed_labels = gesture_subset.parse_label_list(args.letters)
    if allowed_labels is None:
        allowed_labels = gesture_subset.get_reduced_gesture_set()
        logger.info("Using default reduced gesture subset: %s", allowed_labels)
    train_samples, train_labels = load_dataset(args.train_dir, args.num_letters, allowed_labels)
    val_samples, val_labels = load_dataset(args.val_dir, args.num_letters, allowed_labels) if args.val_dir else ([], [])
    test_samples, test_labels = load_dataset(args.test_dir, args.num_letters, allowed_labels) if args.test_dir else ([], [])
    cnn_model_type = 'cnn_bilstm' if args.model == 'both' else args.model
    run_rf = args.model in {'random_forest', 'both'}
    run_cnn = args.model in {'cnn_lstm', 'cnn_bilstm', 'both'}

    rf_step_size = args.step_size if args.step_size is not None else args.window_size // 2
    cnn_step_size = args.step_size if args.step_size is not None else 8
    if args.step_size is None:
        logger.info("Using default Random Forest step_size=%d and CNN step_size=%d", rf_step_size, cnn_step_size)

    label_to_idx = {lab: i for i, lab in enumerate(sorted(set(train_labels)))}
    idx_to_label = {idx: lab for lab, idx in label_to_idx.items()}

    rf_test_sample_ids = np.empty((0,), dtype=np.int64)
    rf_test_truth = np.empty((0,), dtype=np.int64)
    rf_test_preds = np.empty((0,), dtype=np.int64)
    rf_test_conf = np.empty((0,), dtype=np.float32)
    cnn_test_sample_ids = np.empty((0,), dtype=np.int64)
    cnn_test_truth = np.empty((0,), dtype=np.int64)
    cnn_test_preds = np.empty((0,), dtype=np.int64)
    cnn_test_conf = np.empty((0,), dtype=np.float32)

    if run_rf:
        X_train_rf, y_train_rf, train_sample_ids_rf, _, _ = prepare_window_level_data_for_random_forest(
            train_samples, train_labels, args.window_size, rf_step_size, label_to_idx=label_to_idx
        )
        X_val_rf, y_val_rf, val_sample_ids_rf, _, _ = prepare_window_level_data_for_random_forest(
            val_samples, val_labels, args.window_size, rf_step_size, label_to_idx=label_to_idx
        ) if val_samples else (np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), label_to_idx, idx_to_label)
        X_test_rf, y_test_rf, rf_test_sample_ids, _, _ = prepare_window_level_data_for_random_forest(
            test_samples, test_labels, args.window_size, rf_step_size, label_to_idx=label_to_idx
        ) if test_samples else (np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), label_to_idx, idx_to_label)

        clf = model_utils.get_model('random_forest', n_estimators=args.n_estimators, max_depth=args.max_depth)
        logger.info("Training RandomForest on %d windows...", X_train_rf.shape[0])
        clf.fit(X_train_rf, y_train_rf)

        evaluate_random_forest_window_majority_vote(clf, X_train_rf, y_train_rf, train_sample_ids_rf, idx_to_label, 'train')
        if val_samples:
            evaluate_random_forest_window_majority_vote(clf, X_val_rf, y_val_rf, val_sample_ids_rf, idx_to_label, 'validation')
        if test_samples:
            rf_test_sample_ids, rf_test_truth, rf_test_preds, rf_test_conf, _, _ = evaluate_random_forest_window_majority_vote(
                clf, X_test_rf, y_test_rf, rf_test_sample_ids, idx_to_label, 'test'
            )

        rf_save_path = args.save_weights_rf if args.save_weights_rf else (args.save_weights if args.model != 'both' else None)
        if rf_save_path:
            if joblib is None:
                logger.warning("joblib not available; cannot save RandomForest model")
            else:
                joblib.dump({'model': clf, 'label_to_idx': label_to_idx}, rf_save_path)
                logger.info("Saved RandomForest model to %s", rf_save_path)

    if run_cnn:
        if tf is None:
            raise ImportError("TensorFlow is required for deep models")
        X_train_cnn, y_train_cnn, train_sample_ids_cnn, _, _ = prepare_data_for_deep(
            train_samples, train_labels, args.window_size, cnn_step_size, label_to_idx=label_to_idx
        )
        X_val_cnn, y_val_cnn, val_sample_ids_cnn, _, _ = prepare_data_for_deep(
            val_samples, val_labels, args.window_size, cnn_step_size, label_to_idx=label_to_idx
        ) if val_samples else (np.empty((0, args.window_size, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), label_to_idx, idx_to_label)
        X_test_cnn, y_test_cnn, cnn_test_sample_ids, _, _ = prepare_data_for_deep(
            test_samples, test_labels, args.window_size, cnn_step_size, label_to_idx=label_to_idx
        ) if test_samples else (np.empty((0, args.window_size, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), label_to_idx, idx_to_label)

        num_classes = len(label_to_idx)
        y_train_cat = to_categorical(y_train_cnn, num_classes)
        y_val_cat = to_categorical(y_val_cnn, num_classes) if val_samples else None
        y_test_cat = to_categorical(y_test_cnn, num_classes) if test_samples else None

        input_shape = X_train_cnn.shape[1], X_train_cnn.shape[2]
        model = model_utils.get_model(cnn_model_type, input_shape=input_shape, num_classes=num_classes,
                                       conv_filters=args.conv_filters, kernel_size=args.kernel_size,
                                       lstm_units=args.lstm_units, dropout=args.dropout)
        if args.load_weights and os.path.isfile(args.load_weights):
            logger.info("Loading weights from %s", args.load_weights)
            model.load_weights(args.load_weights)

        logger.info("Training %s model on %d sequences...", model.name, X_train_cnn.shape[0])
        model.fit(X_train_cnn, y_train_cat, validation_data=(X_val_cnn, y_val_cat) if val_samples else None,
                  epochs=args.epochs, batch_size=args.batch_size)

        evaluate_cnn_sample_majority_vote(model, X_train_cnn, y_train_cnn, train_sample_ids_cnn, idx_to_label, 'train')
        if val_samples:
            evaluate_cnn_sample_majority_vote(model, X_val_cnn, y_val_cnn, val_sample_ids_cnn, idx_to_label, 'validation')
        if test_samples:
            cnn_test_sample_ids, cnn_test_truth, cnn_test_preds, cnn_test_conf, _ = evaluate_cnn_sample_majority_vote(
                model, X_test_cnn, y_test_cnn, cnn_test_sample_ids, idx_to_label, 'test'
            )

        cnn_save_path = args.save_weights_cnn if args.save_weights_cnn else (args.save_weights if args.model != 'both' else None)
        if cnn_save_path:
            model.save(cnn_save_path)
            logger.info("Saved Keras model to %s", cnn_save_path)

    if args.model == 'both' and test_samples and rf_test_sample_ids.size and cnn_test_sample_ids.size:
        common_sample_ids = np.intersect1d(rf_test_sample_ids, cnn_test_sample_ids)
        if common_sample_ids.size:
            common_idx_rf = {sample_id: idx for idx, sample_id in enumerate(rf_test_sample_ids)}
            common_idx_cnn = {sample_id: idx for idx, sample_id in enumerate(cnn_test_sample_ids)}
            ensemble_truth = []
            ensemble_preds = []
            ensemble_sample_ids: list[int] = []
            rf_conf_list: list[float] = []
            cnn_conf_list: list[float] = []
            for sample_id in common_sample_ids.tolist():
                rf_idx = common_idx_rf[sample_id]
                cnn_idx = common_idx_cnn[sample_id]
                rf_pred = rf_test_preds[rf_idx]
                cnn_pred = cnn_test_preds[cnn_idx]
                rf_conf = rf_test_conf[rf_idx]
                cnn_conf = cnn_test_conf[cnn_idx]
                true_label = rf_test_truth[rf_idx]
                if rf_pred == cnn_pred:
                    ensemble_pred = rf_pred
                elif rf_conf > cnn_conf:
                    ensemble_pred = rf_pred
                elif cnn_conf > rf_conf:
                    ensemble_pred = cnn_pred
                else:
                    ensemble_pred = cnn_pred
                ensemble_sample_ids.append(sample_id)
                ensemble_truth.append(true_label)
                ensemble_preds.append(ensemble_pred)
                rf_conf_list.append(rf_conf)
                cnn_conf_list.append(cnn_conf)

            ensemble_truth = np.array(ensemble_truth, dtype=np.int64)
            ensemble_preds = np.array(ensemble_preds, dtype=np.int64)
            ensemble_sample_ids = np.array(ensemble_sample_ids, dtype=np.int64)
            rf_conf_array = np.array(rf_conf_list, dtype=np.float32)
            cnn_conf_array = np.array(cnn_conf_list, dtype=np.float32)

            logger.info("=== Ensemble Majority Vote Results ===")
            ensemble_accuracy = float(np.mean(ensemble_preds == ensemble_truth)) if ensemble_truth.size else 0.0
            logger.info("Accuracy: %.4f", ensemble_accuracy)
            if classification_report is not None and ensemble_truth.size:
                label_names = [idx_to_label[i] for i in sorted(idx_to_label)]
                logger.info("Confusion Matrix:")
                logger.info("\n" + str(confusion_matrix(ensemble_truth, ensemble_preds)))
                logger.info("Classification Report:")
                logger.info("\n" + classification_report(ensemble_truth, ensemble_preds, target_names=label_names))
            if args.results_csv:
                save_ensemble_results_csv(
                    args.results_csv,
                    ensemble_sample_ids,
                    ensemble_truth,
                    rf_test_preds[[common_idx_rf[sid] for sid in ensemble_sample_ids.tolist()]],
                    rf_conf_array,
                    cnn_test_preds[[common_idx_cnn[sid] for sid in ensemble_sample_ids.tolist()]],
                    cnn_conf_array,
                    ensemble_preds,
                    idx_to_label,
                )
        else:
            logger.warning("No overlapping sample IDs found between RF and CNN predictions for ensemble evaluation")

    logger.info("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train sign language gesture classifiers.")
    parser.add_argument('--train_dir', required=True, help='Directory containing training samples')
    parser.add_argument('--val_dir', help='Directory containing validation samples')
    parser.add_argument('--test_dir', help='Directory containing test samples')
    parser.add_argument('--model', choices=['random_forest', 'cnn_lstm', 'cnn_bilstm', 'both'], default='random_forest', help='Model type to train or evaluate. Use both to run Random Forest and CNN-BiLSTM together.')
    parser.add_argument('--window_size', type=int, default=32, help='Sliding window length')
    parser.add_argument('--step_size', type=int, help='Step size between windows; defaults to 50%% overlap for RF and 8 for CNN-BiLSTM if omitted')
    # RandomForest params
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees for RandomForest')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum depth of trees for RandomForest')
    parser.add_argument('--save_weights_rf', help='Path to save Random Forest model weights when running both models')
    # Deep model hyperparameters
    parser.add_argument('--conv_filters', type=int, default=32, help='Number of filters for Conv1D layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for Conv1D layer')
    parser.add_argument('--lstm_units', type=int, default=32, help='Number of units in the (Bi)LSTM layer')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate before final dense layers')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for deep models')
    parser.add_argument('--save_weights_cnn', help='Path to save CNN-BiLSTM model weights when running both models')
    parser.add_argument('--save_weights', help='Path to save trained model weights for single-model mode')
    parser.add_argument('--load_weights', help='Path to load pre-trained CNN model weights')
    parser.add_argument('--results_csv', help='CSV path to save ensemble predictions when running both models')
    parser.add_argument('--num_letters', type=int, help='Limit training to the first N letters alphabetically (A, B, C, ...). If not specified, uses the reduced label set by default.')
    parser.add_argument('--letters', help='Comma-separated gesture labels to include, e.g. A,B,C,D,Y. Overrides --num_letters if specified.')

    args = parser.parse_args()
    main(args)

