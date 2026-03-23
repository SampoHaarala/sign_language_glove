"""
Feature extraction module for glove sensor data.

This module provides functions to convert raw, normalised sensor
readings into fixed-length feature vectors suitable for classical
machine learning algorithms. The glove produces a vector of sensor
readings at each time step. We summarise each sensor within a
window using simple statistics such as mean, standard deviation,
minimum, maximum, median, range and slope. Additional features
(variance, RMS) can be added if they prove beneficial.

Note: according to recent sign language glove literature, smoothing
should be applied before normalisation so that the calibrated
minimum and maximum values are not distorted. After low‑pass
filtering and normalisation, feature extraction is performed on
windows of normalised data.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from typing import Iterable, Tuple


def load_sample(file_path: str, ignore_time: bool = True) -> np.ndarray:
    """Load a gesture sample from .txt or .csv into a NumPy array.

    For .txt files: whitespace or comma separated numeric values; no header.
    For .csv files: header row may be present. If it contains a "time" column,
    that column will be ignored by default.

    Parameters
    ----------
    file_path : str
        Path to the sample file.
    ignore_time : bool, default True
        If True, drop the first column if named "time" (or first numeric
        column in CSV with 10 columns for time + 9 sensors).

    Returns
    -------
    np.ndarray
        Array of shape (timesteps, sensors).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    if path.suffix.lower() == ".csv":
        # Use csv module for robust parsing and optional header handling
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row and not row[0].strip().startswith("#")]

        if not rows:
            return np.empty((0, 0), dtype=np.float32)

        # Detect header row with non-numeric content
        header = rows[0]
        has_header = any(not cell.replace(".", "", 1).replace("-", "", 1).isdigit() for cell in header)
        data_rows = rows[1:] if has_header else rows

        parsed = []
        for row in data_rows:
            # for CSV we can support random separators that are normalized by csv.reader
            row = [cell.strip() for cell in row if cell.strip() != ""]
            if not row:
                continue
            try:
                parsed.append([float(x) for x in row])
            except ValueError:
                # ignore malformed rows
                continue

        if not parsed:
            return np.empty((0, 0), dtype=np.float32)

        arr = np.array(parsed, dtype=np.float32)

        if ignore_time and arr.shape[1] >= 2:
            colnames = [c.strip().lower() for c in header] if has_header else []
            time_column = None
            if colnames and "time" in colnames:
                time_column = colnames.index("time")
            elif arr.shape[1] == 10:
                time_column = 0

            if time_column is not None and time_column < arr.shape[1]:
                arr = np.delete(arr, time_column, axis=1)

        return arr

    # default: text file
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            try:
                row = [float(x) for x in parts]
                data.append(row)
            except ValueError:
                continue

    return np.array(data, dtype=np.float32)


def sliding_windows(data: np.ndarray, window_size: int, step_size: int) -> Iterable[np.ndarray]:
    """Yield overlapping windows from a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (timesteps, sensors).
    window_size : int
        Length of each window.
    step_size : int
        Step size between consecutive windows.

    Yields
    ------
    np.ndarray
        Windows of shape (window_size, sensors).
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive integers")
    num_samples = data.shape[0]
    for start in range(0, num_samples - window_size + 1, step_size):
        yield data[start:start + window_size]

def compute_slope(signal: np.ndarray) -> float:
    """Compute the linear slope of a 1D signal using simple difference.

    The slope is defined as (last - first) / length. This captures
    whether the finger is moving upward or downward over the window.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional array of sensor values.

    Returns
    -------
    float
        Slope of the signal.
    """
    if len(signal) < 2:
        return 0.0
    return float(signal[-1] - signal[0]) / (len(signal) - 1)

def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract statistical features from a single window.

    For each sensor (column) in the window, compute:
      - mean
      - standard deviation
      - minimum
      - maximum
      - median
      - range (max - min)
      - slope (first and last value)
      - root mean square (RMS)

    Parameters
    ----------
    window : np.ndarray
        Array of shape (window_size, sensors).

    Returns
    -------
    np.ndarray
        Feature vector of shape (sensors * num_features,).
    """
    # Ensure 2D
    if window.ndim != 2:
        raise ValueError("window must be 2D (timesteps x sensors)")
    sensors = window.shape[1]
    feats = []
    for i in range(sensors):
        col = window[:, i]
        mean = np.mean(col)
        std = np.std(col)
        min_val = np.min(col)
        max_val = np.max(col)
        median = np.median(col)
        signal_range = max_val - min_val
        slope = compute_slope(col)
        rms = np.sqrt(np.mean(col**2))
        feats.extend([mean, std, min_val, max_val, median, signal_range, slope, rms])
    return np.array(feats, dtype=np.float32)

def extract_features_from_sample(sample: np.ndarray, window_size: int, step_size: int) -> Tuple[np.ndarray, list[str]]:
    """Generate a feature matrix for all windows within a sample.

    Parameters
    ----------
    sample : np.ndarray
        Raw sample array of shape (timesteps, sensors).
    window_size : int
        Number of time steps per window.
    step_size : int
        Step size between windows.

    Returns
    -------
    Tuple[np.ndarray, list[str]]
        Feature matrix of shape (num_windows, sensors * num_features).
    """
    windows = list(sliding_windows(sample, window_size, step_size))
    if not windows:
        raise ValueError("Sample is shorter than window size")
    feature_matrix = np.stack([extract_features(w) for w in windows])
    return feature_matrix

