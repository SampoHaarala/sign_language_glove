"""
ASL Dataset Cleaner
====================

This utility inspects sign language gesture recordings collected with
the accompanying ``asl_data_collector.py`` script and removes
sub‑standard samples.  The goal is to preserve only those recordings
that capture a meaningful hand movement and exhibit sensor behaviour
consistent with a reference dataset.  Samples that are empty,
constant, too short or too dissimilar to known examples are discarded.

Overview
--------

Each gesture sample produced by the data collection tool is stored as
a CSV file with the format ``<subject>-<label>-<trial>.csv``.  The
first column contains timestamps and the remaining nine columns are
floating‑point readings from the glove sensors.  This script parses
those files, extracts simple summary statistics for each sensor and
compares them against statistics computed from a reference dataset.

A reference dataset directory can be supplied via the ``--reference``
argument.  It should contain one or more subdirectories of clean
recordings whose filenames follow either ``<subject>-<letter>-<trial>.csv``
(new format) or ``<sampleId>-<label>.txt`` (legacy format from the
training scripts).  The cleaner builds per‑letter statistics—mean,
standard deviation and typical sequence length—from the reference
files.  When cleaning a new session the script flags files under the
following conditions:

* **No movement**: the average absolute change between consecutive
  samples across all sensors is below ``--min_variation`` (default
  0.05).  This catches recordings where the user failed to perform
  the gesture.
* **Constant sensor**: any sensor column has variance below
  ``--constant_tol`` (default ``1e-6``).  This indicates a faulty
  sensor stuck at a single value.
* **Length mismatch**: the number of rows differs from the typical
  length in the reference dataset for the same letter by more than
  ``--length_tol`` fraction (default 0.25).  Very short or long
  recordings are likely incomplete or errant.
* **Outlier**: the sample's feature vector (concatenated means and
  standard deviations of the nine sensors) deviates from the
  reference mean by more than ``--distance_threshold`` standard
  deviations.  This coarse distance measure detects unusual shapes
  without requiring an exact match.

Accepted samples are copied into the output directory (``--output``).
Rejected files are written into a ``discarded`` subdirectory for
manual inspection.  A log summarises the decisions for each file.

Usage
-----

::

   python asl_data_cleaner.py --input /path/to/session \
                             --reference /path/to/reference_dataset \
                             --output /path/to/cleaned_session

Additional optional arguments allow fine‑tuning the thresholds.  Run
``python asl_data_cleaner.py --help`` for details.

Note
----

This cleaner deliberately errs on the side of retaining more data.
Thresholds are intentionally conservative; borderline samples may be
kept rather than removed.  You can adjust the parameters to enforce
stricter filtering.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_sensor_file(file_path: Path) -> np.ndarray | None:
    """Load a CSV or TXT file containing sensor readings.

    The first column may be a timestamp which is ignored.  The
    remaining columns must be numeric.  Files with mismatched row
    lengths are skipped.

    Parameters
    ----------
    file_path : Path
        Path to the file to load.

    Returns
    -------
    np.ndarray | None
        Array of shape (timesteps, sensors) or ``None`` if the file
        could not be parsed.
    """
    data: List[List[float]] = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Detect header: if the first row contains non‑numeric
            # tokens (e.g. "time"), skip it
            header = next(reader, None)
            if header is None:
                return None
            # Check if header is numeric; if not, drop it and use
            # remaining rows
            try:
                [float(x) for x in header]
                first_row_is_data = True
            except ValueError:
                first_row_is_data = False
            if first_row_is_data:
                # header is actually data; rewind to include it
                row_vals = [float(x.strip()) for x in header if x.strip()]
                # Drop first column if this appears to be a timestamp
                if len(row_vals) > 1:
                    data.append(row_vals[1:])
            for row in reader:
                # Convert to floats and skip empty strings
                vals = [x.strip() for x in row if x.strip()]
                if not vals:
                    continue
                try:
                    floats = [float(x) for x in vals]
                except ValueError:
                    continue
                # If first column appears to be a timestamp remove it
                if len(floats) > 1:
                    data.append(floats[1:])
        if not data:
            return None
        # Ensure all rows have same length
        row_len = len(data[0])
        for row in data:
            if len(row) != row_len:
                return None
        return np.array(data, dtype=np.float32)
    except Exception:
        return None


def extract_features(sample: np.ndarray) -> np.ndarray:
    """Compute simple summary statistics from a sample.

    Returns a feature vector containing the mean and standard deviation
    of each sensor column.  The resulting vector has length
    ``2 * sensors``.

    Parameters
    ----------
    sample : np.ndarray
        Sensor matrix of shape (timesteps, sensors).

    Returns
    -------
    np.ndarray
        Feature vector of shape (2 * sensors,).
    """
    means = sample.mean(axis=0)
    stds = sample.std(axis=0)
    return np.concatenate([means, stds])


def build_reference_stats(reference_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Compute per‑label reference statistics from a directory of good samples.

    Scans all files in ``reference_dir`` recursively.  For each file it
    infers the gesture label from its filename and accumulates feature
    vectors and sample lengths.  The returned dictionaries map
    labels (e.g. ``"A"``) to the mean feature vector, feature standard
    deviation and average sequence length.

    Parameters
    ----------
    reference_dir : Path
        Root directory containing clean recordings.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]
        (ref_mean, ref_std, ref_len) where each is a mapping from
        gesture label to statistics.
    """
    feature_acc: Dict[str, List[np.ndarray]] = {}
    length_acc: Dict[str, List[int]] = {}

    for file_path in reference_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if not (file_path.suffix.lower() in {".csv", ".txt"}):
            continue
        label = infer_label_from_filename(file_path.name)
        if label is None:
            continue
        sample = parse_sensor_file(file_path)
        if sample is None:
            continue
        feats = extract_features(sample)
        feature_acc.setdefault(label, []).append(feats)
        length_acc.setdefault(label, []).append(len(sample))

    ref_mean: Dict[str, np.ndarray] = {}
    ref_std: Dict[str, np.ndarray] = {}
    ref_len: Dict[str, float] = {}

    for label, feats in feature_acc.items():
        stack = np.stack(feats)
        ref_mean[label] = stack.mean(axis=0)
        # Avoid division by zero: minimum std is small positive
        ref_std[label] = np.maximum(stack.std(axis=0), 1e-8)
        ref_len[label] = float(np.mean(length_acc[label]))
    return ref_mean, ref_std, ref_len


def infer_label_from_filename(filename: str) -> str | None:
    """Extract the gesture label from a sample filename.

    Recognises two patterns:

      * ``<subject>-<letter>-<trial>.csv`` → returns ``letter`` (e.g. ``"A"``)
      * ``<id>-<label>.txt`` → returns ``label``

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
    str | None
        The extracted label or ``None`` if no pattern matches.
    """
    # New format: subject-letter-trial.ext
    parts = filename.rsplit(".", 1)[0].split("-")
    if len(parts) == 3:
        return parts[1]
    # Old format: id-label.ext (label may include hyphens)
    if len(parts) >= 2:
        return parts[-1]
    return None


def assess_sample(sample: np.ndarray, label: str | None, ref_mean: Dict[str, np.ndarray], ref_std: Dict[str, np.ndarray], ref_len: Dict[str, float],
                  min_variation: float, constant_tol: float, length_tol: float, distance_threshold: float) -> Tuple[bool, str]:
    """Evaluate a sample against quality criteria.

    Parameters
    ----------
    sample : np.ndarray
        Sensor matrix of shape (timesteps, sensors).
    label : str | None
        Gesture label inferred from filename.  If ``None`` the
        reference‑based outlier check is skipped.
    ref_mean : Dict[str, np.ndarray]
        Mean feature vectors per label from the reference dataset.
    ref_std : Dict[str, np.ndarray]
        Standard deviation vectors per label from the reference dataset.
    ref_len : Dict[str, float]
        Mean sequence length per label from the reference dataset.
    min_variation : float
        Threshold for average absolute change between consecutive
        samples.
    constant_tol : float
        Threshold below which sensor variance is considered constant.
    length_tol : float
        Allowed fractional deviation of sequence length from the
        reference mean.
    distance_threshold : float
        Maximum Mahalanobis‑like distance (in standard deviation
        units) to accept a sample.  Distances above this value are
        deemed outliers.

    Returns
    -------
    Tuple[bool, str]
        ``(accept, reason)`` where ``accept`` indicates whether the
        sample passes all checks and ``reason`` provides a short
        explanation for rejection if applicable.
    """
    # Check for no movement: mean absolute difference across sensors
    diffs = np.abs(np.diff(sample, axis=0))
    if diffs.size == 0:
        return False, "empty or single row"
    avg_variation = float(diffs.mean())
    if avg_variation < min_variation:
        return False, f"low variation ({avg_variation:.4f} < {min_variation})"
    # Check for constant sensors
    variances = sample.var(axis=0)
    if float(np.min(variances)) < constant_tol:
        return False, "constant sensor"
    # Check length consistency
    if label and label in ref_len:
        ref_length = ref_len[label]
        length = len(sample)
        if abs(length - ref_length) > ref_length * length_tol:
            return False, f"length mismatch ({length} vs {ref_length:.1f})"
    # Check outlier distance
    if label and label in ref_mean:
        feat = extract_features(sample)
        mu = ref_mean[label]
        sigma = ref_std[label]
        # Standardised distance (Mahalanobis with diagonal covariance)
        z = (feat - mu) / sigma
        dist = float(math.sqrt(float((z * z).sum())))
        if dist > distance_threshold:
            return False, f"outlier (distance {dist:.2f} > {distance_threshold})"
    return True, "ok"


def clean_dataset(input_dir: Path, reference_dir: Path, output_dir: Path, min_variation: float,
                  constant_tol: float, length_tol: float, distance_threshold: float) -> None:
    """Clean a dataset directory by removing low quality samples.

    Parameters
    ----------
    input_dir : Path
        Path to the session directory containing raw CSV files.  All
        subdirectories are scanned recursively.
    reference_dir : Path
        Directory containing clean recordings used to compute
        reference statistics.
    output_dir : Path
        Directory into which accepted samples are copied.  A
        ``discarded`` subdirectory is created inside ``output_dir`` for
        rejected files.
    min_variation : float
        Minimum average variation required to accept a sample.
    constant_tol : float
        Maximum variance threshold below which a sensor is considered
        constant.
    length_tol : float
        Fractional tolerance on sequence length compared to reference.
    distance_threshold : float
        Maximum allowed distance from reference features.
    """
    # Build reference statistics
    if reference_dir.exists():
        print(f"Building reference statistics from {reference_dir}…")
        ref_mean, ref_std, ref_len = build_reference_stats(reference_dir)
    else:
        ref_mean, ref_std, ref_len = {}, {}, {}
    output_dir.mkdir(parents=True, exist_ok=True)
    discarded_dir = output_dir / "discarded"
    discarded_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []
    # Iterate over files in input_dir recursively
    for file_path in input_dir.rglob("*.csv"):
        label = infer_label_from_filename(file_path.name)
        sample = parse_sensor_file(file_path)
        if sample is None:
            reason = "could not parse"
            accept = False
        else:
            accept, reason = assess_sample(sample, label, ref_mean, ref_std, ref_len,
                                           min_variation, constant_tol, length_tol, distance_threshold)
        rel_path = file_path.relative_to(input_dir)
        target_dir = output_dir / rel_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        if accept:
            shutil.copy2(file_path, target_dir / file_path.name)
            log_lines.append(f"OK: {rel_path} → kept (label={label})")
        else:
            # Save into discarded directory preserving hierarchy
            disc_target = discarded_dir / rel_path.parent
            disc_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, disc_target / file_path.name)
            log_lines.append(f"DROP: {rel_path} → {reason}")
    # Write summary log
    log_file = output_dir / "cleaning_log.txt"
    with log_file.open("w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"Cleaning complete. See {log_file} for details.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean ASL gesture recordings by filtering low quality samples.")
    parser.add_argument("--input", dest="input_dir", type=str, default="", help="Path to a single raw session directory")
    parser.add_argument("--batch_input", dest="batch_input_dir", type=str, default="", help="Path to a parent directory containing multiple test folders to process")
    parser.add_argument("--reference", dest="reference_dir", type=str, default="", help="Directory of clean reference samples")
    parser.add_argument("--output", dest="output_dir", type=str, default="", help="Directory to write cleaned data (for single input mode)")
    parser.add_argument("--batch_output", dest="batch_output_dir", type=str, default="", help="Parent directory for batch output (for batch mode)")
    parser.add_argument("--min_variation", type=float, default=0.05, help="Minimum mean absolute variation across sensors (default 0.05)")
    parser.add_argument("--constant_tol", type=float, default=1e-6, help="Variance threshold to detect constant sensors (default 1e-6)")
    parser.add_argument("--length_tol", type=float, default=0.25, help="Allowed fractional deviation from reference length (default 0.25)")
    parser.add_argument("--distance_threshold", type=float, default=10.0, help="Maximum z-score distance from reference features (default 10.0)")
    args = parser.parse_args()
    
    reference_dir = Path(args.reference_dir) if args.reference_dir else Path()
    
    # Batch mode
    if args.batch_input_dir:
        batch_input = Path(args.batch_input_dir)
        batch_output = Path(args.batch_output_dir) if args.batch_output_dir else batch_input.parent / f"{batch_input.name}_cleaned"
        
        if not batch_input.exists():
            print(f"Error: batch input directory {batch_input} does not exist.")
            return
        
        print(f"Starting batch processing of {batch_input}…")
        subdirs = [d for d in batch_input.iterdir() if d.is_dir() and d.name != "discarded"]
        if not subdirs:
            print("No subdirectories found in batch input directory.")
            return
        
        print(f"Found {len(subdirs)} subdirectories to process.")
        for i, subdir in enumerate(subdirs, 1):
            output_subdir = batch_output / subdir.name
            print(f"\n[{i}/{len(subdirs)}] Processing {subdir.name}…")
            clean_dataset(subdir, reference_dir, output_subdir,
                         min_variation=args.min_variation,
                         constant_tol=args.constant_tol,
                         length_tol=args.length_tol,
                         distance_threshold=args.distance_threshold)
        print(f"\nBatch processing complete. Cleaned data in {batch_output}")
    
    # Single directory mode
    elif args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        clean_dataset(input_dir, reference_dir, output_dir,
                     min_variation=args.min_variation,
                     constant_tol=args.constant_tol,
                     length_tol=args.length_tol,
                     distance_threshold=args.distance_threshold)
    
    else:
        print("Error: provide either --input and --output for single mode, or --batch_input for batch mode.")
        parser.print_help()


if __name__ == "__main__":
    main()