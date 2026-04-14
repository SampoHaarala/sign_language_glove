"""
ASL Data Normalizer
===================

This utility normalizes saved ASL sensor recordings in CSV files.  It
walks a source directory recursively, normalizes each file's sensor
columns to the range [0, 1] based on that file's observed minimum and
maximum values, and writes the normalized files to an output directory
while preserving the original subdirectory structure.

Usage:

    python asl_data_normalizer.py --input /path/to/raw_session --output /path/to/normalized_session

If the input file contains a timestamp column as the first value in each
row, the timestamp is preserved and only the sensor values are normalized.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple


def parse_sensor_file(file_path: Path) -> Tuple[Optional[List[float]], Optional[List[List[float]]]]:
    """Load a CSV file and return the optional timestamp column and raw sensor rows."""
    rows: List[List[float]] = []
    timestamps: List[float] = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return None, None
            # Detect header rows by checking if first row tokens are numeric
            try:
                [float(x) for x in header if x.strip()]
                first_row_is_data = True
            except ValueError:
                first_row_is_data = False
            if first_row_is_data:
                row = [x.strip() for x in header if x.strip()]
                if len(row) >= 2:
                    # Assume first column is time when there are more than 9 values
                    if len(row) > 9:
                        timestamps.append(float(row[0]))
                        rows.append([float(x) for x in row[1:]])
                    else:
                        rows.append([float(x) for x in row])
            for line in reader:
                parts = [x.strip() for x in line if x.strip()]
                if not parts:
                    continue
                try:
                    floats = [float(x) for x in parts]
                except ValueError:
                    continue
                if len(floats) > 9:
                    timestamps.append(floats[0])
                    rows.append(floats[1:])
                else:
                    rows.append(floats)
        if not rows:
            return None, None
        sensor_length = len(rows[0])
        for row in rows:
            if len(row) != sensor_length:
                return None, None
        return timestamps if timestamps else None, rows
    except Exception:
        return None, None


def normalize_sensor_rows(rows: List[List[float]]) -> List[List[float]]:
    """Normalize sensor rows per column to the interval [0, 1]."""
    if not rows:
        return []
    sensor_count = len(rows[0])
    columns = [[row[col] for row in rows] for col in range(sensor_count)]
    normalized_columns: List[List[float]] = []
    for values in columns:
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            normalized_columns.append([0.5] * len(values))
        else:
            normalized_columns.append([(value - min_val) / (max_val - min_val) for value in values])
    normalized_rows = [[normalized_columns[col][row_idx] for col in range(sensor_count)] for row_idx in range(len(rows))]
    return normalized_rows


def write_normalized_file(output_path: Path, timestamps: Optional[List[float]], normalized_rows: List[List[float]]) -> None:
    """Write normalized rows back to a CSV file, preserving timestamps if present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for idx, row in enumerate(normalized_rows):
            if timestamps is not None:
                writer.writerow([f"{timestamps[idx]:.6f}"] + [f"{value:.6f}" for value in row])
            else:
                writer.writerow([f"{value:.6f}" for value in row])


def normalize_dataset(input_dir: Path, output_dir: Path) -> None:
    """Normalize all CSV files in an input directory into the output directory."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for input_path in sorted(input_dir.rglob("*.csv")):
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        timestamps, rows = parse_sensor_file(input_path)
        if rows is None:
            print(f"Skipping invalid file: {relative_path}")
            skipped += 1
            continue
        normalized_rows = normalize_sensor_rows(rows)
        write_normalized_file(output_path, timestamps, normalized_rows)
        processed += 1
    print(f"Normalization complete: {processed} files processed, {skipped} files skipped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize saved ASL sensor CSV files to [0, 1].")
    parser.add_argument("--input", dest="input_dir", required=True, help="Directory containing raw saved CSV files")
    parser.add_argument("--output", dest="output_dir", required=True, help="Directory to write normalized CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    normalize_dataset(input_dir, output_dir)


if __name__ == "__main__":
    main()