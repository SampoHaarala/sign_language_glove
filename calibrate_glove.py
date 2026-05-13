"""
Sensor Calibration Tool for Sign Language Glove
===============================================

This script calibrates the sensor glove by collecting sensor data while
the glove is held steady (typically with fingers straight). It measures
the baseline sensor values, computes statistics (min, max, range, stability),
and saves calibration data to a JSON file.

The calibration process includes:
1. A preparation period where the user can position the glove
2. A countdown before data collection begins
3. Collection of sensor readings over a specified duration
4. Computation of sensor statistics (range, stability/std dev)
5. Saving to a JSON file with calibration values and metadata

Usage
-----

1.  Install dependencies:

    .. code:: bash

       pip install pyserial

2.  Connect the ESP32 glove to your computer and identify the serial port.

3.  Run the script:

    .. code:: bash

       python calibrate_glove.py --port <SERIAL_PORT> --duration 10 --prep_time 5

    Replace ``<SERIAL_PORT>`` with your device path (e.g., COM3 or /dev/ttyUSB0).
    ``--duration`` specifies how long to collect calibration data (default 10 seconds).
    ``--prep_time`` specifies the preparation period before calibration (default 5 seconds).

The script writes calibration data to ``calibration.json`` in the following format:

.. code:: json

    {
        "timestamp": "2026-05-13T10:30:45",
        "metadata": {
            "sensor_ranges": [0, 1023],
            "duration_seconds": 10,
            "num_samples": 300,
            "preparation_time": 5
        },
        "calibration": {
            "sensor_0": {"min": 450, "max": 520, "range": 70, "mean": 485, "std": 18},
            "sensor_1": {"min": 410, "max": 530, "range": 120, "mean": 470, "std": 25},
            ...
        },
        "notes": "Hold glove steady with fingers straight during collection."
    }

Metadata fields (in the "metadata" section) are preserved but not used when
loading the calibration file. The "calibration" section contains the actual
sensor statistics used for normalization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from statistics import mean, stdev

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None


def list_serial_ports() -> list[str]:
    """Return a list of available serial port names."""
    if serial is None:
        return []
    ports = [p.device for p in list_ports.comports()]
    return ports


class SensorReader(threading.Thread):
    """Background thread that continuously reads lines from a serial port."""

    def __init__(self, port: str, baudrate: int = 115200):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.queue = Queue()
        self.running = False
        self.serial_conn = None

    def run(self) -> None:
        """Main thread loop reading from serial port."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            print(f"Connected to {self.port} at {self.baudrate} baud.")
            while self.running:
                try:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.queue.put(line)
                except Exception as e:
                    print(f"Error reading from serial: {e}")
                    break
        except serial.SerialException as e:
            print(f"Failed to open serial port {self.port}: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.running = False

    def stop(self) -> None:
        """Stop the reader thread."""
        self.running = False

def percentile(values: list[float], p: float) -> float:
    """Return percentile p from a list without requiring numpy."""
    if not values:
        raise ValueError("Cannot calculate percentile of empty list")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    index = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower

    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def parse_sensor_line(line: str) -> list[float] | None:
    """Parse a line of sensor data from the ESP32.

    Expected format:
    counter,sensor0,sensor1,sensor2,sensor3,sensor4
    """
    try:
        values = [float(x.strip()) for x in line.split(",")]
        if len(values) != 6:
            print(f"Skipping malformed line: {line!r}")
            return None
        return values[1:]
    except ValueError:
        print(f"Skipping non-numeric line: {line!r}")
        return None


def collect_phase(reader: SensorReader, duration: int, prep_time: int, instruction: str) -> list[list[float]]:
    """Collect one calibration phase."""
    print("\n=== Calibration Phase ===")
    print(instruction)
    print(f"Preparation time: {prep_time} seconds")

    for i in range(prep_time, 0, -1):
        print(f"Calibration starts in {i}s...", end="\r")
        time.sleep(1)

    for i in range(3, 0, -1):
        print(f"Countdown: {i}...", end="\r")
        time.sleep(1)

    print("Recording!")

    data = []
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            line = reader.queue.get(timeout=0.1)
            sensors = parse_sensor_line(line)
            if sensors:
                data.append(sensors)
                elapsed = time.time() - start_time
                print(f"Collected {len(data)} samples ({elapsed:.1f}s)", end="\r")
        except Empty:
            pass

    print(f"\nPhase complete! Collected {len(data)} samples.\n")
    return data


def collect_calibration_data(reader: SensorReader, duration: int, prep_time: int) -> tuple[list[list[float]], list[list[float]]]:
    """Collect open-hand and fist calibration data."""
    open_data = collect_phase(
        reader,
        duration,
        prep_time,
        "Open your palm fully. Keep all fingers straight. This estimates high sensor values.",
    )

    fist_data = collect_phase(
        reader,
        duration,
        prep_time,
        "Close your hand into a tight fist. This estimates low sensor values.",
    )

    return open_data, fist_data

def compute_statistics(open_data: list[list[float]], fist_data: list[list[float]]) -> dict:
    """Compute robust calibration statistics for each sensor.

    The usable calibration range is NOT based on raw min/max because
    unstable sensors can produce outliers.

    Instead:
    - robust min = 10th percentile of fist readings
    - robust max = 90th percentile of open-hand readings
    """

    if not open_data or not fist_data:
        return {}

    num_sensors = min(len(open_data[0]), len(fist_data[0]))
    stats = {}

    for sensor_idx in range(num_sensors):
        open_values = [row[sensor_idx] for row in open_data]
        fist_values = [row[sensor_idx] for row in fist_data]

        robust_min = percentile(fist_values, 10)
        robust_max = percentile(open_values, 90)

        # Safety fallback if calibration direction is inverted or too narrow
        if robust_max <= robust_min:
            combined = open_values + fist_values
            robust_min = percentile(combined, 10)
            robust_max = percentile(combined, 90)

        if robust_max <= robust_min:
            robust_min = min(open_values + fist_values)
            robust_max = max(open_values + fist_values)

        all_values = open_values + fist_values

        stats[f"sensor_{sensor_idx}"] = {
            "min": float(robust_min),
            "max": float(robust_max),
            "range": float(robust_max - robust_min),
            "open_mean": float(mean(open_values)),
            "fist_mean": float(mean(fist_values)),
            "open_std": float(stdev(open_values)) if len(open_values) > 1 else 0.0,
            "fist_std": float(stdev(fist_values)) if len(fist_values) > 1 else 0.0,
            "observed_min": float(min(all_values)),
            "observed_max": float(max(all_values)),
        }

    return stats


def save_calibration(calibration_stats: dict, output_file: str, duration: int, prep_time: int, num_samples: int) -> None:
    """Save calibration data to a JSON file.
    
    Parameters
    ----------
    calibration_stats : dict
        Statistics computed from calibration data.
    output_file : str
        Path to save the calibration file.
    duration : int
        Duration of calibration data collection.
    prep_time : int
        Preparation time.
    num_samples : int
        Number of samples collected.
    """
    calibration_data = {
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "sensor_ranges": [0, 1023],
            "duration_seconds": duration,
            "num_samples": num_samples,
            "preparation_time": prep_time,
            "notes": "Metadata section is preserved but not used when loading calibration. See 'calibration' section for actual sensor statistics."
        },
        "calibration": calibration_stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"Calibration saved to {output_file}")


def load_calibration(calibration_file: str) -> dict | None:
    """Load calibration data from a JSON file.
    
    Only reads the 'calibration' section; metadata is ignored.
    
    Parameters
    ----------
    calibration_file : str
        Path to the calibration file.
    
    Returns
    -------
    dict | None
        Calibration statistics, or None if file not found.
    """
    if not os.path.isfile(calibration_file):
        return None
    
    try:
        with open(calibration_file, 'r') as f:
            data = json.load(f)
        return data.get("calibration", None)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None


def main(args: argparse.Namespace) -> None:
    """Main calibration routine."""
    if serial is None:
        print("Error: PySerial is required. Install with: pip install pyserial")
        sys.exit(1)
    
    # Detect port if not specified
    if not args.port:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found. Please specify a port with --port")
            sys.exit(1)
        print("Available ports:", ", ".join(ports))
        args.port = ports[0]
        print(f"Using first port: {args.port}")
    
    # Start sensor reader
    reader = SensorReader(args.port)
    reader.start()
    time.sleep(1)  # Give thread time to connect
    
    if not reader.running:
        print("Failed to connect to serial port.")
        sys.exit(1)
    
    try:
        # Collect calibration data
        open_data, fist_data = collect_calibration_data(reader, args.duration, args.prep_time)

        if not open_data or not fist_data:
            print("No sensor data collected in one or both calibration phases.")
            sys.exit(1)
        
        # Compute statistics
        print("Computing sensor statistics...")
        calibration_stats = compute_statistics(open_data, fist_data)
        
        # Print summary
        print("\n=== Calibration Summary ===")
        for sensor_name, stats in calibration_stats.items():
            print(f"{sensor_name}:")
            print(
                f"  Robust range: {stats['min']:.1f} - {stats['max']:.1f} "
                f"(span: {stats['range']:.1f})"
            )
            print(
                f"  Open hand: mean {stats['open_mean']:.1f}, "
                f"std {stats['open_std']:.2f}"
            )
            print(
                f"  Fist:      mean {stats['fist_mean']:.1f}, "
                f"std {stats['fist_std']:.2f}"
            )
            print(
                f"  Observed raw extremes: "
                f"{stats['observed_min']:.1f} - {stats['observed_max']:.1f}"
            )
        
        # Save calibration
        save_calibration(
                        calibration_stats,
                        args.output,
                        args.duration,
                        args.prep_time,
                        len(open_data) + len(fist_data),
                        )
        
    finally:
        reader.stop()
        reader.join(timeout=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate sensor glove by collecting baseline readings.")
    parser.add_argument('--port', help='Serial port (e.g., COM3 or /dev/ttyUSB0). Auto-detected if not specified.')
    parser.add_argument('--duration', type=int, default=10, help='Duration of calibration data collection in seconds (default: 10)')
    parser.add_argument('--prep_time', type=int, default=5, help='Preparation period before calibration in seconds (default: 5)')
    parser.add_argument('--output', default='calibration.json', help='Output file path (default: calibration.json)')
    
    args = parser.parse_args()
    main(args)
