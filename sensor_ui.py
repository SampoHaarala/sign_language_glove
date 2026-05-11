"""
Serial reader for sensor glove data

This script listens to a serial port, receives comma‑separated values
from an Arduino/ESP32 sketch (counter followed by sensor readings),
and keeps track of the minimum and maximum value observed for each
sensor.  It prints the current readings along with the running
minimum and maximum values to the console.

Usage:
    python3 sensor_ui.py [serial_port]

Install pyserial before running:
    pip install pyserial
"""

import sys
import time
from typing import List, Optional

try:
    import serial  # type: ignore
except ImportError as e:
    raise SystemExit(
        "pyserial is required to run this script. Install it with 'pip install pyserial'."
    ) from e


def update_min_max(values: List[int], mins: List[Optional[int]], maxs: List[Optional[int]]) -> None:
    """Update running minimum and maximum lists in place.

    Parameters
    ----------
    values : List[int]
        Latest sensor readings.
    mins : List[Optional[int]]
        Running minimum values for each sensor.  None indicates no value seen yet.
    maxs : List[Optional[int]]
        Running maximum values for each sensor.  None indicates no value seen yet.
    """
    for i, val in enumerate(values):
        if mins[i] is None or val < mins[i]:
            mins[i] = val
        if maxs[i] is None or val > maxs[i]:
            maxs[i] = val


def main(port: str = "/dev/ttyUSB0", baud: int = 115200, num_sensors: int = 9) -> None:
    """Listen to the serial port and display sensor statistics.

    Parameters
    ----------
    port : str
        Serial port device (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows).
    baud : int
        Serial baud rate.  Must match the Arduino sketch (default 115200).
    num_sensors : int
        Expected number of sensor channels following the counter value.
    """
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        raise SystemExit(f"Could not open serial port {port}: {e}")

    mins: List[Optional[int]] = [None] * num_sensors
    maxs: List[Optional[int]] = [None] * num_sensors

    print(f"Listening on {port} at {baud} baud...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            try:
                line_bytes = ser.readline()
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                break
            if not line_bytes:
                continue
            try:
                line = line_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not line:
                continue
            parts = line.split(',')
            # Expect a counter followed by sensor readings
            if len(parts) != num_sensors + 1:
                continue
            try:
                values = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            update_min_max(values, mins, maxs)
            # Display current values and statistics
            print(f"Current sensor readings : {values}")
            print(f"Minimum values so far    : {mins}")
            print(f"Maximum values so far    : {maxs}\n")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()


if __name__ == "__main__":
    port_arg = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    main(port_arg)