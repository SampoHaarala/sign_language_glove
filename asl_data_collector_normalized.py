"""
ASL Data Collection Tool with Normalization
===========================================

This script implements a data‑collection utility for building a dataset of
American Sign Language (ASL) gestures recorded from a glove with nine
sensors connected to an ESP32 microcontroller.  The program presents the
user with pictures of each ASL letter (A–Z) in a randomised order and
records approximately three seconds of time‑series sensor data for each
repetition.  Each gesture sample is saved to its own comma‑separated
values (CSV) file following the format ``<subjectID>-<label>-<trial>.csv``.

The application creates a new directory for every session to keep
different recording runs separate.  Within the session directory a
subdirectory named after the subject ID is created; all gesture files
for that person are stored there.  Sensor data are read from the
specified serial port (for example ``/dev/ttyUSB0`` or ``COM3``) using
PySerial.

This version includes automatic normalization of sensor data to the range [0, 1]
based on the minimum and maximum values observed in each recording session.

Usage
-----

1.  Install the required dependencies:

    .. code:: bash

       pip install pyserial pillow

2.  Place PNG images for each letter of the ASL alphabet (A.png, B.png,
    …, Z.png) into an ``images`` directory located in the same folder
    as this script.  Public domain ASL letter icons can be downloaded
    from Wikimedia Commons (e.g. the image for "A" is available
    at the link documented in the accompanying report【346850666905833†L120-L140】).

3.  Run the script from the terminal:

    .. code:: bash

       python asl_data_collector_normalized.py --port <SERIAL_PORT> --subject <SUBJECT_ID>

    Replace ``<SERIAL_PORT>`` with the device path for your ESP32 and
    ``<SUBJECT_ID>`` with a meaningful identifier for the participant.

At startup the program asks how many repetitions of the alphabet to
record and the directory in which to write the session.  It then
shuffles the list of letters and records each gesture.  A status bar
indicates progress.  While recording the user sees a countdown, then
the sensor values are captured for three seconds and written to a
CSV file.

Note: This script assumes that the ESP32 transmits ten comma‑separated
sensor readings per line (counter + 9 sensor values) at a regular sampling rate.  Modify the
``read_sensor_line`` function if your microcontroller uses a different
format.

"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import gesture_subset_abcd_y as gesture_subset
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None  # type: ignore


def list_serial_ports() -> list[str]:
    """Return a list of available serial port names.

    This helper enumerates connected serial devices using PySerial.  If
    PySerial is not available a fallback empty list is returned.
    """
    if serial is None:
        return []
    ports = [p.device for p in list_ports.comports()]
    return ports


class SensorReader(threading.Thread):
    """Background thread that continuously reads lines from a serial port.

    The latest line is stored in a queue for retrieval.  If an
    exception occurs (e.g. port disconnects), it is propagated to the
    main thread via an exception attribute.
    """

    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.queue: Queue[str] = Queue()
        self.exception: Exception | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self.queue.put(line)
        except Exception as exc:  # noqa: BLE001
            self.exception = exc

    def stop(self) -> None:
        self._stop_event.set()

    def get_line(self, timeout: float = 0.1) -> str | None:
        """Return the latest line read from the serial port.

        If no line is available within the timeout, ``None`` is returned.
        """
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None


def read_sensor_line(line: str) -> list[float] | None:
    """Parse a line of sensor data into a list of floats.

    The expected format is ten comma‑separated numerical values (counter + 9 sensors).
    Returns the 9 sensor values, ignoring the counter. Invalid lines return ``None``.
    Adjust this function if your ESP32 uses a different output format.
    """
    parts = [p.strip() for p in line.split(",")]
    expected_len = 1 + 5  # counter + sensors
    if len(parts) != expected_len:
        return None
    try:
        return [float(p) for p in parts[1:]]
    except ValueError:
        return None
    
def normalize_static(rows: list[list[float]]) -> list[list[float]]:
    # Fallback: clip to 0–4065 and divide by 4065
    max_adc = 4065.0
    return [[max(0.0, min(val, max_adc)) / max_adc for val in row] for row in rows]


def compute_calibration_range(rows: list[list[float]], window_size: int = 8) -> tuple[list[float], list[float]]:
    """Compute per-sensor min/max values from calibration recordings.

    Rather than using a single extreme reading, this computes the average
    of the smallest and largest windows of values. That reduces the
    effect of one noisy low or high sample during calibration.
    """
    if not rows:
        return [0.0] * 9, [4065.0] * 9

    sensor_count = len(rows[0])
    min_vals: list[float] = []
    max_vals: list[float] = []
    for index in range(sensor_count):
        values = sorted(row[index] for row in rows)
        if len(values) <= window_size:
            min_vals.append(sum(values) / len(values))
            max_vals.append(sum(values) / len(values))
            continue

        min_vals.append(sum(values[:window_size]) / window_size)
        max_vals.append(sum(values[-window_size:]) / window_size)

    return min_vals, max_vals


def normalize_data(rows: list[list[float]], min_vals: list[float] | None = None, max_vals: list[float] | None = None) -> list[list[float]]:
    """Normalize sensor data to [0, 1] using calibration bounds or fixed ADC range."""
    if not rows:
        return rows

    if min_vals is None or max_vals is None:
        return normalize_static(rows)

    normalized_rows: list[list[float]] = []
    for row in rows:
        normalized_row: list[float] = []
        for value, min_value, max_value in zip(row, min_vals, max_vals):
            if max_value <= min_value:
                normalized_row.append(0.0)
            else:
                normalized = (value - min_value) / (max_value - min_value)
                normalized_row.append(max(0.0, min(1.0, normalized)))
        normalized_rows.append(normalized_row)

    return normalized_rows


class DataCollectorGUI:
    """Graphical interface for collecting ASL gesture data.

    This class handles the Tkinter window, displays letter images,
    manages countdown timers and recording progress, and writes CSV
    files to the appropriate directories.  It communicates with the
    sensor reader thread to obtain incoming data.
    """

    def __init__(
        self,
        ser,
        subject_id: str,
        session_dir: Path,
        repeat_count: int,
        record_seconds: float = 3.0,
        calibration_seconds: float | None = None,
        letters=None,
    ):
        self.ser = ser
        self.subject_id = subject_id
        self.session_dir = session_dir
        self.repeat_count = repeat_count
        self.record_seconds = record_seconds
        self.calibration_duration = calibration_seconds if calibration_seconds is not None else record_seconds
        self.letters = letters if letters is not None else gesture_subset.get_reduced_gesture_set()
        self.root = tk.Tk()
        self.root.title("ASL Data Collector (Normalized)")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        # Create subject directory
        self.subject_dir = session_dir / subject_id
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        # Load images
        self.images: dict[str, ImageTk.PhotoImage] = {}
        self._load_images()
        # Prepare order of letters
        self.samples: list[tuple[str, int]] = []
        for i in range(repeat_count):
            for letter in self.letters:
                self.samples.append((letter, i + 1))
        random.shuffle(self.samples)
        self.current_index = -1
        # UI elements
        self.letter_label = tk.Label(self.root, font=("Arial", 24))
        self.letter_label.pack(pady=10)
        self.canvas = tk.Label(self.root)
        self.canvas.pack()
        self.status_label = tk.Label(self.root, text="Press Start to begin recording", font=("Arial", 14))
        self.status_label.pack(pady=10)
        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack()
        self.plot_canvas = tk.Canvas(self.root, width=560, height=240, bg="white", highlightthickness=1, highlightbackground="black")
        self.plot_canvas.pack(pady=10)
        self.plot_canvas.create_text(280, 120, text="Sensor timeline will appear here after each recording.", fill="gray", font=("Arial", 10))
        self.start_button = tk.Button(self.root, text="Start Session", command=self.on_start_button, font=("Arial", 14))
        self.start_button.pack(pady=10)
        self.current_letter: str = ""
        self.current_trial: int = 0
        self.sensor_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
        ]
        self.calibration_open_data: list[list[float]] = []
        self.calibration_closed_data: list[list[float]] = []
        self.min_sensor_values: list[float] | None = None
        self.max_sensor_values: list[float] | None = None
        # Sensor reader thread
        self.reader = SensorReader(ser)
        self.reader.start()
        # Exit handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_images(self) -> None:
        """Load all letter images from the images directory.

        If an image is missing for a letter, a placeholder image with
        the letter drawn as text will be generated dynamically.
        """
        images_dir = Path(__file__).resolve().parent / "images"
        for letter in self.letters:
            image_path = images_dir / f"{letter}.png"
            if image_path.exists():
                try:
                    img = Image.open(image_path)
                    img = img.resize((300, 400), Image.LANCZOS)
                    # Composite transparency on white background to avoid black edges
                    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
                        img = img.convert("RGBA")
                        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background.convert("RGB")
                    else:
                        img = img.convert("RGB")
                    self.images[letter] = ImageTk.PhotoImage(img, master=self.root)
                    print(f"Loaded image for {letter}: {image_path}")
                    continue
                except Exception as exc:
                    print(f"Failed to load image {image_path}: {exc}")
                    img = None
            else:
                print(f"Missing image file, using placeholder for {letter}: {image_path}")
                img = None

            # Generate placeholder when actual image is missing or failed to load
            placeholder = Image.new("RGB", (300, 400), color="white")
            try:
                import PIL.ImageDraw as ImageDraw
                import PIL.ImageFont as ImageFont
                draw = ImageDraw.Draw(placeholder)
                try:
                    font = ImageFont.truetype("arial.ttf", 200)
                except Exception:
                    font = ImageFont.load_default()
                w, h = draw.textsize(letter, font=font)
                draw.text(((300 - w) / 2, (400 - h) / 2), letter, fill="black", font=font)
            except Exception:
                pass
            self.images[letter] = ImageTk.PhotoImage(placeholder, master=self.root)

    def _clear_plot(self) -> None:
        """Reset the timeline plot area to a neutral placeholder state."""
        self.plot_canvas.delete("all")
        self.plot_canvas.create_text(
            280,
            120,
            text="Sensor timeline will appear here after each recording.",
            fill="gray",
            font=("Arial", 10),
        )

    def plot_sensor_timeline(self, data: list[list[float]], timestamps: list[float]) -> None:
        """Draw the normalized sensor values as a time-series chart."""
        self._clear_plot()
        if not data or not timestamps:
            return

        width = int(self.plot_canvas.cget("width"))
        height = int(self.plot_canvas.cget("height"))
        margin = 40
        plot_width = width - margin * 2
        plot_height = height - margin * 2

        # Draw border and grid lines
        self.plot_canvas.create_rectangle(margin, margin, width - margin, height - margin, outline="#444")
        self.plot_canvas.create_line(margin, margin + plot_height / 2, width - margin, margin + plot_height / 2, fill="#ddd")
        self.plot_canvas.create_line(margin + plot_width / 2, margin, margin + plot_width / 2, height - margin, fill="#ddd")
        self.plot_canvas.create_text(margin - 10, margin, text="1.0", anchor="e", fill="#444", font=("Arial", 8))
        self.plot_canvas.create_text(margin - 10, height - margin, text="0.0", anchor="e", fill="#444", font=("Arial", 8))
        self.plot_canvas.create_text(width - margin, height - 10, text=f"{timestamps[-1]:.2f}s", anchor="e", fill="#444", font=("Arial", 8))
        self.plot_canvas.create_text(margin, height - 10, text="0.0s", anchor="w", fill="#444", font=("Arial", 8))
        self.plot_canvas.create_text(width / 2, 15, text="Sensor outputs over time", fill="#000", font=("Arial", 10, "bold"))

        time_min = timestamps[0]
        time_max = timestamps[-1] if timestamps[-1] > time_min else self.record_seconds
        x_scale = plot_width / (time_max - time_min if time_max > time_min else 1.0)

        for sensor_index in range(len(data[0])):
            points = []
            for t, row in zip(timestamps, data):
                x = margin + (t - time_min) * x_scale
                y = height - margin - row[sensor_index] * plot_height
                points.extend((x, y))
            self.plot_canvas.create_line(*points, fill=self.sensor_colors[sensor_index % len(self.sensor_colors)], width=1.5)

        self.plot_canvas.create_text(width - margin + 5, margin + 8, text="1.0", anchor="w", fill="#000", font=("Arial", 8))
        self.plot_canvas.create_text(width - margin + 5, height - margin - 8, text="0.0", anchor="w", fill="#000", font=("Arial", 8))
        self.plot_canvas.create_text(width / 2, height - 20, text="Time (seconds)", fill="#000", font=("Arial", 8))
        self.plot_canvas.create_text(60, margin + 10, text="Normalized sensor values", fill="#000", font=("Arial", 8), anchor="w")

    def run_calibration(self) -> None:
        """Run open-hand and closed-fist calibration before actual recordings."""
        self.progress_label.config(text="Calibration 1/2: Open palm")
        self.letter_label.config(text="Calibration: open your hand")
        self.canvas.config(image="")
        self.status_label.config(text=f"Hold a straight open palm for {self.calibration_duration:.0f} seconds...")
        self.root.after(100, lambda: self.record_calibration_pose("open palm", self._on_open_palm_calibrated))

    def _on_open_palm_calibrated(self, open_data: list[list[float]]) -> None:
        self.calibration_open_data = open_data
        self.progress_label.config(text="Calibration 2/2: Closed fist")
        self.letter_label.config(text="Calibration: make a tight fist")
        self.status_label.config(text=f"Hold a tight fist for {self.calibration_duration:.0f} seconds...")
        self.root.after(100, lambda: self.record_calibration_pose("closed fist", self._on_closed_fist_calibrated))

    def _on_closed_fist_calibrated(self, closed_data: list[list[float]]) -> None:
        self.calibration_closed_data = closed_data
        all_calibration_data = self.calibration_open_data + self.calibration_closed_data
        self.min_sensor_values, self.max_sensor_values = compute_calibration_range(all_calibration_data)
        self.progress_label.config(text="Calibration complete")
        self.status_label.config(text="Calibration saved. Starting gesture recording...")
        self.letter_label.config(text="")
        self.canvas.config(image="")
        self.root.after(1000, self.next_sample)

    def record_calibration_pose(self, pose_name: str, on_complete) -> None:
        """Capture raw sensor readings for a calibration pose."""
        start_time = time.time()
        raw_data: list[list[float]] = []
        while time.time() - start_time < self.calibration_duration:
            line = self.reader.get_line(timeout=0.05)
            if self.reader.exception is not None:
                self.status_label.config(text=f"Serial error: {self.reader.exception}")
                return
            if line:
                values = read_sensor_line(line)
                if values is not None:
                    raw_data.append(values)
        if not raw_data:
            self.status_label.config(text=f"No sensor data collected for {pose_name}. Using default calibration.")
        on_complete(raw_data)

    def on_close(self) -> None:
        """Handle window close event by stopping the sensor reader."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.reader.stop()
            self.root.destroy()

    def next_sample(self) -> None:
        """Advance to the next gesture sample or finish the session."""
        self.current_index += 1
        if self.current_index >= len(self.samples):
            self.status_label.config(text="Data collection complete!")
            self.progress_label.config(text="")
            self.letter_label.config(text="")
            self.canvas.config(image="")
            self.start_button.config(state="disabled")
            return
        self.current_letter, self.current_trial = self.samples[self.current_index]
        letter, trial = self.current_letter, self.current_trial
        # Update progress
        total = len(self.samples)
        self.progress_label.config(text=f"Sample {self.current_index + 1}/{total} (letter {letter}, trial {trial})")
        # Show letter
        self.letter_label.config(text=f"Please sign: {letter}")
        self.canvas.config(image=self.images[letter])
        self.canvas.image = self.images[letter]  # Keep explicit reference to avoid Tk garbage collection
        self.status_label.config(text="Prepare to sign for 3 seconds...")
        self.root.after(3000, lambda: self.start_recording(letter, trial))

    def on_start_button(self) -> None:
        """Handle the start button press and begin calibration."""
        self.start_button.config(state="disabled")
        self.status_label.config(text="Starting calibration...")
        self.root.after(100, self.run_calibration)

    def start_recording(self, letter: str, trial: int) -> None:
        """Begin the 3-second recording period for the current sign."""
        self.status_label.config(text="Recording for 3 seconds...")
        self.root.after(100, lambda: self.record_sample(letter, trial))

    def record_sample(self, letter: str, trial: int) -> None:
        """Capture sensor data for a single sample and save it to a CSV file."""
        filename = f"{self.subject_id}-{letter}-{trial}.csv"
        output_path = self.subject_dir / filename
        start_time = time.time()
        raw_data: list[list[float]] = []
        timestamps: list[float] = []

        while time.time() - start_time < self.record_seconds:
            line = self.reader.get_line(timeout=0.05)
            if self.reader.exception is not None:
                self.status_label.config(text=f"Serial error: {self.reader.exception}")
                return
            if line:
                values = read_sensor_line(line)
                if values is not None:
                    tstamp = time.time() - start_time
                    timestamps.append(tstamp)
                    raw_data.append(values)

        # Normalize the sensor data
        if raw_data:
            normalized_data = normalize_data(raw_data, self.min_sensor_values, self.max_sensor_values)
        else:
            normalized_data = []

        # Write to file with header
        header = ["time"] + [f"sensor{i+1}" for i in range(9)]
        try:
            with output_path.open("w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for tstamp, values in zip(timestamps, normalized_data):
                    row = ",".join([f"{tstamp:.3f}"] + [f"{v:.6f}" for v in values])
                    f.write(row + "\n")
            self.status_label.config(text=f"Saved {output_path.name} (normalized)")
            self.plot_sensor_timeline(normalized_data, timestamps)
        except Exception as exc:  # noqa: BLE001
            self.status_label.config(text=f"Failed to save {output_path.name}: {exc}")
        # Move on
        self.root.after(1000, self.next_sample)

    def run(self) -> None:
        """Run the Tkinter main loop."""
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASL gesture data collection tool with normalization")
    parser.add_argument("--port", help="Serial port for ESP32 (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--subject", help="Identifier for the subject (e.g. participant name)")
    parser.add_argument("--repeats", type=int, default=2, help="Number of times to repeat the alphabet (default 2)")
    parser.add_argument("--letters", help="Comma-separated gesture labels to record, e.g. A,B,C,D,Y. Defaults to the reduced gesture subset.")
    parser.add_argument("--session", required=True, help="Directory to store session data")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate for serial connection (default 115200)")
    parser.add_argument("--calibration_seconds", type=float, default=3.0, help="Seconds to record each calibration pose (default 3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Validate required arguments
    if not args.subject:
        print("Error: --subject is required.")
        sys.exit(1)
    session_dir = Path(args.session)
    # Open serial port
    if serial is None:
        print("PySerial is not installed. Please install it with 'pip install pyserial'.")
        return
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open serial port {args.port}: {exc}")
        return
    allowed_letters = gesture_subset.parse_label_list(args.letters)
    if allowed_letters is None:
        allowed_letters = gesture_subset.get_reduced_gesture_set()
        print(f"Recording gesture subset: {', '.join(allowed_letters)}")
    # Show UI
    gui = DataCollectorGUI(
        ser,
        args.subject,
        session_dir,
        args.repeats,
        calibration_seconds=args.calibration_seconds,
        letters=allowed_letters,
    )
    gui.run()
    # Close serial port on exit
    ser.close()


if __name__ == "__main__":
    main()