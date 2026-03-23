"""
ASL Data Collection Tool
========================

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

Usage
-----

1.  Install the required dependencies:

    .. code:: bash

       pip install pyserial pillow

2.  Place PNG images for each letter of the ASL alphabet (A.png, B.png,
    …, Z.png) into an ``images`` directory located in the same folder
    as this script.  Public domain ASL letter icons can be downloaded
    from Wikimedia Commons (e.g. the image for “A” is available
    at the link documented in the accompanying report【346850666905833†L120-L140】).

3.  Run the script from the terminal:

    .. code:: bash

       python asl_data_collector.py --port <SERIAL_PORT> --subject <SUBJECT_ID>

    Replace ``<SERIAL_PORT>`` with the device path for your ESP32 and
    ``<SUBJECT_ID>`` with a meaningful identifier for the participant.

At startup the program asks how many repetitions of the alphabet to
record and the directory in which to write the session.  It then
shuffles the list of letters and records each gesture.  A status bar
indicates progress.  While recording the user sees a countdown, then
the sensor values are captured for three seconds and written to a
CSV file.

Note: This script assumes that the ESP32 transmits nine comma‑separated
sensor readings per line at a regular sampling rate.  Modify the
``read_sensor_line`` function if your microcontroller uses a different
format.

"""

import argparse
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

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

    def __init__(self, ser: serial.Serial):
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

    The expected format is nine comma‑separated numerical values.  Invalid
    lines return ``None``.  Adjust this function if your ESP32 uses a
    different output format.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 9:
        return None
    try:
        values = [float(p) for p in parts]
        return values
    except ValueError:
        return None


class DataCollectorGUI:
    """Graphical interface for collecting ASL gesture data.

    This class handles the Tkinter window, displays letter images,
    manages countdown timers and recording progress, and writes CSV
    files to the appropriate directories.  It communicates with the
    sensor reader thread to obtain incoming data.
    """

    def __init__(self, ser: serial.Serial, subject_id: str, session_dir: Path, repeat_count: int, record_seconds: float = 3.0):
        self.ser = ser
        self.subject_id = subject_id
        self.session_dir = session_dir
        self.repeat_count = repeat_count
        self.record_seconds = record_seconds
        self.root = tk.Tk()
        self.root.title("ASL Data Collector")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        # Create subject directory
        self.subject_dir = session_dir / subject_id
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        # Load images
        self.images: dict[str, ImageTk.PhotoImage] = {}
        self._load_images()
        # Prepare order of letters
        letters = [chr(ord("A") + i) for i in range(26)]
        self.samples: list[tuple[str, int]] = []
        for i in range(repeat_count):
            for letter in letters:
                self.samples.append((letter, i + 1))
        random.shuffle(self.samples)
        self.current_index = -1
        # UI elements
        self.letter_label = tk.Label(self.root, font=("Arial", 24))
        self.letter_label.pack(pady=10)
        self.canvas = tk.Label(self.root)
        self.canvas.pack()
        self.status_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.status_label.pack(pady=10)
        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack()
        # Sensor reader thread
        self.reader = SensorReader(ser)
        self.reader.start()
        # Start first sample
        self.root.after(1000, self.next_sample)
        # Exit handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_images(self) -> None:
        """Load all letter images from the images directory.

        If an image is missing for a letter, a placeholder image with
        the letter drawn as text will be generated dynamically.
        """
        images_dir = Path(__file__).resolve().parent / "images"
        for letter in [chr(ord("A") + i) for i in range(26)]:
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
            return
        letter, trial = self.samples[self.current_index]
        # Update progress
        total = len(self.samples)
        self.progress_label.config(text=f"Sample {self.current_index + 1}/{total} (letter {letter}, trial {trial})")
        # Show letter
        self.letter_label.config(text=f"Please sign: {letter}")
        self.canvas.config(image=self.images[letter])
        self.canvas.image = self.images[letter]  # Keep explicit reference to avoid Tk garbage collection
        # Start countdown before recording
        self.countdown = 3
        self.status_label.config(text=f"Starting in {self.countdown}")
        self._do_countdown(letter, trial)

    def _do_countdown(self, letter: str, trial: int) -> None:
        """Update countdown timer before recording."""
        if self.countdown > 0:
            self.status_label.config(text=f"Starting in {self.countdown}")
            self.countdown -= 1
            self.root.after(1000, lambda: self._do_countdown(letter, trial))
        else:
            self.status_label.config(text="Recording…")
            self.root.after(100, lambda: self.record_sample(letter, trial))

    def record_sample(self, letter: str, trial: int) -> None:
        """Capture sensor data for a single sample and save it to a CSV file."""
        filename = f"{self.subject_id}-{letter}-{trial}.csv"
        output_path = self.subject_dir / filename
        start_time = time.time()
        rows: list[str] = []
        while time.time() - start_time < self.record_seconds:
            line = self.reader.get_line(timeout=0.05)
            if self.reader.exception is not None:
                self.status_label.config(text=f"Serial error: {self.reader.exception}")
                return
            if line:
                values = read_sensor_line(line)
                if values is not None:
                    tstamp = time.time() - start_time
                    row = ",".join([f"{tstamp:.3f}"] + [f"{v:.6f}" for v in values])
                    rows.append(row)
        # Write to file with header
        header = ["time"] + [f"sensor{i+1}" for i in range(9)]
        try:
            with output_path.open("w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for row in rows:
                    f.write(row + "\n")
            self.status_label.config(text=f"Saved {output_path.name}")
        except Exception as exc:  # noqa: BLE001
            self.status_label.config(text=f"Failed to save {output_path.name}: {exc}")
        # Move on
        self.root.after(1000, self.next_sample)

    def run(self) -> None:
        """Run the Tkinter main loop."""
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASL gesture data collection tool")
    parser.add_argument("--port", help="Serial port for ESP32 (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--subject", help="Identifier for the subject (e.g. participant name)")
    parser.add_argument("--repeats", type=int, default=2, help="Number of times to repeat the alphabet (default 2)")
    parser.add_argument("--session", default=None, help="Directory to store session data (default: choose via dialog)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate for serial connection (default 115200)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Validate subject ID
    if not args.subject:
        print("Error: --subject is required.")
        sys.exit(1)
    # Choose session directory if not provided
    if args.session:
        session_dir = Path(args.session)
    else:
        temp_root = tk.Tk()
        temp_root.withdraw()
        messagebox.showinfo(
            "Select Session Folder",
            "Choose a directory to save this data collection session.",
            parent=temp_root,
        )
        folder = filedialog.askdirectory(title="Select session folder", parent=temp_root)
        temp_root.destroy()

        if not folder:
            print("No folder selected. Exiting.")
            return

    session_dir = Path(folder)
    # Open serial port
    if serial is None:
        print("PySerial is not installed. Please install it with 'pip install pyserial'.")
        return
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open serial port {args.port}: {exc}")
        return
    # Show UI
    gui = DataCollectorGUI(ser, args.subject, session_dir, args.repeats)
    gui.run()
    # Close serial port on exit
    ser.close()


if __name__ == "__main__":
    main()