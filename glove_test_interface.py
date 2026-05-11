"""
Glove Test Interface
====================

A real-time visualization and testing tool for the ASL sensor glove.
Displays all 9 sensor readings, live-updating timelines, statistics,
and other useful debugging information during glove testing.

Features
--------
* Real-time 9-sensor value display with numerical readouts
* Live-updating timeline chart showing sensor history (last 3 seconds)
* Live statistics: min, max, mean, std dev for each sensor
* Sensor activity indicator (highlights changing sensors)
* Optional CSV data export for captured sessions
* Configurable sampling rate and history window
* Connection status and data quality metrics
* Simple start/stop controls

Usage
-----
1. Connect the glove to an ESP32 via serial port.

2. Run the script:

   .. code:: bash

      python glove_test_interface.py --port COM3 --baud 115200

3. Observe sensor values updating in real-time. The timeline chart
   refreshes every 100ms and displays the last 3 seconds of data.

4. Click "Export Data" to save the session to a CSV file.

Notes
-----
* The script reads 10 comma-separated values per line
  (counter + 9 sensors). Adjust read_sensor_line() if your format differs.
* History is kept in a rolling buffer of the last 3 seconds (~90 samples
  at 30Hz). Older samples are discarded automatically.
* Statistics update in real-time as new data arrives.
"""

import argparse
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from dataclasses import dataclass

import tkinter as tk
from tkinter import messagebox, filedialog

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None  # type: ignore


@dataclass
class SensorReading:
    """A single timestamped sensor reading."""

    timestamp: float
    values: list[float]


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

    def get_line(self, timeout: float = 0.05) -> str | None:
        """Return the latest line read from the serial port."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None


def read_sensor_line(line: str) -> list[float] | None:
    """Parse a line of sensor data into a list of 9 floats.

    Expected format: 10 comma-separated values (counter + 9 sensors).
    Returns only the 9 sensor values, ignoring the counter.
    Returns None if the line is invalid.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 10:
        return None
    try:
        values = [float(p) for p in parts[1:]]
        return values
    except ValueError:
        return None


def calculate_stats(values: list[float]) -> dict:
    """Calculate min, max, mean, and std dev for a list of values."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_val = variance**0.5
    return {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}


class GloveTestInterface:
    """Real-time glove testing and visualization interface."""

    def __init__(self, ser: serial.Serial, history_seconds: float = 3.0):
        self.ser = ser
        self.history_seconds = history_seconds
        self.history: deque = deque(maxlen=int(30 * history_seconds))
        self.start_time = time.time()

        self.root = tk.Tk()
        self.root.title("Glove Test Interface")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        self.is_running = False
        self.exported_data: list[SensorReading] = []

        self._setup_ui()

        self.reader = SensorReader(ser)
        self.reader.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._update_loop)

    def _setup_ui(self) -> None:
        """Build the Tkinter UI layout."""

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

        self.sensor_names = [
            "Index Palm",
            "Middle Palm",
            "Nameless Palm",
            "Pinky Palm",
            "Pinky Finger",
            "Nameless Finger",
            "Index Finger",
            "Middle Finger",
            "Thumb"
        ]

        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(title_frame, text="Glove Test Interface", font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        self.status_label = tk.Label(title_frame, text="Ready", fg="green", font=("Arial", 10))
        self.status_label.pack(side=tk.RIGHT)

        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        self.start_button = tk.Button(button_frame, text="Start Recording", command=self._toggle_recording, bg="#4CAF50", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Export Data", command=self._export_data, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear History", command=self._clear_history, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=5)

        # Main content: left (values) and right (chart)
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel: sensor values and stats
        left_frame = tk.Frame(content_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        tk.Label(left_frame, text="Sensor Values", font=("Arial", 12, "bold")).pack()
        
        self.sensor_frames = []
        self.sensor_value_labels = []
        self.sensor_bars = []
        
        for i in range(9):
            sensor_frame = tk.Frame(left_frame, bg="#f0f0f0", relief=tk.SUNKEN, bd=1)
            sensor_frame.pack(fill=tk.X, pady=3)
            self.sensor_frames.append(sensor_frame)

            sensor_label = tk.Label(sensor_frame, text=self.sensor_names[i], font=("Arial", 10, "bold"), fg="#333", bg="#f0f0f0")
            sensor_label.pack(anchor=tk.W, padx=5, pady=2)

            value_label = tk.Label(sensor_frame, text="0.000", font=("Arial", 14, "bold"), fg="#1f77b4", bg="#f0f0f0")
            value_label.pack(anchor=tk.W, padx=5)
            self.sensor_value_labels.append(value_label)

            canvas = tk.Canvas(sensor_frame, width=300, height=20, bg="#e8e8e8", highlightthickness=0)
            canvas.pack(fill=tk.X, padx=5, pady=2)
            self.sensor_bars.append(canvas)

        # Stats section
        tk.Label(left_frame, text="Statistics (Last 3s)", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.stats_text = tk.Text(left_frame, height=15, width=40, font=("Courier", 8), state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Right panel: timeline chart
        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="Timeline (Last 3 seconds)", font=("Arial", 12, "bold")).pack()
        self.chart_canvas = tk.Canvas(right_frame, bg="white", highlightthickness=1, highlightbackground="#ccc", height=500)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)

        # Metadata at bottom
        meta_frame = tk.Frame(self.root, bg="#f9f9f9")
        meta_frame.pack(fill=tk.X, padx=10, pady=5)
        self.meta_label = tk.Label(meta_frame, text="Samples: 0 | Time: 0.0s | Quality: OK", font=("Arial", 9), bg="#f9f9f9")
        self.meta_label.pack(anchor=tk.W)

    def _toggle_recording(self) -> None:
        """Start or stop recording data."""
        self.is_running = not self.is_running
        if self.is_running:
            self.exported_data = []
            self.start_button.config(text="Stop Recording", bg="#f44336")
            self.status_label.config(text="Recording...", fg="orange")
        else:
            self.start_button.config(text="Start Recording", bg="#4CAF50")
            self.status_label.config(text="Stopped", fg="red")

    def _export_data(self) -> None:
        """Save recorded data to a CSV file."""
        if not self.exported_data:
            messagebox.showwarning("No Data", "No data to export. Start recording first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"glove_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                header = ["time"] + [f"sensor{i+1}" for i in range(9)]
                f.write(",".join(header) + "\n")
                for reading in self.exported_data:
                    row = [f"{reading.timestamp:.3f}"] + [f"{v:.6f}" for v in reading.values]
                    f.write(",".join(row) + "\n")
            messagebox.showinfo("Success", f"Data exported to {file_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to export data: {exc}")

    def _clear_history(self) -> None:
        """Clear the sensor history buffer."""
        self.history.clear()
        self.exported_data = []
        self.start_time = time.time()

    def _update_loop(self) -> None:
        """Main update loop: read sensor data and refresh UI."""
        # Collect all available sensor readings
        for _ in range(10):
            line = self.reader.get_line(timeout=0.01)
            if self.reader.exception is not None:
                self.status_label.config(text=f"Serial Error: {self.reader.exception}", fg="red")
                break
            if line:
                values = read_sensor_line(line)
                if values is not None:
                    elapsed = time.time() - self.start_time
                    reading = SensorReading(timestamp=elapsed, values=values)
                    self.history.append(reading)
                    if self.is_running:
                        self.exported_data.append(reading)

        # Update UI
        if self.history:
            self._update_sensor_displays()
            self._update_stats()
            self._draw_timeline()
            self._update_metadata()

        self.root.after(100, self._update_loop)

    def _update_sensor_displays(self) -> None:
        """Update the sensor value labels and bar graphs."""
        if not self.history:
            return

        latest = self.history[-1]
        for i, value in enumerate(latest.values):
            # Update value label
            self.sensor_value_labels[i].config(text=f"{value:.3f}")

            # Update bar graph
            canvas = self.sensor_bars[i]
            canvas.delete("all")
            bar_width = int((value / 4065) * 280)
            canvas.create_rectangle(0, 2, bar_width, 18, fill=self.sensor_colors[i], outline=self.sensor_colors[i])
            canvas.create_rectangle(0, 2, 280, 18, outline="#ccc", width=1)
            canvas.create_text(285, 10, text=f"{value:.0f}", font=("Courier", 8), anchor=tk.W)

    def _update_stats(self) -> None:
        """Calculate and display statistics for each sensor over the history window."""
        if not self.history:
            return

        # Transpose history to get per-sensor columns
        sensor_columns = [[] for _ in range(9)]
        for reading in self.history:
            for i, val in enumerate(reading.values):
                sensor_columns[i].append(val)

        stats_lines = []
        for i, col in enumerate(sensor_columns):
            stats = calculate_stats(col)
            stats_lines.append(f"{self.sensor_names[i]:15s}: min={stats['min']:.0f} max={stats['max']:.0f} mean={stats['mean']:.0f} std={stats['std']:.0f}")

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", "\n".join(stats_lines))
        self.stats_text.config(state=tk.DISABLED)

    def _draw_timeline(self) -> None:
        """Draw the live-updating timeline chart."""
        canvas = self.chart_canvas
        width = int(canvas.winfo_width())
        height = int(canvas.winfo_height())

        if width < 10 or height < 10:
            return

        canvas.delete("all")

        if not self.history:
            canvas.create_text(width // 2, height // 2, text="Waiting for data...", fill="gray")
            return

        margin = 40
        plot_width = width - margin * 2
        plot_height = height - margin * 2

        # Draw axes and grid
        canvas.create_rectangle(margin, margin, width - margin, height - margin, outline="#444", width=2)
        canvas.create_line(margin, margin + plot_height // 2, width - margin, margin + plot_height // 2, fill="#eee", dash=(4, 4))
        canvas.create_text(margin - 10, margin, text="4065", anchor="e", fill="#666", font=("Arial", 8))
        canvas.create_text(margin - 10, height - margin, text="0", anchor="e", fill="#666", font=("Arial", 8))

        # Time axis labels
        if self.history:
            time_range = self.history[-1].timestamp - self.history[0].timestamp
            if time_range > 0:
                x_scale = plot_width / time_range
                canvas.create_text(width - margin, height - 10, text=f"{self.history[-1].timestamp:.1f}s", anchor="e", fill="#666", font=("Arial", 8))
                canvas.create_text(margin, height - 10, text=f"{self.history[0].timestamp:.1f}s", anchor="w", fill="#666", font=("Arial", 8))

                # Draw sensor lines
                for sensor_idx in range(9):
                    points = []
                    for reading in self.history:
                        x = margin + (reading.timestamp - self.history[0].timestamp) * x_scale
                        y = height - margin - (reading.values[sensor_idx] / 4065) * plot_height
                        points.extend((x, y))

                    if len(points) >= 4:
                        canvas.create_line(*points, fill=self.sensor_colors[sensor_idx], width=1.5)

        # Labels
        canvas.create_text(width // 2, margin - 15, text="Raw Sensor Values (0-4065) vs Time", font=("Arial", 10, "bold"))
        canvas.create_text(25, margin + 10, text="Value", anchor="w", font=("Arial", 8), angle=90)
        canvas.create_text(width // 2, height - 5, text="Time (seconds)", font=("Arial", 8), anchor="n")

    def _update_metadata(self) -> None:
        """Update sample count and quality metadata."""
        if not self.history:
            return

        num_samples = len(self.history)
        elapsed_time = self.history[-1].timestamp if self.history else 0.0
        quality = "OK"

        # Simple quality heuristic
        if num_samples < 20:
            quality = "Low"
        elif num_samples > 100:
            quality = "Good"

        meta_text = f"Samples: {num_samples} | Time: {elapsed_time:.1f}s | Quality: {quality}"
        if self.is_running:
            meta_text += f" | Recording: {len(self.exported_data)} samples"

        self.meta_label.config(text=meta_text)

    def on_close(self) -> None:
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.reader.stop()
            self.root.destroy()

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time glove testing interface")
    parser.add_argument("--port", default="COM3", help="Serial port for ESP32 (default: COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--history", type=float, default=3.0, help="History window in seconds (default: 3.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if serial is None:
        print("Error: PySerial is not installed. Install it with: pip install pyserial")
        sys.exit(1)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Connected to {args.port} at {args.baud} baud")
    except serial.SerialException as exc:
        print(f"Error: Could not open serial port {args.port}: {exc}")
        sys.exit(1)

    gui = GloveTestInterface(ser, history_seconds=args.history)
    gui.run()
    ser.close()


if __name__ == "__main__":
    main()
