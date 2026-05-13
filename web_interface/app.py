"""
web_interface/app.py

This module implements a small FastAPI server that reads sensor data from
an ESP32‑based glove via a serial connection (or generates fake data
when running in simulation mode), normalises the readings using a
calibration file, maintains a rolling buffer of the most recent
measurements and uses a previously trained scikit‑learn model to make
gesture predictions.  The server exposes a WebSocket endpoint for
streaming live updates to a browser dashboard as well as a handful of
HTTP endpoints for serving the static frontend and basic health and
configuration information.

Key features:
 - Reads comma‑separated sensor rows of the form
   ``counter,sensor0,sensor1,sensor2,sensor3,sensor4``.  The
   first field is ignored; the five remaining values are raw
   ADC readings.
 - Supports optional calibration via a JSON file with per‑sensor
   ``min`` and ``max`` entries.  Without calibration the values are
   clamped to the 0–4065 range and scaled to 0–1.
 - Maintains a rolling buffer of ``WINDOW_SIZE`` samples for
   predictions and a longer history for plotting.
 - Extracts simple statistical features (mean and standard deviation
   for each sensor) from the most recent window and performs
   predictions using a loaded Random Forest model.  The saved model
   should be created by the training pipeline and must include a
   ``label_to_idx`` mapping to decode prediction indices back to
   human‑readable letters.
 - Uses WebSockets to push sensor data, buffer status and
   predictions to any connected frontend clients in real time.

Running the server:

    python app.py --port COM3 --baud 115200 --model models/random_forest.joblib --calibration calibration.json

    # Or in simulation mode
    python app.py --simulate

The simulation mode does not require a serial device and produces
smoothly varying mock sensor data; this is useful for testing the
frontend without the glove connected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover
    serial = None  # type: ignore


# -----------------------------------------------------------------------------
# Configuration constants
#
WINDOW_SIZE = 32  # number of samples required for a prediction
HISTORY_SIZE = 300  # length of history buffer for plotting
NUM_SENSORS = 5  # glove sends five sensor values per line
DEFAULT_ADC_MAX = 4065.0  # maximum ADC value for ESP32 S3


def compute_features(window: List[List[float]]) -> np.ndarray:
    """Compute simple statistical features from a sliding window.

    Each window is a list of length ``WINDOW_SIZE`` where each element
    contains ``NUM_SENSORS`` normalised values.  The feature vector
    concatenates the per‑sensor mean and standard deviation, producing
    a ``2 * NUM_SENSORS``‑dimensional array.
    """
    arr = np.asarray(window, dtype=np.float64)
    # Guard against empty input; return zeros if no data
    if arr.size == 0:
        return np.zeros(NUM_SENSORS * 2, dtype=np.float64)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    return np.concatenate([means, stds], dtype=np.float64)


def load_calibration(calibration_file: Optional[str]) -> Tuple[List[float], List[float]]:
    """Load per‑sensor calibration minima and maxima from a JSON file.

    The calibration file is expected to contain a mapping like::

        {
            "sensor_0": {"min": 123.0, "max": 2345.0, ...},
            "sensor_1": {"min": 100.0, "max": 2000.0, ...},
            ...
        }

    If calibration is not provided or a sensor entry is missing, the
    defaults (0.0 for min and ``DEFAULT_ADC_MAX`` for max) are used.
    """
    mins: List[float] = []
    maxs: List[float] = []
    if calibration_file:
        try:
            with open(calibration_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx in range(NUM_SENSORS):
                sensor_key = f"sensor_{idx}"
                entry = data.get(sensor_key, {})
                mins.append(float(entry.get("min", 0.0)))
                maxs.append(float(entry.get("max", DEFAULT_ADC_MAX)))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):  # pragma: no cover
            # Fallback to defaults if file is missing or malformed
            mins = [0.0 for _ in range(NUM_SENSORS)]
            maxs = [DEFAULT_ADC_MAX for _ in range(NUM_SENSORS)]
    else:
        # Default calibration: full ADC range
        mins = [0.0 for _ in range(NUM_SENSORS)]
        maxs = [DEFAULT_ADC_MAX for _ in range(NUM_SENSORS)]
    return mins, maxs


def normalise(values: List[float], min_vals: List[float], max_vals: List[float]) -> List[float]:
    """Normalise raw sensor readings to the [0, 1] range.

    For each sensor value, apply the transformation::

        (v - min_val) / (max_val - min_val)

    Values falling outside the calibrated range are clipped.  If a
    sensor's ``max_val`` equals or falls below its ``min_val`` the
    value is instead normalised using the default ADC range.
    """
    out = []
    for i, v in enumerate(values):
        v_float = float(v)
        min_v = min_vals[i]
        max_v = max_vals[i]
        if max_v <= min_v:
            # Degenerate calibration; fallback to raw scaling
            norm = v_float / DEFAULT_ADC_MAX
        else:
            norm = (v_float - min_v) / (max_v - min_v)
        # Clip to [0, 1]
        if norm < 0.0:
            norm = 0.0
        elif norm > 1.0:
            norm = 1.0
        out.append(norm)
    return out


class PredictionModel:
    """Wrapper around a scikit‑learn classifier loaded from joblib.

    Handles loading the model from disk, preparing an index‑to‑label
    mapping and performing predictions on feature vectors.
    """

    def __init__(self, model_path: Optional[str]):
        self.model = None  # underlying classifier
        self.idx_to_label: Dict[int, str] = {}
        if model_path and joblib:
            try:
                model_data = joblib.load(model_path)
                # The training pipeline saves a dict containing the model and label mapping
                if isinstance(model_data, dict):
                    self.model = model_data.get("model") or model_data.get("clf")
                    label_to_idx: Optional[Dict[str, int]] = model_data.get("label_to_idx")
                    if label_to_idx:
                        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
                else:
                    # The model itself may be saved directly
                    self.model = model_data
            except Exception:  # pragma: no cover
                # Leave the model as None on any loading error
                self.model = None

    def predict(self, window: List[List[float]]) -> Tuple[str, Dict[str, float], float]:
        """Predict the label and probabilities for the most recent window.

        Returns a tuple ``(label, probability_dict, confidence)``.  If no
        model is loaded, ``label`` will be ``"?"`` and the
        probability dictionary will be empty.  Confidence is the
        probability of the predicted label if available; otherwise 0.
        """
        if not self.model:
            return "?", {}, 0.0
        # Compute feature vector
        feats = compute_features(window)
        feats = feats.reshape(1, -1)
        try:
            # Use predict_proba if available
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(feats)[0]
                pred_idx = int(np.argmax(probas))
                confidence = float(probas[pred_idx])
                label = self.idx_to_label.get(pred_idx, str(pred_idx))
                prob_dict: Dict[str, float] = {
                    self.idx_to_label.get(i, str(i)): float(p)
                    for i, p in enumerate(probas)
                }
                return label, prob_dict, confidence
            # Fall back to predict if probability method is missing
            pred = self.model.predict(feats)[0]
            label = self.idx_to_label.get(int(pred), str(pred))
            return label, {label: 1.0}, 1.0
        except Exception:  # pragma: no cover
            return "?", {}, 0.0


def create_app(args: argparse.Namespace) -> FastAPI:
    """Application factory for the live sensor dashboard."""
    app = FastAPI()

    # Prepare calibration and model
    min_vals, max_vals = load_calibration(args.calibration)
    pred_model = PredictionModel(args.model)

    # Application state
    app.state.min_vals = min_vals
    app.state.max_vals = max_vals
    app.state.pred_model = pred_model
    app.state.rolling_buffer: List[List[float]] = []  # latest WINDOW_SIZE rows
    app.state.history: List[List[float]] = []  # longer plot history
    app.state.websockets: set[WebSocket] = set()
    app.state.running = True
    app.state.simulate = args.simulate

    # Serial connection handle stored in state
    app.state.serial = None
    # When not simulating, attempt to open serial port during startup

    @app.on_event("startup")
    async def on_startup() -> None:
        """FastAPI startup handler; set up serial and background reader."""
        # Only create serial port if not simulating
        if not app.state.simulate:
            if serial is None:
                print("Warning: pyserial not installed; running in simulation mode")
                app.state.simulate = True
            else:
                try:
                    # Open serial connection.  A timeout of 1 second avoids blocking
                    app.state.serial = serial.Serial(
                        args.port,
                        args.baud,
                        timeout=1,
                    )
                    # Give the MCU time to reset after opening the port
                    time.sleep(2.0)
                    # Flush any old data
                    if hasattr(app.state.serial, "reset_input_buffer"):
                        app.state.serial.reset_input_buffer()
                except Exception as e:  # pragma: no cover
                    print(f"Failed to open serial port {args.port}: {e}. Falling back to simulation.")
                    app.state.simulate = True
                    app.state.serial = None
        # Start the reader task
        loop = asyncio.get_event_loop()
        loop.create_task(serial_reader(app))

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        """FastAPI shutdown handler; cleanly stop the reader."""
        app.state.running = False
        if app.state.serial:
            try:
                app.state.serial.close()
            except Exception:  # pragma: no cover
                pass

    # HTTP route: serve the main dashboard page
    @app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        base_dir = Path(__file__).resolve().parent
        static_dir = base_dir / "static"
        index_path = static_dir / "index.html"
        return FileResponse(index_path)

    # HTTP route: serve static files (JS, CSS, images)
    @app.get("/static/{path:path}")
    async def static_files(path: str) -> FileResponse:
        base_dir = Path(__file__).resolve().parent
        file_path = (base_dir / "static" / path).resolve()
        return FileResponse(file_path)

    # Health endpoint: useful for monitoring
    @app.get("/health")
    async def health() -> JSONResponse:
        status = {
            "running": bool(app.state.running),
            "simulate": bool(app.state.simulate),
            "connected_clients": len(app.state.websockets),
        }
        return JSONResponse(content=status)

    # Configuration endpoint
    @app.get("/config")
    async def config() -> JSONResponse:
        conf = {
            "window_size": WINDOW_SIZE,
            "history_size": HISTORY_SIZE,
            "num_sensors": NUM_SENSORS,
            "model_loaded": bool(app.state.pred_model.model),
        }
        return JSONResponse(content=conf)

    # WebSocket endpoint for live streaming
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        # Add to the global set of active websockets
        app.state.websockets.add(ws)
        try:
            # Keep the connection open; we don't expect messages from client
            while True:
                try:
                    # Wait for incoming messages to detect client disconnect
                    await ws.receive_text()
                except WebSocketDisconnect:
                    # Client disconnected gracefully
                    break
                except Exception:
                    # Ignore other errors; the sender will close when send fails
                    await asyncio.sleep(0.1)
        finally:
            # Remove from the set on disconnect
            app.state.websockets.discard(ws)

    return app


async def serial_reader(app: FastAPI) -> None:
    """Background task that reads from serial (or generates fake data) and
    dispatches updates to all connected WebSocket clients.

    This coroutine runs until ``app.state.running`` becomes ``False``.
    """
    loop = asyncio.get_event_loop()
    # Keep track of a phase offset for generating smooth simulated data
    phase = 0.0
    while app.state.running:
        # Acquire a new set of values from either serial or simulation
        if app.state.simulate:
            # Generate fake sensor values that vary slowly between 0 and 1
            phase += 0.05
            raw_values = [
                (np.sin(phase + i) + 1.0) / 2.0 * DEFAULT_ADC_MAX
                for i in range(NUM_SENSORS)
            ]
            timestamp = time.time()
        else:
            ser = app.state.serial
            if ser is None:
                # Serial unexpectedly unavailable; wait and retry
                await asyncio.sleep(0.1)
                continue
            # Read a line from the serial port in a thread to avoid blocking the event loop
            try:
                line_bytes = await loop.run_in_executor(None, ser.readline)
            except Exception:
                # On any serial read error, sleep briefly and continue
                await asyncio.sleep(0.1)
                continue
            if not line_bytes:
                # Timeout or no data
                await asyncio.sleep(0.01)
                continue
            # Decode and strip whitespace
            try:
                line_str = line_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            # Split by comma and parse floats
            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) != 1 + NUM_SENSORS:
                # Unexpected format; skip
                continue
            try:
                # Skip the first value (counter) and parse the rest
                raw_values = [float(x) for x in parts[1:]]
                timestamp = time.time()
            except ValueError:
                # Non‑numeric line
                continue
        # Normalise values using calibration
        norm_values = normalise(raw_values, app.state.min_vals, app.state.max_vals)
        # Append to rolling buffer and history
        rb = app.state.rolling_buffer
        hist = app.state.history
        rb.append(norm_values)
        if len(rb) > WINDOW_SIZE:
            rb.pop(0)
        hist.append(norm_values)
        if len(hist) > HISTORY_SIZE:
            hist.pop(0)

        # Determine readiness and perform prediction if enough data
        if len(rb) == WINDOW_SIZE and app.state.pred_model.model:
            ready = True
            label, prob_dict, conf = app.state.pred_model.predict(rb)
        else:
            ready = False
            label, prob_dict, conf = "", {}, 0.0

        # Build payload
        payload: Dict[str, object] = {
            "timestamp": timestamp,
            "sensors": norm_values,
            "ready": ready,
        }
        if ready:
            payload.update(
                {
                    "prediction": label,
                    "confidence": conf,
                    "probabilities": prob_dict,
                }
            )
        else:
            payload.update(
                {
                    "samples_collected": len(rb),
                    "samples_needed": WINDOW_SIZE,
                }
            )
        # Dispatch payload to all connected websockets
        to_remove: List[WebSocket] = []
        if app.state.websockets:
            for ws in list(app.state.websockets):
                try:
                    await ws.send_json(payload)
                except Exception:
                    # Mark for removal on send failure
                    to_remove.append(ws)
            # Remove any closed websockets
            for ws in to_remove:
                app.state.websockets.discard(ws)
        # Wait briefly before reading the next sample; adjust to achieve ~20 Hz
        await asyncio.sleep(0.05)


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the dashboard server."""
    parser = argparse.ArgumentParser(description="Run the sign language glove dashboard server")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for the glove (e.g. COM3 or /dev/ttyACM0). Required unless --simulate is set.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate for the serial connection. Ignored when --simulate is set.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a joblib file containing the trained model and label mapping.",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to a JSON calibration file from calibrate_glove.py.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode without a serial device.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the web server.",
    )
    parser.add_argument(
        "--port-web",
        type=int,
        default=8000,
        help="TCP port for the web server.",
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    # When not simulating ensure that a serial port is provided
    if not args.simulate and not args.port:
        raise SystemExit("Error: --port is required unless --simulate is specified")
    # Create and run the app
    application = create_app(args)
    import uvicorn  # type: ignore

    uvicorn.run(application, host=args.host, port=args.port_web, log_level="info")