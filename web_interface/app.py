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
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the project root is importable when running from the web_interface folder
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import feature_extractor
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
DEFAULT_WINDOW_SIZE = 32  # default number of samples required for a prediction
HISTORY_SIZE = 150  # length of history buffer for plotting
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


def get_default_classes() -> List[str]:
    """Return a default list of gesture labels.

    If the project provides a `gesture_subset_abcd_y` module, use
    its `get_reduced_gesture_set()` function.  Otherwise fall back
    to a hardcoded list of common ASL letters.  This helper allows
    CNN‑based models to map prediction indices back to letters when
    the label mapping is not embedded in the model file.
    """
    try:
        from gesture_subset_abcd_y import get_reduced_gesture_set  # type: ignore

        classes = get_reduced_gesture_set()
        if isinstance(classes, list) and classes:
            return [str(c) for c in classes]
    except Exception:
        pass
    # Fallback to a minimal set of letters commonly used in the project
    return ["A", "B", "C", "D", "Y"]


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
    """Wrapper around classification models for the sign language glove.

    This class supports both classical scikit‑learn models (Random Forest)
    loaded via joblib and deep learning models (CNN‑LSTM or
    CNN‑BiLSTM) loaded via TensorFlow/Keras.  A ``model_type``
    parameter determines which prediction pathway to use.  For
    classical models, simple statistical features are computed from
    the window before inference; for CNN‑based models, the raw window
    of shape (WINDOW_SIZE, NUM_SENSORS) is passed directly to
    ``model.predict()``.  A list of class labels must be provided
    (either via the saved ``label_to_idx`` mapping in a joblib file
    or explicitly) to decode indices back to gesture letters.
    """

    def __init__(
        self,
        model_path: Optional[str],
        model_type: str = "rf",
        classes: Optional[List[str]] = None,
    ) -> None:
        self.model_type = model_type.lower()
        self.model = None  # underlying classifier or Keras model
        # Determine class labels
        if classes is None:
            classes = get_default_classes()
        self.idx_to_label: Dict[int, str] = {idx: label for idx, label in enumerate(classes)}
        # Load model depending on type
        if not model_path:
            return
        if self.model_type == "rf":
            # Load scikit‑learn model saved via joblib
            if joblib is None:
                return
            try:
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get("model") or model_data.get("clf")
                    label_to_idx: Optional[Dict[str, int]] = model_data.get("label_to_idx")
                    if label_to_idx:
                        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
                else:
                    self.model = model_data
            except Exception:
                self.model = None
        elif self.model_type in ("cnn_lstm", "cnn_bilstm", "cnn", "keras"):
            # Attempt to load a Keras model
            try:
                # Delay import of tensorflow to avoid unnecessary dependency if unused
                import tensorflow as tf  # type: ignore

                self.model = tf.keras.models.load_model(model_path)
            except Exception:
                self.model = None
        else:
            # Unknown model type
            self.model = None

    def predict(self, window: List[List[float]]) -> Tuple[str, Dict[str, float], float]:
        """Predict the label and probabilities for the most recent window.

        Returns ``(label, probability_dict, confidence)``.  If no model is
        loaded or inference fails, returns a placeholder label "?" and
        empty probability dictionary with zero confidence.
        """
        if not self.model:
            return "?", {}, 0.0
        try:
            if self.model_type == "rf":
                # Use the project's feature extractor for Random Forest inputs.
                arr = np.asarray(window, dtype=np.float32)
                feats = feature_extractor.extract_features(arr).reshape(1, -1)
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
                # Fall back to predict() if no probabilities available
                pred = self.model.predict(feats)[0]
                label = self.idx_to_label.get(int(pred), str(pred))
                return label, {label: 1.0}, 1.0
            else:
                # CNN‑based models expect a 3D tensor (batch, timesteps, sensors)
                arr = np.asarray(window, dtype=np.float32)
                arr = arr.reshape((1, arr.shape[0], arr.shape[1]))
                # Predict probabilities; suppress verbose logging
                probas = self.model.predict(arr, verbose=0)
                # Some models return nested arrays
                probas = np.asarray(probas).reshape(-1)
                pred_idx = int(np.argmax(probas))
                confidence = float(probas[pred_idx])
                label = self.idx_to_label.get(pred_idx, str(pred_idx))
                prob_dict = {
                    self.idx_to_label.get(i, str(i)): float(p)
                    for i, p in enumerate(probas)
                }
                return label, prob_dict, confidence
        except Exception:
            return "?", {}, 0.0


def create_app(args: argparse.Namespace) -> FastAPI:
    """Application factory for the live sensor dashboard."""
    app = FastAPI()

    # Prepare calibration and model
    min_vals, max_vals = load_calibration(args.calibration)
    # Determine class labels from --classes argument (comma separated) if provided
    if args.classes:
        class_list = [s.strip() for s in args.classes.split(",") if s.strip()]
    else:
        class_list = None
    pred_model = PredictionModel(args.model, model_type=args.model_type, classes=class_list)

    # Application state
    app.state.min_vals = min_vals
    app.state.max_vals = max_vals
    app.state.pred_model = pred_model
    app.state.rolling_buffer: List[List[float]] = []  # latest window_size rows
    app.state.history: List[List[float]] = []  # longer plot history
    app.state.websockets: set[WebSocket] = set()
    app.state.running = True
    app.state.simulate = args.simulate
    app.state.window_size = args.window_size

    # TCP server configuration; if provided, the server will accept
    # incoming connections from the glove over WiFi.  The
    # network_server attribute holds the asyncio.Server instance and
    # client_tasks stores tasks spawned for each connected client.
    app.state.tcp_host = args.tcp_host or "0.0.0.0"
    app.state.tcp_port = args.tcp_port
    app.state.network_server = None
    app.state.tcp_server_running = False
    app.state.glove_connected = False
    app.state.valid_tcp_lines_received = 0
    app.state.last_tcp_line_time: Optional[float] = None
    app.state.first_valid_tcp_line_logged = False
    app.state.client_tasks: set[asyncio.Task] = set()

    # Serial connection handle stored in state
    app.state.serial = None
    # When not simulating, attempt to open serial port during startup

    @app.on_event("startup")
    async def on_startup() -> None:
        """FastAPI startup handler; set up input sources."""
        loop = asyncio.get_event_loop()
    
        # Only open USB serial if --port was explicitly provided.
        # In WiFi/TCP mode, missing serial must NOT enable simulation.
        if not app.state.simulate and args.port:
            if serial is None:
                print("Warning: pyserial not installed; serial input disabled")
                app.state.serial = None
            else:
                try:
                    app.state.serial = serial.Serial(
                        args.port,
                        args.baud,
                        timeout=1,
                    )
                    time.sleep(2.0)
                    if hasattr(app.state.serial, "reset_input_buffer"):
                        app.state.serial.reset_input_buffer()
                    app.state.glove_connected = True
                    print(f"Serial input enabled on {args.port}")
                except Exception as e:
                    print(f"Failed to open serial port {args.port}: {e}. Serial input disabled.")
                    app.state.serial = None

        # Start serial/simulation reader only when actually needed.
        if app.state.simulate or app.state.serial:
            loop.create_task(serial_reader(app))

        # Start TCP server if configured.
        if app.state.tcp_port is not None:
            try:
                server = await asyncio.start_server(
                    lambda r, w: handle_client(app, r, w),
                    host=app.state.tcp_host,
                    port=app.state.tcp_port,
                )
                app.state.network_server = server
                app.state.tcp_server_running = True
                loop.create_task(server.serve_forever())
                # Print a tuple-style address for consistency with expectations.
                addr = server.sockets[0].getsockname() if server.sockets else (app.state.tcp_host, app.state.tcp_port)
                print(f"Listening for glove TCP connections on {addr}")
            except Exception as e:
                app.state.tcp_server_running = False
                print(f"Failed to start TCP server on {app.state.tcp_host}:{app.state.tcp_port}: {e}")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        """FastAPI shutdown handler; cleanly stop the reader."""
        app.state.running = False
        if app.state.serial:
            try:
                app.state.serial.close()
            except Exception:  # pragma: no cover
                pass
        # Shut down the TCP server and any client reader tasks
        if app.state.network_server:
            try:
                app.state.network_server.close()
                await app.state.network_server.wait_closed()
            except Exception:
                pass
        # Cancel any client tasks
        for task in list(app.state.client_tasks):
            task.cancel()

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
            "tcp_host": app.state.tcp_host,
            "tcp_port": app.state.tcp_port,
            "tcp_server_running": bool(app.state.tcp_server_running),
            "glove_connected": bool(app.state.glove_connected),
            "valid_tcp_lines_received": int(app.state.valid_tcp_lines_received),
            "last_tcp_line_time": app.state.last_tcp_line_time,
            "model_loaded": bool(app.state.pred_model.model),
            "window_size": int(app.state.window_size),
        }
        return JSONResponse(content=status)

    # Configuration endpoint
    @app.get("/config")
    async def config() -> JSONResponse:
        conf = {
            "window_size": int(app.state.window_size),
            "history_size": HISTORY_SIZE,
            "num_sensors": NUM_SENSORS,
            "model_loaded": bool(app.state.pred_model.model),
            "tcp_host": app.state.tcp_host,
            "tcp_port": app.state.tcp_port,
            "tcp_server_running": bool(app.state.tcp_server_running),
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
        if len(rb) > app.state.window_size:
            rb.pop(0)
        hist.append(norm_values)
        if len(hist) > HISTORY_SIZE:
            hist.pop(0)

        # Determine readiness and perform prediction if enough data
        if len(rb) == app.state.window_size and app.state.pred_model.model:
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
            "source": "serial" if not app.state.simulate else "simulate",
            "simulate": app.state.simulate,
            "model_loaded": bool(app.state.pred_model.model),
            "window_size": int(app.state.window_size),
            "glove_connected": app.state.glove_connected,
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
                    "samples_needed": app.state.window_size,
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


async def handle_client(app: FastAPI, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a single TCP client connection from the glove.

    This coroutine reads lines from the network client, parses raw
    sensor values, normalises them, updates the application buffers,
    performs predictions when the rolling buffer is full and
    broadcasts updates to connected websockets.  It mirrors the
    functionality of ``serial_reader`` for TCP connections.
    """
    # Register this task so it can be cancelled on shutdown
    current_task = asyncio.current_task()
    if current_task is not None:
        app.state.client_tasks.add(current_task)
    addr = None
    try:
        addr = writer.get_extra_info("peername")
        app.state.glove_connected = True
        print(f"Glove connected from {addr}")
        while app.state.running:
            try:
                line_bytes = await reader.readline()
            except Exception:
                break
            if not line_bytes:
                # Client closed connection
                break
            try:
                line_str = line_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) != 1 + NUM_SENSORS:
                print(f"Malformed TCP line skipped: {line_str}")
                continue
            try:
                raw_values = [float(x) for x in parts[1:]]
                timestamp = time.time()
            except ValueError:
                print(f"Malformed TCP line skipped: {line_str}")
                continue
            if app.state.valid_tcp_lines_received == 0 and not app.state.first_valid_tcp_line_logged:
                print(f"First valid TCP line received: {line_str}")
                app.state.first_valid_tcp_line_logged = True
            app.state.valid_tcp_lines_received += 1
            app.state.last_tcp_line_time = timestamp
            # Normalise
            norm_values = normalise(raw_values, app.state.min_vals, app.state.max_vals)
            # Update buffers
            rb: List[List[float]] = app.state.rolling_buffer
            hist: List[List[float]] = app.state.history
            rb.append(norm_values)
            if len(rb) > app.state.window_size:
                rb.pop(0)
            hist.append(norm_values)
            if len(hist) > HISTORY_SIZE:
                hist.pop(0)
            # Prediction
            if len(rb) == app.state.window_size and app.state.pred_model.model:
                ready = True
                label, prob_dict, conf = app.state.pred_model.predict(rb)
            else:
                ready = False
                label, prob_dict, conf = "", {}, 0.0
            payload: Dict[str, object] = {
                "timestamp": timestamp,
                "sensors": norm_values,
                "ready": ready,
                "source": "tcp",
                "simulate": app.state.simulate,
                "model_loaded": bool(app.state.pred_model.model),
                "window_size": int(app.state.window_size),
                "glove_connected": app.state.glove_connected,
            }
            if ready:
                payload.update({"prediction": label, "confidence": conf, "probabilities": prob_dict})
            else:
                payload.update({"samples_collected": len(rb), "samples_needed": app.state.window_size})
            # Broadcast to websockets
            to_remove: List[WebSocket] = []
            for ws in list(app.state.websockets):
                try:
                    await ws.send_json(payload)
                except Exception:
                    to_remove.append(ws)
            for ws in to_remove:
                app.state.websockets.discard(ws)
        # End of while
    except asyncio.CancelledError:
        # Task cancelled; exit gracefully
        pass
    finally:
        if addr:
            print(f"Glove disconnected from {addr}")
        app.state.glove_connected = False
        # Clean up writer
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        # Remove from client_tasks
        if current_task is not None:
            app.state.client_tasks.discard(current_task)


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
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of samples per prediction window. Defaults to 32.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode without a serial device.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rf",
        choices=["rf", "cnn_lstm", "cnn_bilstm"],
        help="Type of model to load: 'rf' for Random Forest, 'cnn_lstm' or 'cnn_bilstm' for deep learning models.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of gesture labels corresponding to model output indices. Overrides defaults.",
    )
    parser.add_argument(
        "--tcp-host",
        type=str,
        default=None,
        help="Host interface for the TCP server to listen on (for WiFi glove). Default is '0.0.0.0' when --tcp-port is provided.",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=None,
        help="TCP port to listen for glove connections over WiFi. If provided, the server will accept sensor data via TCP instead of serial.",
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
    # When not simulating and not using TCP, ensure that a serial port is provided
    if not args.simulate and args.tcp_port is None and not args.port:
        raise SystemExit(
            "Error: --port is required unless --simulate or --tcp-port is specified"
        )
    # Create and run the app
    application = create_app(args)
    import uvicorn  # type: ignore

    uvicorn.run(application, host=args.host, port=args.port_web, log_level="info")