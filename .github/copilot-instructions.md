# Copilot Instructions for sign_language_glove

This repository is a small example project showing how to train and
run gesture‑recognition models for a sensor glove.  The code is meant
to be readable and extensible; it contains no complex build system or
heavy tooling.  The sections below describe the "big picture" and some
of the project‑specific conventions that will help an AI agent be
productive quickly.

---

## Project Overview 🔍

* **Purpose**: train classical (RandomForest) or deep (CNN‑LSTM/BiLSTM)
  models on time‑series data from a bend‑sensor glove, then convert the
deep models to TensorFlow Lite for deployment on microcontrollers.

* **Core scripts/modules**
  * `train_model.py` – command‑line entry point that loads datasets,
    prepares features or sequences, builds a model via `model_utils`,
    trains, evaluates and optionally saves weights.
  * `model_utils.py` – factory functions for constructing either a
    `RandomForestClassifier` or a small Keras sequential model with
    convolution + (Bi)LSTM layers.  Dependencies (scikit‑learn,
    TensorFlow) are optional and guarded by `try/except`.
  * `feature_extractor.py` – sliding‑window helper plus statistical
    feature computation used by the classical pipeline.
  * `inference_esp32.ino` – example Arduino sketch showing how to run
a converted `.tflite` model on an ESP32.  This file is mostly static;
update the `model_data` array after using `xxd` on a TFLite file.

* **Data flow**
  1. Raw text files in directories (`<sampleId>-<label>.txt`) are
     parsed by `load_dataset()` in `train_model.py`.
  2. For classical models, samples are segmented with
     `sliding_windows()` and summarized via `extract_features()`.
     Windows are averaged per sample to yield one feature vector.
  3. For deep models, windows become individual training sequences.
  4. `model_utils.get_model()` returns the requested model; training
     happens in `train_model.main()`.
  5. Trained weights may be saved (`.h5` for Keras, `.pkl` for
     scikit‑learn) or converted externally to TFLite/ESP32.

* **Why this structure?**
  The separation of feature extraction, model construction and the
  CLI script keeps the pipeline flexible: you can reuse the model
  factory with different data generators or swap the feature
  extraction logic for a more advanced one without touching the
  training script.

---

## Dataset & preprocessing conventions 🚩

* **Filenames** must be `<sampleId>-<label>.txt`; the label is the
  string after the last hyphen.  Non-`.txt` files are ignored.
* Files contain a fixed number of timesteps; each line has one value
  per sensor separated by spaces or commas.  `load_sample()` skips
  malformed lines silently.
* **Smoothing/filtering**: apply a moving‑average or low‑pass filter
  _before normalising_ the signals.  The README comments and module
  docstrings repeat this; normalising after smoothing can distort the
  calibration range.
* `window_size` and `step_size` are used consistently across the
  codebase for segmentation; the defaults are 32 and 16 respectively.

---

## Development workflows 🛠️

* **Training models** – run `python train_model.py` with the required
  dataset directories and desired flags.  Examples are in the README
  (re‑quoted below for quick reference):

  ```bash
  python3 train_model.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --test_dir /path/to/test \
    --model cnn_bilstm \
    --window_size 32 --step_size 16 \
    --save_weights /path/to/save/model.h5
  ```

* **Optional dependencies** are handled gracefully; imports in
  `train_model.py` and `model_utils.py` fall back to `None` if the
  package is missing.  Code that needs them checks and raises a clear
  `ImportError`.
* **Saving/loading weights** – use `--save_weights` and
  `--load_weights`.  Deep models call `model.save()`/`load_weights()`;
  classical models use `joblib.dump()` if available.
* **ESP32 inference** – after training, convert to `.tflite` (`tf.lite
  TFLiteConverter`) and run `xxd -i model.tflite > model_data.cc` to
  generate a C array.  Paste the bytes into `inference_esp32.ino` and
  flash the sketch; the example program reads a fixed window of
  sensor values and prints the predicted class number.

*No build system, no automated tests.*  Because of the small size of
this project, the primary "build" step is installing dependencies
(`pip install tensorflow scikit-learn numpy` etc.).  Linting/formatting
is not enforced but the code uses reasonably consistent style (PEP8,
Google-style docstrings).  When modifying or adding modules, mimic the
existing `try/except` import pattern and unit-test in a separate
environment if needed.

---

## Patterns & conventions 📦

* **Imports**: top-level modules import project files with plain
  `import feature_extractor` / `import model_utils`.  Avoid relative
  imports because the scripts are intended to be run from the project
  root.
* **Logging**: most modules obtain `logger = logging.getLogger(__name__)`
  and use `logger.info()` / `warning()`.  Do not use `print()` in
  production code; mimic existing logging calls for consistency.
* **Argument parsing**: `argparse` is used solely in `train_model.py`.
  When adding new CLI options, update the help text there; other
  modules are agnostic to command-line args.
* **Model sizing**: for deep models, `model_size` corresponds to both
  convolution filters and LSTM units; for random forests it maps to
  `n_estimators`.  This is enforced by the parameter names in the
  training script (e.g. `--conv_filters`, `--lstm_units`).
* **File structure**: three Python modules plus one Arduino sketch.
  No tests directory or packaging metadata exists.  If you add new
  functionality, keep it under the project root and update the
  README accordingly.

---

## What an AI agent should know 🧠

1. **Big picture**: small ML pipeline; nothing is hidden in opaque
   frameworks.  You can read every line of code to understand how
   samples are loaded, how features are computed, and how models are
   trained.
2. **How to extend**: to add a new model type, implement a build
   function in `model_utils.py` and tweak `get_model()`; to add new
   features, modify `feature_extractor.py` and the corresponding data
   preparation logic in `train_model.py`.
3. **Deployment target**: the only deployment example is an ESP32
   sketch; there is no CI, no Dockerfile.  Keep changes simple and
   self-contained.
4. **Non‑Python files**: the Arduino sketch is static.  Do not attempt
   to generate C code programmatically unless the task explicitly
   involves it.

---

## Updating these instructions

If you add files, workflows, or conventions not covered above, please
expand this document rather than leaving notes in the README.  An AI
reading this file should be able to start writing or modifying code
without digging through the entire Git history.

---

*Feel free to request clarification or additional sections.*  📝
