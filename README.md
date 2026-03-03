# Dependencies

Tensorflow

# Sign Language Glove Gesture Recognition

This repository contains example code for building machine‑learning models that classify hand gestures captured by a sensor glove.  The glove uses resistive yarn sensors sewn into the fingers to measure bend, producing a time‑series of values for each sensor.  The models in this project convert sequences of normalised sensor readings into gesture labels and can be trained on user‑supplied datasets.

## Project structure

```
sign_language_glove/
├── README.md              – project description, dataset format and usage instructions
├── train_model.py         – command‑line script for training models
├── model_utils.py         – helper functions to build deep and classical models
├── feature_extractor.py   – functions for computing statistical features from raw signals
└── inference_esp32.ino    – example Arduino sketch for running a trained TensorFlow Lite model on an ESP32
```

### Dataset format

* **Filename** – each sample file is named `<sampleId>-<label>.txt`, where `sampleId` is an identifier and `label` is the gesture label (e.g. `001‑hello.txt`).
* **Data structure** – each file contains a fixed number of time steps.  Every time step holds an array of normalised sensor readings (one value per sensor).  For example, a file with 5 sensors and 100 time steps looks like:

  ```
  0.02 0.10 0.45 0.33 0.80
  0.03 0.09 0.46 0.32 0.81
  …
  0.01 0.12 0.44 0.35 0.79
  ```

  Each line corresponds to one time step.  The glove calibration process should normalise sensor values into a common range.  Values outside the physical min/max range should be clipped before saving.

### Pre‑processing recommendations

Prior work on glove‑based sign recognition often applies a mean or moving average filter to reduce high‑frequency noise before normalising the signals【941938726998831†L750-L768】.  Because your normalisation relies on physically calibrated min/max values, it is best to apply any additional moving average *before* normalisation.  Filtering the raw sensor signal suppresses sensor noise without altering the calibrated range; normalisation then maps the smoothed signal into a consistent range.  Applying a moving average after normalisation could distort the calibrated boundaries and lead to clipping.

### Training script

Use `train_model.py` to train either a classical machine‑learning model (Random Forest) or a deep neural network (CNN‑LSTM or CNN‑BiLSTM).  The script reads sample files from the specified directories, extracts features when necessary, builds the chosen model, trains it, evaluates it on a validation set and reports accuracy and a confusion matrix.

Example usage:

```bash
# Train a CNN‑BiLSTM model
python3 train_model.py \
  --train_dir /path/to/train_samples \
  --val_dir /path/to/validation_samples \
  --test_dir /path/to/test_samples \
  --model_type cnn_bilstm \
  --model_size 64 \
  --save_weights /path/to/save/my_model.h5

# Train a Random Forest on statistical features
python3 train_model.py \
  --train_dir /path/to/train_samples \
  --val_dir /path/to/validation_samples \
  --test_dir /path/to/test_samples \
  --model_type random_forest \
  --model_size 200 \
  --save_weights /path/to/save/rf_model.pkl
```

Arguments:

* `--train_dir`, `--val_dir`, `--test_dir` – directories containing sample files for training, validation and testing.  Each file must be named `<sampleId>-<label>.txt` and contain a matrix of sensor readings as described above.
* `--model_type` – one of `random_forest`, `cnn_lstm`, or `cnn_bilstm`.
* `--model_size` – for the deep models this is the number of convolution filters and recurrent units; for the Random Forest it is the number of trees.
* `--save_weights` – path to save the trained model weights (e.g. `.h5` for Keras models or `.pkl` for scikit‑learn).  If not provided the weights are not saved.
* `--load_weights` – optional path to an existing weight file to initialise the model before training.  Useful for fine‑tuning.

### Running on ESP32

The file `inference_esp32.ino` shows how to run inference on an ESP32 using TensorFlow Lite for Microcontrollers.  After training a CNN‑LSTM/BiLSTM model in Python, convert it to TensorFlow Lite (`.tflite`) and then to a C array using the `xxd` tool.  Update the `model_data` array in `inference_esp32.ino` with your model bytes.  The sketch reads a sequence of normalised sensor values, feeds them to the model and outputs the predicted label.

This project is provided as a starting point.  Feel free to extend it by adding more sophisticated feature extraction, hyperparameter tuning, or support for additional microcontrollers.
