/*
 * Inference sketch for ESP32 using TensorFlow Lite Micro.
 *
 * This code demonstrates how to load a compiled gesture recognition
 * model (exported as a TensorFlow Lite flatbuffer) onto an ESP32
 * microcontroller and perform real‑time inference. The glove's
 * resistive yarn sensors should be read via analog inputs and the
 * readings normalised in the same way as during training. A sliding
 * window of recent readings is fed into the model to predict the
 * current gesture.
 *
 * To generate the `model_data.h` header, convert your trained Keras
 * model to a TFLite file and then use the `xxd` utility or
 * `tensorflow/lite/micro/tools/make/download_extra_files` scripts to
 * convert it into a C array. Copy the array into model_data.h as
 * `unsigned char g_model[]` and set `g_model_len` accordingly.
 */

#include <Arduino.h>
#include "model_data.h"    // Generated model array
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Configuration parameters (must match training)
constexpr int kNumSensors = 9;      // update with actual number of glove sensors
constexpr int kWindowSize = 32;     // sliding window length used during training
constexpr int kNumClasses = 26;     // number of gesture classes (A-Z)

// ADC pins for XIAO ESP32 S3 sensors (GPIO numbers matching data collection)
const int sensorPins[kNumSensors] = {1, 2, 3, 4, 5, 6, 9, 10, 11};

// Circular buffer to store recent readings
float sensorBuffer[kWindowSize][kNumSensors];
int bufferIndex = 0;
bool bufferFilled = false;

// TensorFlow Lite Micro objects
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
// Arena for working memory; size depends on model. Increase if needed.
constexpr int kTensorArenaSize = 16 * 1024;
static uint8_t tensorArena[kTensorArenaSize];

// Label names corresponding to output indices (A-Z)
const char* gestureLabels[kNumClasses] = {
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
  "U", "V", "W", "X", "Y", "Z"
};

void setup() {
  Serial.begin(115200);
  // Load model
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (true) {}
  }
  // Set up operator resolver; only include operators used in the model
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv1D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  // You may need to add other ops (e.g., LSTM) depending on the model
  resolver.AddLSTM();
  resolver.AddBidirectionalSequenceLSTM();

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensorArena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate TFLite tensors");
    while (true) {}
  }
  Serial.println("Interpreter initialised");
}

// Read sensor values, normalise and store in buffer
void readSensors() {
  for (int i = 0; i < kNumSensors; ++i) {
    int raw = analogRead(sensorPins[i]);
    // Map raw ADC reading (0-4095) to normalised range [0,1]
    float normalised = raw / 4095.0f;
    sensorBuffer[bufferIndex][i] = normalised;
  }
  bufferIndex = (bufferIndex + 1) % kWindowSize;
  if (bufferIndex == 0) {
    bufferFilled = true;
  }
}

void runInference() {
  TfLiteTensor* input = interpreter->input(0);
  // Copy buffer into model input (assuming input dims are [1, kWindowSize, kNumSensors, 1] or similar)
  for (int t = 0; t < kWindowSize; ++t) {
    for (int s = 0; s < kNumSensors; ++s) {
      int index = t * kNumSensors + s;
      input->data.f[index] = sensorBuffer[(bufferIndex + t) % kWindowSize][s];
    }
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }
  // Retrieve output and find argmax
  TfLiteTensor* output = interpreter->output(0);
  int predicted = 0;
  float maxScore = output->data.f[0];
  for (int i = 1; i < kNumClasses; ++i) {
    float score = output->data.f[i];
    if (score > maxScore) {
      maxScore = score;
      predicted = i;
    }
  }
  Serial.print("Predicted gesture: ");
  Serial.print(predicted);
  Serial.print(" (");
  Serial.print(gestureLabels[predicted]);
  Serial.print(") Score:");
  Serial.println(maxScore, 4);
}

void loop() {
  readSensors();
  // Once the buffer is full we can run inference at each new reading
  if (bufferFilled) {
    runInference();
  }
  delay(10); // adjust sampling rate as needed
}
