void setup() {
  Serial.begin(115200);  // Match Python default baud rate
}

// Use the same ADC pins as your original code, but with correct GPIO mapping for XIAO ESP32 S3
const int sensorPins[9] = {1, 2, 3, 4, 5, 6, 9, 10, 11};  // GPIO numbers for A0, A1, A2, A3, A4, A5, A8, A9, A10

static int counter = 0;

void loop() {
  // Read 9 sensor values
  int sensorValues[9];

  for (int i = 0; i < 9; i++) {
    sensorValues[i] = analogRead(sensorPins[i]);
  }

  // Send data as comma-separated values (9 sensors as expected by Python)
  Serial.print(counter++);
  for (int i = 0; i < 9; i++) {
    Serial.print(",");
    Serial.print(sensorValues[i]);
  }
  Serial.println();

  delay(50);  // 20Hz sampling rate
}