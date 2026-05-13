// XIAO ESP32-S3 glove streamer
// Serial: always prints clean CSV for training
// WiFi TCP: sends same CSV when WiFi + TCP are available
// CSV format: counter,A0,A2,A4,A8,A9
// No WiFi/debug messages are printed to Serial

#include <Arduino.h>
#include <WiFi.h>

// WiFi credentials
const char* WIFI_SSID     = "4G-Gateway-E84606-2";
const char* WIFI_PASSWORD = "A3FXJ2Q3";

// Computer running FastAPI TCP server
// Must match Python --tcp-port, NOT --port-web
const char* SERVER_HOST = "0.0.0.0";
const uint16_t SERVER_PORT = 9001;

// Sensor pins
const int sensorPins[5] = {
  A0,  // Thumb
  A2,  // Index
  A4,  // Middle
  A8,  // Ring
  A9   // Pinky
};

const int NUM_SENSORS = 5;

// Timing
const unsigned long SAMPLE_INTERVAL_MS = 50;       // 20 Hz
const unsigned long WIFI_RETRY_MS = 30000;         // retry WiFi every 30 s
const unsigned long TCP_RETRY_MS = 3000;           // retry TCP every 3 s

WiFiClient client;

unsigned long counter = 0;
unsigned long lastSampleMs = 0;
unsigned long lastWiFiAttemptMs = 0;
unsigned long lastTcpAttemptMs = 0;

bool wifiConnectStarted = false;

void startWiFiIfNeeded() {
  wl_status_t status = WiFi.status();
  unsigned long now = millis();

  if (status == WL_CONNECTED) {
    wifiConnectStarted = false;
    return;
  }

  // A connection attempt is already active. Let it finish or timeout.
  if (wifiConnectStarted && (now - lastWiFiAttemptMs < WIFI_RETRY_MS)) {
    return;
  }

  // Start/restart WiFi attempt, but only after timeout.
  WiFi.disconnect(false);
  delay(50);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  wifiConnectStarted = true;
  lastWiFiAttemptMs = now;
}

void connectTcpIfNeeded() {
  unsigned long now = millis();

  if (WiFi.status() != WL_CONNECTED) {
    if (client.connected()) {
      client.stop();
    }
    return;
  }

  if (client.connected()) {
    return;
  }

  // Do not try TCP connect every sample.
  if (now - lastTcpAttemptMs < TCP_RETRY_MS) {
    return;
  }

  client.stop();
  client.connect(SERVER_HOST, SERVER_PORT);
  lastTcpAttemptMs = now;
}

String readSensorCsvLine() {
  String line = String(counter++);

  for (int i = 0; i < NUM_SENSORS; i++) {
    int value = analogRead(sensorPins[i]);
    line += ",";
    line += String(value);
  }

  return line;
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  analogReadResolution(12);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  wifiConnectStarted = true;
  lastWiFiAttemptMs = millis();
}

void loop() {
  startWiFiIfNeeded();
  connectTcpIfNeeded();

  unsigned long now = millis();

  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;

    String line = readSensorCsvLine();

    // Always clean CSV for data collection/training
    Serial.println(line);

    // Send same CSV over WiFi TCP if available
    if (WiFi.status() == WL_CONNECTED && client.connected()) {
      client.println(line);
    }
  }
}