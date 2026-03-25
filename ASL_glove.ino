void setup() {
  Serial.begin(9600);
}

static int counter = 0;

void loop() {
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);

  // int normalised = value * 8 - 15000;

#define READ_FINGER(n) analogRead(n)

#define SCALE(v, o, n) ((v - o) * n)

  digitalWrite(D3, HIGH);
  digitalWrite(D4, HIGH);
  digitalWrite(D5, HIGH);

  Serial.printf(
    "%d,%d,%d,%d,%d,%d\n",
    counter++,
    // SCALE(READ_FINGER(A0), 0, 1),
    SCALE(READ_FINGER(A1), 0, 1),
    SCALE(READ_FINGER(A2), 0, 1),
    SCALE(READ_FINGER(A3), 0, 1),
    SCALE(READ_FINGER(A4), 0, 1),
    SCALE(READ_FINGER(A5), 0, 1)
  );

  delay(50);
}

int readFinger(int n) {
  digitalWrite(n + 3, HIGH);
  int res = analogRead(n);
  digitalWrite(n + 3, LOW);

  return res;
}