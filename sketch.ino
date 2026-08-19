#include <Wire.h>
#include <Arduino_RouterBridge.h>

// ==========================================
// CONFIG — TARGET RATES (see explanation below for why these were chosen)
// ==========================================
// MPU6050: DLPF disabled -> internal gyro output rate = 8 kHz.
//   SMPLRT_DIV = 39  =>  8000 / (1 + 39) = 200 Hz internal refresh,
//   matched to our 200 Hz poll schedule below.
//   Accel range: +-2g (unchanged from original code)
//   Gyro range:  +-250 dps (unchanged from original code)
//   DLPF:        disabled (DLPF_CFG = 0), as in original code
constexpr uint8_t MPU_SMPLRT_DIV = 39;

constexpr uint32_t IMU_INTERVAL_US   = 5000;    // 200 Hz target
constexpr uint32_t AUDIO_INTERVAL_US = 125;     // 8000 Hz target
constexpr uint32_t MLX_INTERVAL_US   = 500000;  // 2 Hz target

// Batch sizes: NOT sent one sample at a time -- Bridge RPC has per-call
// overhead, so audio/IMU samples are buffered on-MCU and flushed as one
// CSV-chunk string per batch. These sizes are a starting guess and are
// the first thing to tune if Monitor output shows actual Hz << target Hz.
constexpr uint8_t IMU_BATCH_SIZE   = 20;  // -> one Bridge call per 100ms @ 200Hz
constexpr uint8_t AUDIO_BATCH_SIZE = 64;  // -> one Bridge call per ~8ms @ 8kHz (aggressive, verify)

// --- MLX90614 Bit-Bang Pins (D2/D3) — unchanged from original ---
constexpr uint8_t MLX_SDA_PIN = 2;
constexpr uint8_t MLX_SCL_PIN = 3;
constexpr uint8_t MLX90614_ADDR = 0x5A;
constexpr uint16_t I2C_DELAY_US = 2;

// --- MAX9814 Microphone ---
constexpr int MIC_PIN = A0;

// --- MPU6050 ---
constexpr uint8_t MPU6050_ADDR = 0x68;

float lastAmbientTempC = 0.0;
float lastObjectTempC  = 0.0;

// ==========================================
// MLX90614 BIT-BANG DRIVER — UNCHANGED from working version
// ==========================================

void sdaHigh() { pinMode(MLX_SDA_PIN, INPUT_PULLUP); }
void sdaLow()  { pinMode(MLX_SDA_PIN, OUTPUT); digitalWrite(MLX_SDA_PIN, LOW); }
void sclHigh() { pinMode(MLX_SCL_PIN, INPUT_PULLUP); }
void sclLow()  { pinMode(MLX_SCL_PIN, OUTPUT); digitalWrite(MLX_SCL_PIN, LOW); }

void i2cStart() {
  sdaHigh(); sclHigh(); delayMicroseconds(I2C_DELAY_US);
  sdaLow();  delayMicroseconds(I2C_DELAY_US);
  sclLow();  delayMicroseconds(I2C_DELAY_US);
}

void i2cStop() {
  sdaLow();  delayMicroseconds(I2C_DELAY_US);
  sclHigh(); delayMicroseconds(I2C_DELAY_US);
  sdaHigh(); delayMicroseconds(I2C_DELAY_US);
}

bool i2cWriteByte(uint8_t b) {
  for (int i = 0; i < 8; i++) {
    if (b & 0x80) sdaHigh(); else sdaLow();
    delayMicroseconds(I2C_DELAY_US);
    sclHigh(); delayMicroseconds(I2C_DELAY_US);
    sclLow();  delayMicroseconds(I2C_DELAY_US);
    b <<= 1;
  }
  sdaHigh();
  delayMicroseconds(I2C_DELAY_US);
  sclHigh();
  delayMicroseconds(I2C_DELAY_US);
  bool ack = (digitalRead(MLX_SDA_PIN) == LOW);
  sclLow();
  delayMicroseconds(I2C_DELAY_US);
  return ack;
}

uint8_t i2cReadByte(bool sendAck) {
  uint8_t val = 0;
  sdaHigh();
  for (int i = 0; i < 8; i++) {
    sclHigh();
    delayMicroseconds(I2C_DELAY_US);
    val = (val << 1) | digitalRead(MLX_SDA_PIN);
    sclLow();
    delayMicroseconds(I2C_DELAY_US);
  }
  if (sendAck) sdaLow(); else sdaHigh();
  delayMicroseconds(I2C_DELAY_US);
  sclHigh(); delayMicroseconds(I2C_DELAY_US);
  sclLow();
  sdaHigh();
  return val;
}

uint8_t calculatePEC(uint8_t *data, uint8_t len) {
  uint8_t crc = 0;
  for (uint8_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; j++) {
      if (crc & 0x80) crc = (crc << 1) ^ 0x07;
      else crc <<= 1;
    }
  }
  return crc;
}

bool readMLXRegister(uint8_t reg, float &tempC, uint8_t &pec) {
  bool ack1 = false, ack2 = false, ack3 = false;
  uint8_t lowByte = 0, highByte = 0;

  uint8_t addrW = (MLX90614_ADDR << 1) | 0;
  uint8_t addrR = (MLX90614_ADDR << 1) | 1;

  i2cStart();
  ack1 = i2cWriteByte(addrW);
  if (ack1) ack2 = i2cWriteByte(reg);

  sdaHigh(); delayMicroseconds(I2C_DELAY_US);
  sclHigh(); delayMicroseconds(I2C_DELAY_US);

  i2cStart();
  ack3 = i2cWriteByte(addrR);

  if (ack3) {
    lowByte  = i2cReadByte(true);
    highByte = i2cReadByte(true);
    pec      = i2cReadByte(false);
  }
  i2cStop();

  if (ack1 && ack2 && ack3) {
    uint8_t buf[5] = { addrW, reg, addrR, lowByte, highByte };
    if (calculatePEC(buf, 5) != pec) return false;

    uint16_t rawTemp = lowByte | (highByte << 8);
    if (highByte & 0x80) return false;

    tempC = (rawTemp * 0.02) - 273.15;
    return true;
  }
  return false;
}

bool readMLXRegisterWithRetry(uint8_t reg, float &tempC, uint8_t &pec) {
  for (uint8_t attempt = 0; attempt < 5; attempt++) {
    if (readMLXRegister(reg, tempC, pec)) return true;
    sdaHigh(); sclHigh(); delay(3);
  }
  return false;
}

// ==========================================
// MPU6050
// ==========================================

void writeMPURegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

void initMPU6050() {
  writeMPURegister(0x6B, 0x80); delay(50);              // Reset
  writeMPURegister(0x6B, 0x01); delay(10);              // Wake Up
  writeMPURegister(0x6A, 0x00); delay(10);               // Disable Master Mode
  writeMPURegister(0x37, 0x02); delay(10);               // I2C Bypass Mode

  writeMPURegister(0x1C, 0x00);                          // Accel +-2g
  writeMPURegister(0x1B, 0x00);                          // Gyro +-250 dps
  writeMPURegister(0x1A, 0x00);                          // DLPF disabled -> 8kHz internal gyro rate
  writeMPURegister(0x19, MPU_SMPLRT_DIV);                 // SMPLRT_DIV=39 -> 200Hz internal refresh
}

// ==========================================
// NON-BLOCKING MULTI-RATE ACQUISITION
// ==========================================

struct ImuSample   { uint32_t ts_us; int16_t ax, ay, az, gx, gy, gz; };
struct AudioSample { uint32_t ts_us; uint16_t val; };

ImuSample   imuBuf[IMU_BATCH_SIZE];
AudioSample audioBuf[AUDIO_BATCH_SIZE];
uint8_t imuBufCount = 0;
uint8_t audioBufCount = 0;

char txBuf[1400]; // shared scratch buffer for building CSV chunks before a Bridge.call

uint32_t nextImuTime = 0;
uint32_t nextAudioTime = 0;
uint32_t nextMlxTime = 0;

uint32_t rateWindowStartMs = 0;
uint16_t imuCountThisSec = 0, audioCountThisSec = 0, mlxCountThisSec = 0;
uint32_t imuMissed = 0, audioMissed = 0;

void flushImu() {
  int offset = 0;
  for (uint8_t i = 0; i < imuBufCount; i++) {
    offset += snprintf(txBuf + offset, sizeof(txBuf) - offset,
                        "%lu,%d,%d,%d,%d,%d,%d\n",
                        (unsigned long)imuBuf[i].ts_us,
                        imuBuf[i].ax, imuBuf[i].ay, imuBuf[i].az,
                        imuBuf[i].gx, imuBuf[i].gy, imuBuf[i].gz);
  }
  bool ok = false;
  Bridge.call("imu_batch", txBuf).result(ok);
  if (!ok) Monitor.println("WARN: imu_batch send failed");
  imuBufCount = 0;
}

void flushAudio() {
  int offset = 0;
  for (uint8_t i = 0; i < audioBufCount; i++) {
    offset += snprintf(txBuf + offset, sizeof(txBuf) - offset,
                        "%lu,%u\n",
                        (unsigned long)audioBuf[i].ts_us, audioBuf[i].val);
  }
  bool ok = false;
  Bridge.call("audio_batch", txBuf).result(ok);
  if (!ok) Monitor.println("WARN: audio_batch send failed");
  audioBufCount = 0;
}

void serviceImu(uint32_t ts) {
  int16_t ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) == 0 &&
      Wire.requestFrom((uint8_t)MPU6050_ADDR, (uint8_t)14) == 14) {
    ax = (Wire.read() << 8) | Wire.read();
    ay = (Wire.read() << 8) | Wire.read();
    az = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read(); // skip internal temp
    gx = (Wire.read() << 8) | Wire.read();
    gy = (Wire.read() << 8) | Wire.read();
    gz = (Wire.read() << 8) | Wire.read();
  }
  if (imuBufCount < IMU_BATCH_SIZE) {
    imuBuf[imuBufCount++] = { ts, ax, ay, az, gx, gy, gz };
  }
  if (imuBufCount >= IMU_BATCH_SIZE) flushImu();
}

void serviceAudio(uint32_t ts) {
  uint16_t val = analogRead(MIC_PIN);
  if (audioBufCount < AUDIO_BATCH_SIZE) {
    audioBuf[audioBufCount++] = { ts, val };
  }
  if (audioBufCount >= AUDIO_BATCH_SIZE) flushAudio();
}

void serviceMlx() {
  float amb = 0, obj = 0;
  uint8_t pecA = 0, pecO = 0;
  bool okA = readMLXRegisterWithRetry(0x06, amb, pecA);
  bool okO = readMLXRegisterWithRetry(0x07, obj, pecO);
  if (okA) lastAmbientTempC = amb;
  if (okO) lastObjectTempC = obj;

  char line[64];
  snprintf(line, sizeof(line), "%lu,%.2f,%.2f",
           (unsigned long)millis(), lastObjectTempC, lastAmbientTempC);
  bool ok = false;
  Bridge.call("temp_row", line).result(ok);
  if (!ok) Monitor.println("WARN: temp_row send failed");
}

// ==========================================
// SETUP & LOOP
// ==========================================

void setup() {
  Monitor.begin();

  pinMode(MIC_PIN, INPUT);
  analogReadResolution(14); // Uno Q ADC is 14-bit (0-16383) per board docs

  sdaHigh();
  sclHigh();

  Wire.begin();
  Wire.setClock(100000); // Fast mode -- UNVERIFIED on this board's I2C driver.
                          // If MPU reads start failing/hanging, revert to 100000.

  initMPU6050();
  Bridge.begin();

  uint32_t now = micros();
  nextImuTime = now;
  nextAudioTime = now;
  nextMlxTime = now;
  rateWindowStartMs = millis();

  Monitor.println("RetroFit acquisition v2: IMU=200Hz(target) Audio=8000Hz(target) Temp=2Hz(target), magnetometer disabled");
}

void loop() {
  uint32_t now = micros();

  // --- Audio: highest priority, checked first every iteration ---
  if ((int32_t)(now - nextAudioTime) >= 0) {
    serviceAudio(now);
    uint32_t missed = 0;
    while ((int32_t)(now - nextAudioTime) >= 0) { nextAudioTime += AUDIO_INTERVAL_US; missed++; }
    if (missed > 1) audioMissed += (missed - 1);
    audioCountThisSec++;
  }

  // --- IMU: second priority ---
  if ((int32_t)(now - nextImuTime) >= 0) {
    serviceImu(now);
    uint32_t missed = 0;
    while ((int32_t)(now - nextImuTime) >= 0) { nextImuTime += IMU_INTERVAL_US; missed++; }
    if (missed > 1) imuMissed += (missed - 1);
    imuCountThisSec++;
  }

  // --- MLX90614: lowest priority, slow, tolerates its own blocking retries ---
  if ((int32_t)(now - nextMlxTime) >= 0) {
    serviceMlx();
    nextMlxTime += MLX_INTERVAL_US;
    mlxCountThisSec++;
  }

  // --- Live rate monitor: printed once per second, ACTUAL measured counts ---
  uint32_t nowMs = millis();
  if (nowMs - rateWindowStartMs >= 1000) {
    Monitor.print("RATE imu="); Monitor.print(imuCountThisSec); Monitor.print("Hz");
    Monitor.print(" audio="); Monitor.print(audioCountThisSec); Monitor.print("Hz");
    Monitor.print(" temp="); Monitor.print(mlxCountThisSec); Monitor.print("Hz");
    Monitor.print(" imu_missed_total="); Monitor.print(imuMissed);
    Monitor.print(" audio_missed_total="); Monitor.println(audioMissed);
    imuCountThisSec = 0; audioCountThisSec = 0; mlxCountThisSec = 0;
    rateWindowStartMs = nowMs;
  }
}