#include <Wire.h>
#include <Arduino_RouterBridge.h>

constexpr uint32_t IMU_INTERVAL_US   = 20000;   // 50 Hz IMU target
constexpr uint32_t AUDIO_INTERVAL_US = 2000;    // 500 Hz Audio target
constexpr uint32_t MLX_INTERVAL_US   = 500000;  // 2 Hz Temp target

constexpr uint8_t IMU_BATCH_SIZE   = 5;
constexpr uint8_t AUDIO_BATCH_SIZE = 20;

constexpr uint8_t MLX_SDA_PIN = 2;
constexpr uint8_t MLX_SCL_PIN = 3;
constexpr uint8_t MLX90614_ADDR = 0x5A;
constexpr uint16_t I2C_DELAY_US = 2;

constexpr int MIC_PIN = A0;
constexpr uint8_t MPU6050_ADDR = 0x68;

float lastAmbientTempC = 28.0;
float lastObjectTempC  = 29.5;

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
  for (uint8_t attempt = 0; attempt < 2; attempt++) {
    if (readMLXRegister(reg, tempC, pec)) return true;
    sdaHigh(); sclHigh(); delay(2);
  }
  return false;
}

void writeMPURegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

void initMPU6050() {
  writeMPURegister(0x6B, 0x80); delay(20);              // Reset
  writeMPURegister(0x6B, 0x01); delay(10);              // Wake Up
  writeMPURegister(0x6A, 0x00); delay(5);               // Disable Master Mode
  writeMPURegister(0x37, 0x02); delay(5);               // I2C Bypass Mode

  writeMPURegister(0x1C, 0x00);                          // Accel +-2g
  writeMPURegister(0x1B, 0x00);                          // Gyro +-250 dps
  writeMPURegister(0x1A, 0x03);                          // DLPF 42Hz for smooth machine tracking
  writeMPURegister(0x19, 39);
}

struct ImuSample   { uint32_t ts_us; int16_t ax, ay, az, gx, gy, gz; };
struct AudioSample { uint32_t ts_us; uint16_t val; };

ImuSample   imuBuf[IMU_BATCH_SIZE];
AudioSample audioBuf[AUDIO_BATCH_SIZE];
uint8_t imuBufCount = 0;
uint8_t audioBufCount = 0;

char txImuBuf[300];
char txAudioBuf[300];

uint32_t nextImuTime = 0;
uint32_t nextAudioTime = 0;
uint32_t nextMlxTime = 0;

void flushImu() {
  int offset = 0;
  for (uint8_t i = 0; i < imuBufCount; i++) {
    offset += snprintf(txImuBuf + offset, sizeof(txImuBuf) - offset,
                        "%lu,%d,%d,%d,%d,%d,%d\n",
                        (unsigned long)imuBuf[i].ts_us,
                        imuBuf[i].ax, imuBuf[i].ay, imuBuf[i].az,
                        imuBuf[i].gx, imuBuf[i].gy, imuBuf[i].gz);
  }
  bool ok = false;
  Bridge.call("imu_batch", txImuBuf).result(ok);
  imuBufCount = 0;
}

void flushAudio() {
  int offset = 0;
  for (uint8_t i = 0; i < audioBufCount; i++) {
    offset += snprintf(txAudioBuf + offset, sizeof(txAudioBuf) - offset,
                        "%lu,%u\n",
                        (unsigned long)audioBuf[i].ts_us, audioBuf[i].val);
  }
  bool ok = false;
  Bridge.call("audio_batch", txAudioBuf).result(ok);
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
    Wire.read(); Wire.read();
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
}

void setup() {
  Bridge.begin(); // Bridge MUST start first so Linux communication is ready immediately!

  pinMode(MIC_PIN, INPUT);
  analogReadResolution(14);

  sdaHigh();
  sclHigh();

  Wire.begin();
  Wire.setClock(100000);

  initMPU6050();

  uint32_t now = micros();
  nextImuTime = now;
  nextAudioTime = now;
  nextMlxTime = now;
}

void loop() {
  uint32_t now = micros();

  // 1. Audio check
  if ((int32_t)(now - nextAudioTime) >= 0) {
    serviceAudio(now);
    nextAudioTime += AUDIO_INTERVAL_US;
  }

  // 2. IMU check
  if ((int32_t)(now - nextImuTime) >= 0) {
    serviceImu(now);
    nextImuTime += IMU_INTERVAL_US;
  }

  // 3. MLX temperature check
  if ((int32_t)(now - nextMlxTime) >= 0) {
    serviceMlx();
    nextMlxTime += MLX_INTERVAL_US;
  }
}
