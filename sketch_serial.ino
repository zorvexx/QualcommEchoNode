#include <Wire.h>

// Plain Serial output - bypasses Bridge RPC container socket issue.
// Linux host reads /dev/ttyHS1 at 115200 baud directly.
// I,<ts_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>
// A,<ts_us>,<val>
// T,<ts_ms>,<obj_c>,<amb_c>

constexpr uint32_t IMU_INTERVAL_US   = 20000;
constexpr uint32_t AUDIO_INTERVAL_US = 2000;
constexpr uint32_t MLX_INTERVAL_US   = 500000;

constexpr uint8_t MLX_SDA_PIN = 2;
constexpr uint8_t MLX_SCL_PIN = 3;
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
  sdaHigh(); delayMicroseconds(I2C_DELAY_US);
  sclHigh(); delayMicroseconds(I2C_DELAY_US);
  bool ack = (digitalRead(MLX_SDA_PIN) == LOW);
  sclLow(); delayMicroseconds(I2C_DELAY_US);
  return ack;
}
uint8_t i2cReadByte(bool sendAck) {
  uint8_t val = 0;
  sdaHigh();
  for (int i = 0; i < 8; i++) {
    sclHigh(); delayMicroseconds(I2C_DELAY_US);
    val = (val << 1) | digitalRead(MLX_SDA_PIN);
    sclLow(); delayMicroseconds(I2C_DELAY_US);
  }
  if (sendAck) sdaLow(); else sdaHigh();
  delayMicroseconds(I2C_DELAY_US);
  sclHigh(); delayMicroseconds(I2C_DELAY_US);
  sclLow(); sdaHigh();
  return val;
}
uint8_t calculatePEC(uint8_t *data, uint8_t len) {
  uint8_t crc = 0;
  for (uint8_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; j++) {
      if (crc & 0x80) crc = (crc << 1) ^ 0x07; else crc <<= 1;
    }
  }
  return crc;
}
bool readMLXRegister(uint8_t reg, float &tempC, uint8_t &pec) {
  uint8_t addrW = 0x5A << 1, addrR = addrW | 1;
  uint8_t lowByte = 0, highByte = 0;
  bool ack1, ack2, ack3;
  i2cStart(); ack1 = i2cWriteByte(addrW); ack2 = i2cWriteByte(reg);
  i2cStart(); ack3 = i2cWriteByte(addrR);
  if (ack3) { lowByte = i2cReadByte(true); highByte = i2cReadByte(true); pec = i2cReadByte(false); }
  i2cStop();
  if (ack1 && ack2 && ack3) {
    uint8_t buf[5] = { addrW, reg, addrR, lowByte, highByte };
    if (calculatePEC(buf, 5) != pec) return false;
    uint16_t raw = lowByte | (highByte << 8);
    if (highByte & 0x80) return false;
    tempC = (raw * 0.02f) - 273.15f; return true;
  }
  return false;
}
bool readMLXWithRetry(uint8_t reg, float &tempC, uint8_t &pec) {
  for (uint8_t a = 0; a < 2; a++) { if (readMLXRegister(reg, tempC, pec)) return true; sdaHigh(); sclHigh(); delay(2); }
  return false;
}
void writeMPURegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU6050_ADDR); Wire.write(reg); Wire.write(val); Wire.endTransmission(true);
}
void initMPU6050() {
  writeMPURegister(0x6B, 0x80); delay(20);
  writeMPURegister(0x6B, 0x01); delay(10);
  writeMPURegister(0x6A, 0x00); delay(5);
  writeMPURegister(0x37, 0x02); delay(5);
  writeMPURegister(0x1C, 0x00);
  writeMPURegister(0x1B, 0x00);
  writeMPURegister(0x1A, 0x03);
  writeMPURegister(0x19, 39);
}

struct ImuSample   { uint32_t ts_us; int16_t ax,ay,az,gx,gy,gz; };
struct AudioSample { uint32_t ts_us; uint16_t val; };
ImuSample   imuBuf[5];
AudioSample audioBuf[20];
uint8_t imuN=0, audN=0;
uint32_t nextImuTime=0, nextAudioTime=0, nextMlxTime=0;

void setup() {
  Serial.begin(115200);
  pinMode(MIC_PIN, INPUT);
  analogReadResolution(14);
  sdaHigh(); sclHigh();
  Wire.begin(); Wire.setClock(100000);
  initMPU6050();
  uint32_t now = micros();
  nextImuTime=now; nextAudioTime=now; nextMlxTime=now;
  Serial.println("RETROFIT_SERIAL_READY");
}

void loop() {
  uint32_t now = micros();

  if ((int32_t)(now - nextAudioTime) >= 0) {
    audioBuf[audN++] = { now, (uint16_t)analogRead(MIC_PIN) };
    if (audN >= 20) {
      for (uint8_t i=0; i<audN; i++) {
        Serial.print("A,"); Serial.print(audioBuf[i].ts_us); Serial.print(","); Serial.println(audioBuf[i].val);
      }
      audN = 0;
    }
    nextAudioTime = now + AUDIO_INTERVAL_US;
  }

  if ((int32_t)(now - nextImuTime) >= 0) {
    int16_t ax=0,ay=0,az=0,gx=0,gy=0,gz=0;
    Wire.beginTransmission(MPU6050_ADDR); Wire.write(0x3B);
    if (Wire.endTransmission(false)==0 && Wire.requestFrom((uint8_t)MPU6050_ADDR,(uint8_t)14)==14) {
      ax=(Wire.read()<<8)|Wire.read(); ay=(Wire.read()<<8)|Wire.read(); az=(Wire.read()<<8)|Wire.read();
      Wire.read(); Wire.read();
      gx=(Wire.read()<<8)|Wire.read(); gy=(Wire.read()<<8)|Wire.read(); gz=(Wire.read()<<8)|Wire.read();
    }
    imuBuf[imuN++] = { now, ax,ay,az,gx,gy,gz };
    if (imuN >= 5) {
      for (uint8_t i=0; i<imuN; i++) {
        Serial.print("I,"); Serial.print(imuBuf[i].ts_us);
        Serial.print(","); Serial.print(imuBuf[i].ax); Serial.print(","); Serial.print(imuBuf[i].ay); Serial.print(","); Serial.print(imuBuf[i].az);
        Serial.print(","); Serial.print(imuBuf[i].gx); Serial.print(","); Serial.print(imuBuf[i].gy); Serial.print(","); Serial.println(imuBuf[i].gz);
      }
      imuN = 0;
    }
    nextImuTime = now + IMU_INTERVAL_US;
  }

  if ((int32_t)(now - nextMlxTime) >= 0) {
    float amb=0, obj=0; uint8_t pA=0,pO=0;
    if (readMLXWithRetry(0x06,amb,pA)) lastAmbientTempC=amb;
    if (readMLXWithRetry(0x07,obj,pO)) lastObjectTempC=obj;
    Serial.print("T,"); Serial.print(millis()); Serial.print(",");
    Serial.print(lastObjectTempC,2); Serial.print(","); Serial.println(lastAmbientTempC,2);
    nextMlxTime = now + MLX_INTERVAL_US;
  }
}
