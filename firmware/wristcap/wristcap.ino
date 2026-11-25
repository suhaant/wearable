// WristCap firmware: LSM6DSOX IMU -> BLE notify stream at 100 Hz.
// Board: Adafruit Feather ESP32 V2 (works on any ESP32 with BLE).
// Libraries: Adafruit LSM6DS, NimBLE-Arduino (2.x).
//
// Packet (20 bytes, little-endian, fits the default BLE MTU):
//   uint32 seq        increments every sample, gaps = dropped packets
//   uint32 t_us       device micros() when the sample was read
//   int16  ax,ay,az   acceleration in milli-g
//   int16  gx,gy,gz   angular rate in milli-rad/s

#include <Wire.h>
#include <Adafruit_LSM6DSOX.h>
#include <NimBLEDevice.h>

#define DEVICE_NAME  "WristCap"
#define SERVICE_UUID "8a1c0001-5b6e-4c3a-9f2d-1e7c3b9a0001"
#define DATA_UUID    "8a1c0002-5b6e-4c3a-9f2d-1e7c3b9a0001"

#define SERIAL_DEBUG 1          // print a reading every 200 ms (milestone 1 check)
static const uint32_t SAMPLE_HZ = 100;
static const uint32_t PERIOD_US = 1000000UL / SAMPLE_HZ;

#pragma pack(push, 1)
struct Packet {
  uint32_t seq;
  uint32_t t_us;
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};
#pragma pack(pop)
static_assert(sizeof(Packet) == 20, "packet must fit the default 20-byte BLE payload");

Adafruit_LSM6DSOX imu;
NimBLECharacteristic* dataChr = nullptr;
volatile bool connected = false;
uint32_t seq = 0;
uint32_t nextTick = 0;

static int16_t clamp16(float v) {
  if (v > 32767.0f) return 32767;
  if (v < -32768.0f) return -32768;
  return (int16_t)lroundf(v);
}

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* server, NimBLEConnInfo& info) override {
    connected = true;
    // Ask for a 7.5-15 ms connection interval so 100 Hz notifications keep up.
    server->updateConnParams(info.getConnHandle(), 6, 12, 0, 400);
    Serial.println("BLE connected");
  }
  void onDisconnect(NimBLEServer* server, NimBLEConnInfo& info, int reason) override {
    connected = false;
    Serial.printf("BLE disconnected (reason %d), advertising again\n", reason);
    NimBLEDevice::startAdvertising();
  }
};

void setup() {
  Serial.begin(115200);
  delay(500);

#ifdef NEOPIXEL_I2C_POWER
  // Feather ESP32 V2: the STEMMA QT port is unpowered until this pin goes HIGH.
  pinMode(NEOPIXEL_I2C_POWER, OUTPUT);
  digitalWrite(NEOPIXEL_I2C_POWER, HIGH);
  delay(10);
#endif

  Wire.begin();
  Wire.setClock(400000);
  if (!imu.begin_I2C()) {
    Serial.println("LSM6DSOX not found. Check the QT cable is seated on both ends.");
    while (true) delay(1000);
  }
  imu.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  imu.setGyroRange(LSM6DS_GYRO_RANGE_1000_DPS);
  imu.setAccelDataRate(LSM6DS_RATE_208_HZ);
  imu.setGyroDataRate(LSM6DS_RATE_208_HZ);
  Serial.println("IMU ready");

  NimBLEDevice::init(DEVICE_NAME);
  NimBLEServer* server = NimBLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  NimBLEService* svc = server->createService(SERVICE_UUID);
  dataChr = svc->createCharacteristic(DATA_UUID, NIMBLE_PROPERTY::NOTIFY);
  svc->start();

  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->setName(DEVICE_NAME);
  adv->addServiceUUID(SERVICE_UUID);
  adv->enableScanResponse(true);
  adv->start();
  Serial.println("Advertising as " DEVICE_NAME);

  nextTick = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextTick) < 0) return;      // not time yet
  nextTick += PERIOD_US;
  if ((int32_t)(now - nextTick) > (int32_t)PERIOD_US) {
    nextTick = now + PERIOD_US;                    // fell far behind: resync, don't burst
  }

  sensors_event_t a, g, t;
  imu.getEvent(&a, &g, &t);

  const float toMilliG = 1000.0f / 9.80665f;
  Packet p;
  p.seq  = seq++;
  p.t_us = now;
  p.ax = clamp16(a.acceleration.x * toMilliG);
  p.ay = clamp16(a.acceleration.y * toMilliG);
  p.az = clamp16(a.acceleration.z * toMilliG);
  p.gx = clamp16(g.gyro.x * 1000.0f);
  p.gy = clamp16(g.gyro.y * 1000.0f);
  p.gz = clamp16(g.gyro.z * 1000.0f);

  if (connected) {
    dataChr->setValue((uint8_t*)&p, sizeof(p));
    dataChr->notify();
  }

#if SERIAL_DEBUG
  if (p.seq % 20 == 0) {
    Serial.printf("seq=%lu  acc[mg]=%6d %6d %6d  gyro[mrad/s]=%6d %6d %6d  %s\n",
                  (unsigned long)p.seq, p.ax, p.ay, p.az, p.gx, p.gy, p.gz,
                  connected ? "BLE" : "-");
  }
#endif
}
