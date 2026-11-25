# WristCap

A wrist-worn motion capture band that streams arm movement over Bluetooth LE for robot-learning data collection. It records timestamped IMU data, estimates orientation with sensor fusion, and ships with automated calibration and a pass/fail self-test.

![tests](https://github.com/suhaant/wearable/actions/workflows/tests.yml/badge.svg)

## Results

Measured on hardware with `selftest.py` (60 s, band stationary):

| Metric | Result | Target |
|---|---|---|
| Sample rate over BLE | _TBD_ Hz | ≥ 95 Hz |
| Packet loss | _TBD_ % | < 1% |
| Max arrival gap | _TBD_ ms | < 100 ms |
| Yaw drift, uncalibrated | _TBD_ °/min | n/a |
| Yaw drift, calibrated | _TBD_ °/min | < 2 °/min |

## Hardware (~$33)

| Part | Adafruit PID |
|---|---|
| ESP32 Feather V2 (BLE, USB-C) | 5400 |
| LSM6DSOX 6-DoF IMU, STEMMA QT | 4438 |
| STEMMA QT cable, 50 mm | 4399 |

No soldering: the IMU plugs into the Feather's STEMMA QT port. Power from USB or a USB power bank. The firmware enables the Feather V2's STEMMA QT power pin automatically.

## How it works

```
LSM6DSOX --I2C 400kHz--> ESP32 --BLE notify, 100 Hz, 20-byte packets--> laptop
                                                                        |-- Madgwick filter (6-DoF) -> orientation
                                                                        |-- CSV logging
                                                                        |-- calibration + self-test
```

**Packet (20 bytes, fits the default BLE MTU):** `uint32 seq`, `uint32 device_us`, accel x/y/z (`int16`, milli-g), gyro x/y/z (`int16`, milli-rad/s). Sequence numbers make packet loss measurable; device timestamps make the fusion independent of BLE delivery jitter.

**Fusion:** a 6-DoF Madgwick filter. Gravity corrects roll and pitch. Yaw is gyro-only (no magnetometer), so gyro bias shows up directly as yaw drift, which is what calibration removes.

## Setup

**Firmware**
1. Install the Arduino IDE.
2. Board package: Boards Manager, install **esp32 by Espressif**. Select board **Adafruit Feather ESP32 V2**.
3. Libraries: **Adafruit LSM6DS** (accept its dependencies) and **NimBLE-Arduino** (2.x).
4. Open `firmware/wristcap/wristcap.ino`, upload, and open Serial Monitor at 115200. You should see `IMU ready`, `Advertising as WristCap`, and readings every 200 ms.

**Host (Python 3.10+)**
```bash
cd host
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
On macOS, allow Bluetooth access for your terminal app when prompted (System Settings, Privacy & Security, Bluetooth).

## Usage

Every script accepts `--sim` to run against a simulated band, so the pipeline can be tested without hardware.

```bash
python calibrate.py                        # 10 s flat and still -> calibration.json
python selftest.py                         # 60 s still -> PASS/FAIL table + selftest_report.json
python visualize.py                        # live 3D orientation; press z to zero heading
python receive.py --label cup_pickup --duration 20   # record to data/*.csv
pytest                                     # unit tests (no hardware)
```

`selftest.py` exits with code 0 on PASS and 1 on FAIL, so it can gate a production line or a CI job.

## Next steps

- Second band on the other arm, time-synced over one BLE central
- Sync recordings with phone video for labeled demonstrations
- Magnetometer or vision-based heading to remove remaining yaw drift
- Custom PCB and enclosure; batch self-test for manufacturing
