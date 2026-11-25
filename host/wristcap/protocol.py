"""Wire format shared with the firmware. Keep in sync with wristcap.ino."""
import struct

DEVICE_NAME = "WristCap"
SERVICE_UUID = "8a1c0001-5b6e-4c3a-9f2d-1e7c3b9a0001"
DATA_UUID = "8a1c0002-5b6e-4c3a-9f2d-1e7c3b9a0001"
SAMPLE_HZ = 100

PACKET = struct.Struct("<IIhhhhhh")  # seq, t_us, ax, ay, az (mg), gx, gy, gz (mrad/s)

# One sample = one row: host time (s), seq, device time (us), accel (g), gyro (rad/s)
COLUMNS = ["host_t", "seq", "dev_t_us", "ax_g", "ay_g", "az_g", "gx_rads", "gy_rads", "gz_rads"]


def _clamp16(v):
    return max(-32768, min(32767, int(round(v))))


def pack(seq, t_us, accel_g, gyro_rads):
    """Build a packet exactly like the firmware does (used by the simulator and tests)."""
    return PACKET.pack(
        seq & 0xFFFFFFFF,
        t_us & 0xFFFFFFFF,
        *(_clamp16(v * 1000) for v in accel_g),
        *(_clamp16(v * 1000) for v in gyro_rads),
    )


def parse(data, host_t):
    """Decode one packet into a row matching COLUMNS."""
    seq, t_us, ax, ay, az, gx, gy, gz = PACKET.unpack(data)
    return (host_t, seq, t_us,
            ax / 1000, ay / 1000, az / 1000,
            gx / 1000, gy / 1000, gz / 1000)
