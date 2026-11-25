"""6-DoF Madgwick orientation filter plus quaternion helpers.

Quaternions are (w, x, y, z) and rotate the sensor frame into the world frame.
"""
import numpy as np


def quat_mult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def euler_deg(q):
    """(roll, pitch, yaw) in degrees."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


class Madgwick:
    """Gyro integration corrected toward gravity. Without a magnetometer, yaw is
    gyro-only, so gyro bias shows up directly as yaw drift."""

    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro, accel, dt):
        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro
        q_dot = 0.5 * np.array([
            -q1 * gx - q2 * gy - q3 * gz,
            q0 * gx + q2 * gz - q3 * gy,
            q0 * gy - q1 * gz + q3 * gx,
            q0 * gz + q1 * gy - q2 * gx,
        ])
        a_norm = np.linalg.norm(accel)
        if a_norm > 1e-6:
            ax, ay, az = np.asarray(accel) / a_norm
            f = np.array([
                2 * (q1 * q3 - q0 * q2) - ax,
                2 * (q0 * q1 + q2 * q3) - ay,
                2 * (0.5 - q1 * q1 - q2 * q2) - az,
            ])
            jac = np.array([
                [-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                [0.0, -4 * q1, -4 * q2, 0.0],
            ])
            step = jac.T @ f
            s_norm = np.linalg.norm(step)
            if s_norm > 1e-12:
                q_dot -= self.beta * step / s_norm
        q = self.q + q_dot * dt
        self.q = q / np.linalg.norm(q)
        return self.q


def device_seconds(dev_t_us):
    """Device timestamps (uint32 micros, may wrap) -> seconds from the first sample."""
    d = np.diff(np.asarray(dev_t_us, dtype=np.int64)) % (1 << 32)
    return np.concatenate([[0.0], np.cumsum(d) / 1e6])


def run_filter(rows, gyro_bias=(0.0, 0.0, 0.0), beta=0.1):
    """Run Madgwick over recorded rows. Returns an (N, 4) array of quaternions."""
    rows = np.asarray(rows, dtype=float)
    t = device_seconds(rows[:, 2])
    dts = np.diff(t, prepend=t[0] - 0.01)
    bias = np.asarray(gyro_bias, dtype=float)
    f = Madgwick(beta)
    out = np.empty((len(rows), 4))
    for i, r in enumerate(rows):
        out[i] = f.update(r[6:9] - bias, r[3:6], dts[i])
    return out
