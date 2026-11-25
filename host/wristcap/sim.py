"""Fake WristCap that emits real packets. Lets you run every script before the
hardware arrives (--sim) and gives the unit tests realistic data."""
import numpy as np

from .fusion import quat_mult, quat_to_matrix
from .protocol import SAMPLE_HZ, pack, parse


class SimDevice:
    def __init__(self, rate=SAMPLE_HZ, gyro_bias_dps=(0.4, -0.25, 0.3), gyro_noise_dps=0.05,
                 accel_noise_g=0.004, loss=0.002, motion=False, seed=0):
        self.rate = rate
        self.bias = np.radians(gyro_bias_dps)
        self.gyro_noise = np.radians(gyro_noise_dps)
        self.accel_noise = accel_noise_g
        self.loss = loss
        self.motion = motion
        self.rng = np.random.default_rng(seed)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.seq = 0
        self.t_us = 1_000_000

    def _true_rate(self, t):
        if not self.motion:
            return np.zeros(3)
        return np.array([1.2 * np.sin(1.1 * t), 0.8 * np.sin(0.7 * t + 1), 1.5 * np.sin(0.5 * t)])

    def step(self):
        """Advance one sample. Returns a packet (bytes) or None if it was 'dropped'."""
        dt = 1.0 / self.rate
        t = self.seq * dt
        w = self._true_rate(t)
        self.q = self.q + 0.5 * quat_mult(self.q, np.r_[0.0, w]) * dt
        self.q /= np.linalg.norm(self.q)
        accel = quat_to_matrix(self.q)[2] + self.rng.normal(0, self.accel_noise, 3)  # gravity in sensor frame
        gyro = w + self.bias + self.rng.normal(0, self.gyro_noise, 3)
        pkt = pack(self.seq, self.t_us, accel, gyro)
        self.seq += 1
        self.t_us += int(dt * 1e6) + int(self.rng.integers(-20, 21))
        return None if self.rng.random() < self.loss else pkt

    def generate(self, seconds):
        """Offline: rows as if received over BLE (host time = device time + small latency)."""
        rows = []
        for _ in range(int(seconds * self.rate)):
            pkt = self.step()
            if pkt is not None:
                host_t = self.t_us / 1e6 + 0.008 + abs(self.rng.normal(0, 0.003))
                rows.append(parse(pkt, host_t))
        return np.array(rows)
