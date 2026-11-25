"""Metrics used by calibrate.py and selftest.py. Pure functions, no hardware needed."""
import numpy as np

from .fusion import device_seconds, euler_deg, run_filter


def sample_rate_hz(rows):
    t = device_seconds(rows[:, 2])
    return (len(rows) - 1) / t[-1] if t[-1] > 0 else 0.0


def packet_loss_pct(rows):
    seq = rows[:, 1].astype(np.int64)
    expected = seq.max() - seq.min() + 1
    return 100.0 * (1 - len(np.unique(seq)) / expected)


def arrival_gaps_ms(rows):
    """Host-side inter-arrival times. BLE delivers in bursts, so max gap matters."""
    d = np.diff(rows[:, 0]) * 1000
    return {"mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": float(d.max())}


def gyro_bias(rows):
    return rows[:, 6:9].mean(axis=0)


def gyro_noise_dps(rows):
    return float(np.degrees(rows[:, 6:9].std(axis=0)).max())


def accel_norm_error_pct(rows):
    return float(100 * (np.linalg.norm(rows[:, 3:6], axis=1).mean() - 1.0))


def yaw_drift_deg_per_min(rows, bias=(0.0, 0.0, 0.0)):
    """Slope of yaw over a stationary recording, from a line fit."""
    quats = run_filter(rows, gyro_bias=bias)
    yaw = np.unwrap(np.radians([euler_deg(q)[2] for q in quats]))
    t = device_seconds(rows[:, 2])
    slope_rad_s = np.polyfit(t, yaw, 1)[0]
    return float(abs(np.degrees(slope_rad_s)) * 60)
