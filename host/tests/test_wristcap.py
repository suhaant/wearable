"""Unit tests that run without hardware (and in CI)."""
import numpy as np
import pytest

from wristcap import analysis
from wristcap.fusion import Madgwick, euler_deg, run_filter
from wristcap.protocol import PACKET, pack, parse
from wristcap.sim import SimDevice


def test_packet_fits_default_ble_payload():
    assert PACKET.size == 20


def test_pack_parse_roundtrip():
    row = parse(pack(42, 123456, (0.01, -0.5, 1.0), (0.1, -2.0, 3.25)), host_t=5.0)
    assert row[:3] == (5.0, 42, 123456)
    assert row[3:6] == pytest.approx((0.01, -0.5, 1.0), abs=1e-3)
    assert row[6:9] == pytest.approx((0.1, -2.0, 3.25), abs=1e-3)


def test_pack_clamps_out_of_range_values():
    row = parse(pack(0, 0, (40.0, -40.0, 0), (0, 0, 0)), host_t=0)
    assert row[3] == pytest.approx(32.767) and row[4] == pytest.approx(-32.768)


def test_filter_stays_level_when_still():
    f = Madgwick()
    for _ in range(500):
        f.update((0, 0, 0), (0, 0, 1), 0.01)
    assert euler_deg(f.q) == pytest.approx([0, 0, 0], abs=0.1)


def test_filter_integrates_yaw_rate():
    f = Madgwick()
    for _ in range(200):  # 0.5 rad/s for 2 s
        f.update((0, 0, 0.5), (0, 0, 1), 0.01)
    assert np.radians(euler_deg(f.q)[2]) == pytest.approx(1.0, abs=0.01)


def test_filter_finds_tilt_from_gravity():
    f = Madgwick(beta=0.5)
    for _ in range(2000):  # sensor rolled 90 deg: gravity along +y
        f.update((0, 0, 0), (0, 1, 0), 0.01)
    assert euler_deg(f.q)[0] == pytest.approx(90, abs=1.0)


def test_metrics_on_simulated_recording():
    rows = SimDevice(loss=0.01, seed=1).generate(30)
    assert analysis.sample_rate_hz(rows) == pytest.approx(100, rel=0.02)
    assert analysis.packet_loss_pct(rows) == pytest.approx(1.0, abs=0.6)
    assert np.degrees(analysis.gyro_bias(rows)) == pytest.approx([0.4, -0.25, 0.3], abs=0.02)


def test_calibration_reduces_yaw_drift():
    rows = SimDevice(seed=2).generate(30)
    bias = analysis.gyro_bias(rows)
    raw = analysis.yaw_drift_deg_per_min(rows)
    cal = analysis.yaw_drift_deg_per_min(rows, bias)
    assert raw > 10       # 0.3 deg/s z bias is about 18 deg/min
    assert cal < 2
    assert len(run_filter(rows)) == len(rows)
