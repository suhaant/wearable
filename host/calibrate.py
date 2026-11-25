"""Measure gyro bias while the band sits still, save to calibration.json.

  python calibrate.py            # leave the band flat and untouched
  python calibrate.py --sim
"""
import argparse
import json
import time

import numpy as np

from wristcap import analysis, stream

MAX_STILL_NOISE_DPS = 1.0  # above this, the band probably moved during calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()

    print(f"Keep the band flat and still for {args.seconds:.0f} seconds...")
    rows = stream.collect(args.seconds, sim=args.sim)

    noise = analysis.gyro_noise_dps(rows)
    if noise > MAX_STILL_NOISE_DPS:
        raise SystemExit(f"Too much motion (gyro noise {noise:.2f} deg/s). Hold still and rerun.")

    bias = analysis.gyro_bias(rows)
    raw_drift = analysis.yaw_drift_deg_per_min(rows)
    cal_drift = analysis.yaw_drift_deg_per_min(rows, bias)
    cal = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": int(len(rows)),
        "gyro_bias_rads": bias.round(6).tolist(),
        "gyro_bias_dps": np.degrees(bias).round(3).tolist(),
        "gyro_noise_dps": round(noise, 4),
        "accel_norm_error_pct": round(analysis.accel_norm_error_pct(rows), 3),
    }
    with open(stream.CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)

    print(f"Gyro bias (deg/s): {cal['gyro_bias_dps']}")
    print(f"Gyro noise: {noise:.3f} deg/s | accel norm error: {cal['accel_norm_error_pct']:.2f}%")
    print(f"Yaw drift: {raw_drift:.2f} deg/min raw -> {cal_drift:.2f} deg/min calibrated")
    print("Saved calibration.json")


if __name__ == "__main__":
    main()
