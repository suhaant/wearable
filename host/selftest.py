"""Automated pass/fail check on a stationary recording. Exit code 0 = PASS.

  python selftest.py                 # 60 s, band flat and still
  python selftest.py --sim --seconds 20
"""
import argparse
import json
import sys
import time

from wristcap import analysis, stream
from wristcap.protocol import SAMPLE_HZ

# (name, unit, threshold, passes-if)
CHECKS = [
    ("sample_rate", "Hz", SAMPLE_HZ * 0.95, lambda v, th: v >= th),
    ("packet_loss", "%", 1.0, lambda v, th: v < th),
    ("max_arrival_gap", "ms", 100.0, lambda v, th: v < th),
    ("gyro_noise", "deg/s", 0.5, lambda v, th: v < th),
    ("accel_norm_error", "%", 5.0, lambda v, th: abs(v) < th),
    ("yaw_drift_calibrated", "deg/min", 2.0, lambda v, th: v < th),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()

    cal = stream.load_calibration()
    if cal is None:
        print("No calibration.json found. Run calibrate.py first for a fair drift number.")
    bias = stream.gyro_bias_from_cal()

    print(f"Self-test: keep the band flat and still for {args.seconds:.0f} seconds...")
    rows = stream.collect(args.seconds, sim=args.sim)

    measured = {
        "sample_rate": analysis.sample_rate_hz(rows),
        "packet_loss": analysis.packet_loss_pct(rows),
        "max_arrival_gap": analysis.arrival_gaps_ms(rows)["max"],
        "gyro_noise": analysis.gyro_noise_dps(rows),
        "accel_norm_error": analysis.accel_norm_error_pct(rows),
        "yaw_drift_calibrated": analysis.yaw_drift_deg_per_min(rows, bias),
    }
    raw_drift = analysis.yaw_drift_deg_per_min(rows)

    print(f"\n{'check':<22}{'value':>10}  {'limit':>12}  result")
    print("-" * 54)
    results, all_pass = {}, True
    for name, unit, th, ok in CHECKS:
        v = measured[name]
        passed = bool(ok(v, th))
        all_pass &= passed
        op = ">=" if name == "sample_rate" else "<"
        print(f"{name:<22}{v:>10.2f}  {op} {th:>6.1f} {unit:<7} {'PASS' if passed else 'FAIL'}")
        results[name] = {"value": round(v, 3), "limit": th, "unit": unit, "pass": passed}
    print("-" * 54)
    print(f"yaw drift uncalibrated: {raw_drift:.2f} deg/min (for comparison)")
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": args.seconds,
              "samples": int(len(rows)), "simulated": args.sim, "overall_pass": all_pass,
              "yaw_drift_uncalibrated": round(raw_drift, 3), "checks": results}
    with open("selftest_report.json", "w") as f:
        json.dump(report, f, indent=2)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
