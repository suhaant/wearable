"""Record a session to CSV with live orientation.

  python receive.py --label cup_pickup --duration 20
  python receive.py --sim --motion           # no hardware needed
"""
import argparse
import csv
import os
import time

import numpy as np

from wristcap import analysis, stream
from wristcap.fusion import Madgwick, euler_deg
from wristcap.protocol import COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C)")
    ap.add_argument("--label", default="session", help="task name, used in the filename")
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of BLE")
    ap.add_argument("--motion", action="store_true", help="simulator moves instead of sitting still")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", f"{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    bias = stream.gyro_bias_from_cal()
    filt = Madgwick()
    rows, last = [], {"dev_t": None, "print": time.time()}

    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(COLUMNS + ["qw", "qx", "qy", "qz"])

    def on_row(row):
        dev_t = row[2]
        dt = 0.01 if last["dev_t"] is None else ((dev_t - last["dev_t"]) % (1 << 32)) / 1e6
        last["dev_t"] = dev_t
        q = filt.update(np.array(row[6:9]) - bias, row[3:6], dt)
        writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row] + [f"{v:.5f}" for v in q])
        rows.append(row)
        now = time.time()
        if now - last["print"] >= 1.0:
            r, p, y = euler_deg(q)
            arr = np.array(rows[-200:])
            print(f"{len(rows):6d} samples | {analysis.sample_rate_hz(arr):5.1f} Hz | "
                  f"roll {r:6.1f}  pitch {p:6.1f}  yaw {y:6.1f}")
            last["print"] = now

    try:
        stream.run(on_row, args.duration, sim=args.sim, motion=args.motion)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()

    if len(rows) > 10:
        arr = np.array(rows)
        print(f"\nSaved {len(rows)} samples to {path}")
        print(f"Rate {analysis.sample_rate_hz(arr):.1f} Hz | loss {analysis.packet_loss_pct(arr):.2f}% | "
              f"max gap {analysis.arrival_gaps_ms(arr)['max']:.0f} ms")


if __name__ == "__main__":
    main()
