"""Live 3D view of the band's orientation. Press 'z' to zero the heading.

  python visualize.py
  python visualize.py --sim --motion
"""
import argparse
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from wristcap import stream
from wristcap.fusion import Madgwick, euler_deg, quat_mult, quat_to_matrix

# A band-shaped box: long along x (forearm), wide along y, thin along z.
L, W, H = 2.0, 1.2, 0.3
CORNERS = np.array([[sx * L, sy * W, sz * H] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--motion", action="store_true")
    args = ap.parse_args()

    bias = stream.gyro_bias_from_cal()
    filt = Madgwick()
    state = {"q": filt.q.copy(), "dev_t": None, "n": 0, "yaw0": 0.0}
    lock = threading.Lock()
    stop = threading.Event()

    def on_row(row):
        dev_t = row[2]
        dt = 0.01 if state["dev_t"] is None else ((dev_t - state["dev_t"]) % (1 << 32)) / 1e6
        state["dev_t"] = dev_t
        q = filt.update(np.array(row[6:9]) - bias, row[3:6], dt)
        with lock:
            state["q"] = q.copy()
            state["n"] += 1

    worker = threading.Thread(target=stream.run, kwargs=dict(
        on_row=on_row, sim=args.sim, motion=args.motion, stop_event=stop), daemon=True)
    worker.start()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-1.5, 1.5)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (toward hand)")
    lines = [ax.plot([], [], [], lw=2, color="tab:blue")[0] for _ in EDGES]
    arrow = ax.plot([], [], [], lw=3, color="tab:orange")[0]  # points toward the hand
    counter = {"t": time.time(), "n": 0, "hz": 0.0}

    def display_quat():
        with lock:
            q = state["q"].copy()
        half = -np.radians(state["yaw0"]) / 2
        return quat_mult(np.array([np.cos(half), 0, 0, np.sin(half)]), q)

    def on_key(event):
        if event.key == "z":
            with lock:
                state["yaw0"] = euler_deg(state["q"])[2]

    def update(_):
        q = display_quat()
        pts = CORNERS @ quat_to_matrix(q).T
        for line, (i, j) in zip(lines, EDGES):
            line.set_data_3d(pts[[i, j], 0], pts[[i, j], 1], pts[[i, j], 2])
        tip = quat_to_matrix(q) @ np.array([1.4, 0, 0])
        arrow.set_data_3d([0, tip[0]], [0, tip[1]], [0, tip[2]])
        now = time.time()
        if now - counter["t"] >= 1.0:
            with lock:
                n = state["n"]
            counter["hz"] = (n - counter["n"]) / (now - counter["t"])
            counter["t"], counter["n"] = now, n
        r, p, y = euler_deg(q)
        ax.set_title(f"{counter['hz']:.0f} Hz | roll {r:5.0f}  pitch {p:5.0f}  yaw {y:5.0f}  (z = zero yaw)")
        return lines + [arrow]

    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)  # noqa: F841
    plt.show()
    stop.set()


if __name__ == "__main__":
    main()
