"""Live 3D forearm + hand that mirrors the band's orientation.
Press 'z' to zero the heading (point your arm forward first).

  python visualize.py
  python visualize.py --sim --motion
"""
import argparse
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from wristcap import stream
from wristcap.fusion import Madgwick, euler_deg, quat_mult, quat_to_matrix

# Sensor frame: x = toward the hand, y = thumb side, z = back of the hand.
# The model rotates around the band (where the sensor actually is), keeping it centered.
SKIN, BAND, NAIL = "#e0ac8a", "#2b2d42", "#c98f6f"
QUAD_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float)


def tapered(x0, x1, hw0, hh0, hw1, hh1):
    """Box whose cross-section shrinks from (hw0, hh0) at x0 to (hw1, hh1) at x1."""
    pts = []
    for x, hw, hh in ((x0, hw0, hh0), (x1, hw1, hh1)):
        for y in (-hw, hw):
            for z in (-hh, hh):
                pts.append([x, y, z])
    return np.array(pts, dtype=float)


def build_arm():
    """List of (8 corner points, color) making up the forearm and hand."""
    parts = [
        (tapered(0.0, 2.6, 0.55, 0.40, 0.38, 0.24), SKIN),   # forearm
        (box(2.15, 2.55, -0.42, 0.42, -0.28, 0.32), BAND),   # the band itself
        (box(2.6, 3.5, -0.42, 0.42, -0.13, 0.13), SKIN),     # palm
        (box(2.75, 3.45, 0.42, 0.62, -0.11, 0.07), SKIN),    # thumb base
        (box(3.45, 3.85, 0.45, 0.62, -0.10, 0.06), NAIL),    # thumb tip
    ]
    for y, length in ((0.31, 0.62), (0.10, 0.78), (-0.11, 0.74), (-0.31, 0.58)):  # index..pinky
        parts.append((box(3.5, 3.5 + length, y - 0.085, y + 0.085, -0.09, 0.09), SKIN))
    return parts


PIVOT = np.array([2.35, 0.0, 0.0])  # center of the band
ARM = [(c - PIVOT, col) for c, col in build_arm()]


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

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    lim = 2.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-120)
    ax.set_axis_off()

    # Floor grid for depth cues
    g = np.linspace(-lim, lim, 9)
    for v in g:
        ax.plot([v, v], [-lim, lim], [-lim, -lim], color="0.85", lw=0.6)
        ax.plot([-lim, lim], [v, v], [-lim, -lim], color="0.85", lw=0.6)

    meshes = []
    for corners, color in ARM:
        poly = Poly3DCollection([corners[list(f)] for f in QUAD_FACES],
                                facecolor=color, edgecolor="#5c4033", linewidths=0.4)
        ax.add_collection3d(poly)
        meshes.append((poly, corners))

    counter = {"t": time.time(), "n": 0, "hz": 0.0}
    title = fig.suptitle("", fontsize=12)

    def display_quat():
        with lock:
            q = state["q"].copy()
            yaw0 = state["yaw0"]
        half = -np.radians(yaw0) / 2
        return quat_mult(np.array([np.cos(half), 0, 0, np.sin(half)]), q)

    def on_key(event):
        if event.key == "z":
            with lock:
                state["yaw0"] = euler_deg(state["q"])[2]

    def update(_):
        q = display_quat()
        R = quat_to_matrix(q)
        for poly, corners in meshes:
            pts = corners @ R.T
            poly.set_verts([pts[list(f)] for f in QUAD_FACES])
        now = time.time()
        if now - counter["t"] >= 1.0:
            with lock:
                n = state["n"]
            counter["hz"] = (n - counter["n"]) / (now - counter["t"])
            counter["t"], counter["n"] = now, n
        r, p, y = euler_deg(q)
        title.set_text(f"WristCap live  |  {counter['hz']:.0f} Hz  |  "
                       f"roll {r:5.0f}°  pitch {p:5.0f}°  yaw {y:5.0f}°   (z = zero heading)")
        return [m[0] for m in meshes]

    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)  # noqa: F841
    plt.show()
    stop.set()


if __name__ == "__main__":
    main()
