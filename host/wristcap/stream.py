"""Data sources: the real board over BLE, or the simulator. Both call on_row(row)."""
import asyncio
import json
import os
import time

import numpy as np

from .protocol import DATA_UUID, DEVICE_NAME, parse

CAL_FILE = os.path.join(os.path.dirname(__file__), "..", "calibration.json")


async def _wait(duration, stop_event):
    if duration:
        await asyncio.sleep(duration)
    elif stop_event is not None:
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
    else:
        await asyncio.Event().wait()  # until Ctrl+C


async def ble_stream(on_row, duration=None, stop_event=None, name=DEVICE_NAME):
    from bleak import BleakClient, BleakScanner

    print(f"Scanning for {name}...")
    device = await BleakScanner.find_device_by_name(name, timeout=15.0)
    if device is None:
        raise SystemExit(f"{name} not found. Is the board powered and advertising? "
                         "(Serial monitor should say 'Advertising as WristCap'.)")
    async with BleakClient(device) as client:
        print(f"Connected to {device.address}")
        await client.start_notify(DATA_UUID, lambda _, data: on_row(parse(bytes(data), time.time())))
        try:
            await _wait(duration, stop_event)
        finally:
            if client.is_connected:
                await client.stop_notify(DATA_UUID)


async def sim_stream(on_row, duration=None, stop_event=None, motion=False):
    from .sim import SimDevice

    sim = SimDevice(motion=motion, seed=int(time.time()))
    print("Using simulated WristCap")
    period = 1.0 / sim.rate
    start = next_t = time.perf_counter()
    while True:
        if duration and time.perf_counter() - start >= duration:
            break
        if stop_event is not None and stop_event.is_set():
            break
        pkt = sim.step()
        if pkt is not None:
            on_row(parse(pkt, time.time()))
        next_t += period
        await asyncio.sleep(max(0.0, next_t - time.perf_counter()))


def run(on_row, duration=None, sim=False, motion=False, stop_event=None):
    coro = (sim_stream(on_row, duration, stop_event, motion=motion) if sim
            else ble_stream(on_row, duration, stop_event))
    asyncio.run(coro)


def collect(duration, sim=False):
    rows = []
    run(rows.append, duration, sim=sim)
    if len(rows) < 10:
        raise SystemExit("Received almost no data. Check the connection and try again.")
    return np.array(rows, dtype=float)


def load_calibration():
    try:
        with open(CAL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def gyro_bias_from_cal():
    cal = load_calibration()
    return np.array(cal["gyro_bias_rads"]) if cal else np.zeros(3)
