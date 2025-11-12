import time

import numpy as np


def test_logging_rate(hz: int = 1000, duration: float = 3.0) -> np.ndarray:
    dt = 1.0 / hz
    next_t = time.perf_counter()
    stamps = []

    end_time = next_t + duration

    while True:
        now = time.perf_counter()
        if now >= end_time:
            break

        # catch-up: log exact timing
        while now >= next_t:
            stamps.append(time.perf_counter())
            next_t += dt

        # yield without sleeping too long
        time.sleep(0)

    stamps = np.array(stamps)
    dts = np.diff(stamps)

    print(f"\nRequested rate: {hz} Hz")
    print(f"Actual rate:    {1 / np.mean(dts):.2f} Hz")
    print(f"Samples:        {len(stamps)}")
    print(f"dt mean:        {np.mean(dts) * 1e6:.1f} µs")
    print(f"dt std:         {np.std(dts) * 1e6:.1f} µs")
    print(f"dt min:         {np.min(dts) * 1e6:.1f} µs")
    print(f"dt max:         {np.max(dts) * 1e6:.1f} µs (jitter)")
    print(f"missed ticks:   {np.sum(dts > 2 * dt):d}")

    return dts


# Example: Test 1 kHz
dts = test_logging_rate(1000, 5.0)
