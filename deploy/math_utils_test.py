import matplotlib.pyplot as plt
import numpy as np
from gs_env.common.utils.math_utils import cross_correlation


def main() -> None:
    # --- parameters ---
    fs = 100.0  # Hz sampling rate
    dt = 1.0 / fs
    N = 600  # samples
    f0 = 1  # Hz sine frequency
    true_lag = 0.174  # seconds (non-integer multiple of dt)
    noise_std = 0.01  # Gaussian noise
    flipped = False  # allow negative lags

    # --- generate clean sine and delayed version ---
    t = np.arange(N) * dt
    a = np.sin(2 * np.pi * f0 * t)  # reference
    b = np.sin(2 * np.pi * f0 * (t - true_lag))  # delayed
    a_noisy = a + np.random.normal(0, noise_std, size=N)
    b_noisy = b + np.random.normal(0, noise_std, size=N)
    if flipped:
        b_noisy *= -1

    # --- call your cross-correlation ---
    lag_samples = cross_correlation(a_noisy, b_noisy, allow_flip=flipped)
    est_lag = lag_samples * dt

    print(f"True lag: {true_lag:+.4f} s")
    print(f"Estimated lag: {est_lag:+.4f} s  ({lag_samples:+.3f} samples)")
    print(f"Error: {(est_lag - true_lag) * 1e3:+.2f} ms")

    # --- visualize ---
    plt.figure(figsize=(8, 4))
    plt.plot(t, a_noisy, label="Signal a")
    plt.plot(t, b_noisy, label="Signal b (delayed)")
    plt.plot(
        t, np.interp(t + est_lag, t, b_noisy), "--", label=f"b shifted by est. {est_lag:+.3f}s"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Cross-correlation lag test")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
