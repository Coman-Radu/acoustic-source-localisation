"""
Source waveforms: recorded drone audio and synthetic stand-ins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample, sosfiltfilt

from .config import BAND_HIGH, BAND_LOW, SAMPLE_RATE

__all__ = ["bandpass", "load_wav", "harmonic_drone", "tone"]


def bandpass(x: np.ndarray,
             f_low: float = BAND_LOW,
             f_high: float = BAND_HIGH,
             fs: int = SAMPLE_RATE,
             order: int = 4) -> np.ndarray:
    """
    Zero-phase band-pass filter along the first axis.

    Uses ``sosfiltfilt`` so the filter adds no group delay -- important here,
    because every estimator in this project measures inter-channel delay and a
    phase-distorting filter would bias the result.  The upper edge should stay
    below the array's spatial Nyquist (see
    :func:`arraylib.config.spatial_nyquist_frequency`).
    """
    nyq = fs / 2.0
    f_high = min(f_high, 0.99 * nyq)
    if not 0 < f_low < f_high:
        raise ValueError(f"need 0 < f_low < f_high <= {nyq}, "
                         f"got {f_low} and {f_high}")
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def load_wav(path: str | Path,
             duration: float | None = None,
             fs: int = SAMPLE_RATE,
             band: tuple[float, float] | None = (BAND_LOW, BAND_HIGH),
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a WAV file as a mono float signal, resampled to ``fs``.

    Args:
        path: WAV file to read.
        duration: Target length in seconds.  A shorter file is tiled, a longer
            one truncated.  ``None`` keeps the native length.
        fs: Target sample rate in Hz.
        band: ``(low, high)`` band-pass applied after resampling, or ``None``
            to skip filtering.

    Returns:
        ``(t, signal)`` with ``t`` in seconds and ``signal`` peak-normalised.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The recorded drone audio is not committed to "
            f"this repository -- see data/README.md for how to obtain it.")

    sr, data = wavfile.read(path)

    # Integer PCM -> float in [-1, 1]; floating-point WAVs are already scaled.
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(abs(np.iinfo(data.dtype).min))
    else:
        data = data.astype(np.float64)

    if data.ndim > 1:            # take the first channel of a multichannel file
        data = data[:, 0]

    if sr != fs:
        data = resample(data, int(round(len(data) * fs / sr)))

    if duration is not None:
        want = int(round(duration * fs))
        if len(data) < want:
            data = np.tile(data, want // len(data) + 1)[:want]
        else:
            data = data[:want]

    if band is not None:
        data = bandpass(data, band[0], band[1], fs)

    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    return np.arange(len(data)) / fs, data


def harmonic_drone(duration: float,
                   fs: int = SAMPLE_RATE,
                   f0: float = 180.0,
                   n_harmonics: int = 24,
                   am_depth: float = 0.15,
                   am_rate: float = 3.0,
                   seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic drone-like source: a harmonic stack with slow amplitude modulation.

    Rotor noise is dominated by the blade-passing frequency and its harmonics,
    with beating between rotors turning up as amplitude modulation.  This is a
    deterministic stand-in for the recorded audio, so experiments stay runnable
    without the dataset.

    The default harmonic count is deliberately high: with ``f0 = 180 Hz`` a
    stack of 24 harmonics reaches 4.3 kHz, so real energy lands inside the
    1-3.5 kHz band where this aperture actually has directivity.  A short stack
    puts everything below 1 kHz, where the array is effectively blind and every
    estimator degrades for reasons that have nothing to do with the estimator.

    Returns:
        ``(t, signal)``, peak-normalised.
    """
    t = np.arange(int(round(duration * fs))) / fs
    amps = np.array([1.0 / (k ** 0.8) for k in range(1, n_harmonics + 1)])
    sig = sum(a * np.sin(2 * np.pi * k * f0 * t)
              for k, a in zip(range(1, n_harmonics + 1), amps))
    sig *= 1.0 + am_depth * np.sin(2 * np.pi * am_rate * t)

    if seed is not None:
        sig = sig + 0.01 * np.random.default_rng(seed).standard_normal(len(t))

    return t, sig / (np.max(np.abs(sig)) + 1e-12)


def tone(duration: float,
         frequency: float = 3000.0,
         fs: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Pure sinusoid, used to verify estimators against closed-form answers."""
    t = np.arange(int(round(duration * fs))) / fs
    return t, np.sin(2 * np.pi * frequency * t)
