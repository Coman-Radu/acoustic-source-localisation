"""
Forward models for what the array receives.

Two propagation models are provided:

``simulate_stationary``
    Far-field plane wave with a fractional sample delay per element.  Valid
    when the source is many apertures away and effectively still over the
    analysis window.

``simulate_moving``
    Doppler time-scaling model, ``x_m(t) = s(alpha(t) * t + tau_m) + n_m(t)``.
    A moving source does not merely shift the waveform, it warps the time
    axis, and that warp is what breaks delay-based estimators.  This is the
    model the project's Doppler study is built on.
"""

from __future__ import annotations

import numpy as np

from .config import CHANNELS, SAMPLE_RATE, SPEED_OF_SOUND
from .geometry import HARDWARE_ORDER, direction_unit_vector, positions_array

__all__ = [
    "add_noise_at_snr",
    "simulate_stationary",
    "simulate_multi_stationary",
    "doppler_factor",
    "simulate_moving",
    "compensate_time_scaling",
]


def add_noise_at_snr(signals: np.ndarray,
                     snr_db: float,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add white Gaussian noise at a given broadband SNR.

    SNR is referenced to the mean power of ``signals`` as a whole, so every
    channel receives noise of the same variance -- matching independent
    self-noise in each preamplifier.
    """
    rng = rng or np.random.default_rng()
    signal_power = float(np.mean(signals ** 2))
    if signal_power <= 0:
        return signals
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return signals + rng.normal(scale=np.sqrt(noise_power), size=signals.shape)


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Delay a signal by a (possibly fractional) number of samples.

    Linear interpolation is used.  At 48 kHz one sample is 7.1 mm of path
    length, and the whole array is only 126 mm across, so sub-sample accuracy
    is essential -- rounding to whole samples would quantise azimuth into
    steps of roughly 10 deg.
    """
    n = len(signal)
    idx = np.arange(n, dtype=float) - delay_samples
    return np.interp(idx, np.arange(n, dtype=float), signal, left=0.0, right=0.0)


def simulate_stationary(source_signal: np.ndarray,
                        azimuth_deg: float,
                        distance: float = 10.0,
                        snr_db: float | None = 10.0,
                        fs: int = SAMPLE_RATE,
                        c: float = SPEED_OF_SOUND,
                        order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                        spherical_spreading: bool = True,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Array response to one stationary far-field source.

    Args:
        source_signal: ``[n_samples]`` emitted waveform.
        azimuth_deg: True source azimuth, compass convention.
        distance: Range to the array centre, in metres.
        snr_db: Broadband SNR; ``None`` for a noiseless simulation.
        fs: Sample rate in Hz.
        c: Speed of sound in m/s.
        order: Channel layout to simulate.
        spherical_spreading: Apply the 1/r amplitude taper across the array.

    Returns:
        ``[n_samples, CHANNELS]`` array signals.
    """
    positions = positions_array(order=order)
    direction = direction_unit_vector(azimuth_deg)

    # Projection of each element onto the look direction. An element displaced
    # *towards* the source sits closer to it by this much, so it receives the
    # waveform EARLIER -- hence the negative delay below.
    path = positions @ direction                      # [CHANNELS], metres
    delays = -path / c * fs                           # samples, advance if path > 0

    out = np.zeros((len(source_signal), CHANNELS))
    for ch in range(CHANNELS):
        delayed = _fractional_delay(source_signal, delays[ch])
        if spherical_spreading:
            # Range to this element, shorter for elements towards the source.
            out[:, ch] = delayed * (distance / (distance - path[ch]))
        else:
            out[:, ch] = delayed

    if snr_db is not None:
        out = add_noise_at_snr(out, snr_db, rng)
    return out


def simulate_multi_stationary(source_signal: np.ndarray,
                              azimuths_deg: list[float] | np.ndarray,
                              distance: float = 10.0,
                              snr_db: float | None = 10.0,
                              decorrelate: bool = True,
                              fs: int = SAMPLE_RATE,
                              c: float = SPEED_OF_SOUND,
                              order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                              rng: np.random.Generator | None = None,
                              ) -> np.ndarray:
    """
    Superposition of several stationary sources.

    Args:
        decorrelate: Give each source an independent circular time shift.
            Subspace methods such as MUSIC and ESPRIT assume the sources are
            not mutually coherent; feeding them identical copies of one
            waveform makes the signal covariance rank-deficient and the
            estimate collapses.  Leave this on unless you are deliberately
            studying coherent sources.
    """
    rng = rng or np.random.default_rng()
    total = np.zeros((len(source_signal), CHANNELS))

    for i, az in enumerate(azimuths_deg):
        sig = source_signal
        if decorrelate:
            sig = np.roll(source_signal, int(rng.integers(0, len(source_signal))))
        total += simulate_stationary(sig, az, distance, snr_db=None, fs=fs,
                                     c=c, order=order)

    if snr_db is not None:
        total = add_noise_at_snr(total, snr_db, rng)
    return total


def doppler_factor(radial_velocity: np.ndarray | float,
                   c: float = SPEED_OF_SOUND) -> np.ndarray | float:
    """
    Doppler time-scaling factor ``alpha = c / (c - v_r)``.

    ``v_r`` is positive when the source approaches.  ``alpha > 1`` compresses
    the waveform in time (raising pitch), ``alpha < 1`` stretches it.
    """
    v = np.asarray(radial_velocity, dtype=float)
    if np.any(v >= c):
        raise ValueError("radial velocity must stay below the speed of sound")
    return c / (c - v)


def simulate_moving(source_fn,
                    duration: float,
                    azimuth_deg: float,
                    radial_velocity,
                    distance: float = 10.0,
                    snr_db: float | None = 10.0,
                    fs: int = SAMPLE_RATE,
                    c: float = SPEED_OF_SOUND,
                    order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                    rng: np.random.Generator | None = None,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Array response to a source with time-varying radial velocity.

    Implements ``x_m(t) = s(alpha(t) * t + tau_m)``: the emitted waveform is
    evaluated on a warped time axis, so the Doppler shift falls out of the
    model rather than being imposed as a separate frequency offset.  Applying
    a frequency shift *and* a retarded time would double-count the effect.

    Args:
        source_fn: Callable mapping an array of (warped) times to waveform
            samples -- the emitted signal ``s(u)``.
        duration: Capture length in seconds.
        azimuth_deg: Source azimuth, held constant (far-field, small motion
            across the aperture).
        radial_velocity: Scalar m/s, or a callable ``v_r(t)`` for an
            accelerating source.  Positive means approaching.
        distance: Range to the array centre in metres.
        snr_db: Broadband SNR; ``None`` for noiseless.
        order: Channel layout to simulate.

    Returns:
        ``(t, signals, alpha)`` -- time vector, ``[n_samples, CHANNELS]``
        signals, and the Doppler factor at each instant.
    """
    t = np.arange(int(round(duration * fs))) / fs

    v_r = radial_velocity(t) if callable(radial_velocity) \
        else np.full_like(t, float(radial_velocity))
    alpha = np.asarray(doppler_factor(v_r, c), dtype=float)

    positions = positions_array(order=order)
    direction = direction_unit_vector(azimuth_deg)
    # Advance for elements towards the source, matching simulate_stationary.
    tau = positions @ direction / c                   # [CHANNELS], seconds

    out = np.zeros((len(t), CHANNELS))
    for ch in range(CHANNELS):
        out[:, ch] = source_fn(alpha * t + tau[ch])

    if snr_db is not None:
        out = add_noise_at_snr(out, snr_db, rng)
    return t, out, alpha


def compensate_time_scaling(signals: np.ndarray,
                            alpha: np.ndarray,
                            fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Undo Doppler time-scaling by resampling onto a uniform emission-time grid.

    Integrating the Doppler factor gives the warped time ``u(t) = int alpha``;
    inverting that mapping and resampling every channel onto a uniform ``u``
    grid restores a common waveform across the array, which is what delay-based
    estimators assume.  The same warp is applied to all channels, so the
    inter-channel delays that carry the direction information are preserved.

    Args:
        signals: ``[n_samples, n_channels]`` received signals.
        alpha: ``[n_samples]`` Doppler factor used (or estimated).
        fs: Sample rate in Hz.

    Returns:
        ``[n_samples, n_channels]`` compensated signals.
    """
    n = signals.shape[0]
    if len(alpha) != n:
        raise ValueError(f"alpha has {len(alpha)} samples, signals have {n}")

    t = np.arange(n) / fs
    u = np.cumsum(alpha) / fs                 # warped time, monotonic
    u_uniform = np.arange(n) / fs
    t_at_u = np.interp(u_uniform, u, t, left=t[0], right=t[-1])

    out = np.empty_like(signals)
    for ch in range(signals.shape[1]):
        out[:, ch] = np.interp(t_at_u, t, signals[:, ch])
    return out
