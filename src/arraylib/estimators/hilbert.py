"""
Delay estimation from the analytic signal.

The Hilbert transform turns a real recording into an analytic signal whose
instantaneous phase and envelope are both directly available.  Two delay
estimators follow from that:

``phase_delay``
    Compare instantaneous phase between channels.  Very precise, but it only
    measures delay modulo one period of the carrier, so it needs a baseline
    shorter than half a wavelength to stay unambiguous.

``envelope_delay``
    Correlate the amplitude envelopes instead.  The envelope varies far more
    slowly than the carrier, so it is unambiguous over long baselines, at the
    cost of much coarser resolution.

Used together they resolve each other's weakness: the envelope fixes the
integer cycle count and the phase refines within it.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ..config import SAMPLE_RATE, SPEED_OF_SOUND
from .gcc_phat import gcc_phat

__all__ = ["analytic", "dominant_frequency", "phase_delay", "envelope_delay",
           "unwrapped_delay"]


def analytic(signals: np.ndarray) -> np.ndarray:
    """Analytic signal of each channel, via the Hilbert transform."""
    return hilbert(np.asarray(signals, dtype=float), axis=0)


def dominant_frequency(signal: np.ndarray,
                       fs: int = SAMPLE_RATE,
                       band: tuple[float, float] | None = None) -> float:
    """
    Frequency of the strongest spectral component, in Hz.

    Phase-based delay needs a carrier frequency to convert phase into time,
    and for rotor noise the sensible choice is the dominant harmonic.
    """
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    if band is not None:
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not mask.any():
            raise ValueError(f"band {band} contains no bin")
        spectrum = np.where(mask, spectrum, 0.0)
    return float(freqs[int(np.argmax(spectrum))])


def phase_delay(sig1: np.ndarray,
                sig2: np.ndarray,
                frequency: float,
                fs: int = SAMPLE_RATE) -> float:
    """
    Delay of ``sig2`` relative to ``sig1`` from mean instantaneous phase.

    Strictly valid only for a narrowband signal.  Across a broadband rotor
    harmonic stack the amplitude-weighted phase mixes contributions from
    several harmonics while the conversion uses a single ``frequency``, which
    leaves a bias of order one sample -- use :func:`unwrapped_delay` or
    :func:`arraylib.estimators.gcc_phat` for broadband sources.

    Args:
        frequency: Carrier frequency used to convert phase to time.

    Returns:
        Delay in seconds, wrapped into +/- half a period of ``frequency``.
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")

    a1, a2 = analytic(sig1), analytic(sig2)

    # Phase of the cross-product averages the phase difference while
    # weighting by amplitude, which is far steadier than differencing two
    # separately unwrapped phase tracks.
    #
    # For a2(t) = a1(t - D) at carrier f, a2 * conj(a1) has phase -2*pi*f*D,
    # so the delay is the negated phase.
    cross = a2 * np.conj(a1)
    phase = float(np.angle(np.sum(cross)))

    return -phase / (2.0 * np.pi * frequency)


def envelope_delay(sig1: np.ndarray,
                   sig2: np.ndarray,
                   fs: int = SAMPLE_RATE,
                   max_tau: float | None = None) -> float:
    """
    Delay of ``sig2`` relative to ``sig1`` from envelope cross-correlation.

    Unambiguous over any baseline, but limited by how fast the envelope
    actually changes -- for a steady rotor tone the envelope is nearly flat
    and this estimate becomes poorly conditioned.
    """
    e1 = np.abs(analytic(sig1))
    e2 = np.abs(analytic(sig2))
    e1 = e1 - e1.mean()
    e2 = e2 - e2.mean()
    return gcc_phat(e1, e2, fs=fs, max_tau=max_tau)[0]


def unwrapped_delay(sig1: np.ndarray,
                    sig2: np.ndarray,
                    frequency: float,
                    baseline: float,
                    fs: int = SAMPLE_RATE,
                    c: float = SPEED_OF_SOUND) -> float:
    """
    Phase-precision delay with the cycle ambiguity resolved geometrically.

    The wrapped phase gives a delay known only to within one period.  The true
    delay cannot exceed ``baseline / c``, so only a handful of candidate cycle
    counts are physically possible; this picks the candidate closest to the
    coarse GCC-PHAT estimate and then keeps the precise phase value.

    Args:
        frequency: Carrier frequency in Hz.
        baseline: Separation of the two elements in metres.

    Returns:
        Delay in seconds.
    """
    period = 1.0 / frequency
    limit = baseline / c

    wrapped = phase_delay(sig1, sig2, frequency, fs)
    coarse = gcc_phat(sig1, sig2, fs=fs, max_tau=1.25 * limit)[0]

    # Candidate cycle counts that keep the delay physically possible.
    k_max = int(np.ceil(limit / period)) + 1
    candidates = wrapped + np.arange(-k_max, k_max + 1) * period
    candidates = candidates[np.abs(candidates) <= limit + 0.5 * period]
    if len(candidates) == 0:
        return coarse

    return float(candidates[int(np.argmin(np.abs(candidates - coarse)))])
