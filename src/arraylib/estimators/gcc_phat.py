"""
Generalised cross-correlation with phase transform (GCC-PHAT).
"""

from __future__ import annotations

import numpy as np

from ..config import SAMPLE_RATE, SPEED_OF_SOUND

__all__ = ["gcc_phat", "pairwise_delays", "parabolic_peak"]


def parabolic_peak(y: np.ndarray, k: int) -> float:
    """
    Sub-sample peak location by fitting a parabola through three points.

    The correlation peak rarely lands on an exact sample.  For the 42 mm
    baselines used here a whole sample at 48 kHz is already ~10 deg of
    azimuth, so interpolating the peak is what makes delay-based estimation
    usable at all.

    Returns:
        Offset in samples from ``k``, in ``[-0.5, 0.5]``.
    """
    if k <= 0 or k >= len(y) - 1:
        return 0.0
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = a - 2.0 * b + c
    if denom == 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def gcc_phat(sig1: np.ndarray,
             sig2: np.ndarray,
             fs: int = SAMPLE_RATE,
             max_tau: float | None = None,
             interp: int = 4,
             floor: float = 1e-3,
             ) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the delay of ``sig2`` relative to ``sig1``.

    PHAT weighting divides the cross-spectrum by its own magnitude, keeping
    only phase.  That makes the correlation peak sharp and largely independent
    of the source spectrum, which matters for drone noise: a plain
    cross-correlation is dominated by whichever rotor harmonic happens to be
    loudest and smears badly in reverberation.

    The magnitude floor is *relative* to the strongest bin.  Whitening with an
    absolute floor is the standard trap here: a rotor-harmonic spectrum leaves
    most bins essentially empty, and dividing those by a tiny constant lifts
    pure numerical noise to the same unit magnitude as real signal, which
    buries the correlation peak entirely.  Scaling the floor to the peak
    magnitude instead leaves empty bins suppressed.

    Args:
        sig1: Reference channel, ``[n]``.
        sig2: Delayed channel, ``[n]``.
        fs: Sample rate in Hz.
        max_tau: Largest plausible delay in seconds.  Lags beyond this are
            discarded, which removes most spurious peaks -- pass
            ``baseline / c`` for the pair.
        interp: Zero-padding factor; higher gives a finer lag grid.
        floor: Magnitude floor as a fraction of the strongest bin.

    Returns:
        ``(tau, lags, correlation)`` with ``tau`` in seconds, positive when
        ``sig2`` arrives later than ``sig1``; ``lags`` in seconds, ascending.
    """
    n = len(sig1)
    if len(sig2) != n:
        raise ValueError(f"signals differ in length: {n} vs {len(sig2)}")

    nfft = int(2 ** np.ceil(np.log2(2 * n * interp)))
    S1 = np.fft.rfft(sig1, n=nfft)
    S2 = np.fft.rfft(sig2, n=nfft)

    # Ordering S2 * conj(S1) puts the peak at a POSITIVE lag when sig2 is the
    # later of the two, so the returned lag axis reads naturally.
    cross = S2 * np.conj(S1)
    magnitude = np.abs(cross)
    cross = cross / (magnitude + floor * (magnitude.max() + 1e-300))

    corr = np.fft.fftshift(np.fft.irfft(cross, n=nfft))
    lags = (np.arange(nfft) - nfft // 2) / float(fs)

    search = np.ones(nfft, dtype=bool)
    if max_tau is not None:
        search = np.abs(lags) <= max_tau
        if not search.any():
            raise ValueError(f"max_tau={max_tau}s excludes every lag")

    masked = np.where(search, corr, -np.inf)
    k = int(np.argmax(masked))
    tau = lags[k] + parabolic_peak(corr, k) / float(fs)
    return float(tau), lags, corr


def pairwise_delays(signals: np.ndarray,
                    pairs: list[tuple[int, int]],
                    positions: np.ndarray,
                    fs: int = SAMPLE_RATE,
                    c: float = SPEED_OF_SOUND,
                    margin: float = 1.25,
                    ) -> np.ndarray:
    """
    Measured delay for each channel pair.

    Each pair's search window is limited to its own baseline divided by the
    speed of sound (times ``margin``), since no plane wave can produce a
    larger delay across that pair.

    Args:
        signals: ``[n_samples, n_channels]``.
        pairs: ``(i, j)`` channel pairs.
        positions: ``[n_channels, 2]`` element positions in metres.
        margin: Slack on the geometric bound, to tolerate noise.

    Returns:
        ``[n_pairs]`` delays in seconds, positive when ``j`` arrives later.
    """
    out = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        baseline = float(np.linalg.norm(positions[i] - positions[j]))
        out[k] = gcc_phat(signals[:, i], signals[:, j], fs=fs,
                          max_tau=margin * baseline / c)[0]
    return out
