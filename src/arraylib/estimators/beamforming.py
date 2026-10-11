"""
Beamforming DOA estimators: delay-and-sum, MVDR (Capon) and SRP-PHAT.
"""

from __future__ import annotations

import numpy as np

from ..config import BAND_HIGH, BAND_LOW, SAMPLE_RATE, SPEED_OF_SOUND, ScanConfig
from ..geometry import direction_unit_vector, steering_matrix
from ..sources import bandpass
from .gcc_phat import gcc_phat

__all__ = ["delay_and_sum", "mvdr", "srp_phat", "scan_grid"]


def _normalise(power: np.ndarray) -> np.ndarray:
    """
    Map a spatial spectrum onto [0, 1].

    Subtracting the floor as well as scaling the peak matters for the
    low-contrast beamformers: a 126 mm aperture gives delay-and-sum a very
    broad main lobe sitting on a high pedestal, and peak-prominence thresholds
    are meaningless unless that pedestal is removed first.  Applying the same
    transform to every estimator keeps one set of thresholds valid across all
    of them.
    """
    power = np.asarray(power, dtype=float)
    power = power - power.min()
    return power / (power.max() + 1e-20)


def scan_grid(scan: ScanConfig | None = None) -> np.ndarray:
    """Azimuth scan grid in degrees, ``[0, 360)``."""
    scan = scan or ScanConfig()
    return np.arange(0.0, 360.0, scan.azimuth_resolution)


def _frequency_snapshots(signals: np.ndarray,
                         nfft: int,
                         fs: int,
                         band: tuple[float, float],
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Short-time Fourier snapshots for subspace and MVDR processing.

    Returns:
        ``(freqs, X)`` where ``X`` is ``[n_freqs, n_channels, n_snapshots]``
        restricted to the requested band.
    """
    n_samples, n_channels = signals.shape
    nfft = min(nfft, n_samples)
    hop = max(1, nfft // 2)
    starts = range(0, n_samples - nfft + 1, hop)

    window = np.hanning(nfft)[:, None]
    snaps = [np.fft.rfft(signals[s:s + nfft, :] * window, n=nfft, axis=0)
             for s in starts]
    if not snaps:
        raise ValueError(f"need at least {nfft} samples, got {n_samples}")

    X = np.stack(snaps, axis=-1)                       # [freqs, ch, snapshots]
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    keep = (freqs >= band[0]) & (freqs <= band[1])
    if not keep.any():
        raise ValueError(f"band {band} contains no FFT bin at nfft={nfft}")
    return freqs[keep], X[keep]


def delay_and_sum(signals: np.ndarray,
                  positions: np.ndarray,
                  scan: ScanConfig | None = None,
                  fs: int = SAMPLE_RATE,
                  c: float = SPEED_OF_SOUND,
                  band: tuple[float, float] | None = (BAND_LOW, BAND_HIGH),
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic delay-and-sum (Bartlett) beamformer, in the time domain.

    Steers the array by applying the fractional delay each element would see
    for a given look direction, sums, and reports output power.  Wideband and
    completely robust, but its resolution is fixed by the aperture -- about
    35 deg for this 126 mm array at 2 kHz.

    Args:
        band: Pre-filter applied before steering, so this estimator sees the
            same band as the frequency-domain ones.  Without it the array's
            undirectional response below 1 kHz dominates the output power and
            the beamformer collapses towards the centroid of the sources.
            ``None`` disables filtering.

    Returns:
        ``(azimuths, power)`` normalised to ``[0, 1]``.
    """
    if band is not None:
        signals = bandpass(signals, band[0], band[1], fs)
    azimuths = scan_grid(scan)
    directions = direction_unit_vector(azimuths)          # [n_az, 2]
    delays = directions @ positions.T / c * fs            # [n_az, n_ch], samples

    n = signals.shape[0]
    idx = np.arange(n, dtype=float)
    power = np.empty(len(azimuths))

    for a in range(len(azimuths)):
        acc = np.zeros(n)
        for ch in range(signals.shape[1]):
            # Each element received the waveform advanced by path/c, so sample
            # it that much later to bring all channels back into alignment.
            acc += np.interp(idx - delays[a, ch], idx, signals[:, ch],
                             left=0.0, right=0.0)
        acc /= signals.shape[1]
        power[a] = float(np.mean(acc ** 2))

    return azimuths, _normalise(power)


def mvdr(signals: np.ndarray,
         positions: np.ndarray,
         scan: ScanConfig | None = None,
         fs: int = SAMPLE_RATE,
         c: float = SPEED_OF_SOUND,
         nfft: int = 1024,
         band: tuple[float, float] = (BAND_LOW, BAND_HIGH),
         diagonal_loading: float = 0.01,
         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimum-variance distortionless-response (Capon) beamformer.

    Minimises output power subject to unit gain on the look direction, which
    places nulls on interferers and resolves sources far closer together than
    delay-and-sum can.  The price is a covariance inverse per frequency bin:
    with few snapshots the estimate is ill-conditioned, so a fraction
    ``diagonal_loading`` of the mean eigenvalue is added to the diagonal.

    Returns:
        ``(azimuths, power)`` normalised to a peak of 1.
    """
    azimuths = scan_grid(scan)
    freqs, X = _frequency_snapshots(signals, nfft, fs, band)
    n_freqs, n_ch, n_snap = X.shape

    # Per-bin spatial covariance, averaged over snapshots.
    R = np.einsum("fcs,fds->fcd", X, np.conj(X)) / n_snap
    load = diagonal_loading * np.trace(R, axis1=1, axis2=2).real / n_ch
    R += load[:, None, None] * np.eye(n_ch)[None]
    R_inv = np.linalg.inv(R)

    A = steering_matrix(azimuths, freqs, positions, c)     # [n_az, n_freqs, n_ch]

    # power(az) = mean_f 1 / (a^H R^-1 a)
    denom = np.einsum("afc,fcd,afd->af", np.conj(A), R_inv, A).real
    power = np.mean(1.0 / (np.abs(denom) + 1e-20), axis=1)

    return azimuths, _normalise(power)


def srp_phat(signals: np.ndarray,
             positions: np.ndarray,
             pairs: list[tuple[int, int]],
             scan: ScanConfig | None = None,
             fs: int = SAMPLE_RATE,
             c: float = SPEED_OF_SOUND,
             interp: int = 4,
             band: tuple[float, float] | None = (BAND_LOW, BAND_HIGH),
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Steered response power with PHAT weighting.

    Accumulates each pair's GCC-PHAT correlation at the delay a candidate
    direction predicts.  Equivalent to delay-and-sum in the far field, but
    because the correlations are computed once and only *sampled* per
    direction, the angular grid is cheap to refine -- and PHAT whitening makes
    it markedly more robust to coloured source spectra than plain
    delay-and-sum.  This is the estimator that held up best on moving sources.

    Args:
        pairs: Channel pairs to accumulate.  Wider baselines give sharper
            correlation structure, so ``"perimeter"`` pairs outperform
            ``"adjacent"`` ones here despite costing more.
        band: Pre-filter applied before correlation, matching the other
            estimators.  ``None`` disables filtering.

    Returns:
        ``(azimuths, power)`` normalised to ``[0, 1]``.
    """
    if band is not None:
        signals = bandpass(signals, band[0], band[1], fs)
    azimuths = scan_grid(scan)
    directions = direction_unit_vector(azimuths)

    correlations, lag_grids = [], []
    for i, j in pairs:
        _, lags, corr = gcc_phat(signals[:, i], signals[:, j], fs=fs,
                                 interp=interp)
        correlations.append(corr)
        lag_grids.append(lags)

    power = np.zeros(len(azimuths))
    for a, direction in enumerate(directions):
        tau = positions @ direction / c                 # [n_ch]
        total = 0.0
        for k, (i, j) in enumerate(pairs):
            predicted = tau[i] - tau[j]
            # Interpolate the correlation at the predicted lag.
            total += float(np.interp(predicted, lag_grids[k], correlations[k]))
        power[a] = total

    return azimuths, _normalise(power)
