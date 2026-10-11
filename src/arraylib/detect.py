"""
Activity detection and peak picking on a spatial spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .config import ScanConfig

__all__ = ["sound_present", "Peak", "find_sources", "angular_error",
           "refine_peak"]


def refine_peak(spectrum: np.ndarray, index: int) -> float:
    """
    Sub-grid offset of a spectral peak, in grid steps.

    Fits a parabola through the peak and its two neighbours, wrapping around
    the circular azimuth axis.  Without this the best any scanning estimator
    can report is the nearest grid point, which puts a hard floor under its
    error -- on a 2 deg grid that floor is 1 deg, large enough to hide the
    entire effect of SNR on accuracy.

    Returns:
        Offset in grid steps, clipped to ``[-0.5, 0.5]``.
    """
    n = len(spectrum)
    a = float(spectrum[(index - 1) % n])
    b = float(spectrum[index])
    c = float(spectrum[(index + 1) % n])
    denom = a - 2.0 * b + c
    if denom == 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def sound_present(signals: np.ndarray, threshold: float = 0.002) -> bool:
    """
    Cheap RMS gate, used to skip DOA estimation on silent frames.

    Args:
        signals: ``[n_samples]`` or ``[n_samples, n_channels]``.
        threshold: RMS level below which the frame counts as silent.
    """
    return bool(np.sqrt(np.mean(np.asarray(signals) ** 2)) > threshold)


class Peak(dict):
    """
    One detected source: azimuth in degrees, normalised power, SNR in dB.

    Subclasses ``dict`` so results stay trivially serialisable to JSON while
    still allowing attribute access.
    """

    __slots__ = ()

    def __init__(self, azimuth: float, power: float, snr_db: float):
        super().__init__(azimuth=float(azimuth), power=float(power),
                         snr_db=float(snr_db))

    azimuth = property(lambda self: self["azimuth"])
    power = property(lambda self: self["power"])
    snr_db = property(lambda self: self["snr_db"])

    def __repr__(self) -> str:
        return (f"Peak(azimuth={self.azimuth:.1f}, power={self.power:.3f}, "
                f"snr_db={self.snr_db:.1f})")


def find_sources(azimuths: np.ndarray,
                 spectrum: np.ndarray,
                 scan: ScanConfig | None = None,
                 max_sources: int | None = None,
                 refine: bool = True,
                 ) -> tuple[list[Peak], float]:
    """
    Pick source directions out of a spatial spectrum.

    A peak is accepted only if it clears an absolute height, stands out from
    its surroundings by ``min_prominence``, sits at least ``min_separation``
    away from a stronger peak, and exceeds the noise floor by
    ``snr_threshold_db``.  The noise floor is taken as a low percentile of the
    spectrum itself, which is robust as long as sources occupy a minority of
    the scanned directions.

    The azimuth grid is treated as circular, so a peak straddling 0/360 deg is
    still found: the spectrum is tiled three times and peaks are taken from the
    middle copy.  Padding by only a few samples is not enough -- a broad main
    lobe then runs into the array edge, its base is truncated, and scipy
    computes a prominence near zero for what is actually the global maximum.

    Args:
        azimuths: ``[n]`` scan grid in degrees, uniformly spaced.
        spectrum: ``[n]`` spatial power, any positive scaling.
        scan: Thresholds; defaults to :class:`ScanConfig`.
        max_sources: Keep at most this many, strongest first.
        refine: Interpolate each peak to sub-grid precision
            (see :func:`refine_peak`).

    Returns:
        ``(peaks, noise_floor)`` with peaks sorted by descending power.
    """
    scan = scan or ScanConfig()
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()

    resolution = float(azimuths[1] - azimuths[0])
    noise_floor = float(np.percentile(spectrum, scan.noise_percentile))

    # Tile three times so every peak in the middle copy has a full descent on
    # both sides, then keep only the middle copy's peaks.
    n = len(spectrum)
    tiled = np.concatenate([spectrum, spectrum, spectrum])

    idx, _ = find_peaks(tiled,
                        height=scan.min_peak_height,
                        prominence=scan.min_prominence,
                        distance=max(1, int(round(scan.min_separation / resolution))))
    idx = [i - n for i in idx if n <= i < 2 * n]

    peaks: list[Peak] = []
    for i in idx:
        power = spectrum[i]
        snr_db = (10.0 * np.log10(power / noise_floor)
                  if noise_floor > 0 else float("inf"))
        if snr_db >= scan.snr_threshold_db:
            azimuth = azimuths[i]
            if refine:
                azimuth = (azimuth
                           + refine_peak(spectrum, i) * resolution) % 360.0
            peaks.append(Peak(azimuth, power, snr_db))

    peaks.sort(key=lambda p: p.power, reverse=True)
    if max_sources is not None:
        peaks = peaks[:max_sources]
    return peaks, noise_floor


def angular_error(estimated_deg: float | np.ndarray,
                  true_deg: float | np.ndarray) -> np.ndarray:
    """
    Signed smallest-arc difference between two azimuths, in ``[-180, 180)``.

    Plain subtraction would report 350 deg of error for a 10 deg miss across
    the 0/360 seam, so every error figure in this project goes through here.

    An exactly antipodal pair returns ``-180``; +180 and -180 describe the same
    arc, so compare magnitudes rather than signed values at that boundary.
    """
    diff = (np.asarray(estimated_deg, dtype=float)
            - np.asarray(true_deg, dtype=float))
    return (diff + 180.0) % 360.0 - 180.0
