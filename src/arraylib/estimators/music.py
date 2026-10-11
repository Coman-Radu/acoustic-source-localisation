"""
MUSIC: multiple signal classification.
"""

from __future__ import annotations

import numpy as np

from ..config import BAND_HIGH, BAND_LOW, SAMPLE_RATE, SPEED_OF_SOUND, ScanConfig
from ..geometry import steering_matrix
from .beamforming import _frequency_snapshots, _normalise, scan_grid

__all__ = ["music", "estimate_num_sources"]


def estimate_num_sources(eigenvalues: np.ndarray,
                         n_snapshots: int,
                         max_sources: int | None = None) -> int:
    """
    Choose the signal-subspace dimension by MDL.

    MUSIC needs to know how many sources are present, and guessing wrong is
    the usual cause of spurious or missing peaks.  The minimum description
    length criterion picks the split between signal and noise eigenvalues by
    testing where the remaining eigenvalues stop looking equal (white).

    Args:
        eigenvalues: Covariance eigenvalues, descending.
        n_snapshots: Snapshots used to form the covariance.
        max_sources: Upper bound on the answer.

    Returns:
        Estimated number of sources, at least 1.
    """
    ev = np.sort(np.asarray(eigenvalues, dtype=float).real)[::-1]
    ev = np.maximum(ev, 1e-300)
    m = len(ev)
    cap = m - 1 if max_sources is None else min(max_sources, m - 1)

    best_k, best_mdl = 1, np.inf
    for k in range(0, cap + 1):
        noise = ev[k:]
        q = len(noise)
        if q < 1:
            break
        # Likelihood term: geometric vs arithmetic mean of noise eigenvalues.
        log_geo = float(np.mean(np.log(noise)))
        log_arith = float(np.log(np.mean(noise)))
        likelihood = -n_snapshots * q * (log_geo - log_arith)
        n_params = k * (2 * m - k) + 1
        mdl = likelihood + 0.5 * n_params * np.log(n_snapshots)
        if mdl < best_mdl:
            best_mdl, best_k = mdl, k

    return max(1, best_k)


def music(signals: np.ndarray,
          positions: np.ndarray,
          num_sources: int | None = None,
          scan: ScanConfig | None = None,
          fs: int = SAMPLE_RATE,
          c: float = SPEED_OF_SOUND,
          nfft: int = 1024,
          band: tuple[float, float] = (BAND_LOW, BAND_HIGH),
          ) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Broadband MUSIC pseudo-spectrum, incoherently averaged over frequency.

    The array covariance is eigendecomposed per frequency bin; directions
    orthogonal to the noise subspace produce sharp peaks.  Resolution well
    beyond the aperture limit is possible, at the cost of needing the source
    count and of assuming mutually incoherent sources.

    This differs from the project's original implementation, which evaluated a
    single steering vector at a fixed 1 kHz regardless of the signal content.
    That is only valid for a narrowband source; drone rotor noise spans the
    whole band, so steering is done per bin here.

    Args:
        num_sources: Signal-subspace dimension.  ``None`` estimates it by MDL.

    Returns:
        ``(azimuths, pseudo_spectrum, num_sources_used)``.
    """
    azimuths = scan_grid(scan)
    freqs, X = _frequency_snapshots(signals, nfft, fs, band)
    n_freqs, n_ch, n_snap = X.shape

    R = np.einsum("fcs,fds->fcd", X, np.conj(X)) / n_snap

    # Hermitian eigendecomposition, ascending -> flip to descending.
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    eigenvalues = eigenvalues[:, ::-1]
    eigenvectors = eigenvectors[:, :, ::-1]

    if num_sources is None:
        # Estimate on the strongest bin, which has the best-conditioned
        # covariance, then apply that order across the band.
        strongest = int(np.argmax(eigenvalues[:, 0]))
        num_sources = estimate_num_sources(eigenvalues[strongest], n_snap,
                                           max_sources=n_ch // 2)
    num_sources = int(np.clip(num_sources, 1, n_ch - 1))

    noise_space = eigenvectors[:, :, num_sources:]        # [f, ch, ch-k]
    A = steering_matrix(azimuths, freqs, positions, c)    # [az, f, ch]

    # projection(az, f) = || En^H a ||^2 ; spectrum = 1 / projection
    proj = np.einsum("afc,fck->afk", np.conj(A), noise_space)
    denom = np.sum(np.abs(proj) ** 2, axis=2)             # [az, f]
    spectrum = np.mean(1.0 / (denom + 1e-20), axis=1)

    return azimuths, _normalise(spectrum), num_sources
