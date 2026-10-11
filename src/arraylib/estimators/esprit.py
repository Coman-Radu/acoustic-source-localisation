"""
ESPRIT: estimation of signal parameters via rotational invariance techniques.

2-D ESPRIT on the rectangular grid, recovering both direction cosines and
hence azimuth and elevation without scanning an angular grid.

Note on the original implementation
-----------------------------------
The version this replaces assigned the *same* signal-subspace matrix to both
subarrays::

    Es1 = Es   # for subarray 1
    Es2 = Es   # for subarray 2 (same, but represents displaced array)
    Psi = lstsq(Es1, Es2)

With identical operands the least-squares solution is exactly the identity, so
every eigenvalue of ``Psi`` is 1, every phase is 0, and the reported angles
carried no information about the source -- confirmed by feeding it simulated
sources from 0 to 270 deg and getting either nothing back or angles unrelated
to the truth.  The rotational invariance only appears if the two subarrays are
*selected out of* a common subspace, which is what ``_rotation_phases`` does
below via selection matrices.
"""

from __future__ import annotations

import numpy as np

from ..config import BAND_HIGH, BAND_LOW, SAMPLE_RATE, SPEED_OF_SOUND
from ..geometry import HARDWARE_ORDER, doublet_subarrays, positions_array
from .beamforming import _frequency_snapshots
from .music import estimate_num_sources

__all__ = ["esprit", "EspritResult"]


class EspritResult(dict):
    """Azimuths and elevations in degrees, plus the direction cosines."""

    __slots__ = ()

    def __init__(self, azimuth, elevation, u, v, num_sources):
        super().__init__(azimuth=list(map(float, azimuth)),
                         elevation=list(map(float, elevation)),
                         u=list(map(float, u)), v=list(map(float, v)),
                         num_sources=int(num_sources))

    azimuth = property(lambda self: self["azimuth"])
    elevation = property(lambda self: self["elevation"])

    def __repr__(self) -> str:
        pairs = ", ".join(f"{a:.1f}/{e:.1f}"
                          for a, e in zip(self.azimuth, self.elevation))
        return f"EspritResult(az/el deg: {pairs})"


def _selection_matrix(channels: list[int], n_channels: int) -> np.ndarray:
    """Row-selection matrix picking ``channels`` out of a full-array vector."""
    J = np.zeros((len(channels), n_channels))
    J[np.arange(len(channels)), channels] = 1.0
    return J


def _rotation_operator(Es: np.ndarray,
                       sub1: list[int],
                       sub2: list[int]) -> np.ndarray:
    """
    Least-squares rotation operator between two displaced subarrays.

    Both subarrays are selected from the *same* signal subspace ``Es``, so
    ``Es2 = Es1 @ Psi`` holds with ``Psi`` similar to the diagonal matrix of
    inter-subarray phase shifts.  ``pinv`` keeps this stable when ``Es1`` is
    near rank-deficient (too few snapshots, or fewer real sources than the
    assumed order).
    """
    J1 = _selection_matrix(sub1, Es.shape[0])
    J2 = _selection_matrix(sub2, Es.shape[0])
    return np.linalg.pinv(J1 @ Es) @ (J2 @ Es)


def _paired_phases(Psi_x: np.ndarray,
                   Psi_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Phases of both rotation operators, paired per source.

    This is the step that makes 2-D ESPRIT work.  ``Psi_x`` and ``Psi_y`` are
    similarity transforms of two *different* diagonal matrices in one shared
    eigenbasis -- the basis set by the sources.  Diagonalising each one
    separately therefore returns correct values of ``u`` and ``v`` but in
    unrelated orders, and there is no way to tell afterwards which ``u``
    belongs to which ``v``.

    Sorting the two lists independently is not a fix, and fails in an easily
    recognised way: two sources at 30 and 120 deg have
    ``u = (0.50, 0.87)`` and ``v = (0.87, -0.50)``, so sorting pairs
    ``(0.50, -0.50)`` and ``(0.87, 0.87)`` and reports 135 and 45 deg.

    Instead, ``Psi_x`` is diagonalised once and its eigenvector matrix ``T``
    is used to transform ``Psi_y`` into the same basis; the diagonal of
    ``T^-1 Psi_y T`` then lines up source by source with the eigenvalues of
    ``Psi_x``.

    Returns:
        ``(phase_x, phase_y)``, each ``[num_sources]`` radians, index-aligned.
    """
    eigenvalues_x, T = np.linalg.eig(Psi_x)
    # Transform Psi_y into the eigenbasis of Psi_x; it should come out close
    # to diagonal, and its diagonal is then paired with eigenvalues_x.
    Psi_y_rotated = np.linalg.solve(T, Psi_y @ T)
    eigenvalues_y = np.diag(Psi_y_rotated)
    return np.angle(eigenvalues_x), np.angle(eigenvalues_y)


def esprit(signals: np.ndarray,
           num_sources: int | None = None,
           fs: int = SAMPLE_RATE,
           c: float = SPEED_OF_SOUND,
           nfft: int = 1024,
           band: tuple[float, float] = (BAND_LOW, BAND_HIGH),
           order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
           ) -> EspritResult:
    """
    Estimate source directions by 2-D ESPRIT.

    Works on the dominant frequency bin: the rotation phase depends on
    frequency, so pairing the x and y phases for the same source requires a
    single bin rather than a band average.  The bin with the most signal power
    is chosen automatically.

    The two direction cosines follow from the x- and y-displaced subarray
    pairs::

        u = phase_x * c / (2*pi*f*d)      (east component)
        v = phase_y * c / (2*pi*f*d)      (north component)

    from which azimuth is ``atan2(u, v)`` in compass convention and elevation
    is ``arccos(hypot(u, v))``.  The two axes are paired per source by joint
    diagonalisation (see :func:`_paired_phases`), which is what lets the
    estimator handle more than one source correctly.  Sources sharing almost
    the same ``u`` leave ``Psi_x`` with near-degenerate eigenvalues and the
    pairing becomes ill-conditioned.

    Args:
        num_sources: Number of sources; ``None`` estimates it by MDL.
        order: Channel layout of ``signals``.

    Returns:
        :class:`EspritResult`.
    """
    positions = positions_array(order=order)
    freqs, X = _frequency_snapshots(signals, nfft, fs, band)
    n_freqs, n_ch, n_snap = X.shape

    R = np.einsum("fcs,fds->fcd", X, np.conj(X)) / n_snap
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    eigenvalues = eigenvalues[:, ::-1]
    eigenvectors = eigenvectors[:, :, ::-1]

    # Pick the bin carrying the most power: best-conditioned subspace, and it
    # keeps the x and y phase estimates at a common frequency.
    bin_idx = int(np.argmax(eigenvalues[:, 0]))
    frequency = float(freqs[bin_idx])

    if num_sources is None:
        num_sources = estimate_num_sources(eigenvalues[bin_idx], n_snap,
                                           max_sources=n_ch // 2)
    num_sources = int(np.clip(num_sources, 1, n_ch // 2))

    Es = eigenvectors[bin_idx][:, :num_sources]          # [n_ch, k]

    sub1x, sub2x, dx = doublet_subarrays("x", order)
    sub1y, sub2y, dy = doublet_subarrays("y", order)

    Psi_x = _rotation_operator(Es, sub1x, sub2x)
    Psi_y = _rotation_operator(Es, sub1y, sub2y)
    phase_x, phase_y = _paired_phases(Psi_x, Psi_y)

    # phase = 2*pi*f*d*cosine/c  ->  cosine = phase*c/(2*pi*f*d)
    scale_x = c / (2.0 * np.pi * frequency * float(np.linalg.norm(dx)))
    scale_y = c / (2.0 * np.pi * frequency * float(np.linalg.norm(dy)))

    u = phase_x * scale_x
    v = phase_y * scale_y

    radius = np.hypot(u, v)
    azimuth = np.degrees(np.arctan2(u, v)) % 360.0        # 0 = North, clockwise
    elevation = np.degrees(np.arccos(np.clip(radius, 0.0, 1.0)))

    return EspritResult(azimuth, elevation, u, v, num_sources)
