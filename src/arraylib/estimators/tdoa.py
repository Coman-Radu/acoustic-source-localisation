"""
Direct TDOA geometry: row/column delays to azimuth and elevation.

This is the method the project started from -- measure the delay along the
array's rows to get one direction cosine and along its columns to get the
other.  It is cheap and needs no angular scan, but it assumes a single source,
so it is the first method to fail when a second aircraft appears.
"""

from __future__ import annotations

import numpy as np

from ..config import SAMPLE_RATE, SPEED_OF_SOUND
from ..geometry import HARDWARE_ORDER, columns, positions_array, rows
from .gcc_phat import gcc_phat

__all__ = ["direction_cosine_from_delay", "tdoa_direction", "least_squares_direction"]


def direction_cosine_from_delay(delay: float,
                                baseline: float,
                                c: float = SPEED_OF_SOUND) -> float:
    """
    Direction cosine along a baseline from the delay measured across it.

    For a plane wave, ``delay = baseline * cos(angle) / c``, so the cosine is
    ``delay * c / baseline``.  Values are clipped to [-1, 1]: noise routinely
    pushes the raw ratio slightly outside the physically possible range, and
    the original code produced NaNs at exactly the broadside angles where the
    estimate should have been *best*.
    """
    if baseline <= 0:
        raise ValueError(f"baseline must be positive, got {baseline}")
    return float(np.clip(delay * c / baseline, -1.0, 1.0))


def _line_delay(signals: np.ndarray,
                line: tuple[int, ...],
                positions: np.ndarray,
                fs: int,
                c: float) -> tuple[float, float]:
    """
    Average delay per unit distance along one row or column.

    Every pair on the line is measured and the per-metre slopes are averaged,
    which is more robust than relying on the single widest pair.

    Returns:
        ``(slope, baseline)`` -- seconds per metre, and the full line length.
    """
    slopes = []
    for a in range(len(line)):
        for b in range(a + 1, len(line)):
            i, j = line[a], line[b]
            separation = positions[j] - positions[i]
            distance = float(np.linalg.norm(separation))
            if distance <= 0:
                continue
            tau, _, _ = gcc_phat(signals[:, i], signals[:, j], fs=fs,
                                 max_tau=1.25 * distance / c)
            # tau is the delay of j relative to i; the element further along
            # the axis towards the source leads, hence the sign flip.
            slopes.append(-tau / distance)

    baseline = float(np.linalg.norm(positions[line[-1]] - positions[line[0]]))
    return float(np.mean(slopes)) if slopes else 0.0, baseline


def tdoa_direction(signals: np.ndarray,
                   fs: int = SAMPLE_RATE,
                   c: float = SPEED_OF_SOUND,
                   order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                   ) -> dict[str, float]:
    """
    Azimuth and elevation of a single source from row and column delays.

    Rows run east-west and give the eastward direction cosine ``u``; columns
    run north-south and give the northward cosine ``v``.  Azimuth follows as
    ``atan2(u, v)`` and elevation as ``arccos(hypot(u, v))``.

    Returns:
        ``{"azimuth", "elevation", "u", "v"}`` with angles in degrees.
    """
    positions = positions_array(order=order)

    row_slopes = [_line_delay(signals, line, positions, fs, c)[0]
                  for line in rows(order)]
    col_slopes = [_line_delay(signals, line, positions, fs, c)[0]
                  for line in columns(order)]

    # slope is seconds per metre along the axis; cosine = slope * c
    u = float(np.clip(np.mean(row_slopes) * c, -1.0, 1.0))
    v = float(np.clip(np.mean(col_slopes) * c, -1.0, 1.0))

    radius = float(np.hypot(u, v))
    return {
        "azimuth": float(np.degrees(np.arctan2(u, v)) % 360.0),
        "elevation": float(np.degrees(np.arccos(min(radius, 1.0)))),
        "u": u,
        "v": v,
    }


def least_squares_direction(signals: np.ndarray,
                            pairs: list[tuple[int, int]],
                            positions: np.ndarray,
                            fs: int = SAMPLE_RATE,
                            c: float = SPEED_OF_SOUND,
                            ) -> dict[str, float]:
    """
    Single-source direction by least squares over every supplied pair.

    Each pair contributes one linear equation in the direction vector,
    ``(p_i - p_j) . d = c * tau_ij``, and the stacked system is solved in one
    step.  Using all pairs at once rather than row and column averages
    separately makes better use of the diagonal baselines, which are the
    longest the array has.

    Returns:
        ``{"azimuth", "elevation", "u", "v", "residual"}``.
    """
    A = np.empty((len(pairs), 2))
    b = np.empty(len(pairs))

    for k, (i, j) in enumerate(pairs):
        separation = positions[i] - positions[j]
        distance = float(np.linalg.norm(separation))
        tau, _, _ = gcc_phat(signals[:, i], signals[:, j], fs=fs,
                             max_tau=1.25 * distance / c)
        A[k] = separation
        b[k] = c * tau

    solution, residuals, *_ = np.linalg.lstsq(A, b, rcond=None)
    u, v = float(solution[0]), float(solution[1])

    # The solution need not be a unit vector under noise; its direction is
    # what carries the bearing, so normalise before reading off the angles.
    norm = float(np.hypot(u, v))
    if norm > 0:
        un, vn = u / norm, v / norm
    else:
        un, vn = 0.0, 0.0

    return {
        "azimuth": float(np.degrees(np.arctan2(un, vn)) % 360.0),
        "elevation": float(np.degrees(np.arccos(min(norm, 1.0)))),
        "u": u,
        "v": v,
        "residual": float(residuals[0]) if len(np.atleast_1d(residuals)) else float("nan"),
    }
