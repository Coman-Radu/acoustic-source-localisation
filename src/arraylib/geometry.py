"""
Array geometry: element positions, channel mapping and steering vectors.

This module is the single source of truth for where each microphone sits.
The original code carried six separate copies of a ``get_mic_positions_4x4``
function that had drifted apart; consolidating them here revealed that the
simulation scripts and the real-time scripts disagreed about which capture
channel corresponded to which physical element (see :data:`SIM_ORDER` and
``docs/array-geometry.md``).
"""

from __future__ import annotations

import numpy as np

from .config import CHANNELS, GRID_SHAPE, MIC_SPACING, SPEED_OF_SOUND

__all__ = [
    "HARDWARE_ORDER",
    "SIM_ORDER",
    "mic_positions",
    "positions_array",
    "rows",
    "columns",
    "doublet_subarrays",
    "microphone_pairs",
    "steering_vector",
    "steering_matrix",
    "direction_unit_vector",
]


# Channel order as wired on the prototype, read out row by row from the
# southern-most row to the northern-most, west to east within each row.
# Element labels on the PCB are MIC01..MIC16; the capture interface presents
# them in this (non-sequential) channel order:
#
#   row 3 (y = +63 mm, North):  MIC09 MIC08 MIC11 MIC10  ->  ch  8  7 10  9
#   row 2 (y = +21 mm):         MIC07 MIC06 MIC13 MIC12  ->  ch  6  5 12 11
#   row 1 (y = -21 mm):         MIC05 MIC04 MIC15 MIC14  ->  ch  4  3 14 13
#   row 0 (y = -63 mm, South):  MIC03 MIC02 MIC01 MIC16  ->  ch  2  1  0 15
#
# x runs west to east at -63, -21, +21, +63 mm.
HARDWARE_ORDER: tuple[tuple[int, ...], ...] = (
    (2, 1, 0, 15),    # row 0, southern-most
    (4, 3, 14, 13),   # row 1
    (6, 5, 12, 11),   # row 2
    (8, 7, 10, 9),    # row 3, northern-most
)

# The ordering used by the original paper-experiment simulations, which walked
# the perimeter and then filled in the four interior elements.  It describes
# the *same* physical grid but assigns 15 of the 16 channels differently, so
# it is kept only to reproduce the original simulation figures exactly.  Never
# use it to interpret a real recording.
SIM_ORDER: tuple[tuple[int, ...], ...] = (
    (1, 3, 5, 7),
    (0, 2, 4, 6),
    (15, 13, 11, 9),
    (14, 12, 10, 8),
)


def mic_positions(spacing: float = MIC_SPACING,
                  order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                  ) -> dict[int, tuple[float, float]]:
    """
    Map capture channel -> (x, y) position in metres, origin at array centre.

    Args:
        spacing: Element spacing of the uniform grid, in metres.
        order: Channel layout, row-major from south to north.  Defaults to
            the as-wired :data:`HARDWARE_ORDER`.

    Returns:
        ``{channel: (x, y)}`` for all :data:`CHANNELS` elements.
    """
    n_rows, n_cols = GRID_SHAPE
    if len(order) != n_rows or any(len(r) != n_cols for r in order):
        raise ValueError(f"order must be {n_rows}x{n_cols}, got "
                         f"{len(order)}x{len(order[0]) if order else 0}")

    # Centre the grid on the origin: offsets are +/-0.5 and +/-1.5 * spacing.
    offsets = (np.arange(n_cols) - (n_cols - 1) / 2.0) * spacing
    y_offsets = (np.arange(n_rows) - (n_rows - 1) / 2.0) * spacing

    positions: dict[int, tuple[float, float]] = {}
    for r, row in enumerate(order):
        for c, channel in enumerate(row):
            positions[channel] = (float(offsets[c]), float(y_offsets[r]))

    if len(positions) != CHANNELS:
        raise ValueError(f"channel layout is not a permutation of "
                         f"0..{CHANNELS - 1}: got {sorted(positions)}")
    return positions


def positions_array(spacing: float = MIC_SPACING,
                    order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                    ) -> np.ndarray:
    """Element positions as a ``[CHANNELS, 2]`` array indexed by channel."""
    pos = mic_positions(spacing, order)
    return np.array([pos[ch] for ch in range(CHANNELS)], dtype=float)


def rows(order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER
         ) -> tuple[tuple[int, ...], ...]:
    """Channels grouped by row (constant y), south to north."""
    return order


def columns(order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER
            ) -> tuple[tuple[int, ...], ...]:
    """Channels grouped by column (constant x), west to east."""
    return tuple(zip(*order))


def doublet_subarrays(axis: str = "x",
                      order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                      spacing: float = MIC_SPACING,
                      ) -> tuple[list[int], list[int], np.ndarray]:
    """
    Two maximally-overlapping subarrays displaced by one element spacing.

    ESPRIT needs a pair of identical subarrays related by a known
    translation.  On a 4x4 grid, dropping the last column gives subarray 1
    and dropping the first column gives subarray 2, so the two are identical
    up to a shift of one ``spacing`` along x (and likewise for y).

    Args:
        axis: ``"x"`` for an east-west displacement, ``"y"`` for north-south.
        order: Channel layout.
        spacing: Element spacing in metres.

    Returns:
        ``(subarray1, subarray2, displacement)`` where the subarray lists are
        channel indices in matching order and ``displacement`` is the 2-vector
        separating element *k* of subarray 1 from element *k* of subarray 2.
    """
    if axis == "x":
        sub1 = [ch for row in order for ch in row[:-1]]
        sub2 = [ch for row in order for ch in row[1:]]
        displacement = np.array([spacing, 0.0])
    elif axis == "y":
        sub1 = [ch for row in order[:-1] for ch in row]
        sub2 = [ch for row in order[1:] for ch in row]
        displacement = np.array([0.0, spacing])
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return sub1, sub2, displacement


def microphone_pairs(selection: str = "adjacent",
                     order: tuple[tuple[int, ...], ...] = HARDWARE_ORDER,
                     ) -> list[tuple[int, int]]:
    """
    Channel pairs to cross-correlate.

    Args:
        selection: ``"all"`` for every unique pair (120 pairs, most robust and
            slowest), ``"adjacent"`` for neighbours within rows and columns
            (24 pairs, the useful default), or ``"perimeter"`` for pairs drawn
            from the outer ring only (66 pairs, widest baselines).
    """
    if selection == "all":
        return [(i, j) for i in range(CHANNELS) for j in range(i + 1, CHANNELS)]

    if selection == "adjacent":
        pairs: list[tuple[int, int]] = []
        for line in list(rows(order)) + list(columns(order)):
            pairs.extend((line[i], line[i + 1]) for i in range(len(line) - 1))
        return pairs

    if selection == "perimeter":
        ring = list(order[0]) \
            + [row[-1] for row in order[1:-1]] \
            + list(reversed(order[-1])) \
            + [row[0] for row in reversed(order[1:-1])]
        return [(ring[i], ring[j])
                for i in range(len(ring)) for j in range(i + 1, len(ring))]

    raise ValueError(f"unknown selection {selection!r}; expected 'all', "
                     f"'adjacent' or 'perimeter'")


def direction_unit_vector(azimuth_deg: float | np.ndarray
                          ) -> np.ndarray:
    """
    Unit vector pointing towards a compass azimuth.

    Azimuth follows the survey convention used throughout this project:
    0 deg is North (+y), increasing clockwise, so 90 deg is East (+x).

    Returns:
        ``[2]`` for scalar input, or ``[n, 2]`` for array input.
    """
    theta = np.radians(90.0 - np.asarray(azimuth_deg, dtype=float))
    vec = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return vec


def steering_vector(azimuth_deg: float,
                    freqs: np.ndarray,
                    positions: np.ndarray,
                    c: float = SPEED_OF_SOUND) -> np.ndarray:
    """
    Narrowband-per-bin steering vector for a far-field plane wave.

    Args:
        azimuth_deg: Look direction, compass convention.
        freqs: ``[n_freqs]`` frequencies in Hz.
        positions: ``[n_mics, 2]`` element positions in metres.
        c: Speed of sound in m/s.

    Returns:
        ``[n_freqs, n_mics]`` complex array of phase factors.
    """
    direction = direction_unit_vector(azimuth_deg)
    delays = positions @ direction / c          # [n_mics]
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    return np.exp(2j * np.pi * freqs[:, None] * delays[None, :])


def steering_matrix(azimuths_deg: np.ndarray,
                    freqs: np.ndarray,
                    positions: np.ndarray,
                    c: float = SPEED_OF_SOUND) -> np.ndarray:
    """
    Steering vectors for a whole azimuth grid.

    Returns:
        ``[n_azimuths, n_freqs, n_mics]`` complex array.
    """
    directions = direction_unit_vector(np.asarray(azimuths_deg))  # [n_az, 2]
    delays = directions @ positions.T / c                          # [n_az, n_mics]
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    return np.exp(2j * np.pi * freqs[None, :, None] * delays[:, None, :])
