"""
Shared plotting helpers, so every experiment figure looks the same.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["polar_spectrum", "plot_array", "plot_error_series", "save"]

# Consistent colour per estimator across every figure in the repository.
METHOD_COLOURS = {
    "DAS": "#4878a8",
    "MVDR": "#c86432",
    "SRP-PHAT": "#4f9153",
    "MUSIC": "#9c5ca8",
    "ESPRIT": "#b8903c",
    "TDOA": "#777777",
}


def save(fig, path: str | Path, dpi: int = 150) -> Path:
    """Write a figure next to the others and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def polar_spectrum(ax,
                   azimuths: np.ndarray,
                   spectrum: np.ndarray,
                   label: str | None = None,
                   colour: str | None = None,
                   true_azimuths: list[float] | None = None):
    """
    Draw a spatial spectrum on a polar axis in compass orientation.

    Matplotlib's polar axes put 0 at the east and run anticlockwise, while
    every azimuth in this project is a compass bearing: 0 at north, running
    clockwise.  ``set_theta_zero_location`` and ``set_theta_direction`` make
    the plot match the convention used in the maths.
    """
    theta = np.radians(np.append(azimuths, azimuths[0]))
    radius = np.append(spectrum, spectrum[0])

    ax.plot(theta, radius, label=label,
            color=colour or METHOD_COLOURS.get(label or "", None), linewidth=1.6)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])

    for true_az in true_azimuths or []:
        ax.axvline(np.radians(true_az), color="#cc2222", linestyle="--",
                   linewidth=1.0, alpha=0.8)
    return ax


def plot_array(ax, positions: np.ndarray, annotate: bool = True):
    """Scatter the element layout in millimetres."""
    ax.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000,
               marker="s", s=48, color="#4878a8", zorder=3)
    if annotate:
        for ch, (x, y) in enumerate(positions):
            ax.annotate(str(ch), (x * 1000, y * 1000),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color="#555555")
    ax.set_xlabel("x east [mm]")
    ax.set_ylabel("y north [mm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    return ax


def plot_error_series(ax, t: np.ndarray, errors: dict[str, np.ndarray]):
    """Signed angular error against time, one line per method."""
    for name, err in errors.items():
        ax.plot(t, err, label=name, linewidth=1.4,
                color=METHOD_COLOURS.get(name))
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("azimuth error [deg]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax
