"""
Shared setup for the experiment scripts.

Every experiment imports its geometry, source and estimator set from here so
that figures are directly comparable and a change to the array definition
propagates to all of them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from arraylib import config, geometry, simulate, sources          # noqa: E402
from arraylib.detect import angular_error, find_sources           # noqa: E402
from arraylib.estimators import (delay_and_sum, esprit, music,    # noqa: E402
                                 mvdr, srp_phat, tdoa_direction)

FIGURES = REPO_ROOT / "figures"
RECORDED_WAV = REPO_ROOT / "data" / "drone_bebop_sample.wav"

#: Estimators compared throughout, each reduced to a single call signature
#: returning ``(azimuths, spectrum)`` or, for the parametric ones, ``None``.
SCAN_METHODS = ("DAS", "MVDR", "SRP-PHAT", "MUSIC")
ALL_METHODS = SCAN_METHODS + ("ESPRIT",)


def positions() -> np.ndarray:
    """Element positions of the as-wired array."""
    return geometry.positions_array()


def source_signal(duration: float, prefer_recorded: bool = True):
    """
    Load the recorded drone sample if present, else synthesise one.

    Returns:
        ``(signal, label)`` where ``label`` names the source for figure titles.
    """
    if prefer_recorded and RECORDED_WAV.is_file():
        _, sig = sources.load_wav(RECORDED_WAV, duration=duration)
        return sig, "recorded Bebop rotor noise"
    _, sig = sources.harmonic_drone(duration)
    return sig, "synthetic harmonic rotor noise"


def scan(method: str, signals: np.ndarray, pos: np.ndarray,
         pairs, num_sources: int | None = None):
    """
    Run one scanning estimator and return ``(azimuths, spectrum)``.
    """
    if method == "DAS":
        return delay_and_sum(signals, pos)
    if method == "MVDR":
        return mvdr(signals, pos)
    if method == "SRP-PHAT":
        return srp_phat(signals, pos, pairs)
    if method == "MUSIC":
        return music(signals, pos, num_sources=num_sources)[:2]
    raise ValueError(f"{method!r} is not a scanning estimator")


def estimate(method: str, signals: np.ndarray, pos: np.ndarray, pairs,
             num_sources: int = 1) -> list[float]:
    """
    Azimuth estimates from any estimator, strongest first.

    Returns:
        Up to ``num_sources`` azimuths in degrees; may be shorter if the
        estimator found fewer sources.
    """
    if method == "ESPRIT":
        return list(esprit(signals, num_sources=num_sources)["azimuth"])
    if method == "TDOA":
        return [tdoa_direction(signals)["azimuth"]]

    azimuths, spectrum = scan(method, signals, pos, pairs, num_sources)
    peaks, _ = find_sources(azimuths, spectrum, max_sources=num_sources)
    return [p.azimuth for p in peaks]


def match_to_truth(estimates: list[float],
                   truth: list[float]) -> list[float]:
    """
    Pair estimates with true azimuths and return the per-source error.

    Estimators return sources in power order, not in the order they were
    simulated, so a direct zip would report nonsense.  Each true source is
    assigned its nearest unused estimate; a source with no estimate left
    counts as a miss and yields NaN.
    """
    remaining = list(estimates)
    errors: list[float] = []
    for true_az in truth:
        if not remaining:
            errors.append(float("nan"))
            continue
        deltas = [abs(angular_error(e, true_az)) for e in remaining]
        k = int(np.argmin(deltas))
        errors.append(float(angular_error(remaining.pop(k), true_az)))
    return errors


def parse_args(description: str, **extra) -> argparse.Namespace:
    """Common command-line interface for the experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--snr", type=float, default=20.0,
                        help="broadband SNR in dB (default: 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default: 0)")
    parser.add_argument("--duration", type=float, default=0.5,
                        help="capture length per frame in seconds")
    parser.add_argument("--pairs", default="perimeter",
                        choices=["adjacent", "perimeter", "all"],
                        help="channel pairs for SRP-PHAT (default: perimeter)")
    parser.add_argument("--synthetic", action="store_true",
                        help="force the synthetic source even if a WAV exists")
    parser.add_argument("--no-show", action="store_true",
                        help="save the figure without opening a window")
    for name, kwargs in extra.items():
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    return parser.parse_args()
