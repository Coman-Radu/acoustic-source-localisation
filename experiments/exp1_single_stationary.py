#!/usr/bin/env python3
"""
Experiment 1 -- one stationary source.

Baseline accuracy check: every estimator against a single source at a known
bearing.  Establishes the floor each method reaches before multiple sources or
motion are introduced.

    python experiments/exp1_single_stationary.py --azimuth 30 --snr 20
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (ALL_METHODS, FIGURES, SCAN_METHODS, estimate, geometry,
                     parse_args, positions, scan, simulate, source_signal)
from arraylib.detect import angular_error
from arraylib.plotting import METHOD_COLOURS, polar_spectrum, save


def main() -> None:
    args = parse_args(__doc__, azimuth=dict(type=float, default=30.0,
                                            help="true source azimuth in deg"))

    pos = positions()
    pairs = geometry.microphone_pairs(args.pairs)
    signal, label = source_signal(args.duration,
                                  prefer_recorded=not args.synthetic)

    signals = simulate.simulate_stationary(
        signal, args.azimuth, snr_db=args.snr,
        rng=np.random.default_rng(args.seed))

    print("Experiment 1 -- single stationary source")
    print(f"  source      : {label}")
    print(f"  true azimuth: {args.azimuth:.1f} deg")
    print(f"  SNR         : {args.snr:.0f} dB")
    print(f"  SRP pairs   : {args.pairs} ({len(pairs)})")
    print()
    print(f"  {'method':<10} {'estimate':>10} {'error':>10}")

    results: dict[str, tuple[float, float]] = {}
    for method in ALL_METHODS:
        got = estimate(method, signals, pos, pairs, num_sources=1)
        if got:
            err = float(angular_error(got[0], args.azimuth))
            results[method] = (got[0], err)
            print(f"  {method:<10} {got[0]:>7.1f} deg {err:>+7.2f} deg")
        else:
            results[method] = (float("nan"), float("nan"))
            print(f"  {method:<10} {'no peak':>10} {'-':>10}")

    fig = plt.figure(figsize=(11, 4.2))

    ax = fig.add_subplot(1, 2, 1, projection="polar")
    for method in SCAN_METHODS:
        azimuths, spectrum = scan(method, signals, pos, pairs, num_sources=1)
        polar_spectrum(ax, azimuths, spectrum, label=method,
                       true_azimuths=[args.azimuth] if method == SCAN_METHODS[0]
                       else None)
    ax.set_title(f"Spatial spectra, source at {args.azimuth:.0f} deg\n"
                 f"({label}, {args.snr:.0f} dB SNR)", fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    names = list(results)
    errors = [abs(results[n][1]) for n in names]
    ax2.barh(names, errors,
             color=[METHOD_COLOURS.get(n, "#888888") for n in names])
    ax2.set_xlabel("absolute azimuth error [deg]")
    ax2.set_title("Accuracy on a single source", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()
    for i, err in enumerate(errors):
        if np.isfinite(err):
            ax2.text(err, i, f" {err:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    path = save(fig, FIGURES / "exp1_single_stationary.png")
    print(f"\n  figure -> {path.relative_to(FIGURES.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
