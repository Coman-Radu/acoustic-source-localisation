#!/usr/bin/env python3
"""
Experiment 4 -- accuracy against signal-to-noise ratio.

Sweeps broadband SNR to find where each estimator stops working.  This is the
number that matters operationally, because SNR falls as 1/r with range: it
converts directly into a detection radius.

Each SNR is run over several trials and bearings so the result is an average
rather than one lucky draw.

    python experiments/exp4_snr_sweep.py --trials 8
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (ALL_METHODS, FIGURES, estimate, geometry, parse_args,
                     positions, simulate, source_signal)
from arraylib.detect import angular_error
from arraylib.plotting import METHOD_COLOURS, save


def main() -> None:
    args = parse_args(__doc__,
                      trials=dict(type=int, default=6,
                                  help="trials per SNR point"),
                      snr_min=dict(type=float, default=-15.0),
                      snr_max=dict(type=float, default=25.0),
                      snr_step=dict(type=float, default=5.0))

    pos = positions()
    pairs = geometry.microphone_pairs(args.pairs)
    signal, label = source_signal(args.duration,
                                  prefer_recorded=not args.synthetic)

    snrs = np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step)
    # Spread the test bearings around the compass so no method is flattered by
    # a direction that happens to sit on the scan grid.
    bearings = np.linspace(0.0, 360.0, args.trials, endpoint=False) + 7.0

    rms = {m: [] for m in ALL_METHODS}
    failures = {m: [] for m in ALL_METHODS}

    print("Experiment 4 -- accuracy vs SNR")
    print(f"  source : {label}")
    print(f"  trials : {args.trials} bearings per SNR point")
    print(f"  frame  : {args.duration * 1000:.0f} ms")
    print()
    header = "  " + f"{'SNR':>6} " + " ".join(f"{m:>9}" for m in ALL_METHODS)
    print(header)

    for snr in snrs:
        errors = {m: [] for m in ALL_METHODS}
        misses = {m: 0 for m in ALL_METHODS}

        for trial, bearing in enumerate(bearings):
            rng = np.random.default_rng(args.seed + trial)
            signals = simulate.simulate_stationary(signal, bearing,
                                                   snr_db=float(snr), rng=rng)
            for method in ALL_METHODS:
                got = estimate(method, signals, pos, pairs, num_sources=1)
                if got:
                    errors[method].append(abs(angular_error(got[0], bearing)))
                else:
                    misses[method] += 1

        row = f"  {snr:>5.0f}dB "
        for method in ALL_METHODS:
            vals = errors[method]
            value = float(np.sqrt(np.mean(np.square(vals)))) if vals else np.nan
            rms[method].append(value)
            failures[method].append(misses[method] / len(bearings))
            row += f"{value:>8.2f}  " if np.isfinite(value) else f"{'--':>8}  "
        print(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    for method in ALL_METHODS:
        ax1.plot(snrs, rms[method], marker="o", markersize=4, label=method,
                 color=METHOD_COLOURS.get(method))
    ax1.set_xlabel("broadband SNR [dB]")
    ax1.set_ylabel("RMS azimuth error [deg]")
    ax1.set_yscale("log")
    ax1.set_title(f"Accuracy vs SNR ({args.trials} bearings per point)",
                  fontsize=10)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    for method in ALL_METHODS:
        ax2.plot(snrs, np.array(failures[method]) * 100, marker="o",
                 markersize=4, label=method, color=METHOD_COLOURS.get(method))
    ax2.set_xlabel("broadband SNR [dB]")
    ax2.set_ylabel("frames with no detection [%]")
    ax2.set_title("Detection failures vs SNR", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = save(fig, FIGURES / "exp4_snr_sweep.png")
    print(f"\n  figure -> {path.relative_to(FIGURES.parent)}")

    print("\n  lowest SNR holding RMS error under 5 deg:")
    for method in ALL_METHODS:
        ok = [s for s, v in zip(snrs, rms[method]) if np.isfinite(v) and v < 5.0]
        print(f"    {method:<10} {f'{min(ok):.0f} dB' if ok else 'never':>8}")
    plt.close(fig)


if __name__ == "__main__":
    main()
