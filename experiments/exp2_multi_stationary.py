#!/usr/bin/env python3
"""
Experiment 2 -- several stationary sources.

The case the project exists for: more than one aircraft in the air at once.
Separates the estimators that merely find the loudest bearing from those that
genuinely resolve multiple sources, and shows where each one breaks as the
sources close in on each other.

    python experiments/exp2_multi_stationary.py --azimuths 30 75 120
    python experiments/exp2_multi_stationary.py --separation-sweep
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (ALL_METHODS, FIGURES, SCAN_METHODS, estimate, geometry,
                     match_to_truth, parse_args, positions, scan, simulate,
                     source_signal)
from arraylib.plotting import METHOD_COLOURS, polar_spectrum, save


def resolve_sources(signals, pos, pairs, truth):
    """Per-method errors against a known set of true bearings."""
    out = {}
    for method in ALL_METHODS:
        got = estimate(method, signals, pos, pairs, num_sources=len(truth))
        errors = match_to_truth(got, truth)
        out[method] = (got, errors)
    return out


def separation_sweep(signal, pos, pairs, args):
    """
    How close can two sources get before each method stops separating them?

    Both sources are placed symmetrically about due north and walked together
    until the estimators can no longer tell them apart.
    """
    separations = np.arange(10.0, 95.0, 5.0)
    rms = {m: [] for m in ALL_METHODS}
    found = {m: [] for m in ALL_METHODS}

    for sep in separations:
        truth = [(-sep / 2) % 360.0, (sep / 2) % 360.0]
        signals = simulate.simulate_multi_stationary(
            signal, truth, snr_db=args.snr,
            rng=np.random.default_rng(args.seed))
        for method, (got, errors) in resolve_sources(signals, pos, pairs,
                                                     truth).items():
            finite = [e for e in errors if np.isfinite(e)]
            rms[method].append(np.sqrt(np.mean(np.square(finite)))
                               if finite else np.nan)
            found[method].append(len(got))

    return separations, rms, found


def main() -> None:
    args = parse_args(__doc__,
                      azimuths=dict(type=float, nargs="+",
                                    default=[30.0, 75.0, 120.0],
                                    help="true source azimuths in deg"),
                      separation_sweep=dict(action="store_true",
                                            help="sweep two-source separation"))

    pos = positions()
    pairs = geometry.microphone_pairs(args.pairs)
    signal, label = source_signal(args.duration,
                                  prefer_recorded=not args.synthetic)
    truth = list(args.azimuths)

    signals = simulate.simulate_multi_stationary(
        signal, truth, snr_db=args.snr, rng=np.random.default_rng(args.seed))

    print("Experiment 2 -- multiple stationary sources")
    print(f"  source       : {label}")
    print(f"  true azimuths: {', '.join(f'{a:.0f}' for a in truth)} deg")
    print(f"  SNR          : {args.snr:.0f} dB")
    print()
    print(f"  {'method':<10} {'found':>6}  {'estimates':<28} {'RMS error':>10}")

    results = resolve_sources(signals, pos, pairs, truth)
    for method, (got, errors) in results.items():
        finite = [e for e in errors if np.isfinite(e)]
        rms = np.sqrt(np.mean(np.square(finite))) if finite else float("nan")
        shown = ", ".join(f"{g:.1f}" for g in sorted(got)) or "-"
        misses = sum(1 for e in errors if not np.isfinite(e))
        suffix = f"  ({misses} missed)" if misses else ""
        print(f"  {method:<10} {len(got):>6}  {shown:<28} "
              f"{rms:>7.2f} deg{suffix}")

    n_panels = len(SCAN_METHODS)
    if args.separation_sweep:
        fig = plt.figure(figsize=(13, 7))
        grid = (2, n_panels)
    else:
        fig = plt.figure(figsize=(13, 3.8))
        grid = (1, n_panels)

    for i, method in enumerate(SCAN_METHODS):
        ax = fig.add_subplot(grid[0], grid[1], i + 1, projection="polar")
        azimuths, spectrum = scan(method, signals, pos, pairs,
                                  num_sources=len(truth))
        polar_spectrum(ax, azimuths, spectrum, label=method,
                       true_azimuths=truth)
        ax.set_title(method, fontsize=10, pad=14)

    if args.separation_sweep:
        print("\n  sweeping two-source separation ...")
        separations, rms, found = separation_sweep(signal, pos, pairs, args)

        ax = fig.add_subplot(2, 2, 3)
        for method in ALL_METHODS:
            ax.plot(separations, rms[method], marker="o", markersize=3,
                    label=method, color=METHOD_COLOURS.get(method))
        ax.set_xlabel("true separation between sources [deg]")
        ax.set_ylabel("RMS azimuth error [deg]")
        ax.set_title("Accuracy vs source separation", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(2, 2, 4)
        for method in ALL_METHODS:
            ax.plot(separations, found[method], marker="o", markersize=3,
                    label=method, color=METHOD_COLOURS.get(method))
        ax.axhline(2, color="#cc2222", linestyle="--", linewidth=1.0,
                   label="true count")
        ax.set_xlabel("true separation between sources [deg]")
        ax.set_ylabel("sources detected")
        ax.set_title("Did the method resolve both sources?", fontsize=10)
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        print(f"  {'method':<10} smallest separation resolved as 2 sources")
        for method in ALL_METHODS:
            ok = [s for s, n in zip(separations, found[method]) if n >= 2]
            print(f"  {method:<10} "
                  f"{f'{min(ok):.0f} deg' if ok else 'never':>12}")

    fig.suptitle(f"Multiple stationary sources at "
                 f"{', '.join(f'{a:.0f}' for a in truth)} deg "
                 f"({args.snr:.0f} dB SNR)", fontsize=11)
    fig.tight_layout()
    name = ("exp2_multi_stationary_sweep.png" if args.separation_sweep
            else "exp2_multi_stationary.png")
    path = save(fig, FIGURES / name)
    print(f"\n  figure -> {path.relative_to(FIGURES.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
