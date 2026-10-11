#!/usr/bin/env python3
"""
Experiment 5 -- Doppler time-scaling, and which angle it actually damages.

A source with radial velocity rescales the received time axis rather than
merely shifting it: ``x_m(t) = s(alpha(t) * t + tau_m)`` with
``alpha = c / (c - v_r)``.

The interesting part is what that does *not* break.  In the far field the warp
is common to the whole array, so every inter-element delay is divided by the
same ``alpha``.  Azimuth is read from ``atan2(u, v)`` and both direction
cosines shrink by that identical factor, so **azimuth is invariant to Doppler**.
Elevation is read from the *magnitude* ``hypot(u, v)``, which is not
scale-invariant, so the whole bias lands there: a source in the horizontal
plane appears to climb.

This matters because the project's original notes attributed large bearing
errors to Doppler.  Bearing is the one thing it leaves alone; the real
casualties are elevation, and -- when ``alpha`` varies *within* a frame -- the
coherence that every delay estimator depends on.

Compensation resamples onto a uniform emission-time grid, which removes the
scale factor and restores elevation.

    python experiments/exp5_doppler_compensation.py --velocity 60
    python experiments/exp5_doppler_compensation.py --acceleration 300
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (FIGURES, geometry, parse_args, positions, simulate,
                     source_signal)
from arraylib import config
from arraylib.detect import angular_error
from arraylib.estimators import esprit, least_squares_direction
from arraylib.plotting import save

#: Only the estimators that report elevation as well as azimuth are useful
#: here, since elevation is where the Doppler bias shows up.
METHODS = ("ESPRIT", "TDOA-LSQ")


def _estimate(method, signals, pos, pairs):
    """Azimuth and elevation in degrees from one estimator."""
    if method == "ESPRIT":
        r = esprit(signals, num_sources=1)
        return r["azimuth"][0], r["elevation"][0]
    r = least_squares_direction(signals, pairs, pos)
    return r["azimuth"], r["elevation"]


def main() -> None:
    args = parse_args(__doc__,
                      azimuth=dict(type=float, default=40.0,
                                   help="true bearing in deg"),
                      velocity=dict(type=float, default=60.0,
                                    help="maximum radial velocity in m/s"),
                      acceleration=dict(type=float, default=0.0,
                                        help="radial acceleration in m/s^2; "
                                             "non-zero makes alpha vary "
                                             "within each frame"))

    fs = config.SAMPLE_RATE
    pos = positions()
    pairs = geometry.microphone_pairs("all")
    signal, label = source_signal(max(args.duration * 4, 2.0),
                                  prefer_recorded=not args.synthetic)

    def source_fn(u):
        idx = np.clip((np.asarray(u) * fs).astype(np.int64), 0, len(signal) - 1)
        return signal[idx]

    speeds = np.linspace(0.0, args.velocity, 9)

    print("Experiment 5 -- Doppler time-scaling")
    print(f"  source       : {label}")
    print(f"  true bearing : {args.azimuth:.1f} deg, "
          f"true elevation 0.0 deg (horizontal plane)")
    print(f"  frame        : {args.duration * 1000:.0f} ms, "
          f"{args.snr:.0f} dB SNR")
    if args.acceleration:
        print(f"  acceleration : {args.acceleration:.0f} m/s^2 "
              f"(alpha varies within the frame)")
    print()
    print(f"  {'v_r':>6} {'alpha':>7} | "
          + " | ".join(f"{m:^31}" for m in METHODS))
    print(f"  {'[m/s]':>6} {'':>7} | "
          + " | ".join(f"{'az err':>7} {'elev raw':>9} {'elev comp':>10}"
                       for _ in METHODS))

    data = {m: {"az": [], "el_raw": [], "el_comp": []} for m in METHODS}

    for v in speeds:
        rng = np.random.default_rng(args.seed)
        if args.acceleration:
            def radial_velocity(t, v=v):
                return v + args.acceleration * (np.asarray(t)
                                                - args.duration / 2.0)
        else:
            radial_velocity = float(v)

        _, signals, alpha = simulate.simulate_moving(
            source_fn, args.duration, args.azimuth, radial_velocity,
            snr_db=args.snr, rng=rng)
        corrected = simulate.compensate_time_scaling(signals, alpha, fs)

        row = f"  {v:>6.1f} {float(np.mean(alpha)):>7.4f} | "
        cells = []
        for method in METHODS:
            az_raw, el_raw = _estimate(method, signals, pos, pairs)
            _, el_comp = _estimate(method, corrected, pos, pairs)
            az_err = float(angular_error(az_raw, args.azimuth))

            data[method]["az"].append(abs(az_err))
            data[method]["el_raw"].append(el_raw)
            data[method]["el_comp"].append(el_comp)
            cells.append(f"{az_err:>+7.2f} {el_raw:>9.2f} {el_comp:>10.2f}")
        print(row + " | ".join(cells))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    ax = axes[0]
    for method in METHODS:
        ax.plot(speeds, data[method]["az"], marker="o", markersize=4,
                label=method)
    ax.set_xlabel("radial velocity [m/s]")
    ax.set_ylabel("absolute azimuth error [deg]")
    ax.set_title("Azimuth is invariant to Doppler", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for method in METHODS:
        ax.plot(speeds, data[method]["el_raw"], marker="o", markersize=4,
                label=f"{method} raw")
        ax.plot(speeds, data[method]["el_comp"], marker="s", markersize=4,
                linestyle="--", label=f"{method} compensated")
    ax.axhline(0.0, color="#cc2222", linewidth=1.0, label="true elevation")
    ax.set_xlabel("radial velocity [m/s]")
    ax.set_ylabel("apparent elevation [deg]")
    ax.set_title("Elevation takes the whole bias", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # The predicted bias follows from the geometry alone: the direction-cosine
    # magnitude is divided by alpha, so apparent elevation is arccos(1/alpha).
    ax = axes[2]
    alphas = np.array([simulate.doppler_factor(v) for v in speeds])
    predicted = np.degrees(np.arccos(np.clip(1.0 / alphas, 0, 1)))
    ax.plot(speeds, predicted, color="#cc2222", linewidth=2.0,
            label="arccos(1/alpha), predicted")
    for method in METHODS:
        ax.plot(speeds, data[method]["el_raw"], marker="o", markersize=4,
                linestyle="none", label=f"{method} measured")
    ax.set_xlabel("radial velocity [m/s]")
    ax.set_ylabel("apparent elevation [deg]")
    ax.set_title("Measured bias matches the prediction", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    suffix = (f", {args.acceleration:.0f} m/s^2" if args.acceleration else "")
    fig.suptitle(f"Doppler time-scaling at {args.azimuth:.0f} deg bearing"
                 f"{suffix}", fontsize=11)
    fig.tight_layout()
    path = save(fig, FIGURES / "exp5_doppler_compensation.png")
    print(f"\n  figure -> {path.relative_to(FIGURES.parent)}")

    print(f"\n  at {speeds[-1]:.0f} m/s (alpha = {alphas[-1]:.4f}):")
    print(f"    predicted elevation bias arccos(1/alpha) = "
          f"{predicted[-1]:.2f} deg")
    for method in METHODS:
        print(f"    {method:<10} azimuth error "
              f"{data[method]['az'][-1]:.2f} deg | elevation "
              f"{data[method]['el_raw'][-1]:.2f} deg raw -> "
              f"{data[method]['el_comp'][-1]:.2f} deg compensated")
    plt.close(fig)


if __name__ == "__main__":
    main()
