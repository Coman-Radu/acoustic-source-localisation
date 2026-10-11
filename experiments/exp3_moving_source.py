#!/usr/bin/env python3
"""
Experiment 3 -- a moving source.

Tracks a source crossing the array and measures how each estimator degrades
with motion.  Two separate effects are at work and this experiment is built to
tell them apart:

* **motion blur** -- the bearing changes during the analysis window, so the
  frame has no single correct answer;
* **Doppler time-scaling** -- the received waveform is time-warped, which
  breaks the common-waveform assumption that every delay estimator relies on.

Shortening the frame reduces the first and does nothing for the second, so the
frame-length sweep separates them.

    python experiments/exp3_moving_source.py --velocity 10 --frame 0.1
    python experiments/exp3_moving_source.py --frame-sweep
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (ALL_METHODS, FIGURES, estimate, geometry, parse_args,
                     positions, simulate, source_signal)
from arraylib import config
from arraylib.detect import angular_error
from arraylib.plotting import METHOD_COLOURS, plot_error_series, save


def track(signals, pos, pairs, frame_samples, true_azimuth_fn, fs):
    """
    Run every estimator frame by frame over a long capture.

    Returns:
        ``(times, truth, {method: estimates})`` with angles in degrees.
    """
    n_frames = signals.shape[0] // frame_samples
    times, truth = [], []
    tracks = {m: [] for m in ALL_METHODS}

    for f in range(n_frames):
        lo = f * frame_samples
        frame = signals[lo:lo + frame_samples, :]
        # Evaluate truth at the frame centre.
        t_mid = (lo + frame_samples / 2) / fs
        times.append(t_mid)
        truth.append(true_azimuth_fn(t_mid))

        for method in ALL_METHODS:
            got = estimate(method, frame, pos, pairs, num_sources=1)
            tracks[method].append(got[0] if got else np.nan)

    return np.array(times), np.array(truth), {k: np.array(v)
                                              for k, v in tracks.items()}


def main() -> None:
    args = parse_args(__doc__,
                      velocity=dict(type=float, default=10.0,
                                    help="tangential speed in m/s"),
                      distance=dict(type=float, default=50.0,
                                    help="range at closest approach in m"),
                      total=dict(type=float, default=4.0,
                                 help="total capture length in s"),
                      frame=dict(type=float, default=0.1,
                                 help="analysis frame length in s"),
                      frame_sweep=dict(action="store_true",
                                       help="sweep the frame length"))

    fs = config.SAMPLE_RATE
    pos = positions()
    pairs = geometry.microphone_pairs(args.pairs)
    rng = np.random.default_rng(args.seed)

    # Source crosses in front of the array at constant range, so the bearing
    # sweeps while the Doppler factor changes sign at closest approach.
    def true_azimuth(t: float) -> float:
        x = args.velocity * (t - args.total / 2.0)
        return float(np.degrees(np.arctan2(x, args.distance)) % 360.0)

    def radial_velocity(t):
        x = args.velocity * (np.asarray(t) - args.total / 2.0)
        r = np.hypot(x, args.distance)
        # Closing speed is the projection of velocity onto the line of sight;
        # negative while approaching under this sign convention.
        return -args.velocity * x / r

    signal, label = source_signal(args.total * 1.5,
                                  prefer_recorded=not args.synthetic)

    print("Experiment 3 -- moving source")
    print(f"  source     : {label}")
    print(f"  speed      : {args.velocity:.1f} m/s at {args.distance:.0f} m")
    print(f"  bearing     : {true_azimuth(0):.1f} -> "
          f"{true_azimuth(args.total):.1f} deg over {args.total:.1f} s")
    peak_vr = float(np.max(np.abs(radial_velocity(
        np.linspace(0, args.total, 500)))))
    print(f"  peak radial : {peak_vr:.1f} m/s "
          f"(Doppler factor {simulate.doppler_factor(peak_vr):.4f})")

    frames = ([0.05, 0.1, 0.2, 0.4, 0.8] if args.frame_sweep else [args.frame])
    sweep: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
    detail = None

    for frame_len in frames:
        signals = _resimulate_with_sweep(signal, true_azimuth, radial_velocity,
                                         args, fs, rng)

        frame_samples = int(round(frame_len * fs))
        times, truth, tracks = track(signals, pos, pairs, frame_samples,
                                     true_azimuth, fs)

        print(f"\n  frame {frame_len * 1000:.0f} ms "
              f"({frame_samples} samples, "
              f"bearing moves {abs(true_azimuth(frame_len) - true_azimuth(0)):.1f} "
              f"deg per frame)")
        print(f"    {'method':<10} {'RMS error':>10} {'misses':>8}")
        for method in ALL_METHODS:
            err = angular_error(tracks[method], truth)
            finite = err[np.isfinite(err)]
            rms = float(np.sqrt(np.mean(finite ** 2))) if len(finite) else np.nan
            misses = int(np.sum(~np.isfinite(err)))
            sweep[method].append(rms)
            print(f"    {method:<10} {rms:>7.2f} deg {misses:>8}")

        if frame_len == args.frame or detail is None:
            detail = (times, truth, tracks, frame_len)

    times, truth, tracks, frame_len = detail

    fig = plt.figure(figsize=(13, 4.2) if not args.frame_sweep
                     else (13, 8))
    rows = 2 if args.frame_sweep else 1

    ax = fig.add_subplot(rows, 2, 1)
    ax.plot(times, truth, color="#cc2222", linewidth=2.0, label="true bearing")
    for method in ALL_METHODS:
        ax.plot(times, tracks[method], marker=".", markersize=3,
                linewidth=1.0, label=method,
                color=METHOD_COLOURS.get(method))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("azimuth [deg]")
    ax.set_title(f"Tracking a {args.velocity:.0f} m/s pass at "
                 f"{args.distance:.0f} m ({frame_len * 1000:.0f} ms frames)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = fig.add_subplot(rows, 2, 2)
    plot_error_series(ax, times,
                      {m: angular_error(tracks[m], truth)
                       for m in ALL_METHODS})
    ax.set_title("Signed bearing error per frame", fontsize=10)

    if args.frame_sweep:
        ax = fig.add_subplot(2, 1, 2)
        for method in ALL_METHODS:
            ax.plot(np.array(frames) * 1000, sweep[method], marker="o",
                    label=method, color=METHOD_COLOURS.get(method))
        blur = [abs(true_azimuth(f) - true_azimuth(0)) for f in frames]
        ax.plot(np.array(frames) * 1000, blur, color="#cc2222",
                linestyle="--", label="bearing change per frame")
        ax.set_xlabel("analysis frame length [ms]")
        ax.set_ylabel("RMS azimuth error [deg]")
        ax.set_xscale("log")
        ax.set_title("Shorter frames trade noise against motion blur",
                     fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, ncol=3)

    fig.tight_layout()
    name = ("exp3_moving_source_sweep.png" if args.frame_sweep
            else "exp3_moving_source.png")
    path = save(fig, FIGURES / name)
    print(f"\n  figure -> {path.relative_to(FIGURES.parent)}")
    plt.close(fig)


def _resimulate_with_sweep(signal, true_azimuth, radial_velocity, args, fs,
                           rng):
    """
    Build a capture in which the bearing really does sweep.

    ``simulate_moving`` holds the bearing fixed, which captures Doppler but not
    motion blur.  Stitching the capture from short blocks, each simulated at
    its own instantaneous bearing and Doppler factor, reproduces both -- which
    is the whole point of this experiment.
    """
    block = int(round(0.01 * fs))                 # 10 ms blocks
    n_total = int(round(args.total * fs))
    out = np.zeros((n_total, config.CHANNELS))

    for lo in range(0, n_total, block):
        hi = min(lo + block, n_total)
        t_mid = (lo + (hi - lo) / 2) / fs
        azimuth = true_azimuth(t_mid)
        v_r = float(np.atleast_1d(radial_velocity(t_mid))[0])
        alpha = float(simulate.doppler_factor(v_r))

        # Resample this block of the source at the local Doppler factor.
        t_block = np.arange(lo, hi) / fs
        idx = np.clip((alpha * t_block * fs).astype(int), 0, len(signal) - 1)
        chunk = signal[idx]

        out[lo:hi, :] = simulate.simulate_stationary(
            chunk, azimuth, distance=args.distance, snr_db=None)

    return simulate.add_noise_at_snr(out, args.snr, rng)


if __name__ == "__main__":
    main()
