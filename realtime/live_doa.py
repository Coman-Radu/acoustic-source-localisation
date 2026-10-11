#!/usr/bin/env python3
"""
Live direction-of-arrival display for the 16-channel array.

Replaces the nine near-identical real-time scripts the project accumulated --
one per estimator, each with its own copy of the array geometry, its own
capture loop and its own plotting code.  The estimator is now a command-line
option, so the capture path and the display are shared and there is exactly one
definition of the array.

    python realtime/live_doa.py --list-devices
    python realtime/live_doa.py --method music --device 3
    python realtime/live_doa.py --method srp --frame 0.1 --channels 16

Requires ``sounddevice`` and the PortAudio system library:

    sudo apt install libportaudio2       # Debian / Ubuntu
    pip install sounddevice
"""

from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arraylib import config, geometry                              # noqa: E402
from arraylib.detect import find_sources, sound_present            # noqa: E402
from arraylib.estimators import (delay_and_sum, esprit, music,     # noqa: E402
                                 mvdr, srp_phat, tdoa_direction)
from arraylib.plotting import polar_spectrum                       # noqa: E402

METHODS = ("das", "mvdr", "srp", "music", "esprit", "tdoa")


def require_sounddevice():
    """
    Import ``sounddevice``, with an actionable message if it is unavailable.

    The library is an optional dependency: everything except live capture
    works without it, and it fails at import time when PortAudio is missing
    rather than when a device is opened.
    """
    try:
        import sounddevice as sd
    except OSError as exc:
        raise SystemExit(
            f"sounddevice could not load PortAudio ({exc}).\n"
            f"Install the system library first:\n"
            f"    sudo apt install libportaudio2      # Debian / Ubuntu\n"
            f"    brew install portaudio              # macOS"
        ) from exc
    except ImportError as exc:
        raise SystemExit(
            "sounddevice is not installed. Install it with:\n"
            "    pip install sounddevice"
        ) from exc
    return sd


def run_estimator(method: str, frame: np.ndarray, pos: np.ndarray, pairs):
    """
    Run one estimator on a captured frame.

    Returns:
        ``(azimuths, spectrum, peaks)``.  The parametric estimators have no
        spatial spectrum to show, so ``azimuths`` and ``spectrum`` are ``None``
        and only ``peaks`` is populated.
    """
    if method == "das":
        azimuths, spectrum = delay_and_sum(frame, pos)
    elif method == "mvdr":
        azimuths, spectrum = mvdr(frame, pos)
    elif method == "srp":
        azimuths, spectrum = srp_phat(frame, pos, pairs)
    elif method == "music":
        azimuths, spectrum, _ = music(frame, pos)
    elif method == "esprit":
        result = esprit(frame)
        return None, None, [{"azimuth": a, "elevation": e}
                            for a, e in zip(result["azimuth"],
                                            result["elevation"])]
    elif method == "tdoa":
        result = tdoa_direction(frame)
        return None, None, [result]
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    peaks, _ = find_sources(azimuths, spectrum)
    return azimuths, spectrum, [dict(p) for p in peaks]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live DOA display for the 4x4 microphone array.")
    parser.add_argument("--method", default="srp", choices=METHODS,
                        help="estimator to run (default: srp)")
    parser.add_argument("--device", type=int, default=None,
                        help="input device index; see --list-devices")
    parser.add_argument("--channels", type=int, default=config.CHANNELS,
                        help=f"input channels (default: {config.CHANNELS})")
    parser.add_argument("--samplerate", type=int, default=config.SAMPLE_RATE,
                        help=f"capture rate (default: {config.SAMPLE_RATE})")
    parser.add_argument("--frame", type=float, default=0.2,
                        help="analysis frame length in seconds")
    parser.add_argument("--pairs", default="perimeter",
                        choices=["adjacent", "perimeter", "all"],
                        help="channel pairs for SRP-PHAT")
    parser.add_argument("--threshold", type=float, default=0.002,
                        help="RMS gate below which frames are skipped")
    parser.add_argument("--list-devices", action="store_true",
                        help="print the available audio devices and exit")
    parser.add_argument("--text", action="store_true",
                        help="print bearings instead of opening a plot window")
    args = parser.parse_args()

    sd = require_sounddevice()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.channels != config.CHANNELS:
        print(f"warning: the array geometry describes {config.CHANNELS} "
              f"elements but {args.channels} channels were requested; "
              f"estimates will be wrong unless the geometry is adjusted.",
              file=sys.stderr)

    pos = geometry.positions_array()
    pairs = geometry.microphone_pairs(args.pairs)
    frame_samples = int(round(args.frame * args.samplerate))

    audio: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        try:
            audio.put_nowait(indata.copy())
        except queue.Full:
            pass        # drop the frame rather than block the audio thread

    print(f"capturing {args.channels} channels at {args.samplerate} Hz, "
          f"{args.frame * 1000:.0f} ms frames, method = {args.method}")
    print("press Ctrl+C to stop")

    stream = sd.InputStream(device=args.device, channels=args.channels,
                            samplerate=args.samplerate,
                            blocksize=frame_samples, callback=callback)

    if args.text:
        _run_text(stream, audio, args, pos, pairs)
    else:
        _run_plot(stream, audio, args, pos, pairs)


def _run_text(stream, audio, args, pos, pairs) -> None:
    """Print bearings to stdout; useful over SSH with no display."""
    with stream:
        try:
            while True:
                frame = audio.get()
                if not sound_present(frame, args.threshold):
                    continue
                _, _, peaks = run_estimator(args.method, frame, pos, pairs)
                if peaks:
                    print("  ".join(
                        f"{p['azimuth']:6.1f} deg"
                        + (f" el {p['elevation']:5.1f}" if "elevation" in p
                           else "")
                        for p in peaks))
        except KeyboardInterrupt:
            print("\nstopped")


def _run_plot(stream, audio, args, pos, pairs) -> None:
    """Animated polar display of the spatial spectrum and detected bearings."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                           figsize=(6.5, 6.5))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.05)
    (line,) = ax.plot([], [], linewidth=1.6, color="#4878a8")
    markers = ax.scatter([], [], s=90, marker="v", color="#cc2222", zorder=4)
    title = ax.set_title(f"{args.method.upper()} -- waiting for signal",
                         fontsize=11, pad=18)

    state: dict = {"peaks": []}

    def update(_frame):
        drained = None
        while not audio.empty():          # always show the freshest frame
            drained = audio.get_nowait()
        if drained is None:
            return line, markers, title

        if not sound_present(drained, args.threshold):
            title.set_text(f"{args.method.upper()} -- quiet")
            return line, markers, title

        azimuths, spectrum, peaks = run_estimator(args.method, drained, pos,
                                                  pairs)
        state["peaks"] = peaks

        if spectrum is not None:
            theta = np.radians(np.append(azimuths, azimuths[0]))
            line.set_data(theta, np.append(spectrum, spectrum[0]))

        if peaks:
            markers.set_offsets(np.column_stack([
                np.radians([p["azimuth"] for p in peaks]),
                np.full(len(peaks), 1.0)]))
            title.set_text(f"{args.method.upper()} -- "
                           + ", ".join(f"{p['azimuth']:.0f} deg"
                                       for p in peaks))
        else:
            markers.set_offsets(np.empty((0, 2)))
            title.set_text(f"{args.method.upper()} -- no source")
        return line, markers, title

    with stream:
        _anim = animation.FuncAnimation(fig, update, interval=80,
                                        blit=False, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
    print("stopped")


if __name__ == "__main__":
    main()
