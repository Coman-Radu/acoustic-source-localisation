#!/usr/bin/env python3
"""
Test suite for the array-processing library.

Several of these are regression tests for defects found while consolidating
the original scripts; each is marked REGRESSION and names what it guards
against, so the failure mode cannot quietly return.

Runs under pytest, or standalone:

    python -m pytest tests/ -v
    python tests/test_estimators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arraylib import config, geometry, simulate, sources
from arraylib.detect import angular_error, find_sources, refine_peak
from arraylib.estimators import (delay_and_sum, esprit, gcc_phat,
                                 least_squares_direction, music, mvdr,
                                 phase_delay, srp_phat, tdoa_direction)

FS = config.SAMPLE_RATE
POSITIONS = geometry.positions_array()
PAIRS = geometry.microphone_pairs("perimeter")


def _drone(duration=0.4):
    return sources.harmonic_drone(duration)[1]


def _capture(azimuth, snr_db=20.0, seed=0, duration=0.4):
    return simulate.simulate_stationary(
        _drone(duration), azimuth, snr_db=snr_db,
        rng=np.random.default_rng(seed))


# --------------------------------------------------------------- geometry ---

def test_geometry_is_a_uniform_grid():
    """Element positions form a centred 4x4 grid at the stated spacing."""
    pos = geometry.positions_array()
    assert pos.shape == (config.CHANNELS, 2)
    assert np.allclose(pos.mean(axis=0), 0.0), "array is not centred"
    for axis in (0, 1):
        unique = np.unique(np.round(pos[:, axis], 9))
        assert len(unique) == 4
        assert np.allclose(np.diff(unique), config.MIC_SPACING)


def test_channel_layouts_describe_the_same_physical_array():
    """
    REGRESSION: the simulation and real-time code used different channel maps.

    Six copies of the geometry had drifted apart; the simulation ordering
    assigned 15 of 16 channels to different positions than the as-wired one.
    Both must at least describe the same set of physical points.
    """
    hardware = {tuple(p) for p in
                np.round(geometry.positions_array(order=geometry.HARDWARE_ORDER), 9)}
    simulated = {tuple(p) for p in
                 np.round(geometry.positions_array(order=geometry.SIM_ORDER), 9)}
    assert hardware == simulated

    # ...and they are genuinely different orderings, which is why only
    # HARDWARE_ORDER may be used to interpret a real recording.
    assert not np.allclose(geometry.positions_array(geometry.MIC_SPACING,
                                                    geometry.HARDWARE_ORDER),
                           geometry.positions_array(geometry.MIC_SPACING,
                                                     geometry.SIM_ORDER))


def test_doublet_subarrays_have_a_constant_displacement():
    """
    REGRESSION: ESPRIT needs two subarrays related by one fixed translation.

    The original subarrays were built by hand from row lists and did not
    satisfy this, which is part of why the estimator could not work.
    """
    for axis in ("x", "y"):
        sub1, sub2, displacement = geometry.doublet_subarrays(axis)
        assert len(sub1) == len(sub2) == 12
        offsets = POSITIONS[sub2] - POSITIONS[sub1]
        assert np.allclose(offsets, displacement), \
            f"{axis} doublet displacement is not constant"
        assert np.isclose(np.linalg.norm(displacement), config.MIC_SPACING)


def test_steering_vector_matches_hand_computed_phase():
    """Steering phase equals 2*pi*f*(p.d)/c for a known direction."""
    freqs = np.array([2000.0])
    azimuth = 37.0
    a = geometry.steering_vector(azimuth, freqs, POSITIONS)
    direction = geometry.direction_unit_vector(azimuth)
    expected = np.exp(2j * np.pi * freqs[0]
                      * (POSITIONS @ direction) / config.SPEED_OF_SOUND)
    assert np.allclose(a[0], expected)


# ------------------------------------------------------------ simulation ---

def test_snr_is_delivered_as_requested():
    clean = simulate.simulate_stationary(_drone(), 30.0, snr_db=None)
    for target in (0.0, 10.0, 20.0):
        noisy = simulate.add_noise_at_snr(clean, target,
                                          np.random.default_rng(1))
        measured = 10 * np.log10(np.mean(clean ** 2)
                                 / np.mean((noisy - clean) ** 2))
        assert abs(measured - target) < 0.5


def test_element_nearer_the_source_receives_earlier():
    """
    REGRESSION: the propagation delay had the wrong sign.

    With the sign inverted every subspace and beamforming estimator reported
    the reciprocal bearing, 180 deg away from the truth.
    """
    azimuth = 0.0                      # due north
    signals = simulate.simulate_stationary(_drone(), azimuth, snr_db=None)
    direction = geometry.direction_unit_vector(azimuth)
    projections = POSITIONS @ direction

    north = int(np.argmax(projections))      # closest to the source
    south = int(np.argmin(projections))
    lag, _, _ = gcc_phat(signals[:, north], signals[:, south], fs=FS)

    # The southern element is further away, so it arrives later: a positive
    # lag for (north, south) ordering.
    assert lag > 0, "element nearer the source did not arrive first"


def test_doppler_factor_and_compensation_round_trip():
    f0 = 1500.0
    v_r = 25.0
    _, signals, alpha = simulate.simulate_moving(
        lambda u: np.sin(2 * np.pi * f0 * u), 0.3, 45.0, v_r, snr_db=None)

    def peak_frequency(x):
        spectrum = np.abs(np.fft.rfft(x))
        return np.fft.rfftfreq(len(x), 1 / FS)[int(np.argmax(spectrum))]

    observed = peak_frequency(signals[:, 0])
    assert abs(observed - f0 * simulate.doppler_factor(v_r)) < 5.0

    corrected = simulate.compensate_time_scaling(signals, alpha, FS)
    assert abs(peak_frequency(corrected[:, 0]) - f0) < 5.0


# ---------------------------------------------------------------- detect ---

def test_angular_error_takes_the_short_way_round():
    assert np.isclose(angular_error(5.0, 355.0), 10.0)
    assert np.isclose(angular_error(355.0, 5.0), -10.0)
    assert np.isclose(angular_error(10.0, 20.0), -10.0)
    # Antipodal is reported as -180; the magnitude is what callers compare.
    assert np.isclose(abs(angular_error(180.0, 0.0)), 180.0)
    # The result always lies in [-180, 180).
    diffs = angular_error(np.arange(0.0, 360.0), 0.0)
    assert np.all(diffs >= -180.0) and np.all(diffs < 180.0)


def test_peak_at_the_seam_is_found():
    """
    REGRESSION: a source at 0 deg was missed entirely.

    The spectrum was padded by only a few samples, so a broad main lobe ran
    into the array edge, its base was truncated, and scipy computed a
    prominence near zero for what was actually the global maximum.
    """
    azimuths = np.arange(0.0, 360.0, 2.0)
    for true in (0.0, 2.0, 358.0, 180.0):
        delta = (azimuths - true + 180.0) % 360.0 - 180.0
        spectrum = np.exp(-0.5 * (delta / 8.0) ** 2)
        peaks, _ = find_sources(azimuths, spectrum, max_sources=1)
        assert peaks, f"no peak found for a source at {true} deg"
        assert abs(angular_error(peaks[0].azimuth, true)) < 1.0


def test_sub_grid_refinement_beats_the_grid():
    """A peak between grid points is located to better than the grid step."""
    azimuths = np.arange(0.0, 360.0, 2.0)
    true = 7.0
    delta = (azimuths - true + 180.0) % 360.0 - 180.0
    spectrum = np.exp(-0.5 * (delta / 8.0) ** 2)

    coarse, _ = find_sources(azimuths, spectrum, max_sources=1, refine=False)
    fine, _ = find_sources(azimuths, spectrum, max_sources=1, refine=True)
    assert abs(angular_error(coarse[0].azimuth, true)) > 0.5
    assert abs(angular_error(fine[0].azimuth, true)) < 0.1


def test_refine_peak_wraps_around_the_axis():
    spectrum = np.array([1.0, 0.5, 0.1, 0.5])     # peak at index 0
    assert abs(refine_peak(spectrum, 0)) < 1e-9   # symmetric, no offset


# ------------------------------------------------------------- gcc-phat ---

def test_gcc_phat_recovers_a_known_delay():
    signal = _drone(0.3)
    for shift in (-17, -4, 6, 23):
        lag, _, _ = gcc_phat(signal, np.roll(signal, shift), fs=FS)
        assert abs(lag * FS - shift) < 1.5, \
            f"expected {shift} samples, measured {lag * FS:.2f}"


def test_gcc_phat_survives_a_sparse_spectrum():
    """
    REGRESSION: PHAT whitening used an absolute magnitude floor.

    A rotor-harmonic spectrum leaves most bins empty; dividing those by a tiny
    constant lifted numerical noise to full unit magnitude and buried the
    correlation peak, so a known 20-sample delay was reported as -0.25.
    """
    t = np.arange(int(0.2 * FS)) / FS
    tone = np.sin(2 * np.pi * 1200.0 * t)         # one bin, everything else ~0
    lag, _, _ = gcc_phat(tone, np.roll(tone, 11), fs=FS)
    assert abs(lag * FS - 11) < 1.5


def test_max_tau_bounds_the_search_geometrically():
    """
    REGRESSION: an unbounded search returned impossible delays.

    The original analysis recorded TDOAs of +/-10 ms across a 120 mm array,
    29x larger than the geometry permits, which is what produced its very
    large bearing errors.
    """
    signals = _capture(30.0, snr_db=5.0)
    i, j = 2, 15
    baseline = float(np.linalg.norm(POSITIONS[i] - POSITIONS[j]))
    limit = baseline / config.SPEED_OF_SOUND

    lag, _, _ = gcc_phat(signals[:, i], signals[:, j], fs=FS,
                         max_tau=1.25 * limit)
    assert abs(lag) <= 1.3 * limit, \
        f"delay {lag * 1e6:.0f} us exceeds the geometric maximum " \
        f"{limit * 1e6:.0f} us"


# ----------------------------------------------------------- estimators ---

def test_every_estimator_finds_a_single_source():
    """All estimators agree on an isolated source to within a degree."""
    for azimuth in (0.0, 47.0, 123.0, 271.0):
        signals = _capture(azimuth, snr_db=20.0)

        for name, (grid, spectrum) in (
                ("DAS", delay_and_sum(signals, POSITIONS)),
                ("MVDR", mvdr(signals, POSITIONS)),
                ("SRP", srp_phat(signals, POSITIONS, PAIRS)),
                ("MUSIC", music(signals, POSITIONS, num_sources=1)[:2])):
            peaks, _ = find_sources(grid, spectrum, max_sources=1)
            assert peaks, f"{name} found nothing at {azimuth} deg"
            assert abs(angular_error(peaks[0].azimuth, azimuth)) < 1.5, \
                f"{name} off at {azimuth} deg: {peaks[0].azimuth:.1f}"

        got = esprit(signals, num_sources=1)["azimuth"][0]
        assert abs(angular_error(got, azimuth)) < 1.5, \
            f"ESPRIT off at {azimuth} deg: {got:.1f}"

        got = tdoa_direction(signals)["azimuth"]
        assert abs(angular_error(got, azimuth)) < 1.5


def test_esprit_actually_depends_on_the_source_direction():
    """
    REGRESSION: ESPRIT returned angles unrelated to the source.

    The original code assigned the same signal-subspace matrix to both
    subarrays, so the rotation operator was exactly the identity, every
    eigenvalue was 1 and every phase was 0.  This test fails for any
    implementation whose output does not track the true bearing.
    """
    bearings = np.array([20.0, 80.0, 140.0, 200.0, 260.0, 320.0])
    estimates = np.array([esprit(_capture(b, snr_db=25.0),
                                 num_sources=1)["azimuth"][0]
                          for b in bearings])

    errors = np.abs(angular_error(estimates, bearings))
    assert np.all(errors < 2.0), f"ESPRIT errors: {errors.round(2)}"

    # Guard specifically against a constant output.
    assert np.ptp(estimates) > 180.0, \
        "ESPRIT returned nearly the same angle for every direction"


def test_esprit_pairs_two_sources_correctly():
    """
    REGRESSION: sorting u and v independently mispaired the axes.

    Two sources at 30 and 120 deg gave u = (0.50, 0.87) and v = (0.87, -0.50);
    sorting each list separately paired (0.50, -0.50) and (0.87, 0.87) and
    reported 135 and 45 deg.  Joint diagonalisation fixes the pairing.
    """
    truth = [30.0, 120.0]
    signals = simulate.simulate_multi_stationary(
        _drone(0.5), truth, snr_db=25.0, rng=np.random.default_rng(0))
    got = sorted(esprit(signals, num_sources=2)["azimuth"])

    for estimate, true in zip(got, truth):
        assert abs(angular_error(estimate, true)) < 10.0, \
            f"ESPRIT reported {got}, expected about {truth}"


def test_music_resolves_two_close_sources():
    """MUSIC separates sources that the aperture-limited beamformers merge."""
    truth = [40.0, 70.0]
    signals = simulate.simulate_multi_stationary(
        _drone(0.5), truth, snr_db=25.0, rng=np.random.default_rng(0))

    grid, spectrum, _ = music(signals, POSITIONS, num_sources=2)
    peaks, _ = find_sources(grid, spectrum, max_sources=2)
    assert len(peaks) == 2, f"MUSIC found {len(peaks)} of 2 sources"

    found = sorted(p.azimuth for p in peaks)
    for estimate, true in zip(found, truth):
        assert abs(angular_error(estimate, true)) < 5.0


def test_azimuth_is_invariant_to_doppler():
    """
    A far-field Doppler factor scales both direction cosines equally, so
    ``atan2(u, v)`` is unchanged.  Elevation, read from the magnitude, is not.
    """
    azimuth = 40.0
    f0 = 1800.0

    def source(u):
        return np.sin(2 * np.pi * f0 * np.asarray(u))

    elevations = []
    for v_r in (0.0, 20.0, 50.0):
        _, signals, _ = simulate.simulate_moving(
            source, 0.3, azimuth, v_r, snr_db=30.0,
            rng=np.random.default_rng(0))
        result = esprit(signals, num_sources=1)
        assert abs(angular_error(result["azimuth"][0], azimuth)) < 2.0, \
            f"azimuth moved at v_r = {v_r} m/s"
        elevations.append(result["elevation"][0])

    # The bias is real and grows with speed, which is what compensation is for.
    assert elevations[-1] > elevations[0] + 10.0, \
        "expected a growing elevation bias with radial velocity"


def test_phase_delay_is_exact_for_a_narrowband_source():
    """Hilbert phase delay matches the geometric delay for a pure tone."""
    f0 = 2000.0
    azimuth = 30.0
    signals = simulate.simulate_stationary(
        sources.tone(0.3, f0)[1], azimuth, snr_db=40.0,
        rng=np.random.default_rng(0))

    direction = geometry.direction_unit_vector(azimuth)
    i, j = 2, 15
    expected = float((POSITIONS[i] - POSITIONS[j]) @ direction
                     / config.SPEED_OF_SOUND)
    measured = phase_delay(signals[:, i], signals[:, j], f0, FS)
    assert abs((measured - expected) * FS) < 0.2


def test_least_squares_direction_uses_all_baselines():
    signals = _capture(68.0, snr_db=20.0)
    result = least_squares_direction(signals, geometry.microphone_pairs("all"),
                                     POSITIONS)
    assert abs(angular_error(result["azimuth"], 68.0)) < 1.5


def _run_standalone() -> int:
    """Minimal runner so the suite works without pytest installed."""
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}\n          {exc}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
