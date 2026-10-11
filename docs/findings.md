# Defects found while consolidating the original code

The work in this repository began as four separate repositories and roughly
ninety loose scripts written over a six-month internship. Merging them into one
library meant putting implementations of the same quantity side by side for the
first time, and several disagreed. This document records what was wrong, how it
was confirmed, and what changed — both because the defects are interesting in
themselves and because a few of them invalidate conclusions in the original
project notes.

Every claim below is reproducible from `tests/test_estimators.py`, where each
item has a named regression test.

---

## 1. ESPRIT returned angles unrelated to the source

**Severity: the estimator did not work at all.**

The original implementation assigned the same matrix to both subarrays:

```python
Es1 = Es  # For subarray 1
Es2 = Es  # For subarray 2 (same, but represents displaced array)
Psi = np.linalg.lstsq(Es1, Es2, rcond=None)[0]
```

With identical operands the least-squares solution is exactly the identity, so
every eigenvalue of `Psi` is 1 and every phase is 0. ESPRIT recovers direction
*from* those phases, so there was no direction information left to recover.

Confirmed by feeding the original function simulated single sources from 0° to
270°:

| true azimuth | reported |
|---|---|
| 0° | nothing |
| 30° | nothing |
| 60° | nothing |
| 90° | 294.0°, 66.0° |
| 135° | nothing |
| 180° | nothing |
| 225° | 330.0°, 30.0° |
| 270° | 14.0°, 346.0° |

and by evaluating `Psi` directly, which came out as the 3×3 identity with
phases `[0, 0, 0]`.

The underlying problem was structural: the two subarrays were built by hand
from row lists and were not related by a single fixed translation, which is the
one property ESPRIT requires. `geometry.doublet_subarrays` now constructs them
by dropping the first or last column (or row), so the displacement is exactly
one element spacing for every pair — asserted in
`test_doublet_subarrays_have_a_constant_displacement`.

**Fixed** by selecting both subarrays *out of a common signal subspace* with
selection matrices, `Es1 = J1 @ Es` and `Es2 = J2 @ Es`, so the rotational
invariance actually exists. The corrected estimator goes from returning no
usable output at all to **0.04° mean error** on a single source, with the
tightest worst case of the six (0.07°) — it scans no grid, so it carries no
quantisation error.

Guarded by `test_esprit_actually_depends_on_the_source_direction`, which fails
for any implementation whose output does not track the true bearing — including
one that returns a constant.

---

## 2. ESPRIT mispaired the two axes on multiple sources

Once ESPRIT produced real phases, a second defect appeared. 2-D ESPRIT measures
one direction cosine per axis, and the two sets arrive in unrelated orders;
pairing them by sorting is wrong whenever more than one source is present.

Two sources at 30° and 120° have `u = (0.50, 0.87)` and `v = (0.87, −0.50)`.
Sorting each list independently pairs `(0.50, −0.50)` and `(0.87, 0.87)`, giving
135° and 45° — which is exactly what the estimator reported.

**Fixed** by joint diagonalisation: `Psi_x` is diagonalised once and its
eigenvector matrix is used to transform `Psi_y` into the same basis, so the two
diagonals line up source by source. Three sources at 30°/75°/120° now return
28.4°/72.1°/107.8° instead of nonsense.

---

## 3. Six copies of the array geometry had drifted apart

`get_mic_positions_4x4` was duplicated in six files. The five real-time scripts
agreed with each other; the simulation module used by the paper experiments did
not.

Both describe the **same physical array** — a centred 4×4 grid at 42 mm
spacing, 126 mm aperture — but they assign **15 of the 16 capture channels to
different positions**.

In pure simulation this is harmless, because the same mapping is used to
synthesise and to process the signals, so the error cancels. It stops being
harmless the moment real recorded audio is fed into the paper-experiment code
path, which would then produce confidently wrong bearings.

**Fixed** by making `geometry.py` the single source of truth.
`HARDWARE_ORDER` documents the as-wired channel map and is the default;
`SIM_ORDER` is retained, clearly marked, solely so the original simulation
figures remain reproducible.

---

## 4. GCC-PHAT was broken by its own whitening

PHAT weighting divides the cross-spectrum by its magnitude to keep only phase.
The original used an absolute floor:

```python
R_phat = R / (np.abs(R) + 1e-10)
```

Rotor noise is a sparse harmonic spectrum: most bins hold essentially nothing.
Dividing those by `1e-10` lifts pure numerical noise to the same unit magnitude
as real signal, and with thousands of such bins the correlation peak disappears
beneath them.

Measured on a known 20-sample delay, the original returned **−0.25 samples**.

**Fixed** by scaling the floor to the strongest bin
(`magnitude + floor * magnitude.max()`, `floor = 1e-3`), which leaves empty bins
suppressed. The same test now returns 19.04 samples, and a single-bin pure tone
shifted by 11 samples returns 11.0.

---

## 5. The TDOA search was unbounded, and this is what caused the large errors

The original `DOPPLER_ANALYSIS.md` reported 60–126° bearing errors on a moving
source and attributed them to motion blur and Doppler. Three numbers in that
analysis do not survive checking:

| quantity | document | actual | overstated by |
|---|---|---|---|
| angular velocity at closest approach | 350 °/s | 11.46 °/s | **31×** |
| bearing change during a 0.5 s buffer | 175° | 5.73° | **31×** |
| measured TDOA across a 120 mm array | ±10 ms | ±350 µs max possible | **29×** |

The angular velocity is simply `v/r = 10/50 = 0.2 rad/s = 11.46 °/s`; the
figure of 350 °/s does not follow from the stated geometry.

The third row is the diagnostic one. A 120 mm baseline cannot produce a delay
larger than `0.120/343 = 350 µs`. Delays of ±10 ms are **29× physically
impossible**, so those correlation peaks were not arrivals at all — the
cross-correlation search was unbounded and locking onto noise. That, not
Doppler, produced the large errors.

**Fixed** by bounding every search to the pair's own geometry:
`max_tau = margin * baseline / c`. Guarded by
`test_max_tau_bounds_the_search_geometrically`.

With the bound in place, motion blur costs what the geometry says it should.
For a 25 m/s pass at 15 m — a far more aggressive case than the original —
errors grow smoothly from 0.3–0.8° at 50 ms frames to 1.7–5.5° at 800 ms
frames. Real, measurable, and two orders of magnitude short of 60°.

---

## 6. Doppler corrupts elevation, not azimuth

The original notes treat Doppler as a source of bearing error. It is not, and
the reason is worth stating because it has a clean closed form.

In the far field the Doppler factor `α = c/(c − v_r)` is common to the whole
array, so every inter-element delay is divided by the same `α`. Azimuth is
`atan2(u, v)` and both direction cosines shrink by that identical factor, so
**azimuth is invariant**. Elevation comes from the *magnitude* `hypot(u, v)`,
which is not scale-invariant, so the entire bias lands there — a source in the
horizontal plane appears to climb by `arccos(1/α)`.

Measured against that prediction:

| radial velocity | α | predicted bias | ESPRIT measured | azimuth error |
|---|---|---|---|---|
| 15 m/s | 1.0457 | 17.4° | 16.3° | 0.05° |
| 30 m/s | 1.0958 | 24.4° | 23.0° | 0.07° |
| 45 m/s | 1.1510 | 29.8° | 29.3° | 0.15° |
| 60 m/s | 1.2120 | 34.4° | 33.3° | 0.06° |

Azimuth error stays below 0.2° across the whole range while elevation runs away
to 33°. Resampling onto a uniform emission-time grid removes the scale factor
and returns elevation to **0.00°**.

A further consequence: the original `doppler_simulation.py` applied a
Doppler-shifted frequency *and* evaluated at a retarded time
(`sin(2π f_doppler · t_emit)`), which double-counts the effect, while the
README claimed the Doppler emerged naturally from retarded time alone. Only
`Doppler_time_scaling.py` implemented the stated model; that is the one carried
forward into `simulate.simulate_moving`.

---

## 7. Sign convention errors in the forward model

Found by testing the simulator against the estimators rather than by reading
the code. An element displaced *towards* the source is closer to it and must
receive the waveform **earlier**; the simulator applied a positive delay
instead. Because the frequency-domain steering vectors used the physically
correct convention, every subspace and beamforming estimator reported the
reciprocal bearing — consistently 180° wrong, which is why it was easy to spot
once a known-truth test existed.

Delay-and-sum had the matching error in its steering step, and `phase_delay`
had an inverted sign in converting cross-product phase to time.

All three are fixed, and `test_element_nearer_the_source_receives_earlier`
pins the convention down.

---

## 8. Peak detection missed sources at 0°

The spatial spectrum is circular, so the original code padded it by a few
samples before calling `scipy.signal.find_peaks`. The pad was shorter than the
main lobe, so for a source near 0° the lobe ran into the array edge, its base
was truncated at 0.985, and the computed prominence collapsed to 0.015 — below
threshold. The global maximum of the spectrum was discarded as insufficiently
prominent.

**Fixed** by tiling the spectrum three times and taking peaks from the middle
copy, so every peak has a full descent on both sides.

---

## 9. Processing band included frequencies the array cannot use

The array is 126 mm across, which is less than one wavelength below about
2.7 kHz, and directivity collapses as the aperture shrinks relative to the
wavelength. Admitting energy from 200 Hz upwards added an undirectional
pedestal that dragged the beamformers towards the centroid of the sources:
delay-and-sum merged two sources 90° apart into a single peak at 78°.

The band is now 1.0–3.5 kHz — the upper edge below the 4.08 kHz spatial
Nyquist of the 42 mm grid, the lower edge where the aperture starts to provide
directivity — and the same band is applied to the time-domain estimators, which
previously had no front-end filter at all. SRP-PHAT went from five detection
failures out of eight bearings to **zero, at 0.09° mean error**.

---

## 10. Scan-grid quantisation hid the effect of SNR

With a 2° scan grid and off-grid test bearings, every scanning estimator
reported exactly 1.00° RMS error at every SNR from −15 dB to +25 dB: the
quantisation floor dominated and the measurement was uninformative.

**Fixed** by parabolic interpolation of the spectral peak, the same technique
already used on correlation peaks. Accuracy then varies with SNR as it should,
revealing a breakdown threshold near −20 dB and a high-SNR floor of 0.04° for
MVDR and SRP-PHAT.

---

## What did not change

The physics and the method selection in the original work were sound. The
array design, the choice of GCC-PHAT for delay estimation, the decision to
prefer wideband methods for moving sources, and the identification of spatial
Nyquist as the binding constraint on element spacing all hold up. Most of the
defects above are implementation errors sitting underneath correct reasoning —
which is precisely why they were hard to see without putting two
implementations of the same quantity next to each other.
