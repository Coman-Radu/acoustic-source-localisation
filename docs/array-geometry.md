# Array geometry

The prototype is a planar 16-element array on a uniform 42 mm grid, captured at
48 kHz through a MiniDSP multichannel interface.

```
            x (east) →
        -63   -21   +21   +63  mm
      ┌─────────────────────────┐
 +63  │  08    07    10    09   │  row 3  (north)
 +21  │  06    05    12    11   │  row 2
 -21  │  04    03    14    13   │  row 1
 -63  │  02    01    00    15   │  row 0  (south)
      └─────────────────────────┘
   y (north) ↑        numbers are capture channels
```

Defined once in `arraylib.geometry.HARDWARE_ORDER`. Everything else —
steering vectors, subarrays, channel pairs — derives from it.

| property | value |
|---|---|
| elements | 16 (4 × 4) |
| spacing | 42 mm |
| aperture | 126 mm × 126 mm |
| sample rate | 48 kHz |
| spatial Nyquist | 4.08 kHz |
| processing band | 1.0 – 3.5 kHz |

## Why 42 mm

Element spacing sets the upper frequency limit. Grating lobes — false peaks
indistinguishable from the real source — appear once the spacing exceeds half a
wavelength, so the highest usable frequency is

```
f_max = c / (2 d) = 343 / (2 × 0.042) = 4.08 kHz
```

Wider spacing would buy angular resolution at the cost of aliasing the upper
rotor harmonics; closer spacing would raise the ceiling but shrink the aperture
and blunt the resolution. 42 mm places the limit just above the band where
multirotor harmonics are strong.

## Why the band is bounded at *both* ends

The upper bound is spatial Nyquist, 3.5 kHz with a little margin.

The lower bound is less obvious and was found by measurement. Angular
resolution scales with aperture measured in wavelengths, and at 126 mm the
array is smaller than one wavelength below about 2.7 kHz. Low-frequency energy
therefore carries almost no directional information while still contributing
output power, which shows up as an undirectional pedestal under the spatial
spectrum. With the band opened down to 200 Hz, delay-and-sum merged two sources
90° apart into a single peak at their midpoint. Raising the lower edge to
1 kHz resolved them.

So the array has a genuinely narrow usable band: roughly 1–3.5 kHz, a little
under two octaves. This is the central design tension of a small acoustic
array, and it is why the aperture, not the algorithm, sets the resolution floor
for the beamforming methods.

## Resolution in practice

Measured smallest separation at which each estimator reports two distinct
sources (20 dB SNR, synthetic rotor noise):

| estimator | resolves two sources at |
|---|---|
| delay-and-sum | 55° |
| MVDR | 50° |
| SRP-PHAT | 50° |
| MUSIC | 30° |
| ESPRIT | 10° * |

\* ESPRIT always returns exactly as many estimates as the order it is given, so
"resolved" is not a meaningful threshold for it — judge it by RMS error
instead. The others genuinely merge their peaks and report a single source.

The ~50° figure for the beamformers is the aperture limit, not an
implementation shortcoming. MUSIC and ESPRIT beat it because they exploit the
eigenstructure of the covariance rather than scanning a beam, which is the
whole reason for preferring subspace methods on a small array.

## Subarrays for ESPRIT

ESPRIT needs two identical subarrays related by one fixed translation. On a
4 × 4 grid, dropping the last column gives subarray 1 and dropping the first
column gives subarray 2; the two are then identical up to a shift of exactly
one spacing along x. Dropping rows instead gives the y-axis pair.

```
      x-displaced doublet              y-displaced doublet
   ┌───────────┬─┐                  ┌─────────────┐
   │ subarray1 │ │                  │  subarray2  │ ─┐
   │     ┌─┬───┴─┤                  ├─────────────┤  │ 42 mm
   │     │ │ sub2│                  │  subarray1  │ ─┘
   └─────┴─┴─────┘                  └─────────────┘
      → 42 mm
```

Each subarray has 12 of the 16 elements, so up to 12 sources are
theoretically separable, though the covariance estimate degrades long before
that.

`test_doublet_subarrays_have_a_constant_displacement` asserts the displacement
is identical for all 12 element pairs. The original hand-built subarrays did
not satisfy this, which contributed to the ESPRIT failure described in
[findings.md](findings.md#1-esprit-returned-angles-unrelated-to-the-source).

## Channel pair selection

For the delay-based estimators, `geometry.microphone_pairs` offers three sets:

| selection | pairs | notes |
|---|---|---|
| `adjacent` | 24 | neighbours within rows and columns; cheapest, 42 mm baselines only |
| `perimeter` | 66 | outer ring; widest baselines, **the default for SRP-PHAT** |
| `all` | 120 | every unique pair; most robust, slowest |

Baseline length matters more than pair count. Switching SRP-PHAT from
`adjacent` to `perimeter` took it from 0.67° mean error with five detection
failures out of eight bearings to **0.09° with none** — the 126 mm diagonal
baselines resolve delay far better than the 42 mm adjacent ones, because the
delay to be measured is larger relative to the sampling interval.

## Sampling resolution

At 48 kHz one sample is 20.8 µs. The largest delay the array can produce is
`0.126/343 = 367 µs`, so the full 180° span of bearings is covered by only
about **35 samples**. A whole-sample delay estimate would therefore quantise
azimuth into steps of roughly 5–10°, worse near endfire.

Sub-sample interpolation is consequently not an optimisation but a
requirement. Both the correlation peak (`parabolic_peak`) and the spatial
spectrum peak (`refine_peak`) are interpolated parabolically, which is what
brings the estimators down to the sub-degree accuracies reported in the README.

## Coordinate convention

Azimuth is a compass bearing: **0° is north (+y), increasing clockwise**, so
90° is east (+x). Elevation is measured up from the horizontal plane.

This differs from the mathematical convention and from matplotlib's polar
default, both of which start at +x and run anticlockwise.
`arraylib.plotting.polar_spectrum` applies `set_theta_zero_location("N")` and
`set_theta_direction(-1)` so that plots match the maths; forgetting that is an
easy way to produce a figure mirrored about the north–south axis.
