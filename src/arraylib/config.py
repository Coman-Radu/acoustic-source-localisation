"""
Physical and processing constants for the 4x4 microphone array.

All values match the hardware prototype used during the internship: a planar
16-element array on a 42 mm uniform grid, captured through a MiniDSP
multichannel interface at 48 kHz.
"""

from dataclasses import dataclass, field

# --- Physical constants -------------------------------------------------

SPEED_OF_SOUND = 343.0  #: m/s, dry air at ~20 C

# --- Array hardware ----------------------------------------------------

MIC_SPACING = 0.042  #: m, centre-to-centre spacing of the uniform grid
GRID_SHAPE = (4, 4)  #: rows (y) x columns (x)
CHANNELS = 16        #: total elements, == GRID_SHAPE[0] * GRID_SHAPE[1]

# --- Capture ------------------------------------------------------------

SAMPLE_RATE = 48000  #: Hz, interface capture rate


def spatial_nyquist_frequency(spacing: float = MIC_SPACING,
                              c: float = SPEED_OF_SOUND) -> float:
    """
    Highest frequency the array can resolve without spatial aliasing.

    Grating lobes appear once the element spacing exceeds half a wavelength,
    so the usable upper bound is ``c / (2 * spacing)``.  For the 42 mm grid
    this is about 4.08 kHz, which is why the processing band is capped below
    it (see :data:`BAND_HIGH`).
    """
    return c / (2.0 * spacing)


#: Processing band, bounded at both ends by the array itself.
#:
#: Upper edge: grating lobes appear above the 4.08 kHz spatial Nyquist of the
#: 42 mm grid, so 3.5 kHz leaves a little margin.
#:
#: Lower edge: the array is only 126 mm across, which is less than a
#: wavelength below about 2.7 kHz, and directivity collapses as the aperture
#: shrinks relative to the wavelength.  Admitting low frequencies does not add
#: information, it just adds an undirectional pedestal that drags every
#: beamformer towards the centroid of the sources -- measurably so: including
#: 200 Hz upwards made delay-and-sum merge two sources 90 deg apart into a
#: single peak at their midpoint.  1 kHz is the practical floor for this
#: aperture, which also conveniently rejects wind and handling noise.
BAND_LOW = 1000.0
BAND_HIGH = 3500.0


@dataclass
class ScanConfig:
    """Angular scan grid and peak-acceptance thresholds for a DOA scan."""

    azimuth_resolution: float = 2.0  #: deg between scan points
    min_peak_height: float = 0.15    #: normalised power, 0-1
    min_prominence: float = 0.08     #: normalised power a peak must stand out by
    min_separation: float = 10.0     #: deg between accepted peaks
    snr_threshold_db: float = 6.0    #: dB above the estimated noise floor
    noise_percentile: float = 25.0   #: percentile of the spectrum taken as noise


@dataclass
class SourceConfig:
    """Geometry of a simulated acoustic source."""

    distance: float = 10.0        #: m from array centre
    azimuth: float = 30.0         #: deg, 0 = North, increasing clockwise
    snr_db: float = 10.0          #: dB, broadband signal-to-noise at the array


@dataclass
class MovingSourceConfig:
    """Trajectory of a simulated moving source."""

    velocity: float = 5.0         #: m/s tangential speed
    duration: float = 4.0         #: s of simulated capture
    frame_duration: float = 0.2   #: s per DOA frame
    distance: float = 10.0        #: m, constant radius
    azimuth_initial: float = 30.0 #: deg at t = 0
    snr_db: float = 10.0


@dataclass
class MultiSourceConfig:
    """Several stationary sources, evenly spaced in azimuth."""

    num_sources: int = 3
    azimuth_initial: float = 30.0
    azimuth_increment: float = 45.0
    distance: float = 10.0
    snr_db: float = 10.0

    @property
    def azimuths(self) -> list[float]:
        """True azimuths of the configured sources, in degrees."""
        return [(self.azimuth_initial + i * self.azimuth_increment) % 360.0
                for i in range(self.num_sources)]
