"""DOA estimators sharing a common geometry and signal model."""

from . import hilbert, tdoa
from .beamforming import delay_and_sum, mvdr, scan_grid, srp_phat
from .esprit import EspritResult, esprit
from .gcc_phat import gcc_phat, pairwise_delays, parabolic_peak
from .hilbert import (analytic, dominant_frequency, envelope_delay,
                      phase_delay, unwrapped_delay)
from .music import estimate_num_sources, music
from .tdoa import (direction_cosine_from_delay, least_squares_direction,
                   tdoa_direction)

__all__ = [
    # beamforming / spectral
    "delay_and_sum", "mvdr", "srp_phat", "scan_grid",
    # subspace
    "music", "estimate_num_sources", "esprit", "EspritResult",
    # delay based
    "gcc_phat", "pairwise_delays", "parabolic_peak",
    "tdoa", "tdoa_direction", "least_squares_direction",
    "direction_cosine_from_delay",
    # analytic signal
    "hilbert", "analytic", "dominant_frequency", "phase_delay",
    "envelope_delay", "unwrapped_delay",
]
