"""
Acoustic source localisation with a 4x4 microphone array.

Direction-of-arrival estimation for small uncrewed aircraft from their rotor
noise, built around a 16-element planar array on a 42 mm grid.
"""

from . import (config, detect, estimators, geometry, plotting, simulate,
               sources)

__all__ = ["config", "geometry", "sources", "simulate", "detect", "estimators",
           "plotting"]
__version__ = "1.0.0"
