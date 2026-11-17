"""Image de référence : voir :mod:`fuji.core.flatfield` (ce module reste pour compatibilité)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fuji.core.flatfield import FlatFieldStore

if TYPE_CHECKING:
    from fuji.core.microscope import Microscope


def average_frames(microscope: Microscope, n: int = 8) -> np.ndarray:
    """Moyenne de ``n`` images brutes (sans correction du fond)."""
    return microscope.flatfield.measure("flat", n).image


__all__ = ["FlatFieldStore", "average_frames"]
