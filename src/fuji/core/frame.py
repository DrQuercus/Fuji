"""Image acquise + métadonnées complètes (traçabilité, calibration)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


@dataclass
class FrameMetadata:
    timestamp: str = field(default_factory=_now_iso)
    camera: str | None = None
    exposure_ms: float | None = None
    gain: float | None = None
    bit_depth: int | None = None
    binning: float = 1
    pixel_size_um: float | None = None
    objective: str | None = None
    objective_na: float | None = None
    channel: str | None = None
    x_um: float | None = None
    y_um: float | None = None
    z_um: float | None = None
    #: Temps écoulé depuis le début de l'acquisition (s).
    elapsed_s: float | None = None
    #: Indices dans la séquence, ex. ``{"t": 3, "c": 1, "z": 0}``.
    index: dict[str, int] = field(default_factory=dict)
    #: État des autres périphériques (positions de tourelles, lampes...).
    devices: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_rgb(data: np.ndarray) -> bool:
    """Vrai pour une image couleur telle que la livrent les caméras : (lignes, colonnes, 3 ou 4)."""
    return data.ndim == 3 and data.shape[-1] in (3, 4)


@dataclass
class Frame:
    data: np.ndarray
    metadata: FrameMetadata = field(default_factory=FrameMetadata)

    @property
    def is_rgb(self) -> bool:
        return is_rgb(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)
