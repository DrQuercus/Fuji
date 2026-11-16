"""Types de données de l'analyse : images calibrées, étiquettes, tableaux de résultats.

Toutes les mesures tiennent compte de la taille du pixel : une surface est en µm²
si l'image est calibrée, en pixels² sinon (comme « Set Scale » dans ImageJ).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _is_rgb(data: np.ndarray) -> bool:
    return data.ndim == 3 and data.shape[-1] in (3, 4) and data.shape[0] > 4


@dataclass
class Image:
    data: np.ndarray
    pixel_size_um: float | None = None
    name: str = "image"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_rgb(self) -> bool:
        return _is_rgb(self.data)

    @property
    def calibrated(self) -> bool:
        return bool(self.pixel_size_um)

    @property
    def unit(self) -> str:
        return "µm" if self.calibrated else "px"

    @property
    def px(self) -> float:
        """Taille du pixel (1 si non calibrée)."""
        return float(self.pixel_size_um) if self.pixel_size_um else 1.0

    def with_data(self, data: np.ndarray, name: str | None = None, **md: Any) -> Image:
        return replace(self, data=data, name=name or self.name, metadata={**self.metadata, **md})

    def gray(self, method: str = "moyenne") -> np.ndarray:
        """Image en niveaux de gris (float32).

        ``moyenne`` : (R+G+B)/3, comme ImageJ par défaut ;
        ``luminance`` : 0.299 R + 0.587 G + 0.114 B (option « Weighted » d'ImageJ).
        """
        d = self.data
        if not self.is_rgb:
            return d.astype(np.float32, copy=False)
        rgb = d[..., :3].astype(np.float32)
        if method == "luminance":
            return rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        return rgb.mean(axis=-1)

    @classmethod
    def from_frame(cls, frame: Any, name: str | None = None) -> Image:
        md = frame.metadata
        return cls(frame.data, md.pixel_size_um, name or (md.channel or "image"), md.to_dict())

    @classmethod
    def from_file(cls, path: str | Path) -> Image:
        import tifffile

        path = Path(path)
        with tifffile.TiffFile(path) as tif:
            data = tif.series[0].asarray()
            px = None
            if tif.is_ome and tif.ome_metadata:
                import re

                m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata)
                if m:
                    px = float(m.group(1))
            elif tif.is_imagej:
                page = tif.pages[0]
                tag = page.tags.get("XResolution")
                if tag is not None:
                    num, den = tag.value
                    if num:
                        px = den / num
        return cls(data, px, path.stem.replace(".ome", ""))


@dataclass
class Labels:
    """Masque binaire (0/1) ou image d'étiquettes (un numéro par objet)."""

    data: np.ndarray
    pixel_size_um: float | None = None
    name: str = "objets"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary(self) -> bool:
        return int(self.data.max(initial=0)) <= 1

    @property
    def count(self) -> int:
        if self.is_binary:
            from scipy import ndimage as ndi

            return int(ndi.label(self.data > 0)[1])
        return len(np.unique(self.data[self.data > 0]))

    @property
    def px(self) -> float:
        return float(self.pixel_size_um) if self.pixel_size_um else 1.0

    @property
    def unit(self) -> str:
        return "µm" if self.pixel_size_um else "px"


@dataclass
class Table:
    """Tableau de résultats (comme la fenêtre « Results » d'ImageJ)."""

    data: pd.DataFrame
    name: str = "Résultats"
    units: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        df = self.data.copy()
        df.columns = [f"{c} ({self.units[c]})" if self.units.get(c) else c for c in df.columns]
        df.to_csv(path, index_label=" ")
        return path

    def __len__(self) -> int:
        return len(self.data)
