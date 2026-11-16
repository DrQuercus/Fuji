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
        """Lit une image (TIFF, OME-TIFF, TIFF ImageJ, PNG, JPEG...) avec sa calibration."""
        path = Path(path)
        name = path.name.removesuffix(path.suffix).removesuffix(".ome")
        if path.suffix.lower() not in (".tif", ".tiff"):
            from skimage import io

            return cls(np.asarray(io.imread(path)), None, name)
        import tifffile

        with tifffile.TiffFile(path) as tif:
            data = tif.series[0].asarray()
            px = read_pixel_size_um(tif)
        return cls(data, px, name)


#: Facteur de conversion vers le micromètre des unités rencontrées dans les fichiers.
#: ImageJ écrit « micron », ou « µm » sous la forme échappée ``\\u00B5m``.
_TO_UM = {
    "µm": 1.0,
    "\u03bcm": 1.0,  # mu grec
    "\\u00b5m": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
    "inch": 25400.0,
    "in": 25400.0,
}


def read_pixel_size_um(tif: Any) -> float | None:
    """Taille du pixel (µm) d'un TIFF ouvert, d'après ses métadonnées OME ou ImageJ.

    Renvoie ``None`` si le fichier n'est pas calibré ou si son unité est inconnue,
    plutôt qu'une valeur fausse.
    """
    import re

    if tif.is_ome and tif.ome_metadata:
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata)
        if not m:
            return None
        u = re.search(r'PhysicalSizeXUnit="([^"]+)"', tif.ome_metadata)
        factor = _TO_UM.get(u.group(1).strip().lower() if u else "µm")
        return float(m.group(1)) * factor if factor else None
    if tif.is_imagej:
        tag = tif.pages[0].tags.get("XResolution")
        unit = str((tif.imagej_metadata or {}).get("unit", "")).strip().lower()
        factor = _TO_UM.get(unit)
        if tag is None or factor is None:
            return None
        num, den = tag.value
        return den / num * factor if num else None
    return None


@dataclass
class Labels:
    """Masque binaire (0/1) ou image d'étiquettes (un numéro par objet)."""

    data: np.ndarray
    pixel_size_um: float | None = None
    name: str = "objets"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary(self) -> bool:
        """Vrai si l'image ne contient qu'une seule valeur non nulle (masque 0/1, 0/255...)."""
        top = self.data.max(initial=0)
        return bool(top <= 1) or not bool(((self.data != 0) & (self.data != top)).any())

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
        # « utf-8-sig » : Excel reconnaît l'encodage et affiche correctement « µm² »
        df.to_csv(path, index_label=" ", encoding="utf-8-sig")
        return path

    def __len__(self) -> int:
        return len(self.data)
