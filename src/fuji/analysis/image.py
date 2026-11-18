"""Types de données de l'analyse : images calibrées, étiquettes, tableaux de résultats.

Toutes les mesures tiennent compte de la taille du pixel : une surface est en µm²
si l'image est calibrée, en pixels² sinon (comme « Set Scale » dans ImageJ).
"""

from __future__ import annotations

from collections.abc import Sequence
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
    #: ``True`` / ``False`` pour lever l'ambiguïté d'une pile (le dernier axe est-il la
    #: couleur ?) ; ``None`` : deviné d'après la forme du tableau.
    rgb: bool | None = None

    @property
    def is_rgb(self) -> bool:
        if self.rgb is None:
            return _is_rgb(self.data)
        return bool(self.rgb) and self.data.ndim >= 3 and self.data.shape[-1] in (3, 4)

    @property
    def plane_shape(self) -> tuple[int, int]:
        """(hauteur, largeur) d'un plan."""
        return tuple(self.data.shape[-3:-1] if self.is_rgb else self.data.shape[-2:])  # type: ignore[return-value]

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
class Rois:
    """Régions d'intérêt : rectangles, ellipses, polygones, lignes et tracés.

    Les sommets sont en pixels de l'image, sous la forme (ligne, colonne). Les coordonnées
    entières tombent **au centre** des pixels (convention de napari, et des lignes
    d'ImageJ). Un pixel appartient à une région si son centre est à l'intérieur.

    Pour retrouver les sélections d'ImageJ, dont les coordonnées entières tombent sur les
    coins des pixels, utiliser :meth:`rectangle`, :meth:`oval` et :meth:`polygon`.
    """

    shapes: list[tuple[str, np.ndarray]] = field(default_factory=list)
    pixel_size_um: float | None = None
    name: str = "ROI"

    def __len__(self) -> int:
        return len(self.shapes)

    @property
    def px(self) -> float:
        return float(self.pixel_size_um) if self.pixel_size_um else 1.0

    @property
    def unit(self) -> str:
        return "µm" if self.pixel_size_um else "px"

    # --- constructeurs à la manière d'ImageJ (x, y = coin supérieur gauche, en pixels)
    @classmethod
    def rectangle(cls, x: float, y: float, width: float, height: float, **kw: Any) -> Rois:
        """``makeRectangle(x, y, largeur, hauteur)`` d'ImageJ."""
        r0, c0, r1, c1 = y - 0.5, x - 0.5, y + height - 0.5, x + width - 0.5
        return cls([("rectangle", np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]]))], **kw)

    @classmethod
    def oval(cls, x: float, y: float, width: float, height: float, **kw: Any) -> Rois:
        """``makeOval(x, y, largeur, hauteur)`` d'ImageJ."""
        r0, c0, r1, c1 = y - 0.5, x - 0.5, y + height - 0.5, x + width - 0.5
        return cls([("ellipse", np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]]))], **kw)

    @classmethod
    def polygon(cls, xs: Any, ys: Any, **kw: Any) -> Rois:
        """``makePolygon(x1, y1, x2, y2, ...)`` d'ImageJ."""
        v = np.stack([np.asarray(ys, float) - 0.5, np.asarray(xs, float) - 0.5], axis=1)
        return cls([("polygon", v)], **kw)

    @classmethod
    def line(cls, x1: float, y1: float, x2: float, y2: float, **kw: Any) -> Rois:
        """``makeLine(x1, y1, x2, y2)`` d'ImageJ (coordonnées au centre des pixels)."""
        return cls([("line", np.array([[y1, x1], [y2, x2]], float))], **kw)

    # --- géométrie
    def kind(self, index: int) -> str:
        return self.shapes[index][0]

    def vertices(self, index: int) -> np.ndarray:
        return np.asarray(self.shapes[index][1], dtype=np.float64)[:, -2:]

    def is_line(self, index: int) -> bool:
        return self.kind(index) in ("line", "path")

    def outline(self, index: int) -> np.ndarray:
        """Sommets (ligne, colonne) du contour de la région, dans le repère d'ImageJ
        (coins des pixels) : sert au périmètre, au Feret et à l'enveloppe convexe."""
        return self.vertices(index) + 0.5

    def mask(self, index: int, shape: tuple[int, int]) -> np.ndarray:
        """Masque des pixels dont le centre est dans la région (lignes : erreur)."""
        kind, v = self.kind(index), self.vertices(index)
        if kind in ("line", "path"):
            raise ValueError("Une ligne n'a pas de surface : utilisez une région fermée")
        h, w = shape
        out = np.zeros((h, w), bool)
        r_lo, r_hi = max(int(np.ceil(v[:, 0].min())), 0), min(int(np.floor(v[:, 0].max())), h - 1)
        c_lo, c_hi = max(int(np.ceil(v[:, 1].min())), 0), min(int(np.floor(v[:, 1].max())), w - 1)
        if r_hi < r_lo or c_hi < c_lo:
            return out
        rr, cc = np.mgrid[r_lo : r_hi + 1, c_lo : c_hi + 1]
        if kind == "ellipse":
            centre = v.mean(axis=0)
            a, b = (v[1] - v[0]) / 2.0, (v[3] - v[0]) / 2.0  # demi-axes (ellipse tournée possible)
            la, lb = float(np.hypot(*a)), float(np.hypot(*b))
            if la == 0 or lb == 0:
                return out
            dr, dc = rr - centre[0], cc - centre[1]
            u = (dr * a[0] + dc * a[1]) / (la * la)
            t = (dr * b[0] + dc * b[1]) / (lb * lb)
            inside = u * u + t * t <= 1.0
        else:
            inside = _points_in_polygon(rr, cc, v)
        out[r_lo : r_hi + 1, c_lo : c_hi + 1] = inside
        return out

    def length(self, index: int) -> float:
        """Longueur d'une ligne ou périmètre d'une région, en pixels.

        Ellipse : approximation de Ramanujan, comme ImageJ. Polygone : somme des côtés.
        """
        kind, v = self.kind(index), self.vertices(index)
        if kind == "ellipse":
            a, b = float(np.hypot(*(v[1] - v[0]))) / 2.0, float(np.hypot(*(v[3] - v[0]))) / 2.0
            return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))
        closed = kind in ("rectangle", "polygon")
        pts = np.vstack([v, v[:1]]) if closed else v
        return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def _points_in_polygon(rr: np.ndarray, cc: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Règle pair-impair, comme ``PolygonFiller`` d'ImageJ : sur une ligne de pixels, un
    centre situé exactement sur le bord gauche est dedans, sur le bord droit dehors."""
    inside = np.zeros(rr.shape, bool)
    n = len(v)
    for i in range(n):
        r1, c1 = v[i]
        r2, c2 = v[(i + 1) % n]
        if r1 == r2:
            continue
        crosses = (rr >= min(r1, r2)) & (rr < max(r1, r2))
        c_at = c1 + (rr - r1) * (c2 - c1) / (r2 - r1)
        inside ^= crosses & (cc >= c_at)
    return inside


@dataclass
class Points:
    """Points repérés sur une image (maxima, comptage manuel) : (ligne, colonne) en pixels."""

    data: np.ndarray
    pixel_size_um: float | None = None
    name: str = "points"

    def __len__(self) -> int:
        return len(self.data)

    @property
    def px(self) -> float:
        return float(self.pixel_size_um) if self.pixel_size_um else 1.0

    @property
    def unit(self) -> str:
        return "µm" if self.pixel_size_um else "px"


@dataclass
class Plot:
    """Graphique accompagnant un tableau : colonnes à tracer, type et titres des axes."""

    x: str
    y: tuple[str, ...]
    kind: str = "ligne"  # « ligne », « barres » ou « points »
    xlabel: str = ""
    ylabel: str = ""


@dataclass
class Table:
    """Tableau de résultats (comme la fenêtre « Results » d'ImageJ)."""

    data: pd.DataFrame
    name: str = "Résultats"
    units: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    #: graphique à afficher avec le tableau (profil, histogramme...)
    plot: Plot | None = None

    def frame(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Les données prêtes à sortir de Fuji : colonnes choisies (toutes par défaut), dans
        l'ordre du tableau, avec l'unité dans le titre — « Area (µm²) »."""
        keep = [c for c in self.data.columns if columns is None or str(c) in set(columns)]
        df = self.data[keep].copy()
        df.columns = [f"{c} ({self.units[c]})" if self.units.get(c) else c for c in df.columns]
        return df

    def to_csv(
        self,
        path: str | Path,
        *,
        columns: Sequence[str] | None = None,
        decimal: str = ".",
        sep: str | None = None,
    ) -> Path:
        """Écrit le tableau dans un fichier CSV. Avec la virgule décimale (``decimal=","``,
        pour Excel en français), les colonnes sont séparées par des points-virgules."""
        path = Path(path)
        if sep is None:
            sep = ";" if decimal == "," else ","
        # « utf-8-sig » : Excel reconnaît l'encodage et affiche correctement « µm² »
        self.frame(columns).to_csv(path, index_label=" ", encoding="utf-8-sig", sep=sep, decimal=decimal)
        return path

    def __len__(self) -> int:
        return len(self.data)
