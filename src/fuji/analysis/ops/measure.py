"""Mesures : analyse de particules et mesure globale, avec les définitions d'ImageJ.

Colonnes (mêmes noms que la fenêtre « Results » d'ImageJ) :

========== ==================================================================
Area       surface (µm² si calibré), en pixels de l'objet (trous exclus)
Mean...    intensités : Mean, StdDev (n−1), Min, Max, Median, Mode
Skew, Kurt asymétrie et aplatissement de la distribution des intensités
IntDen     Area × Mean ; RawIntDen = somme des valeurs
X, Y       centroïde ; XM, YM : centre de masse (pondéré par l'intensité)
BX, BY     coin du rectangle englobant ; Width, Height
Perim.     périmètre tracé (algorithme d'ImageJ)
Major...   ellipse ajustée : Major, Minor, Angle (0–180°)
Circ.      4π·Area/Perim² (1 = cercle parfait)
Feret      plus grand diamètre ; MinFeret, FeretAngle
AR, Round  Major/Minor ; 4·Area/(π·Major²)
Solidity   Area / surface de l'enveloppe convexe
========== ==================================================================
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .. import ij
from ..image import Image, Labels, Table
from ..registry import Param, operation
from ..shape import feret, fit_ellipse, polygon_area, trace_outline, traced_perimeter

_UNITS_LEN = (
    "X",
    "Y",
    "XM",
    "YM",
    "BX",
    "BY",
    "Width",
    "Height",
    "Perim.",
    "Major",
    "Minor",
    "Feret",
    "MinFeret",
)


def _intensity_plane(intensity: Image | None) -> np.ndarray | None:
    """Valeurs mesurées : l'image, ou son gris 8 bits ``(R+V+B)/3`` si elle est en couleur."""
    if intensity is None:
        return None
    return ij.intensity_plane(intensity.data, intensity.is_rgb)


def intensity_stats(values: np.ndarray) -> dict[str, float]:
    """Statistiques d'intensité d'ImageJ (``ImageStatistics``) sur un ensemble de pixels.

    Particularités reprises d'ImageJ : écart-type sur n − 1 ; pour une image entière, la
    médiane est la valeur du milieu « haute » (pas la moyenne des deux valeurs centrales)
    et le mode la valeur la plus fréquente ; pour une image 32 bits, le mode est le centre
    de la classe la plus peuplée d'un histogramme à 256 classes ; asymétrie et
    aplatissement calculés avec les moments de la population.
    """
    n = values.size
    v = values.astype(np.float64)
    mean = float(v.mean())
    integer = values.dtype.kind in "ui"
    if integer:
        median = float(np.partition(values, n // 2)[n // 2])
        lo = int(values.min())
        mode = float(lo + int(np.argmax(np.bincount((values - lo).astype(np.int64)))))
    else:
        median = float(np.median(v))
        lo_f, hi_f = float(v.min()), float(v.max())
        # des valeurs interpolées le long d'une ligne, sur une zone uniforme, ne diffèrent que
        # du dernier chiffre (40 et 40,000000000000007) : pas de quoi faire 256 classes
        if hi_f - lo_f > 1e-9 * max(1.0, abs(lo_f), abs(hi_f)):
            hist, _ = np.histogram(v, bins=256, range=(lo_f, hi_f))
            width = (hi_f - lo_f) / 256.0
            mode = lo_f + int(np.argmax(hist)) * width + width / 2.0
        else:
            mode = lo_f
    variance = float((v * v).mean() - mean * mean)
    if variance > 0:
        sd = np.sqrt(variance)
        skew = float((((v**3).mean() - 3.0 * mean * (v * v).mean()) + 2.0 * mean**3) / (variance * sd))
        kurt = float(
            ((v**4).mean() - 4.0 * mean * (v**3).mean() + 6.0 * mean * mean * (v * v).mean() - 3.0 * mean**4)
            / (variance * variance)
            - 3.0
        )
    else:
        skew = kurt = float("nan")
    return {
        "Mean": mean,
        "StdDev": float(v.std(ddof=1)) if n > 1 else 0.0,
        "Mode": mode,
        "Min": float(v.min()),
        "Max": float(v.max()),
        "Median": median,
        "Skew": skew,
        "Kurt": kurt,
        "RawIntDen": float(v.sum()),
    }


_intensity_stats = intensity_stats


def measure_objects(
    labels: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    px: float = 1.0,
    include_holes: bool = False,
) -> pd.DataFrame:
    """Mesure chaque objet d'une image d'étiquettes (une ligne par objet)."""
    rows_out: list[dict[str, float]] = []
    slices = ndi.find_objects(labels)
    for lab_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        sub = labels[slc] == lab_id
        if include_holes:
            sub = ndi.binary_fill_holes(sub)
        rr, cc = np.nonzero(sub)
        n = rr.size
        if n == 0:
            continue
        top, left = slc[0].start, slc[1].start
        area = n * px * px
        verts = trace_outline(ndi.binary_fill_holes(sub))
        perim = traced_perimeter(verts, px)
        ell = fit_ellipse(rr, cc)
        fer = feret(verts)
        hull_area = polygon_area(_hull_of(verts))
        row: dict[str, float] = {
            "Label": float(lab_id),
            "Area": area,
            "X": (cc.mean() + left + 0.5) * px,
            "Y": (rr.mean() + top + 0.5) * px,
            "BX": left * px,
            "BY": top * px,
            "Width": sub.shape[1] * px,
            "Height": sub.shape[0] * px,
            "Perim.": perim,
            "Major": ell.major * px,
            "Minor": ell.minor * px,
            "Angle": ell.angle,
            "Circ.": min(1.0, 4 * np.pi * area / perim**2) if perim > 0 else 0.0,
            "Feret": fer.max * px,
            "FeretAngle": fer.angle,
            "MinFeret": fer.min * px,
            "AR": ell.major / ell.minor if ell.minor > 0 else 0.0,
            "Round": 4 * n / (np.pi * ell.major**2) if ell.major > 0 else 0.0,
            "Solidity": n / hull_area if hull_area > 0 else 1.0,
        }
        if intensity is not None:
            vals = intensity[slc][sub]
            stats = _intensity_stats(vals)
            row.update(stats)
            row["IntDen"] = area * stats["Mean"]
            total = stats["RawIntDen"]
            if total > 0:
                row["XM"] = ((cc + left + 0.5) * vals).sum() / total * px
                row["YM"] = ((rr + top + 0.5) * vals).sum() / total * px
            else:
                row["XM"], row["YM"] = row["X"], row["Y"]
        rows_out.append(row)
    df = pd.DataFrame(rows_out)
    if not df.empty:
        df.index = pd.RangeIndex(1, len(df) + 1)
    return df


def _hull_of(verts: np.ndarray) -> np.ndarray:
    from ..shape import _hull

    return _hull(verts) if len(verts) else verts


def _units(unit: str, intensity: bool) -> dict[str, str]:
    units = {"Area": f"{unit}²", "Angle": "°", "FeretAngle": "°"}
    units.update({k: unit for k in _UNITS_LEN})
    if intensity:
        units["IntDen"] = f"{unit}²·niveau"
    return units


@operation("Analyse de particules", category="Mesures", imagej="Analyze > Analyze Particles...")
def analyze_particles(
    mask: Labels,
    intensity: Image | None = None,
    min_area_um2: Annotated[float, Param(label="surface minimale", min=0, unit="µm² ou px²")] = 0.0,
    max_area_um2: Annotated[
        float, Param(label="surface maximale", min=0, unit="µm² ou px²", description="0 = pas de limite")
    ] = 0.0,
    min_circularity: Annotated[float, Param(label="circularité min", min=0, max=1, step=0.05)] = 0.0,
    max_circularity: Annotated[float, Param(label="circularité max", min=0, max=1, step=0.05)] = 1.0,
    exclude_edges: Annotated[bool, Param(label="exclure les bords")] = False,
    include_holes: Annotated[bool, Param(label="inclure les trous")] = False,
) -> tuple[Table, Labels]:
    """Détecte, filtre et mesure les objets d'un masque (comme ImageJ).

    Renvoie le tableau de résultats et l'image des objets retenus (numérotés)."""
    px = mask.px
    data = mask.data
    if mask.is_binary:
        lab, _ = ndi.label(data > 0, structure=np.ones((3, 3)))
    else:
        lab = data.astype(np.int64)
    if exclude_edges:
        edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        lab = np.where(np.isin(lab, edge_ids[edge_ids > 0]), 0, lab)
    plane = _intensity_plane(intensity)
    df = measure_objects(lab, plane, px=px, include_holes=include_holes)
    if not df.empty:
        keep = df["Area"] >= min_area_um2
        if max_area_um2 > 0:
            keep &= df["Area"] <= max_area_um2
        keep &= (df["Circ."] >= min_circularity) & (df["Circ."] <= max_circularity)
        kept_ids = df.loc[keep, "Label"].astype(np.int64).to_numpy()
        df = df[keep].reset_index(drop=True)
        df.index = pd.RangeIndex(1, len(df) + 1)
        # renuméroter les objets retenus 1..N (comme les « Count Masks » d'ImageJ)
        lut = np.zeros(int(lab.max()) + 1, np.int32)
        lut[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
        lab = lut[lab]
        df["Label"] = np.arange(1, len(df) + 1)
    total_px = data.shape[0] * data.shape[1]
    summary = {
        "Count": len(df),
        "Total Area": float(df["Area"].sum()) if len(df) else 0.0,
        "Average Size": float(df["Area"].mean()) if len(df) else 0.0,
        "%Area": float(df["Area"].sum() / (total_px * px * px) * 100) if len(df) else 0.0,
    }
    if len(df) and "Mean" in df:
        summary["Mean"] = float(df["Mean"].mean())
    unit = mask.unit
    table = Table(
        df, name=f"Particules — {mask.name}", units=_units(unit, plane is not None), summary=summary
    )
    objects = Labels(lab.astype(np.int32), mask.pixel_size_um, f"{mask.name} (particules)")
    return table, objects


@operation("Mesurer", category="Mesures", imagej="Analyze > Measure")
def measure(image: Image, mask: Labels | None = None) -> Table:
    """Statistiques d'intensité de l'image entière, ou seulement dans un masque
    (« Limit to threshold »). Pour mesurer des régions dessinées : « Mesurer les ROI »."""
    plane = _intensity_plane(image)
    assert plane is not None
    if mask is not None and mask.data.shape != plane.shape:
        raise ValueError("Le masque n'a pas la taille de l'image")
    sel = mask.data > 0 if mask is not None else np.ones(plane.shape, bool)
    vals = plane[sel]
    px = image.px
    area = vals.size * px * px
    stats = _intensity_stats(vals) if vals.size else {}
    row = {"Area": area, **stats}
    if stats:
        row["IntDen"] = area * stats["Mean"]
    row["%Area"] = 100.0 * sel.mean() if mask is not None else 100.0 * float(np.mean(plane != 0))
    df = pd.DataFrame([row], index=pd.RangeIndex(1, 2))
    return Table(df, name=f"Mesure — {image.name}", units=_units(image.unit, True))
