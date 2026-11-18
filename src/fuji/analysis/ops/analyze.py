"""Mesures d'ImageJ : régions d'intérêt, profils, histogrammes, maxima, colocalisation."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .. import ij
from ..image import Image, Labels, Plot, Points, Rois, Table
from ..maxima import find_maxima
from ..registry import Param, operation
from ..shape import _hull, feret, fit_ellipse, polygon_area, trace_outline
from .measure import _units, intensity_stats

_EIGHT = np.ones((3, 3), bool)


def _plane(image: Image) -> np.ndarray:
    plane = ij.intensity_plane(image.data, image.is_rgb)
    if plane.ndim != 2:
        raise ValueError("Cette mesure demande une image 2D (un plan)")
    return plane


# ------------------------------------------------------------------------ profils
def sample_line(plane: np.ndarray, start: np.ndarray, end: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Valeurs le long d'un segment (``ImageProcessor.getLine`` d'ImageJ).

    ``round(longueur) + 1`` points régulièrement espacés, extrémités comprises, avec
    interpolation bilinéaire. ``width`` > 1 : moyenne sur des lignes parallèles espacées
    d'un pixel, de part et d'autre du segment.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    length = float(np.hypot(*delta))
    n = int(math.floor(length + 0.5))
    t = np.arange(n + 1) / n if n > 0 else np.zeros(1)
    rows, cols = start[0] + t * delta[0], start[1] + t * delta[1]
    lines = max(1, round(width))
    if lines == 1 or length == 0:
        return ndi.map_coordinates(plane.astype(np.float64), [rows, cols], order=1, mode="nearest")
    normal = np.array([-delta[1], delta[0]]) / length
    offsets = np.arange(lines) - (lines - 1) / 2.0
    values = [
        ndi.map_coordinates(
            plane.astype(np.float64), [rows + o * normal[0], cols + o * normal[1]], order=1, mode="nearest"
        )
        for o in offsets
    ]
    return np.mean(values, axis=0)


def roi_profile(plane: np.ndarray, rois: Rois, index: int, width: float = 1.0) -> np.ndarray:
    """Profil d'une ROI : le long d'une ligne ou d'un tracé, ou moyenne des colonnes d'un
    rectangle (comme « Plot Profile » d'ImageJ)."""
    kind, v = rois.kind(index), rois.vertices(index)
    if kind == "line":
        return sample_line(plane, v[0], v[-1], width)
    if kind == "path":
        seg = np.hypot(*np.diff(v, axis=0).T)
        total = float(seg.sum())
        n = int(math.floor(total + 0.5))
        s = np.linspace(0.0, total, n + 1) if n > 0 else np.zeros(1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        rows, cols = np.interp(s, cum, v[:, 0]), np.interp(s, cum, v[:, 1])
        return ndi.map_coordinates(plane.astype(np.float64), [rows, cols], order=1, mode="nearest")
    mask = rois.mask(index, plane.shape)
    if not mask.any():
        raise ValueError("La ROI est en dehors de l'image")
    cols = np.flatnonzero(mask.any(axis=0))
    sums = np.where(mask, plane, 0).sum(axis=0, dtype=np.float64)[cols]
    return sums / mask.sum(axis=0)[cols]


@operation("Profil d'intensité", category="Mesures", imagej="Analyze > Plot Profile")
def plot_profile(
    image: Image,
    roi: Rois,
    width: Annotated[
        float,
        Param(label="largeur de ligne", min=1, max=500, unit="px", description="moyenne sur cette largeur"),
    ] = 1.0,
) -> Table:
    """Intensité le long d'une ligne (ou d'un tracé), ou moyenne verticale d'un rectangle.

    Un tableau et un graphique par opération ; s'il y a plusieurs ROI, une colonne par ROI."""
    if not len(roi):
        raise ValueError(
            "Dessinez d'abord une ligne ou un rectangle sur l'image (outils « Dessiner », en haut du panneau)"
        )
    plane = _plane(image)
    profiles = [roi_profile(plane, roi, i, width) for i in range(len(roi))]
    n = max(len(p) for p in profiles)
    data = {"Distance": np.arange(n) * image.px}
    names = []
    for i, p in enumerate(profiles):
        name = "Valeur" if len(profiles) == 1 else f"ROI {i + 1}"
        names.append(name)
        data[name] = np.concatenate([p, np.full(n - len(p), np.nan)])
    df = pd.DataFrame(data, index=pd.RangeIndex(1, n + 1))
    return Table(
        df,
        name=f"Profil — {image.name}",
        units={"Distance": image.unit},
        plot=Plot("Distance", tuple(names), "ligne", f"Distance ({image.unit})", "Intensité"),
    )


# -------------------------------------------------------------------- histogramme
def _roi_selection(rois: Rois | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Union des régions fermées (``None`` s'il n'y en a pas : toute l'image)."""
    if rois is None:
        return None
    sel = np.zeros(shape, bool)
    found = False
    for i in range(len(rois)):
        if not rois.is_line(i):
            sel |= rois.mask(i, shape)
            found = True
    if not found:
        return None
    if not sel.any():
        raise ValueError("Les ROI sont en dehors de l'image")
    return sel


@operation("Histogramme", category="Mesures", imagej="Analyze > Histogram")
def histogram(
    image: Image,
    roi: Rois | None = None,
    bins: Annotated[
        int,
        Param(label="classes", min=2, max=65536, description="pour les images 16 et 32 bits (8 bits : 256)"),
    ] = 256,
) -> Table:
    """Répartition des valeurs de l'image (ou de la zone des ROI), avec ses statistiques."""
    plane = _plane(image)
    sel = _roi_selection(roi, plane.shape)
    values = plane[sel] if sel is not None else plane.ravel()
    values = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if values.size == 0:
        raise ValueError("Aucun pixel à compter")
    if values.dtype == np.uint8:
        counts = np.bincount(values, minlength=256)
        centres, width = np.arange(256, dtype=np.float64), 1.0
    else:
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
        width = (hi - lo) / bins
        centres = edges[:-1]
    stats = intensity_stats(values)
    df = pd.DataFrame({"Valeur": centres, "Nombre": counts}, index=pd.RangeIndex(1, len(counts) + 1))
    summary = {
        "N": int(values.size),
        "Mean": stats["Mean"],
        "StdDev": stats["StdDev"],
        "Min": stats["Min"],
        "Max": stats["Max"],
        "Mode": stats["Mode"],
        "Largeur de classe": width,
    }
    return Table(
        df,
        name=f"Histogramme — {image.name}",
        summary=summary,
        plot=Plot("Valeur", ("Nombre",), "barres", "Valeur", "Nombre de pixels"),
    )


# ------------------------------------------------------------------- mesure de ROI
def measure_rois(plane: np.ndarray, rois: Rois, px: float = 1.0) -> pd.DataFrame:
    """Une ligne de mesures par ROI, avec les définitions d'ImageJ (``Analyze > Measure``).

    Régions fermées : surface et intensités d'après les pixels dont le centre est dans la
    région ; périmètre d'après le contour dessiné ; ellipse, Feret et descripteurs de forme
    comme dans l'analyse de particules. Lignes : longueur, angle et intensités du profil.
    """
    rows_out: list[dict[str, float | str]] = []
    for i in range(len(rois)):
        kind = rois.kind(i)
        row: dict[str, float | str] = {"ROI": float(i + 1), "Type": kind}
        if rois.is_line(i):
            v = rois.vertices(i)
            values = roi_profile(plane, rois, i)
            row["Length"] = rois.length(i) * px
            dx, dy = v[-1, 1] - v[0, 1], v[0, 0] - v[-1, 0]  # y vers le haut, comme ImageJ
            # tracé à trois points : l'angle au sommet du milieu (outil « Angle » d'ImageJ)
            row["Angle"] = rois.angle(i) if len(v) == 3 else math.degrees(math.atan2(dy, dx))
            row.update(
                {k: s for k, s in intensity_stats(values).items() if k in ("Mean", "StdDev", "Min", "Max")}
            )
            rows_out.append(row)
            continue
        mask = rois.mask(i, plane.shape)
        rr, cc = np.nonzero(mask)
        n = rr.size
        if n == 0:
            rows_out.append(row)
            continue
        top, left = int(rr.min()), int(cc.min())
        height, width = int(rr.max()) - top + 1, int(cc.max()) - left + 1
        area = n * px * px
        values = plane[mask]
        stats = intensity_stats(values)
        total = stats["RawIntDen"]
        perim = rois.length(i) * px
        ell = fit_ellipse(rr - top, cc - left)
        if kind == "ellipse":  # contour en pixels, comme le polygone d'un ovale dans ImageJ
            verts = trace_outline(
                ndi.binary_fill_holes(mask[top : top + height, left : left + width])
            ).astype(float)
            verts += (left, top)
        else:
            verts = rois.outline(i)[:, ::-1]  # (x, y), coins de pixels
        fer = feret(verts)
        hull_area = polygon_area(_hull(verts)) if len(verts) else 0.0
        row.update(
            {
                "Area": area,
                "Mean": stats["Mean"],
                "StdDev": stats["StdDev"],
                "Mode": stats["Mode"],
                "Min": stats["Min"],
                "Max": stats["Max"],
                "X": (cc.mean() + 0.5) * px,
                "Y": (rr.mean() + 0.5) * px,
                "XM": ((cc + 0.5) * values).sum() / total * px if total > 0 else (cc.mean() + 0.5) * px,
                "YM": ((rr + 0.5) * values).sum() / total * px if total > 0 else (rr.mean() + 0.5) * px,
                "Perim.": perim,
                "BX": left * px,
                "BY": top * px,
                "Width": width * px,
                "Height": height * px,
                "Major": ell.major * px,
                "Minor": ell.minor * px,
                "Angle": ell.angle,
                "Circ.": min(1.0, 4 * np.pi * area / perim**2) if perim > 0 else 0.0,
                "Feret": fer.max * px,
                "IntDen": area * stats["Mean"],
                "Median": stats["Median"],
                "Skew": stats["Skew"],
                "Kurt": stats["Kurt"],
                "%Area": 100.0 * float(np.mean(values != 0)),
                "RawIntDen": total,
                "FeretAngle": fer.angle,
                "MinFeret": fer.min * px,
                "AR": ell.major / ell.minor if ell.minor > 0 else 0.0,
                "Round": 4 * n / (np.pi * ell.major**2) if ell.major > 0 else 0.0,
                "Solidity": n / hull_area if hull_area > 0 else float("nan"),
            }
        )
        rows_out.append(row)
    df = pd.DataFrame(rows_out)
    if not df.empty:
        df["ROI"] = df["ROI"].astype(int)
        df.index = pd.RangeIndex(1, len(df) + 1)
    return df


@operation("Mesurer les ROI", category="Mesures", imagej="Analyze > Measure (ROI Manager > Measure)")
def measure_rois_op(image: Image, roi: Rois) -> Table:
    """Mesure chaque région dessinée (rectangle, ellipse, polygone, ligne) : surface,
    intensités, position, périmètre, forme. Mêmes colonnes et définitions qu'ImageJ."""
    if not len(roi):
        raise ValueError("Dessinez d'abord une ROI sur l'image (outils « Dessiner », en haut du panneau)")
    df = measure_rois(_plane(image), roi, image.px)
    units = {**_units(image.unit, True), "Length": image.unit}
    return Table(df, name=f"ROI — {image.name}", units=units)


# ---------------------------------------------------------------------- ROI, masques
@operation("ROI vers masque", category="ROI", imagej="Edit > Selection > Create Mask")
def rois_to_mask(image: Image, roi: Rois) -> Labels:
    """Masque de la taille de l'image : chaque région fermée y reçoit son numéro."""
    shape = image.plane_shape
    out = np.zeros(shape, np.int32)
    for i in range(len(roi)):
        if not roi.is_line(i):
            out[roi.mask(i, shape)] = i + 1
    return Labels(out, image.pixel_size_um, f"{roi.name} (masque)")


@operation(
    "Masque vers ROI",
    category="ROI",
    imagej="Edit > Selection > Create Selection / Analyze Particles (Add to Manager)",
)
def mask_to_rois(
    mask: Labels,
    max_objects: Annotated[int, Param(label="nombre maximal", min=1, max=20000, advanced=True)] = 2000,
) -> Rois:
    """Contour de chaque objet d'un masque, sous forme de polygones modifiables."""
    lab = mask.data
    if mask.is_binary:
        lab, _ = ndi.label(lab > 0, structure=_EIGHT)
    shapes: list[tuple[str, np.ndarray]] = []
    for lab_id, slc in enumerate(ndi.find_objects(lab), start=1):
        if slc is None:
            continue
        if len(shapes) >= max_objects:
            raise ValueError(
                f"Plus de {max_objects} objets : filtrez d'abord le masque (analyse de particules)"
            )
        verts = trace_outline(ndi.binary_fill_holes(lab[slc] == lab_id)).astype(float)
        if len(verts) < 3:
            continue
        verts += (slc[1].start, slc[0].start)
        shapes.append(("polygon", verts[:, ::-1] - 0.5))  # coins de pixels (x, y) → (ligne, colonne)
    return Rois(shapes, mask.pixel_size_um, f"{mask.name} (contours)")


@operation("Effacer avec les ROI", category="ROI", imagej="Edit > Clear / Clear Outside / Fill")
def clear_rois(
    image: Image,
    roi: Rois,
    where: Annotated[Literal["extérieur", "intérieur"], Param(label="zone à effacer")] = "extérieur",
    value: Annotated[
        float, Param(label="valeur", min=-1e9, span="image", description="valeur donnée aux pixels effacés")
    ] = 0.0,
) -> Image:
    """Met à une valeur fixe tout ce qui est hors des ROI (ou dedans)."""
    sel = _roi_selection(roi, image.plane_shape)
    if sel is None:
        raise ValueError("Il faut au moins une région fermée")
    out = image.data.copy()
    target = ~sel if where == "extérieur" else sel
    out[target] = ij.restore_dtype(np.array(value, np.float64), out)
    return image.with_data(out, name=f"{image.name} (effacée)")


# ------------------------------------------------------------------------- maxima
@operation("Trouver les maxima", category="Mesures", imagej="Process > Find Maxima...")
def find_maxima_op(
    image: Image,
    prominence: Annotated[
        float,
        Param(
            label="proéminence",
            min=0,
            max=1e9,
            description="de combien un maximum doit dépasser son voisinage",
        ),
    ] = 10.0,
    light_background: Annotated[
        bool, Param(label="fond clair", description="cherche les points sombres")
    ] = False,
    exclude_edges: Annotated[bool, Param(label="exclure les bords")] = False,
    strict: Annotated[bool, Param(label="strict", advanced=True)] = False,
    segment: Annotated[
        bool,
        Param(
            label="segmenter", description="découpe l'image en une zone par maximum (« Segmented Particles »)"
        ),
    ] = False,
) -> tuple[Points, Table, Labels | None]:
    """Repère et compte les maxima locaux (noyaux, grains, spots), avec la règle d'ImageJ.

    Renvoie les points, leur tableau (position, valeur) et, sur demande, une image
    d'étiquettes qui attribue à chaque maximum sa zone (ligne de partage des eaux ; son
    tracé peut différer de quelques pixels de celui d'ImageJ)."""
    plane = _plane(image)
    positions, values, _zones = find_maxima(
        plane, prominence, strict=strict, exclude_edges=exclude_edges, light_background=light_background
    )
    df = pd.DataFrame(
        {"X": positions[:, 1] * image.px, "Y": positions[:, 0] * image.px, "Valeur": values},
        index=pd.RangeIndex(1, len(positions) + 1),
    )
    table = Table(
        df,
        name=f"Maxima — {image.name}",
        units={"X": image.unit, "Y": image.unit},
        summary={"Count": len(positions)},
    )
    labels = None
    if segment and len(positions):
        from skimage import segmentation

        markers = np.zeros(plane.shape, np.int32)
        markers[positions[:, 0], positions[:, 1]] = np.arange(1, len(positions) + 1)
        relief = plane.astype(np.float64) if light_background else -plane.astype(np.float64)
        zones = segmentation.watershed(relief, markers, watershed_line=True)
        labels = Labels(zones.astype(np.int32), image.pixel_size_um, f"{image.name} (zones des maxima)")
    points = Points(positions.astype(np.float64), image.pixel_size_um, f"{image.name} (maxima)")
    return points, table, labels


@operation(
    "Compter les points", category="Mesures", imagej="Plugins > Analyze > Cell Counter / Multi-point Tool"
)
def count_points(points: Points, image: Image | None = None) -> Table:
    """Tableau des points posés à la main (calque de points) : position, et valeur de
    l'image à cet endroit si une image est fournie. Un calque de points par catégorie."""
    pos = np.asarray(points.data, dtype=np.float64).reshape(-1, max(2, np.asarray(points.data).shape[-1]))[
        :, -2:
    ]
    data: dict[str, np.ndarray] = {"X": pos[:, 1] * points.px, "Y": pos[:, 0] * points.px}
    if image is not None and len(pos):
        plane = _plane(image)
        data["Valeur"] = ndi.map_coordinates(
            plane.astype(np.float64), [pos[:, 0], pos[:, 1]], order=1, mode="nearest"
        )
    df = pd.DataFrame(data, index=pd.RangeIndex(1, len(pos) + 1))
    return Table(
        df,
        name=f"Comptage — {points.name}",
        units={"X": points.unit, "Y": points.unit},
        summary={"Count": len(pos)},
    )


# ----------------------------------------------------------------- colocalisation
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    return float((da * db).sum() / denom) if denom > 0 else float("nan")


def costes_thresholds(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Seuils automatiques de Costes : ``(seuil 1, seuil 2, pente, ordonnée)``.

    Droite de régression orthogonale du canal 2 sur le canal 1, puis on abaisse le seuil le
    long de cette droite jusqu'à ce que les pixels restés sous les seuils ne soient plus
    corrélés (r de Pearson ≤ 0).
    """
    var_a, var_b = float(a.var()), float(b.var())
    cov = float(((a - a.mean()) * (b - b.mean())).mean())
    if cov == 0 or var_a == 0:
        return float(a.max()), float(b.max()), 0.0, float(b.mean())
    slope = (var_b - var_a + math.sqrt((var_b - var_a) ** 2 + 4.0 * cov * cov)) / (2.0 * cov)
    intercept = float(b.mean()) - slope * float(a.mean())

    def below(t: float) -> float:
        sel = (a < t) | (b < slope * t + intercept)
        r = _pearson(a[sel], b[sel]) if sel.sum() > 2 else -1.0
        return -1.0 if math.isnan(r) else r

    lo, hi = float(a.min()), float(a.max())
    if below(hi) <= 0:
        return hi, slope * hi + intercept, slope, intercept
    for _ in range(40):  # recherche par dichotomie du seuil où la corrélation s'annule
        mid = (lo + hi) / 2.0
        if below(mid) > 0:
            hi = mid
        else:
            lo = mid
    return lo, slope * lo + intercept, slope, intercept


def colocalization(
    a: np.ndarray, b: np.ndarray, t1: float | None = None, t2: float | None = None
) -> dict[str, float]:
    """Coefficients de colocalisation de deux canaux (tableaux 1D de pixels appariés).

    ``t1`` / ``t2`` : seuils des canaux (``None`` : seuils de Costes). Les coefficients de
    Manders seuillés tM1 et tM2 donnent la part de l'intensité de chaque canal située là où
    l'autre dépasse son seuil ; M1 et M2 utilisent le seuil zéro.
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    slope = intercept = float("nan")
    if t1 is None or t2 is None:
        t1, t2, slope, intercept = costes_thresholds(a, b)
    above = (a > t1) & (b > t2)
    sum_a, sum_b = float(a.sum()), float(b.sum())
    prod = (a - a.mean()) * (b - b.mean())
    return {
        "Pearson": _pearson(a, b),
        "Pearson (au-dessus des seuils)": _pearson(a[above], b[above]),
        "Pearson (sous les seuils)": _pearson(a[~above], b[~above]),
        "M1": float(a[b > 0].sum() / sum_a) if sum_a > 0 else float("nan"),
        "M2": float(b[a > 0].sum() / sum_b) if sum_b > 0 else float("nan"),
        "tM1": float(a[b > t2].sum() / sum_a) if sum_a > 0 else float("nan"),
        "tM2": float(b[a > t1].sum() / sum_b) if sum_b > 0 else float("nan"),
        "Recouvrement": float((a * b).sum() / math.sqrt(float((a * a).sum()) * float((b * b).sum())))
        if (a * a).sum() > 0 and (b * b).sum() > 0
        else float("nan"),
        "ICQ (Li)": float(np.mean(prod > 0) - 0.5) if np.any(prod != 0) else 0.0,
        "Seuil 1": float(t1),
        "Seuil 2": float(t2),
        "Pente": slope,
        "Ordonnée": intercept,
        "% pixels colocalisés": 100.0 * float(above.mean()),
        "N": float(a.size),
    }


@operation("Colocalisation", category="Mesures", imagej="Analyze > Colocalization > Coloc 2")
def colocalization_op(
    image1: Image,
    image2: Image,
    mask: Labels | None = None,
    thresholds: Annotated[
        Literal["Costes (automatiques)", "manuels", "aucun"], Param(label="seuils")
    ] = "Costes (automatiques)",
    threshold1: Annotated[
        float, Param(label="seuil canal 1", min=-1e9, span="image", description="pour « manuels »")
    ] = 0.0,
    threshold2: Annotated[
        float, Param(label="seuil canal 2", min=-1e9, span="image", description="pour « manuels »")
    ] = 0.0,
) -> Table:
    """Mesure à quel point deux marquages se superposent : corrélation de Pearson,
    coefficients de Manders (M1, M2 et leurs versions seuillées), recouvrement, ICQ.

    Avec un masque, seuls ses pixels comptent (une cellule, par exemple). Soustraire le
    fond des deux images avant de mesurer."""
    a, b = _plane(image1), _plane(image2)
    if a.shape != b.shape:
        raise ValueError("Les deux images n'ont pas la même taille")
    sel = mask.data > 0 if mask is not None else np.ones(a.shape, bool)
    if mask is not None and mask.data.shape != a.shape:
        raise ValueError("Le masque n'a pas la taille des images")
    sel &= np.isfinite(a) & np.isfinite(b)
    if sel.sum() < 3:
        raise ValueError("Pas assez de pixels à comparer")
    va, vb = a[sel], b[sel]
    if thresholds.startswith("Costes"):
        res = colocalization(va, vb)
    elif thresholds == "manuels":
        res = colocalization(va, vb, threshold1, threshold2)
    else:
        res = colocalization(va, vb, float(va.min()) - 1.0, float(vb.min()) - 1.0)
    df = pd.DataFrame([res], index=pd.RangeIndex(1, 2))
    return Table(df, name=f"Colocalisation — {image1.name} / {image2.name}")
