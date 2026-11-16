"""Méthodes de seuillage automatique, avec les noms et conventions d'ImageJ.

« Default » reproduit l'algorithme IsoData modifié d'ImageJ (celui de la fenêtre
Image > Adjust > Threshold en mode « Default »). Les autres méthodes viennent de
scikit-image, sur l'histogramme à 256 classes comme dans ImageJ.
"""

from __future__ import annotations

import numpy as np
from skimage import filters

METHODS = ("Default", "Otsu", "Li", "Yen", "IsoData", "Mean", "Minimum", "Triangle")


def histogram256(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Histogramme à 256 classes entre min et max (comme ImageJ pour 16 bits/float)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    if np.issubdtype(np.asarray(values).dtype, np.uint8):
        lo, hi = 0.0, 255.0
        hist = np.bincount(np.asarray(values).ravel(), minlength=256)[:256]
        return hist.astype(np.int64), lo, hi
    if hi <= lo:
        return np.zeros(256, np.int64), lo, hi
    hist, _ = np.histogram(v, bins=256, range=(lo, hi))
    return hist.astype(np.int64), lo, hi


def ij_default_level(hist: np.ndarray) -> int:
    """Algorithme « Default » d'ImageJ (IsoData modifié, AutoThresholder.IJDefault)."""
    data = hist.astype(np.float64).copy()
    max_value = len(data) - 1
    data[0] = 0  # les extrêmes sont ignorés (zones effacées / saturées)
    data[max_value] = 0
    lo = 0
    while data[lo] == 0 and lo < max_value:
        lo += 1
    hi = max_value
    while data[hi] == 0 and hi > 0:
        hi -= 1
    if lo >= hi:
        return len(data) // 2
    idx = np.arange(len(data), dtype=np.float64)
    moving = lo
    while True:
        below, above = slice(lo, moving + 1), slice(moving + 1, hi + 1)
        s1, s2 = (idx[below] * data[below]).sum(), data[below].sum()
        s3, s4 = (idx[above] * data[above]).sum(), data[above].sum()
        result = (s1 / s2 + s3 / s4) / 2.0 if s2 > 0 and s4 > 0 else float(moving)
        moving += 1
        if not ((moving + 1) <= result and moving < hi - 1):
            break
    return int(round(result))


def threshold_value(values: np.ndarray, method: str = "Default") -> float:
    """Valeur de seuil : les objets *clairs* sont ``>= seuil``."""
    if method not in METHODS:
        raise ValueError(f"Méthode inconnue {method!r} ; méthodes : {METHODS}")
    hist, lo, hi = histogram256(values)
    if hi <= lo:
        return hi
    width = (hi - lo) / 256.0 if not np.issubdtype(np.asarray(values).dtype, np.uint8) else 1.0
    if method == "Default":
        level = ij_default_level(hist)
        # ImageJ : objets = classes > level
        return lo + (level + 1) * width
    fn = {
        "Otsu": filters.threshold_otsu,
        "Li": filters.threshold_li,
        "Yen": filters.threshold_yen,
        "IsoData": filters.threshold_isodata,
        "Mean": filters.threshold_mean,
        "Minimum": filters.threshold_minimum,
        "Triangle": filters.threshold_triangle,
    }[method]
    if method in ("Li", "Mean", "Triangle"):  # ces méthodes travaillent sur les valeurs
        return float(fn(np.asarray(values, dtype=np.float64)))
    centers = lo + (np.arange(256) + 0.5) * width
    level = float(fn(hist=(hist, centers)))
    return level
