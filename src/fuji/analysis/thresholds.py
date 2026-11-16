"""Seuillage automatique : les 17 méthodes d'ImageJ, avec ses noms et ses conventions.

Chaque méthode est une transcription de ``ij.process.AutoThresholder`` (elles-mêmes
portées par G. Landini). Toutes travaillent sur un histogramme à 256 classes et rendent
un **niveau** entier : les objets sombres sont les niveaux ``0..niveau``, les objets
clairs les niveaux ``niveau+1..255``.

Pour une image 16 ou 32 bits, ImageJ commence par la ramener à 8 bits (``convertToByte``)
puis reporte le niveau sur l'échelle d'origine avec ``min + niveau / 255 × (max − min)`` :
c'est ce que fait :func:`threshold_range`.

Les seuillages locaux (:func:`local_threshold`) reprennent le plugin « Auto Local
Threshold » de Fiji.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from . import ij

METHODS = (
    "Default",
    "Huang",
    "Intermodes",
    "IsoData",
    "IJ_IsoData",
    "Li",
    "MaxEntropy",
    "Mean",
    "MinError",
    "Minimum",
    "Moments",
    "Otsu",
    "Percentile",
    "RenyiEntropy",
    "Shanbhag",
    "Triangle",
    "Yen",
)
LOCAL_METHODS = ("Bernsen", "Contrast", "Mean", "Median", "MidGrey", "Niblack", "Phansalkar", "Sauvola")

_EPS = 2.220446049250313e-16
_I = np.arange(256, dtype=np.float64)


# ------------------------------------------------------------------ histogrammes
def histogram256(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Histogramme à 256 classes, et plage (min, max) de l'image.

    8 bits : les 256 niveaux. Autres types : l'image est d'abord ramenée à 8 bits comme
    le fait ImageJ avant un seuillage automatique.
    """
    a = np.asarray(values)
    if a.dtype == np.uint8:
        return np.bincount(a.ravel(), minlength=256)[:256].astype(np.int64), 0.0, 255.0
    finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    if finite.size == 0:
        return np.zeros(256, np.int64), 0.0, 0.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hist = np.zeros(256, np.int64)
        hist[0] = finite.size
        return hist, lo, hi
    return np.bincount(ij.to_byte(a, lo, hi).ravel(), minlength=256)[:256].astype(np.int64), lo, hi


# --------------------------------------------------------------------- méthodes
def ij_default_level(hist: np.ndarray) -> int:
    """« Default » et « IJ_IsoData » : IsoData modifié d'ImageJ (``IJDefault``)."""
    data = np.asarray(hist, dtype=np.float64).copy()
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
    return math.floor(result + 0.5)  # Math.round de Java (Python arrondit au pair)


def _huang(data: np.ndarray) -> int:
    nz = np.flatnonzero(data)
    first, last = int(nz[0]), int(nz[-1])
    if last == first:
        return first
    term = 1.0 / (last - first)
    w = data.astype(np.float64)
    mu0 = np.zeros(256)
    num, tot = np.cumsum(w[first:]), np.cumsum(_I[first:] * w[first:])
    mu0[first:] = tot / num
    mu1 = np.zeros(256)
    idx = np.arange(last, 0, -1)
    mu1[idx - 1] = np.cumsum(idx * w[idx]) / np.cumsum(w[idx])
    ih = _I[None, :]
    it = np.arange(256)[:, None]
    centre = np.where(ih <= it, mu0[:, None], mu1[:, None])
    mu = 1.0 / (1.0 + term * np.abs(ih - centre))
    valid = ~((mu < 1e-06) | (mu > 0.999999))
    safe = np.where(valid, mu, 0.5)
    shannon = np.where(valid, -safe * np.log(safe) - (1.0 - safe) * np.log(1.0 - safe), 0.0)
    return int(np.argmin((shannon * w[None, :]).sum(axis=1)))


def _bimodal(y: np.ndarray) -> bool:
    modes = int(np.count_nonzero((y[:-2] < y[1:-1]) & (y[2:] < y[1:-1])))
    return modes == 2


def _smooth3(y: np.ndarray) -> np.ndarray:
    """Moyenne glissante sur 3 points, zéro au-delà des extrémités."""
    padded = np.concatenate([[0.0], y, [0.0]])
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def _intermodes(data: np.ndarray) -> int:
    nz = np.flatnonzero(data)
    minbin, maxbin = int(nz[0]), int(nz[-1])
    hist = data[minbin : maxbin + 1].astype(np.float64)
    for _ in range(10001):
        if len(hist) < 3 or _bimodal(hist):
            break
        hist = _smooth3(hist)
    else:
        return -1
    if len(hist) < 3:
        return -1
    peaks = np.flatnonzero((hist[:-2] < hist[1:-1]) & (hist[2:] < hist[1:-1])) + 1
    return math.floor(int(peaks.sum()) / 2.0) + minbin


def _minimum(data: np.ndarray) -> int:
    hist = data.astype(np.float64)
    top = int(np.flatnonzero(data)[-1])
    for _ in range(10001):
        if _bimodal(hist):
            break
        hist = _smooth3(hist)
    else:
        return -1
    for i in range(1, top):
        if hist[i - 1] > hist[i] and hist[i + 1] >= hist[i]:
            return i
    return -1


def _isodata(data: np.ndarray) -> int:
    counts = data.astype(np.int64)
    g = 0
    for i in range(1, 256):
        if counts[i] > 0:
            g = i + 1
            break
    weighted = counts * np.arange(256)
    while True:
        totl, low = int(counts[:g].sum()), int(weighted[:g].sum())
        toth, high = int(counts[g + 1 :].sum()), int(weighted[g + 1 :].sum())
        if totl > 0 and toth > 0:
            low //= totl  # divisions entières, comme le code d'origine
            high //= toth
            if g == math.floor((low + high) / 2.0 + 0.5):
                return g
        g += 1
        if g > 254:
            return -1


def _li(data: np.ndarray) -> int:
    w = data.astype(np.float64)
    num_pixels = w.sum()
    new_thresh = float((_I[1:] * w[1:]).sum() / num_pixels)
    threshold = 0
    for _ in range(10000):
        old_thresh = new_thresh
        threshold = int(old_thresh + 0.5)
        num_back = w[: threshold + 1].sum()
        mean_back = (_I[: threshold + 1] * w[: threshold + 1]).sum() / num_back if num_back else 0.0
        num_obj = w[threshold + 1 :].sum()
        mean_obj = (_I[threshold + 1 :] * w[threshold + 1 :]).sum() / num_obj if num_obj else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            temp = (mean_back - mean_obj) / (np.log(mean_back) - np.log(mean_obj))
        if not np.isfinite(temp):
            break
        new_thresh = float(int(temp - 0.5) if temp < -_EPS else int(temp + 0.5))
        if abs(new_thresh - old_thresh) <= 0.5:
            break
    return threshold


def _cumulative(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Histogramme normalisé, cumuls P1 et P2 = 1 − P1, première et dernière classes utiles."""
    norm = data.astype(np.float64) / data.sum()
    p1 = np.cumsum(norm)
    p2 = 1.0 - p1
    first = int(np.argmax(np.abs(p1) >= _EPS))
    last = 255
    for ih in range(255, first - 1, -1):
        if not abs(p2[ih]) < _EPS:
            last = ih
            break
    return norm, p1, p2, first, last


def _entropy_split(norm: np.ndarray, p1: np.ndarray, p2: np.ndarray, it: int) -> float:
    back, obj = norm[: it + 1], norm[it + 1 :]
    back, obj = back[back > 0] / p1[it], obj[obj > 0] / p2[it]
    return float(-(back * np.log(back)).sum() - (obj * np.log(obj)).sum())


def _max_entropy(data: np.ndarray) -> int:
    norm, p1, p2, first, last = _cumulative(data)
    threshold, best = -1, np.finfo(float).tiny
    for it in range(first, last + 1):
        total = _entropy_split(norm, p1, p2, it)
        if best < total:
            best, threshold = total, it
    return threshold


def _mean(data: np.ndarray) -> int:
    w = data.astype(np.float64)
    return math.floor((_I * w).sum() / w.sum())


def _min_error(data: np.ndarray) -> int:
    w = data.astype(np.float64)
    a, b, c = np.cumsum(w), np.cumsum(_I * w), np.cumsum(_I * _I * w)
    threshold, previous = _mean(data), -2
    while threshold != previous:
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = b[threshold] / a[threshold]
            nu = (b[255] - b[threshold]) / (a[255] - a[threshold])
            p = a[threshold] / a[255]
            q = (a[255] - a[threshold]) / a[255]
            sigma2 = c[threshold] / a[threshold] - mu * mu
            tau2 = (c[255] - c[threshold]) / (a[255] - a[threshold]) - nu * nu
            w0 = 1.0 / sigma2 - 1.0 / tau2
            w1 = mu / sigma2 - nu / tau2
            w2 = (mu * mu) / sigma2 - (nu * nu) / tau2 + np.log10((sigma2 * (q * q)) / (tau2 * (p * p)))
            sqterm = w1 * w1 - w0 * w2
            if sqterm < 0:  # pas de solution réelle : on garde le seuil courant
                break
            previous = threshold
            temp = (w1 + np.sqrt(sqterm)) / w0
        if not np.isfinite(temp):
            threshold = previous
            break
        threshold = int(np.clip(math.floor(temp), 0, 255))
    return threshold


def _moments(data: np.ndarray) -> int:
    histo = data.astype(np.float64) / data.sum()
    m0, m1, m2, m3 = 1.0, (_I * histo).sum(), (_I**2 * histo).sum(), (_I**3 * histo).sum()
    cd = m0 * m2 - m1 * m1
    if cd == 0:
        return -1
    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (m0 * -m3 + m2 * m1) / cd
    root = math.sqrt(max(c1 * c1 - 4.0 * c0, 0.0))
    z0, z1 = 0.5 * (-c1 - root), 0.5 * (-c1 + root)
    if z1 == z0:
        return -1
    p0 = (z1 - m1) / (z1 - z0)
    above = np.flatnonzero(np.cumsum(histo) > p0)
    return int(above[0]) if len(above) else -1


def _otsu(data: np.ndarray) -> int:
    histo = data.astype(np.float64) / data.sum()
    cnh = np.cumsum(histo)
    mean = np.cumsum(_I * histo)
    with np.errstate(divide="ignore", invalid="ignore"):
        bcv = mean[-1] * cnh - mean
        bcv = bcv * (bcv / (cnh * (1.0 - cnh)))
    bcv = np.where(np.isfinite(bcv), bcv, -np.inf)
    best = int(np.argmax(bcv))  # premier maximum, comme la comparaison stricte d'origine
    return best if bcv[best] > 0 else -1


def _percentile(data: np.ndarray) -> int:
    w = data.astype(np.float64)
    avec = np.abs(np.cumsum(w) / w.sum() - 0.5)
    best = int(np.argmin(avec))
    return best if avec[best] < 1.0 else -1


def _renyi(data: np.ndarray) -> int:
    norm, p1, p2, first, last = _cumulative(data)
    t_star2, best = 0, 0.0
    for it in range(first, last + 1):
        total = _entropy_split(norm, p1, p2, it)
        if best < total:
            best, t_star2 = total, it

    def best_split(score) -> int:  # type: ignore[no-untyped-def]
        threshold, best_score = 0, 0.0
        for it in range(first, last + 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                back, obj = score(norm[: it + 1], p1[it]), score(norm[it + 1 :], p2[it])
            product = back * obj
            total = scale * (math.log(product) if product > 0.0 else 0.0)
            if total > best_score:
                best_score, threshold = total, it
        return threshold

    scale = 1.0 / (1.0 - 0.5)
    t_star1 = best_split(lambda h, p: float(np.sqrt(h / p).sum()))
    scale = 1.0 / (1.0 - 2.0)
    t_star3 = best_split(lambda h, p: float(((h * h) / (p * p)).sum()))
    t_star1, t_star2, t_star3 = sorted((t_star1, t_star2, t_star3))
    near12, near23 = abs(t_star1 - t_star2) <= 5, abs(t_star2 - t_star3) <= 5
    if near12 == near23:
        beta1, beta2, beta3 = 1, 2, 1
    elif near12:
        beta1, beta2, beta3 = 0, 1, 3
    else:
        beta1, beta2, beta3 = 3, 1, 0
    omega = p1[t_star3] - p1[t_star1]
    return int(
        t_star1 * (p1[t_star1] + 0.25 * omega * beta1)
        + 0.25 * t_star2 * omega * beta2
        + t_star3 * (p2[t_star3] + 0.25 * omega * beta3)
    )


def _shanbhag(data: np.ndarray) -> int:
    norm, p1, p2, first, last = _cumulative(data)
    threshold, best = -1, math.inf
    for it in range(first, last + 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = 0.5 / p1[it]
            back = -(norm[1 : it + 1] * np.log(1.0 - term * p1[:it])).sum() * term
            term = 0.5 / p2[it]
            obj = -(norm[it + 1 :] * np.log(1.0 - term * p2[it + 1 :])).sum() * term
        total = abs(back - obj)
        if total < best:
            best, threshold = total, it
    return threshold


def _triangle(data: np.ndarray) -> int:
    d = data.astype(np.float64).copy()
    n = len(d)
    nz = np.flatnonzero(d)
    lo = int(nz[0])
    if lo > 0:
        lo -= 1  # la droite part du premier niveau vide, pas du premier niveau occupé
    lo2 = int(nz[-1])
    if lo2 < n - 1:
        lo2 += 1
    peak = int(np.argmax(d))
    inverted = (peak - lo) < (lo2 - peak)
    if inverted:  # la queue la plus longue est à droite : on retourne l'histogramme
        d = d[::-1].copy()
        lo, peak = n - 1 - lo2, n - 1 - peak
    if lo == peak:
        return lo
    nx, ny = d[peak], float(lo - peak)
    norm = math.sqrt(nx * nx + ny * ny)
    nx, ny = nx / norm, ny / norm
    offset = nx * lo + ny * d[lo]
    split, best = lo, 0.0
    for i in range(lo + 1, peak + 1):
        distance = nx * i + ny * d[i] - offset
        if distance > best:
            split, best = i, distance
    split -= 1
    return n - 1 - split if inverted else split


def _yen(data: np.ndarray) -> int:
    norm = data.astype(np.float64) / data.sum()
    p1 = np.cumsum(norm)
    p1_sq = np.cumsum(norm * norm)
    p2_sq = np.concatenate([np.cumsum((norm * norm)[:0:-1])[::-1], [0.0]])
    a, b = p1_sq * p2_sq, p1 * (1.0 - p1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crit = -1.0 * np.where(a > 0.0, np.log(np.where(a > 0, a, 1.0)), 0.0) + 2.0 * np.where(
            b > 0.0, np.log(np.where(b > 0, b, 1.0)), 0.0
        )
    best = int(np.argmax(crit))
    return best if crit[best] > np.finfo(float).tiny else -1


_LEVELS = {
    "Default": ij_default_level,
    "IJ_IsoData": ij_default_level,
    "Huang": _huang,
    "Intermodes": _intermodes,
    "IsoData": _isodata,
    "Li": _li,
    "MaxEntropy": _max_entropy,
    "Mean": _mean,
    "MinError": _min_error,
    "Minimum": _minimum,
    "Moments": _moments,
    "Otsu": _otsu,
    "Percentile": _percentile,
    "RenyiEntropy": _renyi,
    "Shanbhag": _shanbhag,
    "Triangle": _triangle,
    "Yen": _yen,
}


def threshold_level(hist: np.ndarray, method: str = "Default") -> int:
    """Niveau (0–255) rendu par ``AutoThresholder.getThreshold`` pour cet histogramme."""
    if method not in _LEVELS:
        raise ValueError(f"Méthode inconnue {method!r} ; méthodes : {METHODS}")
    data = np.asarray(hist, dtype=np.int64)
    if len(data) != 256:
        raise ValueError("L'histogramme doit avoir 256 classes")
    if data.sum() == 0:
        return 0
    level = int(_LEVELS[method](data))
    return 0 if level == -1 else level


def threshold_range(
    values: np.ndarray, method: str = "Default", *, dark_background: bool = True
) -> tuple[float, float]:
    """Bornes (incluses) des pixels retenus, dans l'unité de l'image.

    ``dark_background=True`` : objets clairs sur fond sombre (niveaux au-dessus du seuil) ;
    ``False`` : objets sombres. Même règle que ``ImageProcessor.setAutoThreshold``.
    """
    hist, lo, hi = histogram256(values)
    level = threshold_level(hist, method)
    lower, upper = (min(level + 1.0, 255.0), 255.0) if dark_background else (0.0, float(level))
    if np.asarray(values).dtype == np.uint8:
        return lower, upper
    if hi <= lo:
        return lo, lo
    return lo + lower / 255.0 * (hi - lo), lo + upper / 255.0 * (hi - lo)


def threshold_value(values: np.ndarray, method: str = "Default") -> float:
    """Valeur de seuil : les objets *clairs* sont ``>= seuil``."""
    return threshold_range(values, method, dark_background=True)[0]


# ---------------------------------------------------------------- seuillage local
def local_threshold(
    plane: np.ndarray,
    method: str = "Phansalkar",
    radius: float = 15,
    param1: float = 0.0,
    param2: float = 0.0,
    *,
    bright_objects: bool = True,
) -> np.ndarray:
    """Seuillage local (plugin « Auto Local Threshold » de Fiji), sur une image 8 bits.

    Le seuil de chaque pixel dépend de son voisinage circulaire de rayon ``radius``.
    ``param1`` et ``param2`` valent 0 pour les réglages par défaut de chaque méthode :

    ============ =============================================== ====================
    méthode      seuil local                                     paramètres (défaut)
    ============ =============================================== ====================
    Bernsen      milieu (max+min)/2 si le contraste suffit       contraste min (15)
    Contrast     plus proche du maximum que du minimum           —
    Mean         moyenne − c                                     c (0)
    Median       médiane − c                                     c (0)
    MidGrey      (max+min)/2 − c                                 c (0)
    Niblack      moyenne + k·écart-type − c                      k (±0,2), c (0)
    Phansalkar   moyenne·(1 + p·e^(−q·moy) + k·(σ/r − 1))        k (0,25), r (0,5)
    Sauvola      moyenne·(1 + k·(σ/r − 1))                       k (0,5), r (128)
    ============ =============================================== ====================

    Renvoie le masque des pixels au-dessus du seuil local (``bright_objects``) ou son
    complément, comme l'option « White objects on black background » du plugin.
    """
    if method not in LOCAL_METHODS:
        raise ValueError(f"Méthode inconnue {method!r} ; méthodes : {LOCAL_METHODS}")
    pixels = ij.to_byte(plane).astype(np.float64)
    fp = ij.disk(radius)
    n = float(fp.sum())

    def local_mean(a: np.ndarray) -> np.ndarray:
        return ndi.correlate(a, fp.astype(np.float64), mode="nearest") / n

    def local_var(a: np.ndarray) -> np.ndarray:
        mean = local_mean(a)
        return np.maximum(local_mean(a * a) - mean * mean, 0.0)

    if method in ("Bernsen", "Contrast", "MidGrey"):
        vmax = ndi.maximum_filter(pixels, footprint=fp, mode="nearest")
        vmin = ndi.minimum_filter(pixels, footprint=fp, mode="nearest")
    if method == "Bernsen":
        contrast_threshold = param1 if param1 else 15.0
        mid = np.floor((vmax + vmin) / 2.0)
        above = np.where((vmax - vmin) < contrast_threshold, mid >= 128, pixels >= mid)
    elif method == "Contrast":
        above = np.abs(vmax - pixels) <= np.abs(vmin - pixels)
    elif method == "MidGrey":
        above = pixels > np.trunc((vmax + vmin) / 2.0 - param1)
    elif method == "Mean":
        above = pixels > np.trunc(local_mean(pixels) - param1)
    elif method == "Median":
        above = pixels > np.trunc(ndi.median_filter(pixels, footprint=fp, mode="nearest") - param1)
    elif method == "Niblack":
        k = param1 if param1 else (0.2 if bright_objects else -0.2)
        above = pixels > np.trunc(local_mean(pixels) + k * np.sqrt(local_var(pixels)) - param2)
    elif method == "Sauvola":
        k, r = (param1 if param1 else 0.5), (param2 if param2 else 128.0)
        above = pixels > np.trunc(local_mean(pixels) * (1.0 + k * (np.sqrt(local_var(pixels)) / r - 1.0)))
    else:  # Phansalkar, sur l'image ramenée entre 0 et 1
        k, r = (param1 if param1 else 0.25), (param2 if param2 else 0.5)
        norm = pixels / 255.0
        mean = local_mean(norm)
        std = np.sqrt(local_var(norm))
        above = norm > mean * (1.0 + 2.0 * np.exp(-10.0 * mean) + k * (std / r - 1.0))
    return above if bright_objects else ~above
