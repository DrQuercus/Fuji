"""Conventions numériques d'ImageJ, partagées par toutes les opérations.

ImageJ a ses propres règles d'arrondi, de conversion entre types et de voisinage. Les
reprendre ici, une fois pour toutes, est ce qui permet à Fuji de donner les mêmes
résultats :

- arrondi « moitié vers le haut » (``(int)(v + 0.5)``), et non « au pair » comme NumPy ;
- conversion vers 8 bits de ``TypeConverter`` / ``create8BitImage`` ;
- masque circulaire de ``RankFilters`` ;
- noyau gaussien de ``GaussianBlur`` (queue raccordée à zéro) ;
- gris d'une image couleur : ``(R + V + B) / 3`` en division entière ;
- teinte / saturation / luminosité de ``java.awt.Color.RGBtoHSB``, ramenées à 0–255.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def java_round(values: np.ndarray | float) -> np.ndarray:
    """``(int)(v + 0.5)`` pour des valeurs positives ; ``Math.round`` de Java."""
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def restore_dtype(out: np.ndarray, like: np.ndarray | np.dtype) -> np.ndarray:
    """Revient au type d'origine : arrondi d'ImageJ et saturation aux bornes du type."""
    dtype = like.dtype if isinstance(like, np.ndarray) else np.dtype(like)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        clean = np.nan_to_num(out, nan=0.0, posinf=info.max, neginf=info.min)
        return np.clip(np.floor(clean + 0.5), info.min, info.max).astype(dtype)
    return out.astype(dtype, copy=False)


def per_channel(data: np.ndarray, fn, *, rgb: bool) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Applique ``fn`` à chaque canal d'une image couleur (ImageJ traite R, V, B séparément)."""
    if rgb:
        return np.stack([fn(data[..., c]) for c in range(data.shape[-1])], axis=-1)
    return fn(data)


# --------------------------------------------------------------------- types
def to_byte(plane: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Conversion vers 8 bits avec mise à l'échelle, comme ImageJ.

    16 bits : ``(v − min) × 256 / (max − min + 1) + 0.5`` ; 32 bits : ``(v − min) × 255 /
    (max − min) + 0.5`` ; tronqué puis borné à 255. ``lo`` / ``hi`` : plage d'affichage
    (par défaut le minimum et le maximum de l'image, comme une image fraîchement ouverte).
    """
    if plane.dtype == np.uint8:
        return plane
    finite = plane[np.isfinite(plane)] if plane.dtype.kind == "f" else plane
    if finite.size == 0:
        return np.zeros(plane.shape, np.uint8)
    lo = float(finite.min()) if lo is None else float(lo)
    hi = float(finite.max()) if hi is None else float(hi)
    value = np.maximum(np.nan_to_num(plane.astype(np.float64), nan=lo) - lo, 0.0)
    if plane.dtype.kind in "ui":
        lo, hi = float(int(lo)), float(int(hi))
        scale = 256.0 / (hi - lo + 1.0)
    else:
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
    return np.minimum(np.floor(value * scale + 0.5), 255).astype(np.uint8)


def gray8(rgb: np.ndarray, weighted: bool = False) -> np.ndarray:
    """Gris 8 bits d'une image RGB : ``(R+V+B)/3`` (division entière) ou pondéré
    (0,299 R + 0,587 V + 0,114 B, arrondi), comme « Image > Type > 8-bit »."""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    if weighted:
        return np.floor(r * 0.299 + g * 0.587 + b * 0.114 + 0.5).astype(np.uint8)
    return ((r + g + b) // 3).astype(np.uint8)


def to_byte_rgb(rgb: np.ndarray) -> np.ndarray:
    """Image couleur en 8 bits par canal (une image RGB d'ImageJ est toujours en 8 bits)."""
    c = rgb[..., :3]
    if c.dtype == np.uint8:
        return c
    top = float(c.max()) if c.size else 0.0
    if c.dtype.kind == "f" and top <= 1.0:
        return np.floor(np.clip(c, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    bits = max(8, int(math.ceil(math.log2(top + 1)))) if top > 255 else 8
    return (c.astype(np.float64) / (2**bits - 1) * 255.0 + 0.5).astype(np.uint8)


def intensity_plane(data: np.ndarray, rgb: bool) -> np.ndarray:
    """Valeurs mesurées par ImageJ : l'image elle-même, ou le gris 8 bits d'une image RGB."""
    if rgb:
        if data.dtype == np.uint8:
            return gray8(data)
        return data[..., :3].astype(np.float64).mean(axis=-1)
    return data


def rgb_to_hsb(rgb: np.ndarray) -> np.ndarray:
    """Teinte, saturation, luminosité en 8 bits (``ColorProcessor.getHSB`` d'ImageJ).

    Tout le calcul est fait en simple précision, comme ``java.awt.Color.RGBtoHSB`` : c'est
    nécessaire pour retomber sur les mêmes entiers après la multiplication par 255.
    """
    f32 = np.float32
    c = rgb[..., :3].astype(np.int32)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    cmax, cmin = c.max(axis=-1), c.min(axis=-1)
    delta = (cmax - cmin).astype(f32)
    brightness = cmax.astype(f32) / f32(255.0)
    saturation = np.where(cmax != 0, delta / np.where(cmax != 0, cmax, 1).astype(f32), f32(0.0)).astype(f32)
    safe = np.where(delta > 0, delta, f32(1.0)).astype(f32)
    redc, greenc, bluec = ((cmax - x).astype(f32) / safe for x in (r, g, b))
    hue = np.where(
        r == cmax, bluec - greenc, np.where(g == cmax, f32(2.0) + redc - bluec, f32(4.0) + greenc - redc)
    )
    hue = (hue.astype(f32) / f32(6.0)).astype(f32)
    hue = np.where(hue < 0, hue + f32(1.0), hue).astype(f32)
    hue = np.where(saturation == 0, f32(0.0), hue).astype(f32)
    hsb = np.stack([hue, saturation, brightness], axis=-1)
    return (hsb.astype(np.float64) * 255.0).astype(np.uint8)


def hsb_to_rgb(hsb: np.ndarray) -> np.ndarray:
    """Inverse de :func:`rgb_to_hsb` (``ColorProcessor.setHSB`` / ``Color.HSBtoRGB``)."""
    f32 = np.float32
    h, s, v = ((hsb[..., i].astype(np.float64) / 255.0).astype(f32) for i in range(3))
    h6 = ((h - np.floor(h)) * f32(6.0)).astype(f32)
    f = (h6 - np.floor(h6)).astype(f32)
    p = (v * (f32(1.0) - s)).astype(f32)
    q = (v * (f32(1.0) - s * f)).astype(f32)
    t = (v * (f32(1.0) - (s * (f32(1.0) - f)))).astype(f32)
    sector = h6.astype(np.int32) % 6
    table = {"r": [v, q, p, p, t, v], "g": [t, v, v, q, p, p], "b": [p, p, t, v, v, q]}
    out = np.stack([np.choose(sector, table[k]) for k in "rgb"], axis=-1).astype(f32)
    out = np.where((s == 0)[..., None], v[..., None], out).astype(f32)
    return (out * f32(255.0) + f32(0.5)).astype(np.uint8)


# ------------------------------------------------------------------- voisinages
def disk(radius: float) -> np.ndarray:
    """Masque circulaire d'ImageJ (``RankFilters``) : pixels tels que dx² + dy² ≤ r² + 1.

    Rayon 1 → carré 3×3 ; rayon 2 → 21 pixels (et non les 13 d'un disque « strict »).
    """
    r2 = int(radius * radius) + 1
    k = int(math.sqrt(r2 + 1e-10))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    return (xx * xx + yy * yy) <= r2


def gaussian_kernel(sigma: float, accuracy: float) -> np.ndarray:
    """Noyau 1D de ``GaussianBlur.makeGaussianKernel`` (symétrique, normalisé).

    La gaussienne est coupée là où elle passe sous ``accuracy`` ; sa queue est remplacée
    par un raccord quadratique pour arriver à zéro sans marche.
    """
    radius = int(math.ceil(sigma * math.sqrt(-2.0 * math.log(accuracy)))) + 1
    i = np.arange(radius, dtype=np.float64)
    half = np.exp(-0.5 * i * i / sigma / sigma).astype(np.float32).astype(np.float64)
    if radius > 3:
        sqrt_slope = math.inf
        r = radius
        while r > radius // 2:
            r -= 1
            a = math.sqrt(half[r]) / (radius - r)
            if a < sqrt_slope:
                sqrt_slope = a
            else:
                break
        for r1 in range(r + 2, radius):
            half[r1] = np.float32((radius - r1) * (radius - r1) * sqrt_slope * sqrt_slope)
    total = half[0] + 2.0 * half[1:].sum()
    half = half / total
    return np.concatenate([half[:0:-1], half])


def gaussian_blur(plane: np.ndarray, sigma: float, accuracy: float | None = None) -> np.ndarray:
    """Flou gaussien d'ImageJ sur un plan (résultat en virgule flottante).

    ``accuracy`` : 0,002 pour 8 bits et RGB, 0,0002 sinon (valeurs d'ImageJ). Hors de
    l'image, le pixel du bord est répété. Jusqu'à σ = 4,5 le calcul est celui d'ImageJ ;
    au-delà, ImageJ réduit l'image avant de la flouter pour aller plus vite, et les deux
    résultats peuvent différer d'une fraction de niveau de gris.
    """
    if sigma <= 0:
        return plane.astype(np.float64)
    if accuracy is None:
        accuracy = 0.002 if plane.dtype == np.uint8 else 0.0002
    kernel = gaussian_kernel(sigma, accuracy)
    out = ndi.correlate1d(plane.astype(np.float64), kernel, axis=1, mode="nearest")
    return ndi.correlate1d(out, kernel, axis=0, mode="nearest")
