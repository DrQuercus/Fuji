"""Aide à la mise au point : indicateur de netteté, mémoire du maximum, focus peaking,
et autofocus pour une mise au point motorisée."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi

if TYPE_CHECKING:
    from fuji.core.microscope import Microscope

METRICS = ("tenengrad", "laplacien", "variance normalisée", "brenner")


def subsample_step(shape: tuple[int, ...], max_side: int | None) -> int:
    """Pas de sous-échantillonnage pour que le plus grand côté ne dépasse pas ``max_side``."""
    if not max_side:
        return 1
    return max(1, math.ceil(max(shape[:2]) / max_side))


def to_gray(image: np.ndarray, max_side: int | None = 768) -> np.ndarray:
    """Niveaux de gris float32, réduit par sous-échantillonnage pour la vitesse.

    Le sous-échantillonnage est fait **avant** toute conversion : sur une image 4K
    couleur, convertir d'abord coûterait plus de 100 Mo et figerait l'affichage.
    Le pixel ``(i, j)`` du résultat correspond au pixel ``(i·pas, j·pas)`` de l'image.
    """
    step = subsample_step(image.shape, max_side)
    g = image[::step, ::step] if step > 1 else image
    if g.ndim == 3:
        return g[..., :3].mean(axis=-1, dtype=np.float32)
    return g.astype(np.float32)


def sharpness(image: np.ndarray, method: str = "tenengrad", *, max_side: int | None = 768) -> float:
    """Netteté de l'image (plus grand = plus net), insensible à la luminosité."""
    g = to_gray(image, max_side)
    mean = float(g.mean()) or 1.0
    if method == "tenengrad":
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        return float(np.mean(gx * gx + gy * gy)) / mean**2
    if method == "laplacien":
        return float(ndi.laplace(g).var()) / mean**2
    if method == "variance normalisée":
        return float(g.var()) / mean
    if method == "brenner":
        d = g[:, 2:] - g[:, :-2]
        return float(np.mean(d * d)) / mean**2
    raise ValueError(f"Méthode inconnue {method!r} ; méthodes : {METRICS}")


def focus_peaking(
    image: np.ndarray, percentile: float = 97.0, *, max_side: int | None = 1024
) -> tuple[np.ndarray, int]:
    """Masque des zones les plus nettes (à superposer au live) et son pas d'échantillonnage.

    Calculé sur une image réduite pour rester fluide : un pixel du masque couvre
    ``pas × pas`` pixels de l'image.
    """
    step = subsample_step(image.shape, max_side)
    g = ndi.gaussian_filter(to_gray(image, max_side), 0.8)
    mag = np.hypot(ndi.sobel(g, axis=1), ndi.sobel(g, axis=0))
    return mag > np.percentile(mag, percentile), step


@dataclass
class FocusState:
    value: float
    best: float
    relative: float  # valeur / meilleure (0–1)
    trend: str  # « meilleure », « s'améliore », « se dégrade », « stable »


class FocusTracker:
    """Suit la netteté en direct et se souvient du meilleur point atteint.

    En mise au point manuelle : on tourne la vis jusqu'à dépasser le maximum, puis
    on revient vers le maximum indiqué.
    """

    def __init__(self, method: str = "tenengrad", history: int = 120) -> None:
        self.method = method
        self.history: deque[float] = deque(maxlen=history)
        self.best = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.best = 0.0

    def update(self, image: np.ndarray) -> FocusState:
        v = sharpness(image, self.method)
        prev = self.history[-1] if self.history else None
        self.history.append(v)
        if v >= self.best:
            self.best = v
            trend = "meilleure"
        elif prev is None or abs(v - prev) < 0.01 * max(v, 1e-12):
            trend = "stable"
        else:
            trend = "s'améliore" if v > prev else "se dégrade"
        rel = v / self.best if self.best > 0 else 0.0
        return FocusState(v, self.best, rel, trend)


def autofocus(
    microscope: Microscope,
    *,
    range_um: float = 20.0,
    step_um: float = 2.0,
    method: str = "tenengrad",
) -> float:
    """Balayage grossier puis affinage parabolique (mise au point motorisée uniquement)."""
    focus = microscope.focus
    if focus is None or focus.manual:
        raise RuntimeError("L'autofocus nécessite une mise au point motorisée")
    z0 = focus.get_position()

    def scan(center: float, half: float, step: float) -> tuple[np.ndarray, np.ndarray]:
        zs = np.arange(center - half, center + half + step / 2, step)
        vals = []
        for z in zs:
            focus.move_to(float(z))
            vals.append(sharpness(microscope.snap().data, method))
        return zs, np.asarray(vals)

    try:
        zs, vals = scan(z0, range_um / 2, step_um)
        best = float(zs[int(np.argmax(vals))])
        zs, vals = scan(best, step_um, step_um / 4)
        i = int(np.argmax(vals))
        z_best = float(zs[i])
        if 0 < i < len(zs) - 1:  # sommet de la parabole passant par 3 points
            y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                z_best += 0.5 * (y0 - y2) / denom * (zs[1] - zs[0])
    except BaseException:
        focus.move_to(z0)  # ne pas laisser la mise au point au milieu du balayage
        raise
    focus.move_to(z_best)
    return z_best
