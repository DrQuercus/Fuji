"""Mesure du déplacement entre deux images (recalage de piles, calibration de platine,
raccord des mosaïques)."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

#: en dessous de cette corrélation, un déplacement mesuré ne vaut rien (image vide, sombre, floue)
MIN_CONFIDENCE = 0.3


def _gray(image: np.ndarray) -> np.ndarray:
    return image[..., :3].mean(axis=-1) if image.ndim == 3 else image


def _prepare(image: np.ndarray) -> np.ndarray:
    """Retire l'éclairage (fixe dans l'image, il fausserait la mesure) et adoucit les bords."""
    a = image.astype(np.float64)
    a = a - ndi.gaussian_filter(a, max(2.0, min(a.shape) / 12.0), mode="nearest")
    return a * np.outer(np.hanning(a.shape[0] + 2)[1:-1], np.hanning(a.shape[1] + 2)[1:-1])


def image_shift(reference: np.ndarray, moved: np.ndarray, upsample: int = 10) -> tuple[float, float]:
    """Déplacement du contenu de ``moved`` par rapport à ``reference``, en pixels :
    (vers la droite, vers le bas).

    Corrélation croisée (non normalisée : la version « phase » décroche sur les images
    bruitées), au dixième de pixel par défaut. Les deux images doivent se recouvrir
    largement et contenir des détails.
    """
    from skimage.registration import phase_cross_correlation

    a, b = _gray(reference), _gray(moved)
    step = max(1, max(a.shape) // 1024)  # inutile de corréler 8 millions de pixels
    a, b = a[::step, ::step].astype(np.float64), b[::step, ::step].astype(np.float64)
    prepared = _prepare(a)
    total = np.zeros(2)  # correction (lignes, colonnes) qui ramène ``moved`` sur ``reference``
    current = b
    # la fenêtre qui adoucit les bords biaise un peu la mesure : on corrige, puis on remesure
    for _ in range(3 if upsample > 1 else 1):
        shift, _, _ = phase_cross_correlation(
            prepared, _prepare(current), upsample_factor=upsample, normalization=None
        )
        total += shift
        if np.abs(shift).max() < 0.05:
            break
        current = ndi.shift(b, total, order=1, mode="nearest")
    return -float(total[1]) * step, -float(total[0]) * step


def shift_confidence(reference: np.ndarray, moved: np.ndarray, shift: tuple[float, float]) -> float:
    """Confiance dans un déplacement mesuré par :func:`image_shift` : corrélation (−1 à 1) des
    deux images une fois ``moved`` ramenée sur ``reference``, dans leur partie commune.

    Proche de 1 quand les mêmes détails se superposent ; proche de 0 quand il n'y avait rien à
    reconnaître (champ vide, lampe éteinte, image floue) ou que les images ne se recouvrent
    pas : le déplacement mesuré n'a alors aucun sens. Seuil conseillé : :data:`MIN_CONFIDENCE`.
    """
    a, b = _gray(reference), _gray(moved)
    step = max(1, max(a.shape) // 1024)
    a, b = a[::step, ::step].astype(np.float64), b[::step, ::step].astype(np.float64)
    dx, dy = shift[0] / step, shift[1] / step
    my, mx = math.ceil(abs(dy)) + 2, math.ceil(abs(dx)) + 2
    if a.shape[0] - 2 * my < 16 or a.shape[1] - 2 * mx < 16:
        return 0.0  # presque plus de partie commune
    background = max(2.0, min(a.shape) / 12.0)

    def details(x: np.ndarray) -> np.ndarray:  # ni l'éclairage (fixe), ni le bruit pixel à pixel
        return ndi.gaussian_filter(x, 1.5) - ndi.gaussian_filter(x, background, mode="nearest")

    a = details(a)[my:-my, mx:-mx]
    b = details(ndi.shift(b, (-dy, -dx), order=1, mode="nearest"))[my:-my, mx:-mx]
    a, b = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0
