"""Calibration automatique de la taille du pixel avec un micromètre objet.

Principe :

1. localiser la zone de l'image qui contient les graduations (forte densité de contours) ;
2. trouver l'orientation des traits (direction dominante des contours), affinée en
   cherchant la projection la plus contrastée ;
3. projeter l'image le long des traits : chaque trait devient un pic d'un profil 1D ;
4. localiser chaque pic au sous-pixel, numéroter les traits de proche en proche et
   ajuster une droite ``position = a + période × numéro``.

Toutes les graduations visibles participent : la précision est bien meilleure que
la mesure à la main d'une seule distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy import signal

from .focus import to_gray


@dataclass
class MicrometerCalibration:
    pixel_size_um: float
    period_px: float
    angle_deg: float  # orientation des traits (0 = verticaux)
    n_lines: int
    residual_px: float  # écart-type des positions après ajustement
    relative_uncertainty: float  # incertitude relative estimée sur la taille du pixel

    @property
    def reliable(self) -> bool:
        """Assez de traits et un ajustement cohérent pour faire confiance au résultat."""
        return self.n_lines >= 5 and self.relative_uncertainty < 0.01

    @property
    def summary(self) -> str:
        return (
            f"{self.pixel_size_um:.4f} µm/pixel ({self.n_lines} traits, période "
            f"{self.period_px:.2f} px, ±{self.relative_uncertainty * 100:.2f} %)"
        )


def _ruler_region(g: np.ndarray, block: int = 16) -> tuple[slice, slice]:
    """Zone de l'image qui contient les graduations.

    Les traits fins produisent beaucoup de contours : on garde la plus grande zone
    connexe de forte « énergie de contour ». Indispensable quand la mire n'occupe
    qu'une petite bande d'une grande image (caméra 4K à faible grossissement), sinon
    les traits sont noyés dans le reste de l'image.
    """
    h, w = g.shape
    hb, wb = (h - 1) // block, (w - 1) // block
    if hb < 4 or wb < 4:
        return slice(0, h), slice(0, w)
    gx = np.diff(g[: hb * block + 1, : wb * block + 1], axis=1)[:-1]
    gy = np.diff(g[: hb * block + 1, : wb * block + 1], axis=0)[:, :-1]
    energy = (gx * gx + gy * gy).reshape(hb, block, wb, block).mean(axis=(1, 3))
    energy = ndi.gaussian_filter(energy, 1.0)
    floor = float(np.median(energy))
    top = float(np.percentile(energy, 99.5))
    if top <= 2 * floor:  # contours partout (mire plein champ) ou nulle part
        return slice(0, h), slice(0, w)
    strong = energy > floor + 0.25 * (top - floor)
    labels, n = ndi.label(strong, structure=np.ones((3, 3)))
    if n == 0:
        return slice(0, h), slice(0, w)
    weights = ndi.sum(energy, labels, index=np.arange(1, n + 1))
    rows, cols = ndi.find_objects(labels)[int(np.argmax(weights))]
    r0, r1 = max(0, (rows.start - 1) * block), min(h, (rows.stop + 1) * block)
    c0, c1 = max(0, (cols.start - 1) * block), min(w, (cols.stop + 1) * block)
    return slice(r0, r1), slice(c0, c1)


def _gradient_direction(g: np.ndarray) -> float:
    """Direction dominante des contours (radians, x vers la droite, y vers le bas).

    C'est la direction perpendiculaire aux traits : ils sont nombreux et parallèles,
    donc leurs bords l'emportent sur le reste (ligne de base, chiffres, poussières).
    """
    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    if jxx + jyy <= 0:
        raise ValueError("Image uniforme : aucune graduation visible")
    coherence = math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy)
    if coherence < 0.15:
        raise ValueError("Graduations non reconnues : pas de traits parallèles nets dans l'image")
    return 0.5 * math.atan2(2 * jxy, jxx - jyy)


def _profile(g: np.ndarray, theta: float, row_step: int = 1) -> np.ndarray:
    """Moyenne de l'image le long des traits, en fonction de la position perpendiculaire
    (une valeur par pixel).

    Chaque pixel est réparti entre les deux positions voisines (interpolation linéaire) :
    toutes les lignes contribuent ainsi à chaque position. Avec des classes plus fines,
    chaque classe ne recevrait que certaines lignes, ce qui crée de faux motifs quand
    les traits n'occupent pas toute la hauteur et que l'angle est faible.

    ``row_step`` > 1 : n'utiliser qu'une ligne sur ``row_step`` (estimation rapide).
    """
    h, w = g.shape
    rows = np.arange(0, h, row_step)
    s = np.add.outer(rows * math.sin(theta), np.arange(w) * math.cos(theta)).ravel()
    s -= s.min()
    values = g[::row_step].ravel()
    lower = np.floor(s).astype(np.int64)
    frac = s - lower
    size = int(lower.max()) + 2
    counts = np.bincount(lower, weights=1 - frac, minlength=size) + np.bincount(
        lower + 1, weights=frac, minlength=size
    )
    sums = np.bincount(lower, weights=values * (1 - frac), minlength=size) + np.bincount(
        lower + 1, weights=values * frac, minlength=size
    )
    good = counts > max(2.0, 0.2 * float(np.median(counts[counts > 0])))
    first, last = int(np.argmax(good)), len(good) - int(np.argmax(good[::-1]))
    prof = np.where(good, sums / np.maximum(counts, 1e-9), np.nan)[first:last]
    valid = ~np.isnan(prof)
    return np.interp(np.arange(len(prof)), np.flatnonzero(valid), prof[valid])


def _edge_sharpness(g: np.ndarray, theta: float, row_step: int) -> float:
    """Raideur des bords du profil projeté : maximale quand la projection suit exactement
    les traits (sinon ils s'étalent). Indépendante de la largeur des traits."""
    prof = ndi.gaussian_filter1d(_profile(g, theta, row_step), 0.7)
    return float(np.sum(np.diff(prof) ** 2))


def _refine_direction(g: np.ndarray, theta: float, span_deg: float = 1.5, step_deg: float = 0.25) -> float:
    """Affine l'orientation autour de l'estimation, là où les bords sont les plus raides."""
    row_step = max(1, g.shape[0] // 256)  # quelques centaines de lignes suffisent
    offsets = np.radians(np.arange(-span_deg, span_deg + step_deg / 2, step_deg))
    scores = np.array([_edge_sharpness(g, theta + d, row_step) for d in offsets])
    i = int(np.argmax(scores))
    best = float(offsets[i])
    if 0 < i < len(offsets) - 1:  # sommet de la parabole passant par les trois meilleurs points
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        denom = a - 2 * b + c
        if denom < 0:
            best += 0.5 * (a - c) / denom * math.radians(step_deg)
    return theta + best


def calibrate_micrometer(image: np.ndarray, spacing_um: float = 10.0) -> MicrometerCalibration:
    """Mesure la taille du pixel à partir d'une image de micromètre objet.

    ``spacing_um`` : écart entre deux graduations (10 µm pour un micromètre de
    1 mm divisé en 100).

    Lève ``ValueError`` si moins de trois graduations sont reconnues. Vérifier
    ``reliable`` (au moins 5 traits, incertitude < 1 %) avant d'enregistrer le résultat.
    """
    full = to_gray(image, max_side=None)
    rows, cols = _ruler_region(full)
    g = full[rows, cols]
    theta = _gradient_direction(g)
    theta = _refine_direction(g, theta)
    prof = ndi.gaussian_filter1d(_profile(g, theta), 0.7)

    # traits sombres sur fond clair → creux ; on travaille toujours sur des pics
    if np.mean(prof > prof.mean()) > 0.5:
        prof = -prof
    prof = prof - np.median(prof)
    peaks, props = signal.find_peaks(prof, prominence=0)
    if len(peaks) < 3:
        raise ValueError("Pas assez de graduations détectées : vérifiez la mise au point")
    prom = props["prominences"]
    # les vrais traits dépassent largement le bruit ; les petits traits font au moins un
    # quart de la hauteur des grands (ils sont plus courts, donc moins marqués en projection)
    peaks = peaks[prom > 0.2 * np.sort(prom)[-3]]
    if len(peaks) < 3:
        raise ValueError("Pas assez de graduations détectées : vérifiez la mise au point")

    # position de chaque trait : milieu de sa largeur à mi-hauteur (au sous-pixel). Plus
    # robuste que le sommet pour les traits larges, dont le dessus est plat.
    _, _, left, right = signal.peak_widths(prof, peaks, rel_height=0.5)
    pos_px = (left + right) / 2

    # numérotation de proche en proche, à partir de l'écart médian entre voisins
    # (insensible aux traits manqués et aux faux traits isolés)
    spacing = float(np.median(np.diff(pos_px)))
    n_idx = np.concatenate([[0], np.cumsum(np.round(np.diff(pos_px) / spacing))])
    keep = np.ones(len(pos_px), bool)
    for _ in range(3):  # ajustement robuste : écarter poussières, chiffres...
        coef = np.polyfit(n_idx[keep], pos_px[keep], 1)
        res = pos_px - np.polyval(coef, n_idx)
        new_keep = np.abs(res) < 0.25 * coef[0]
        if new_keep.sum() < 3 or (new_keep == keep).all():
            break
        keep = new_keep
    period = float(coef[0])
    if period <= 0:
        raise ValueError("Graduations non reconnues")
    residual = float(np.std(res[keep]))
    n = int(keep.sum())
    span = float(n_idx[keep].max() - n_idx[keep].min()) or 1.0
    # incertitude sur la pente ≈ résidu / (étendue × racine(n/12))
    rel = residual / (period * span) * math.sqrt(12 / max(n, 1)) if n > 2 else 1.0
    angle = (math.degrees(theta) + 90) % 180 - 90  # traits ⟂ à la direction des contours
    return MicrometerCalibration(spacing_um / period, period, angle, n, residual, rel)
