"""Calibration automatique de la taille du pixel avec un micromètre objet.

Principe : on trouve l'orientation et la période des graduations par transformée
de Fourier, on projette l'image perpendiculairement aux traits, on localise chaque
trait au sous-pixel puis on ajuste une droite position = a + période × numéro.
Toutes les graduations visibles participent : la précision est bien meilleure que
la mesure à la main d'une seule distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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
    def summary(self) -> str:
        return (
            f"{self.pixel_size_um:.4f} µm/pixel ({self.n_lines} traits, période "
            f"{self.period_px:.2f} px, ±{self.relative_uncertainty * 100:.2f} %)"
        )


def _dominant_frequency(g: np.ndarray, min_period: float, max_period: float) -> tuple[float, float]:
    """(période en px, angle du vecteur fréquence en radians, x vers la droite, y vers le bas)."""
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spec = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win)))
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    f = np.hypot(fx, fy)
    valid = (f > 1 / max_period) & (f < 1 / min_period) & (fx >= 0)
    spec = np.where(valid, spec, 0)
    i, j = np.unravel_index(int(np.argmax(spec)), spec.shape)

    # affinage parabolique autour du pic
    def refine(arr: np.ndarray, k: int) -> float:
        if 0 < k < len(arr) - 1:
            a, b, c = arr[k - 1], arr[k], arr[k + 1]
            den = a - 2 * b + c
            return k + (0.5 * (a - c) / den if den != 0 else 0.0)
        return float(k)

    ii = refine(spec[:, j], i)
    jj = refine(spec[i, :], j)
    fyv = (ii - h // 2) / h
    fxv = (jj - w // 2) / w
    return 1 / math.hypot(fxv, fyv), math.atan2(fyv, fxv)


def calibrate_micrometer(
    image: np.ndarray,
    spacing_um: float = 10.0,
    *,
    min_period_px: float = 4.0,
) -> MicrometerCalibration:
    """Mesure la taille du pixel à partir d'une image de micromètre objet.

    ``spacing_um`` : écart entre deux graduations (10 µm pour un micromètre de
    1 mm divisé en 100).
    """
    g = to_gray(image, max_side=None)
    h, w = g.shape
    period0, theta = _dominant_frequency(g, min_period_px, min(h, w) / 3)

    # projection perpendiculaire aux traits (le long du vecteur fréquence)
    yy, xx = np.mgrid[0:h, 0:w]
    s = xx * math.cos(theta) + yy * math.sin(theta)
    s -= s.min()
    bin_px = 0.25
    idx = (s / bin_px).astype(np.int64).ravel()
    counts = np.bincount(idx)
    sums = np.bincount(idx, weights=g.ravel())
    good = counts > max(5, 0.2 * np.median(counts[counts > 0]))
    prof = np.where(good, sums / np.maximum(counts, 1), np.nan)
    first, last = np.argmax(good), len(good) - np.argmax(good[::-1])
    prof = prof[first:last]
    prof = np.interp(np.arange(len(prof)), np.flatnonzero(~np.isnan(prof)), prof[~np.isnan(prof)])
    # traits sombres sur fond clair → les traits sont des creux
    if np.mean(prof > prof.mean()) > 0.5:
        prof = -prof
    prof = prof - np.median(prof)
    dist = max(1, int(0.6 * period0 / bin_px))
    peaks, props = signal.find_peaks(prof, distance=dist, prominence=0)
    if len(peaks) < 3:
        raise ValueError("Pas assez de graduations détectées : vérifiez la mise au point")
    prom = props["prominences"]
    peaks = peaks[prom > 0.3 * np.median(prom)]
    # position sous-pixel (parabole)
    pos = []
    for p in peaks:
        if 0 < p < len(prof) - 1:
            a, b, c = prof[p - 1], prof[p], prof[p + 1]
            den = a - 2 * b + c
            pos.append(p + (0.5 * (a - c) / den if den != 0 else 0.0))
        else:
            pos.append(float(p))
    pos_px = (np.asarray(pos) + first) * bin_px
    # numérotation des traits puis ajustement robuste
    n_idx = np.round((pos_px - pos_px[0]) / period0)
    keep = np.ones(len(pos_px), bool)
    for _ in range(3):
        coef = np.polyfit(n_idx[keep], pos_px[keep], 1)
        res = pos_px - np.polyval(coef, n_idx)
        new_keep = np.abs(res) < 0.25 * coef[0]
        if new_keep.sum() < 3 or (new_keep == keep).all():
            break
        keep = new_keep
        n_idx = np.round((pos_px - coef[1]) / coef[0])
    period = float(coef[0])
    residual = float(np.std(res[keep]))
    n = int(keep.sum())
    span = float(n_idx[keep].max() - n_idx[keep].min()) or 1.0
    # incertitude sur la pente ≈ résidu / (étendue × racine(n/12))
    rel = (residual / (period * span) * math.sqrt(12 / max(n, 1))) if n > 2 else 1.0
    angle = (math.degrees(theta) + 90) % 180 - 90  # traits ⟂ au vecteur fréquence
    return MicrometerCalibration(spacing_um / period, period, angle, n, residual, rel)
