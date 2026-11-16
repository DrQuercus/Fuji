"""Assistant d'alignement de Köhler (lumière transmise).

Le logiciel ne tourne pas les molettes : il *regarde* l'image du diaphragme de
champ avec la caméra et guide l'utilisateur pas à pas.

1. **Fermer** le diaphragme de champ jusqu'à voir ses bords.
2. **Netteté** : monter/descendre le condenseur jusqu'à des bords nets
   (largeur du bord mesurée en direct, avec mémoire du meilleur réglage).
3. **Centrage** : vis de centrage du condenseur (flèche vers le centre).
4. **Ouvrir** le diaphragme juste au-delà du champ.
5. **Diaphragme d'ouverture** : réglé à 70–80 % de l'ouverture numérique de
   l'objectif (ne se voit pas à la caméra sans lentille de Bertrand : étape guidée).

On peut aussi vérifier l'**uniformité de l'éclairage** sur une zone vide de la lame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure

from .focus import subsample_step, to_gray


class KohlerStep(str, Enum):
    CLOSE_FIELD = "fermer"
    FOCUS_CONDENSER = "netteté"
    CENTER_CONDENSER = "centrage"
    OPEN_FIELD = "ouvrir"
    SET_APERTURE = "ouverture"
    DONE = "terminé"


STEP_TITLES = {
    KohlerStep.CLOSE_FIELD: "1. Fermer le diaphragme de champ",
    KohlerStep.FOCUS_CONDENSER: "2. Mettre au point le condenseur",
    KohlerStep.CENTER_CONDENSER: "3. Centrer le condenseur",
    KohlerStep.OPEN_FIELD: "4. Ouvrir le diaphragme de champ",
    KohlerStep.SET_APERTURE: "5. Régler le diaphragme d'ouverture",
    KohlerStep.DONE: "Köhler terminé",
}


@dataclass
class FieldDiaphragmReport:
    visible: bool
    fully_inside: bool
    coverage: float  # fraction de l'image éclairée
    center: tuple[float, float] | None  # (ligne, colonne), pleine résolution
    radius_px: float | None
    offset_px: tuple[float, float] | None  # (dy, dx) centre du diaphragme − centre image
    offset_fraction: float | None  # |décalage| / demi-côté le plus petit
    edge_width_px: float | None  # largeur du bord 10–90 % (petit = net)
    contrast: float | None
    contour: np.ndarray | None = None  # (N, 2) lignes/colonnes, pour l'affichage


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Cercle des moindres carrés (méthode de Kåsa) : (yc, xc, r)."""
    y, x = points[:, 0], points[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    xc, yc = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + xc * xc + yc * yc, 0.0))
    return yc, xc, r


def analyze_field_diaphragm(image: np.ndarray, *, max_side: int = 768) -> FieldDiaphragmReport:
    """Détecte l'image du diaphragme de champ et mesure netteté, centrage, ouverture.

    L'analyse se fait sur une image réduite ; les résultats sont rendus en pixels de
    l'image d'origine.
    """
    full_h, full_w = image.shape[:2]
    g = to_gray(image, max_side)
    scale = float(subsample_step(image.shape, max_side))
    h, w = g.shape
    smooth = ndi.gaussian_filter(g, 1.0)
    try:
        t = filters.threshold_otsu(smooth)
    except ValueError:
        t = float(smooth.mean())
    bright = smooth > t
    lab, n = ndi.label(bright)
    if n == 0:
        return FieldDiaphragmReport(False, False, 0.0, None, None, None, None, None, None)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    region = ndi.binary_fill_holes(lab == int(np.argmax(sizes)))
    coverage = float(region.mean())
    border = np.zeros_like(region)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touches = bool((region & border).any())
    inside_vals = g[ndi.binary_erosion(region, iterations=4)]
    outside_vals = g[~ndi.binary_dilation(region, iterations=4)]
    if inside_vals.size < 50 or outside_vals.size < 50:
        # pas de zone sombre : diaphragme ouvert au-delà du champ
        return FieldDiaphragmReport(False, False, coverage, None, None, None, None, None, None)
    inside_med, outside_med = float(np.median(inside_vals)), float(np.median(outside_vals))
    contrast = inside_med - outside_med
    # un diaphragme fermé laisse l'extérieur sombre ; un simple éclairage inégal ne
    # descend pas sous la moitié de l'intensité du centre
    if outside_med > 0.5 * inside_med or coverage > 0.97:
        return FieldDiaphragmReport(False, False, coverage, None, None, None, None, contrast, None)

    # contour du diaphragme (hors bords de l'image) → cercle ajusté
    contours = measure.find_contours(region.astype(np.float32), 0.5)
    contour = max(contours, key=len)
    margin = 3
    keep = (
        (contour[:, 0] > margin)
        & (contour[:, 0] < h - 1 - margin)
        & (contour[:, 1] > margin)
        & (contour[:, 1] < w - 1 - margin)
    )
    pts = contour[keep]
    if len(pts) < 20:
        return FieldDiaphragmReport(False, False, coverage, None, None, None, None, contrast, None)
    yc, xc, r = _fit_circle(pts)

    # largeur du bord : pente maximale du profil à travers le bord
    gy, gx = np.gradient(ndi.gaussian_filter(g, 0.7))
    grad = np.hypot(gx, gy)
    band = ndi.binary_dilation(region, iterations=2) & ~ndi.binary_erosion(region, iterations=2)
    band &= ~ndi.binary_dilation(border, iterations=margin)
    gmax = float(np.percentile(grad[band], 90)) if band.any() else 0.0
    edge_width = None
    if gmax > 0:
        sigma = contrast / (gmax * math.sqrt(2 * math.pi))
        sigma = math.sqrt(max(sigma**2 - 0.7**2, 0.25))  # retirer le lissage appliqué
        edge_width = 2.563 * sigma * scale  # 10–90 % d'un bord gaussien

    center = (yc * scale, xc * scale)
    offset = (center[0] - full_h / 2, center[1] - full_w / 2)
    offset_frac = math.hypot(*offset) / (min(full_h, full_w) / 2)
    step = max(1, len(pts) // 200)
    return FieldDiaphragmReport(
        visible=True,
        fully_inside=not touches,
        coverage=coverage,
        center=center,
        radius_px=r * scale,
        offset_px=offset,
        offset_fraction=offset_frac,
        edge_width_px=edge_width,
        contrast=contrast,
        contour=pts[::step] * scale,
    )


def recommended_aperture(objective_na: float, condenser_na: float | None = None) -> tuple[float, float]:
    """Ouverture du condenseur conseillée : 70 à 80 % de l'ON de l'objectif."""
    lo, hi = 0.7 * objective_na, 0.8 * objective_na
    if condenser_na:
        lo, hi = min(lo, condenser_na), min(hi, condenser_na)
    return lo, hi


def contrast_index(image: np.ndarray) -> float:
    """Contraste global (écart-type / moyenne) : aide à juger le diaphragme d'ouverture."""
    g = to_gray(image)
    return float(g.std() / (g.mean() or 1.0))


# --------------------------------------------------------------- uniformité
@dataclass
class UniformityReport:
    map: np.ndarray  # intensité relative par bloc (1 = maximum)
    min_max_ratio: float
    corner_center_ratio: float
    tilt_percent: float  # variation d'un bord à l'autre due à une pente
    tilt_direction_deg: float  # direction vers laquelle l'éclairage baisse
    message: str


def illumination_uniformity(image: np.ndarray, grid: tuple[int, int] = (9, 12)) -> UniformityReport:
    """Mesure l'uniformité de l'éclairage sur une lame vide (ou une zone vide)."""
    g = to_gray(image, 1024)
    h, w = g.shape
    gh, gw = grid
    blocks = np.empty(grid, np.float64)
    for i in range(gh):
        for j in range(gw):
            blk = g[i * h // gh : (i + 1) * h // gh, j * w // gw : (j + 1) * w // gw]
            blocks[i, j] = np.median(blk)
    rel = blocks / blocks.max()
    corners = np.mean([rel[0, 0], rel[0, -1], rel[-1, 0], rel[-1, -1]])
    center = rel[gh // 2 - (gh % 2 == 0) : gh // 2 + 1, gw // 2 - (gw % 2 == 0) : gw // 2 + 1].mean()
    # pente : plan ajusté (y vers le bas)
    yy, xx = np.mgrid[0:gh, 0:gw]
    yy = (yy - (gh - 1) / 2) / ((gh - 1) / 2)
    xx = (xx - (gw - 1) / 2) / ((gw - 1) / 2)
    a = np.column_stack([xx.ravel(), yy.ravel(), np.ones(gh * gw)])
    (cx, cy, c0), *_ = np.linalg.lstsq(a, rel.ravel(), rcond=None)
    tilt = 2 * math.hypot(cx, cy) / max(c0, 1e-6) * 100
    direction = math.degrees(math.atan2(cy, -cx)) % 360  # vers où ça baisse (0° = droite)
    ratio = float(rel.min())
    msgs = []
    if ratio > 0.9:
        msgs.append("Éclairage homogène.")
    if tilt > 8:
        side = _direction_name(direction)
        msgs.append(
            f"L'éclairage baisse vers {side} ({tilt:.0f} %) : centrez la lampe "
            "(vis de centrage du collecteur) ou vérifiez le centrage du condenseur."
        )
    if corners / max(center, 1e-6) < 0.8:
        msgs.append(
            "Coins nettement plus sombres (vignettage) : ouvrez le diaphragme de champ "
            "juste au-delà du champ, ou utilisez la correction de champ plat."
        )
    if not msgs:
        msgs.append("Éclairage correct ; une correction de champ plat améliorera les mesures.")
    return UniformityReport(
        rel, ratio, float(corners / max(center, 1e-6)), float(tilt), float(direction), " ".join(msgs)
    )


def _direction_name(deg: float) -> str:
    names = [
        "la droite",
        "le haut à droite",
        "le haut",
        "le haut à gauche",
        "la gauche",
        "le bas à gauche",
        "le bas",
        "le bas à droite",
    ]
    return names[int(((deg + 22.5) % 360) // 45)]


# ----------------------------------------------------------- assistant guidé
@dataclass
class KohlerGuidance:
    step: KohlerStep
    title: str
    message: str
    report: FieldDiaphragmReport | None
    #: Flèche (dy, dx) normalisée : sens dans lequel déplacer l'image du diaphragme.
    arrow: tuple[float, float] | None = None
    #: Netteté du bord : actuelle / meilleure (en largeur de bord, px).
    edge_width_px: float | None = None
    best_edge_width_px: float | None = None
    can_advance: bool = False
    extra: dict[str, float] = field(default_factory=dict)


class KohlerAssistant:
    """Assistant pas à pas (avec mémoire), à alimenter avec les images du live."""

    def __init__(
        self,
        objective_na: float | None = None,
        condenser_na: float | None = None,
        *,
        center_tolerance: float = 0.04,
        stable_frames: int = 4,
    ) -> None:
        self.objective_na = objective_na
        self.condenser_na = condenser_na
        self.center_tolerance = center_tolerance
        self.stable_frames = stable_frames
        self.reset()

    def reset(self) -> None:
        self.step = KohlerStep.CLOSE_FIELD
        self.best_edge: float | None = None
        self._ok_count = 0

    def advance(self) -> None:
        order = list(KohlerStep)
        i = order.index(self.step)
        if i < len(order) - 1:
            self.step = order[i + 1]
            self._ok_count = 0

    def back(self) -> None:
        order = list(KohlerStep)
        i = order.index(self.step)
        if i > 0:
            self.step = order[i - 1]
            self._ok_count = 0

    def _stable(self, ok: bool) -> bool:
        self._ok_count = self._ok_count + 1 if ok else 0
        return self._ok_count >= self.stable_frames

    def update(self, image: np.ndarray) -> KohlerGuidance:
        rep = analyze_field_diaphragm(image)
        # un bord « net » fait quelques pixels de l'image réduite : seuil proportionnel à
        # la taille de l'image, sinon il serait inatteignable avec une caméra 4K
        sharp_enough = max(12.0, 0.015 * min(image.shape[:2]))
        step = self.step
        title = STEP_TITLES[step]
        g = KohlerGuidance(step, title, "", rep)

        if step is KohlerStep.CLOSE_FIELD:
            if rep.visible and rep.fully_inside and rep.coverage < 0.6:
                g.message = "Bords du diaphragme visibles."
                g.can_advance = True
                if self._stable(True):
                    self.advance()
            elif rep.visible:
                self._stable(False)
                g.message = (
                    "Bords du diaphragme visibles : fermez-le encore un peu pour voir le "
                    "polygone lumineux en entier, entouré de noir."
                )
            else:
                self._stable(False)
                g.message = (
                    "Fermez le diaphragme de champ (bague à la base du microscope) "
                    "jusqu'à voir un polygone lumineux entouré de noir. "
                    "S'il reste invisible, montez le condenseur au maximum."
                )
        elif step is KohlerStep.FOCUS_CONDENSER:
            w = rep.edge_width_px if rep.visible else None
            g.edge_width_px = w
            if w is not None and (self.best_edge is None or w < self.best_edge):
                self.best_edge = w
            g.best_edge_width_px = self.best_edge
            if w is None:
                g.message = "Diaphragme perdu de vue : refermez-le un peu."
            else:
                near_best = self.best_edge is not None and w <= self.best_edge * 1.1
                g.message = (
                    "Montez ou descendez le condenseur pour rendre les bords du polygone les "
                    f"plus nets possible. Bord : {w:.1f} px (meilleur : {self.best_edge:.1f} px). "
                    "Dépassez le point net puis revenez-y."
                )
                g.can_advance = near_best and w < sharp_enough
        elif step is KohlerStep.CENTER_CONDENSER:
            if not rep.visible or rep.offset_px is None or rep.offset_fraction is None:
                self._stable(False)
                g.message = "Diaphragme hors du champ : refermez-le un peu pour le voir entier."
            else:
                dy, dx = rep.offset_px
                norm = math.hypot(dy, dx) or 1.0
                g.arrow = (-dy / norm, -dx / norm)
                ok = rep.offset_fraction <= self.center_tolerance
                g.message = (
                    "Centré."
                    if ok
                    else "Tournez les vis de centrage du condenseur pour amener le polygone au "
                    f"centre (décalage : {rep.offset_fraction * 100:.0f} % du champ)."
                )
                g.can_advance = ok
                if self._stable(ok):
                    self.advance()
        elif step is KohlerStep.OPEN_FIELD:
            done = (not rep.visible) or rep.coverage > 0.95
            g.message = (
                "Ouvrez le diaphragme de champ jusqu'à ce que ses bords sortent juste du champ "
                "(aux oculaires : juste au-delà du champ visible ; la caméra voit un champ plus "
                "petit). Ne l'ouvrez pas davantage : cela dégrade le contraste."
            )
            g.can_advance = done
            if self._stable(done):
                self.advance()
        elif step is KohlerStep.SET_APERTURE:
            ci = contrast_index(image)
            g.extra["contrast_index"] = ci
            if self.objective_na:
                lo, hi = recommended_aperture(self.objective_na, self.condenser_na)
                g.message = (
                    f"Réglez le diaphragme d'ouverture du condenseur sur ≈ {lo:.2f}–{hi:.2f} "
                    f"(70–80 % de l'ON {self.objective_na:.2f} de l'objectif). Sans graduation : "
                    "retirez un oculaire, regardez dans le tube et fermez jusqu'à masquer "
                    "20 à 30 % de la pupille. Plus fermé = plus de contraste mais moins de "
                    "résolution."
                )
            else:
                g.message = (
                    "Réglez le diaphragme d'ouverture à 70–80 % de l'ouverture "
                    "numérique de l'objectif (sélectionnez l'objectif pour une valeur)."
                )
            g.can_advance = True
        else:
            g.message = (
                "Éclairage de Köhler réglé. À refaire à chaque changement d'objectif "
                "(au moins le diaphragme d'ouverture)."
            )
        return g
