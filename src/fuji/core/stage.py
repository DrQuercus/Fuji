"""Outils pour une platine motorisée : orientation, centrage, jeu mécanique, mosaïques.

Une platine et une caméra n'ont aucune raison d'avoir les mêmes axes : la caméra peut être
tournée, un axe inversé, et « +X » de la platine déplacer l'image vers la gauche. Tout ce
qui relie l'image à la platine (cliquer pour centrer, assembler une mosaïque) passe donc
par une **matrice d'orientation**, mesurée une fois par :func:`calibrate_stage` :

    [dX, dY] platine (µm) = orientation · [dx, dy] image (µm)

où (dx, dy) est un déplacement dans l'image (x vers la droite, y vers le bas) et
(dX, dY) le déplacement de platine qui amène ce point au centre.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fuji.analysis.registration import MIN_CONFIDENCE, image_shift, shift_confidence

if TYPE_CHECKING:
    from .microscope import Microscope


@dataclass
class StageCalibration:
    """Résultat de la calibration platine ↔ image."""

    #: [dX, dY] platine (µm) = matrice · [dx, dy] image (pixels)
    matrix_um_per_px: np.ndarray
    #: taille du pixel d'après la platine (µm), indépendante de l'objectif déclaré
    pixel_size_um: float
    #: angle entre l'axe X de l'image et l'axe X de la platine (degrés)
    rotation_deg: float
    #: vrai si l'image est en miroir par rapport à la platine (un axe inversé)
    mirrored: bool
    #: taille du pixel connue par ailleurs (objectif calibré), pour comparaison
    expected_pixel_size_um: float | None = None
    #: défaut d'orthogonalité : 0 si les deux axes mesurés sont bien perpendiculaires
    skew: float = 0.0

    @property
    def orientation(self) -> np.ndarray:
        """Matrice sans échelle (rotation et miroir), à enregistrer dans le profil."""
        return self.matrix_um_per_px / self.pixel_size_um

    @property
    def scale_error(self) -> float | None:
        """Écart relatif entre le pixel mesuré par la platine et le pixel calibré."""
        if not self.expected_pixel_size_um:
            return None
        return self.pixel_size_um / self.expected_pixel_size_um - 1.0

    @property
    def summary(self) -> str:
        mirror = ", image en miroir" if self.mirrored else ""
        text = (
            f"caméra tournée de {self.rotation_deg:+.1f}° par rapport à la platine{mirror} ; "
            f"{self.pixel_size_um:.4f} µm/pixel d'après la platine"
        )
        if self.scale_error is not None:
            text += f" ({self.scale_error * 100:+.1f} % par rapport au pixel calibré)"
        return text


def _require_stage(microscope: Microscope):  # type: ignore[no-untyped-def]
    stage = microscope.xy_stage
    if stage is None or stage.manual:
        raise RuntimeError("Cette fonction demande une platine XY motorisée")
    return stage


_NOT_FOLLOWING = (
    "Mesure incohérente : ce que l'on voit ne suit pas la platine (le même pas dans les deux sens ne "
    "déplace pas l'image d'autant). Lame vide, poussières sur le capteur, ou platine qui ne bouge pas ? "
    "Placez une lame bien contrastée et recommencez."
)
_NO_DETAIL = (
    "Mesure impossible : l'image ne contient pas assez de détails nets pour suivre son déplacement. "
    "Placez une lame bien contrastée, faites la mise au point et vérifiez l'éclairage et l'exposition."
)


def calibrate_stage(microscope: Microscope, step_um: float | None = None) -> StageCalibration:
    """Mesure l'orientation de la platine par rapport à l'image.

    La platine fait un petit pas dans chaque sens, en X puis en Y ; le déplacement de l'image
    est mesuré à chaque fois. À faire sur une lame avec des détails nets dans tout le champ.
    La platine revient à sa position de départ. ``step_um`` : taille du pas (par défaut
    environ 15 % du champ, trouvé automatiquement si le pixel n'est pas calibré).

    La mesure est refusée plutôt que fausse : image sans détails, ou image qui ne suit pas la
    platine (poussières sur le capteur, éclairage : ils restent en place quand la lame bouge).
    """
    stage = _require_stage(microscope)
    x0, y0 = stage.get_position()
    h, w = microscope.snap().data.shape[:2]
    px = microscope.pixel_size_um()
    target_px = 0.15 * min(h, w)
    step = float(step_um) if step_um else (target_px * px if px else 5.0)

    def measure(dx: float, dy: float) -> tuple[np.ndarray, float]:
        """Déplacement de l'image pour un pas de la platine, et confiance dans cette mesure.

        Le pas est mesuré « dans le sens de la marche » : la platine recule d'abord d'autant et
        revient, ce qui rattrape le jeu mécanique avant la mesure."""
        with contextlib.suppress(ValueError):  # butée logicielle de ce côté : pas d'élan
            stage.move_to(x0 - dx, y0 - dy)
        stage.move_to(x0, y0)
        reference = microscope.snap().data
        stage.move_to(x0 + dx, y0 + dy)
        moved = microscope.snap().data
        shift = image_shift(reference, moved)
        return np.array(shift), shift_confidence(reference, moved, shift)

    def both_ways(dx: float, dy: float) -> np.ndarray:
        """Même pas dans les deux sens : le contenu de la lame doit partir d'autant de chaque
        côté. Ce qui ne suit pas la platine donne le même « déplacement » des deux côtés."""
        results = []
        for sign in (1.0, -1.0):
            try:
                shift, sure = measure(sign * dx, sign * dy)
            except ValueError:  # hors de la course autorisée de ce côté
                continue
            if sure < MIN_CONFIDENCE:
                raise RuntimeError(_NO_DETAIL)
            results.append(sign * shift)
        if not results:
            raise ValueError("La platine est en butée : impossible de faire le pas de mesure")
        if len(results) == 2:
            forward, backward = results
            if np.linalg.norm(forward - backward) > 0.35 * np.linalg.norm(forward + backward) + 1.0:
                raise RuntimeError(_NOT_FOLLOWING)
        return np.mean(results, axis=0)

    try:
        if not step_um:  # pas trop petit pour être mesuré, ou trop grand pour se recouvrir ?
            for _ in range(6):
                try:
                    shift, sure = measure(step, 0.0)
                except ValueError:  # butée de ce côté : on tâte de l'autre
                    shift, sure = measure(-step, 0.0)
                size, good = float(np.linalg.norm(shift)), sure >= MIN_CONFIDENCE
                if good and 0.05 * min(h, w) <= size <= 0.3 * min(h, w):
                    break
                # rien de reconnu : le pas a peut-être sorti tout le champ, on essaie plus petit
                step *= float(np.clip(target_px / max(size, 0.5), 0.2, 5.0)) if good else 0.25
        ax = both_ways(step, 0.0)
        ay = both_ways(0.0, step)
    finally:
        stage.move_to(x0, y0)
    if np.linalg.norm(ax) < 1.0 or np.linalg.norm(ay) < 1.0:
        raise RuntimeError(
            "L'image n'a pas bougé : la platine se déplace-t-elle ? La lame a-t-elle des détails nets ?"
        )
    # le contenu se déplace à l'opposé du champ : déplacement du champ = − déplacement du contenu
    field = -np.array([[ax[0], ay[0]], [ax[1], ay[1]]])  # colonnes : effet d'un pas en X, en Y
    if abs(np.linalg.det(field)) < 1e-6:
        raise RuntimeError("Les deux axes de la platine déplacent l'image dans la même direction")
    matrix = step * np.linalg.inv(field)
    scale = math.sqrt(abs(np.linalg.det(matrix)))
    cx, cy = matrix[:, 0], matrix[:, 1]
    skew = abs(float(cx @ cy)) / (float(np.linalg.norm(cx)) * float(np.linalg.norm(cy)))
    if skew > 0.5:  # axes à moins de 60° l'un de l'autre : ce n'est pas une platine, c'est une erreur
        raise RuntimeError(
            "Les deux axes de la platine semblent presque parallèles : mesure incohérente. "
            "Recommencez sur une zone riche en détails, et vérifiez le câblage des axes."
        )
    return StageCalibration(
        matrix_um_per_px=matrix,
        pixel_size_um=scale,
        rotation_deg=math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])),
        mirrored=bool(np.linalg.det(matrix) < 0),
        expected_pixel_size_um=px,
        skew=skew,
    )


def stage_matrix(microscope: Microscope) -> np.ndarray:
    """Matrice [dX, dY] platine (µm) = M · [dx, dy] image (pixels), pour l'objectif en place.

    Utilise l'orientation enregistrée dans le profil (identité si elle n'a pas été mesurée).
    """
    px = microscope.pixel_size_um()
    if not px:
        raise RuntimeError(
            "La taille du pixel est inconnue : sélectionnez l'objectif en place ou calibrez-le"
        )
    orientation = microscope.profile.stage_orientation
    return np.asarray(orientation if orientation else np.eye(2), dtype=np.float64) * px


def save_stage_calibration(microscope: Microscope, calibration: StageCalibration) -> None:
    """Enregistre l'orientation mesurée dans le profil du microscope (en mémoire)."""
    microscope.profile.stage_orientation = [[float(v) for v in row] for row in calibration.orientation]
    microscope.profile_changed.emit()


def offset_to_stage(microscope: Microscope, dx_px: float, dy_px: float) -> tuple[float, float]:
    """Déplacement de platine (µm) qui amène au centre un point situé à (dx, dy) pixels du
    centre de l'image."""
    delta = stage_matrix(microscope) @ np.array([dx_px, dy_px], dtype=np.float64)
    return float(delta[0]), float(delta[1])


def center_on_pixel(
    microscope: Microscope, row: float, col: float, shape: tuple[int, ...]
) -> tuple[float, float]:
    """Déplace la platine pour amener le pixel (ligne, colonne) au centre de l'image.

    Renvoie la nouvelle position de la platine."""
    stage = _require_stage(microscope)
    h, w = shape[:2]
    dx, dy = offset_to_stage(microscope, col - (w - 1) / 2.0, row - (h - 1) / 2.0)
    x, y = stage.get_position()
    stage.move_to(x + dx, y + dy)
    return stage.get_position()


def measure_backlash(microscope: Microscope, axis: str = "x", step_um: float | None = None) -> float:
    """Jeu mécanique d'un axe de la platine (µm) : écart entre la même position atteinte en
    venant d'un côté puis de l'autre.

    Si le pilote rattrape déjà le jeu, c'est le jeu résiduel qui est mesuré (proche de zéro).
    La valeur trouvée se reporte dans l'option « jeu » du pilote des moteurs. Le jeu doit
    rester petit devant le champ (moins d'un tiers) : prendre un objectif faible.
    """
    stage = _require_stage(microscope)
    if axis not in ("x", "y"):
        raise ValueError("axis vaut « x » ou « y »")
    matrix = stage_matrix(microscope)
    h, w = microscope.require_camera().sensor_shape()
    # pas nettement plus grand que le jeu attendu, sinon il n'est pas rattrapé en entier
    step = float(step_um) if step_um else 0.25 * min(h, w) * float(microscope.pixel_size_um() or 1.0)
    unit = np.array([1.0, 0.0]) if axis == "x" else np.array([0.0, 1.0])
    x0, y0 = stage.get_position()
    try:
        stage.move_to(*(np.array([x0, y0]) - step * unit))
        stage.move_to(x0, y0)  # arrivée par le côté négatif
        from_below = microscope.snap().data
        stage.move_to(*(np.array([x0, y0]) + step * unit))
        displaced = microscope.snap().data
        stage.move_to(x0, y0)  # arrivée par le côté positif
        from_above = microscope.snap().data
    finally:
        stage.move_to(x0, y0)
    # garde-fou : l'image suit-elle la platine ? (sinon un champ vide donnerait « jeu nul »)
    travel = image_shift(from_below, displaced)
    if shift_confidence(from_below, displaced, travel) < MIN_CONFIDENCE:
        raise RuntimeError(_NO_DETAIL)
    expected = float(np.linalg.norm(np.linalg.solve(matrix, step * unit)))
    if not 0.4 * expected <= math.hypot(*travel) <= 2.5 * expected:
        raise RuntimeError(_NOT_FOLLOWING)
    shift = image_shift(from_below, from_above)
    if shift_confidence(from_below, from_above, shift) < MIN_CONFIDENCE:
        raise RuntimeError(_NO_DETAIL)
    return abs(float((matrix @ -np.array(shift)) @ unit))


# -------------------------------------------------------------------------- mosaïques
@dataclass
class MosaicPlan:
    """Positions d'une mosaïque et place de chaque tuile dans l'image assemblée."""

    #: positions de la platine (µm), dans l'ordre d'acquisition (en serpentin)
    positions: list[tuple[float, float]]
    #: coin haut-gauche de chaque tuile dans la mosaïque, en pixels (ligne, colonne)
    offsets_px: list[tuple[float, float]]
    columns: int
    rows: int
    tile_shape: tuple[int, int]
    overlap: float


def plan_mosaic(
    microscope: Microscope,
    columns: int,
    rows: int,
    overlap: float = 0.1,
    center: tuple[float, float] | None = None,
) -> MosaicPlan:
    """Positions d'une mosaïque de ``columns`` × ``rows`` champs, centrée sur la position
    actuelle (ou ``center``). ``overlap`` : recouvrement entre tuiles voisines (0,1 = 10 %),
    nécessaire pour les raccorder proprement."""
    stage = _require_stage(microscope)
    if columns < 1 or rows < 1:
        raise ValueError("Il faut au moins une tuile")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("Le recouvrement doit être entre 0 et 0,9")
    matrix = stage_matrix(microscope)
    h, w = microscope.require_camera().sensor_shape()
    cx, cy = center if center is not None else stage.get_position()
    step_c, step_r = w * (1.0 - overlap), h * (1.0 - overlap)
    positions: list[tuple[float, float]] = []
    offsets: list[tuple[float, float]] = []
    for r in range(rows):
        order = range(columns) if r % 2 == 0 else range(columns - 1, -1, -1)  # serpentin
        for c in order:
            dx_px = (c - (columns - 1) / 2.0) * step_c
            dy_px = (r - (rows - 1) / 2.0) * step_r
            delta = matrix @ np.array([dx_px, dy_px])
            positions.append((float(cx + delta[0]), float(cy + delta[1])))
            offsets.append((r * step_r, c * step_c))
    limits = stage.limits()
    if limits is not None:
        (xlo, xhi), (ylo, yhi) = limits
        if any(not (xlo <= x <= xhi and ylo <= y <= yhi) for x, y in positions):
            raise ValueError("La mosaïque dépasse la course de la platine : réduisez-la ou recentrez-vous")
    return MosaicPlan(positions, offsets, columns, rows, (h, w), overlap)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _displacement(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[int, int, float]:
    """Décalage entier (lignes, colonnes) du contenu de ``b`` par rapport à ``a`` qui donne la
    meilleure corrélation normalisée, et cette corrélation. Recherche exhaustive, d'abord sur
    des images réduites puis en pleine résolution autour du meilleur candidat."""
    from scipy import ndimage as ndi

    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.ndim == 3:
        a, b = a[..., :3].mean(axis=-1), b[..., :3].mean(axis=-1)
    # l'éclairage est fixe dans l'image : sans le retirer, il « colle » les tuiles à décalage nul
    sigma = max(3.0, min(a.shape) / 8.0)
    a = a - ndi.gaussian_filter(a, sigma, mode="nearest")
    b = b - ndi.gaussian_filter(b, sigma, mode="nearest")

    def search(x: np.ndarray, y: np.ndarray, centre: tuple[int, int], radius: int) -> tuple[int, int, float]:
        best = (centre[0], centre[1], -2.0)
        h, w = x.shape
        for dy in range(centre[0] - radius, centre[0] + radius + 1):
            for dx in range(centre[1] - radius, centre[1] + radius + 1):
                ya, yb = max(0, -dy), min(h, h - dy)
                xa, xb = max(0, -dx), min(w, w - dx)
                if yb - ya < 8 or xb - xa < 8:
                    continue
                score = _ncc(x[ya:yb, xa:xb], y[ya + dy : yb + dy, xa + dx : xb + dx])
                if score > best[2]:
                    best = (dy, dx, score)
        return best

    step = max(1, int(math.ceil(math.sqrt(a.size / 40_000.0))))
    if step > 1:
        dy, dx, _ = search(a[::step, ::step], b[::step, ::step], (0, 0), int(math.ceil(max_shift / step)))
        h, w = a.shape  # affinage sur une fenêtre centrale en pleine résolution
        ch, cw = min(h, 200), min(w, 200)
        r0, c0 = (h - ch) // 2, (w - cw) // 2
        pad = max_shift + step
        ra, rb = max(0, r0 - pad), min(h, r0 + ch + pad)
        ca, cb = max(0, c0 - pad), min(w, c0 + cw + pad)
        return search(a[ra:rb, ca:cb], b[ra:rb, ca:cb], (dy * step, dx * step), step)
    return search(a, b, (0, 0), max_shift)


def refine_offsets(
    tiles: list[np.ndarray], offsets_px: list[tuple[float, float]], max_correction: int | None = None
) -> list[tuple[float, float]]:
    """Corrige la place des tuiles d'après leur contenu commun (une platine n'est jamais
    parfaite : jeu, axes pas tout à fait perpendiculaires, pas imprécis).

    Pour chaque paire de tuiles qui se recouvrent, l'écart réel est mesuré par corrélation
    dans la zone commune ; les mesures peu sûres (zone vide, corrélation faible) sont
    écartées. Toutes les positions sont ensuite ajustées ensemble, pour que l'erreur ne
    s'accumule pas d'une tuile à la suivante. Sans mesure fiable, une tuile garde sa
    place théorique.
    """
    n = len(tiles)
    nominal = np.asarray(offsets_px, dtype=np.float64)
    if n < 2:
        return [(float(r), float(c)) for r, c in nominal]
    h, w = tiles[0].shape[:2]
    rows, rhs, weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            dr, dc = (int(v) for v in np.round(nominal[j] - nominal[i]))
            r0, r1 = max(dr, 0), min(h + dr, h)  # zone commune, dans la tuile i
            c0, c1 = max(dc, 0), min(w + dc, w)
            if r1 - r0 < 16 or c1 - c0 < 16 or (r1 - r0) * (c1 - c0) < 0.03 * h * w:
                continue
            part_i = tiles[i][r0:r1, c0:c1]
            part_j = tiles[j][r0 - dr : r1 - dr, c0 - dc : c1 - dc]
            limit = max_correction if max_correction is not None else max(3, int(0.3 * min(r1 - r0, c1 - c0)))
            dy, dx, score = _displacement(part_i, part_j, limit)
            reliable = score > 0.3 and abs(dy) < limit and abs(dx) < limit  # pas sur le bord de la recherche
            measured = nominal[j] - nominal[i] - (np.array([dy, dx]) if reliable else 0.0)
            row = np.zeros(n)
            row[i], row[j] = -1.0, 1.0
            rows.append(row)
            rhs.append(measured)
            weights.append(score if reliable else 0.05)
    # attache faible à la position théorique, et première tuile fixée
    for k in range(n):
        row = np.zeros(n)
        row[k] = 1.0
        rows.append(row)
        rhs.append(nominal[k])
        weights.append(100.0 if k == 0 else 0.01)
    wts = np.sqrt(np.asarray(weights))[:, None]
    solution, *_ = np.linalg.lstsq(np.asarray(rows) * wts, np.asarray(rhs) * wts, rcond=None)
    return [(float(r), float(c)) for r, c in solution]


def stitch(
    tiles: list[np.ndarray], offsets_px: list[tuple[float, float]], *, blend: bool = True
) -> np.ndarray:
    """Assemble des tuiles en une seule image.

    ``offsets_px`` : coin haut-gauche de chaque tuile (ligne, colonne). Dans les zones de
    recouvrement, les tuiles sont fondues progressivement (``blend``) pour effacer les
    raccords ; sinon la dernière tuile posée l'emporte.
    """
    if not tiles or len(tiles) != len(offsets_px):
        raise ValueError("Il faut autant de positions que de tuiles")
    offsets = np.round(np.asarray(offsets_px, dtype=np.float64)).astype(int)
    offsets -= offsets.min(axis=0)
    h, w = tiles[0].shape[:2]
    extra = tiles[0].shape[2:]
    height, width = int(offsets[:, 0].max()) + h, int(offsets[:, 1].max()) + w
    dtype = tiles[0].dtype
    if not blend:
        out = np.zeros((height, width, *extra), dtype)
        for tile, (r, c) in zip(tiles, offsets, strict=True):
            out[r : r + h, c : c + w] = tile
        return out
    # poids : distance au bord de la tuile (une tuile compte moins près de ses bords)
    wy = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1).astype(np.float32)
    wx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1).astype(np.float32)
    weight = np.minimum.outer(wy, wx)
    total = np.zeros((height, width, *extra), np.float32)
    weights = np.zeros((height, width), np.float32)
    for tile, (r, c) in zip(tiles, offsets, strict=True):
        total[r : r + h, c : c + w] += tile.astype(np.float32) * (weight[..., None] if extra else weight)
        weights[r : r + h, c : c + w] += weight
    safe = np.maximum(weights, 1e-6)
    out_f = total / (safe[..., None] if extra else safe)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.floor(out_f + 0.5), info.min, info.max).astype(dtype)
    return out_f.astype(dtype)
