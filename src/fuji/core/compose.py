"""Images composées en direct à partir du live : le microscope peut rester entièrement manuel.

- :class:`FocusStacker` — **profondeur de champ étendue** : on tourne lentement la mise au
  point à travers l'échantillon ; chaque endroit de l'image garde l'état le plus net vu
  jusque-là. Le résultat est net partout, comme le « EDF » des logiciels de caméra.
- :class:`LiveMosaic` — **panorama** : on déplace la lame à la main ; le logiciel suit le
  déplacement, et chaque champ nouveau est ajouté quand la platine s'immobilise (pas de flou
  de bougé). Le résultat final est assemblé en pleine résolution, raccords fondus.

Les deux reçoivent les images du live une par une (méthode ``add``), dans un fil de travail :
une image qui arrive pendant un calcul est simplement sautée.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

#: en dessous de cette corrélation, deux images ne sont pas reconnues comme la même zone
MIN_MATCH = 0.3


# ---------------------------------------------------------------------------- outils
def _gray(frame: np.ndarray) -> np.ndarray:
    """Niveaux de gris en float32 (moyenne des canaux couleur)."""
    if frame.ndim == 3:
        out = frame[..., 0].astype(np.float32)
        for c in range(1, min(frame.shape[-1], 3)):
            out += frame[..., c]
        return out / min(frame.shape[-1], 3)
    return frame.astype(np.float32)


def _block_mean(gray: np.ndarray, factor: int) -> np.ndarray:
    """Réduction par moyenne de blocs ``factor``×``factor`` (le bord incomplet est ignoré)."""
    if factor <= 1:
        return gray
    h, w = (gray.shape[0] // factor) * factor, (gray.shape[1] // factor) * factor
    return gray[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3), dtype=np.float32)


def _small(frame: np.ndarray, factor: int) -> np.ndarray:
    """Petite image grise pour le suivi : sous-échantillonnage, puis léger lissage."""
    g = _gray(frame[::factor, ::factor])
    return ndi.gaussian_filter(g, 0.8)


def _highpass(g: np.ndarray, size: int = 15) -> np.ndarray:
    """Retire l'éclairage (fixe dans l'image) pour ne comparer que les détails."""
    return g - ndi.uniform_filter(g, size, mode="nearest")


def _window(shape: tuple[int, int]) -> np.ndarray:
    return np.outer(np.hanning(shape[0] + 2)[1:-1], np.hanning(shape[1] + 2)[1:-1]).astype(np.float32)


def _content_shift(ref_hp: np.ndarray, mov_hp: np.ndarray, upsample: int = 4) -> tuple[float, float]:
    """Déplacement (lignes, colonnes) du contenu de ``mov`` par rapport à ``ref`` :
    ``mov[y, x] ≈ ref[y - dy, x - dx]``. Corrélation de phase sur images fenêtrées."""
    from skimage.registration import phase_cross_correlation

    win = _window(ref_hp.shape)
    shift, _, _ = phase_cross_correlation(
        ref_hp * win, mov_hp * win, upsample_factor=upsample, normalization=None
    )
    return -float(shift[0]), -float(shift[1])


def _overlap_ncc(a: np.ndarray, b: np.ndarray, dy: float, dx: float) -> float:
    """Corrélation (−1…1) entre ``a`` et ``b`` sur leur partie commune, ``b`` étant ``a``
    déplacée de (dy, dx). Proche de 1 : même zone, bien superposée."""
    iy, ix = int(round(dy)), int(round(dx))
    h, w = a.shape
    if abs(iy) >= h - 8 or abs(ix) >= w - 8:
        return 0.0
    a_part = a[max(0, -iy) : h - max(0, iy), max(0, -ix) : w - max(0, ix)]
    b_part = b[max(0, iy) : h - max(0, -iy), max(0, ix) : w - max(0, -ix)]
    a_part = a_part - a_part.mean()
    b_part = b_part - b_part.mean()
    denom = math.sqrt(float((a_part * a_part).sum()) * float((b_part * b_part).sum()))
    return float((a_part * b_part).sum() / denom) if denom > 0 else 0.0


def _shift_frame(frame: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Image décalée de (dy, dx) pixels entiers (bords à zéro), et la zone valide."""
    if dy == 0 and dx == 0:
        return frame, (slice(None), slice(None))
    out = np.zeros_like(frame)
    h, w = frame.shape[:2]
    dst = (slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx)))
    src = (slice(max(0, -dy), h + min(0, -dy)), slice(max(0, -dx), w + min(0, -dx)))
    out[dst] = frame[src]
    return out, dst


# ------------------------------------------------------- profondeur de champ étendue
@dataclass
class StackStatus:
    """Où en est la composition, après chaque image du live."""

    frames: int = 0
    used: int = 0
    #: part de l'image améliorée par la dernière image (0 à 1) : tombe à 0 quand on a
    #: dépassé l'échantillon
    contribution: float = 0.0
    #: décalage latéral mesuré (pixels) depuis la première image
    drift_px: tuple[float, float] = (0.0, 0.0)
    accepted: bool = True
    message: str = ""


class FocusStacker:
    """Profondeur de champ étendue, image après image.

    Pour chaque petite zone de l'image, une mesure de netteté (énergie du laplacien, lissée)
    est comparée à la meilleure vue jusque-là ; si la nouvelle image est plus nette à cet
    endroit, ses pixels remplacent ceux du résultat. La décision se prend sur une image
    réduite (rapide même en 4K), les pixels copiés sont ceux de la pleine résolution.

    Le déplacement latéral est toujours mesuré : si la lame a bougé (plus de ``max_drift``
    du champ), l'image est ignorée. ``align`` le compense en plus ; désactivé par défaut,
    car le flou qui change d'une image à l'autre fausse un peu la mesure (sur le simulateur,
    jusqu'à 5 pixels de « dérive » imaginaire) : à n'activer que si la lame glisse vraiment
    pendant la mise au point.
    """

    def __init__(
        self,
        *,
        align: bool = False,
        margin: float = 0.05,
        smooth: float = 5.0,
        detail: float = 1.5,
        max_drift: float = 0.08,
    ) -> None:
        self.align = align
        self.margin = margin
        self.smooth = smooth
        self.detail = detail
        self.max_drift = max_drift
        self.reset()

    def reset(self) -> None:
        self._result: np.ndarray | None = None
        self._best: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._factor = 1
        self._track = 1
        self._prev_hp: np.ndarray | None = None
        self._drift = (0.0, 0.0)
        self.status = StackStatus()

    @property
    def result(self) -> np.ndarray | None:
        """Image composée (même type que les images du live), ou None avant la première."""
        return self._result

    def depth_map(self) -> np.ndarray | None:
        """Numéro de l'image la plus nette, zone par zone (0 = la première) : un relief
        relatif, sans unité (la mise au point manuelle n'est pas mesurée)."""
        return None if self._depth is None else self._depth.copy()

    def _energy(self, g: np.ndarray) -> np.ndarray:
        """Netteté locale : énergie des détails d'une bande de tailles (différence de deux
        flous), moyennée sur un voisinage. Les plus petits détails sont laissés de côté : le
        bruit de la caméra y imiterait de la netteté (essais sur le simulateur : erreur
        réduite d'un tiers par rapport au laplacien)."""
        band = ndi.gaussian_filter(g, self.detail) - ndi.gaussian_filter(g, 3.0 * self.detail)
        return ndi.gaussian_filter(band * band, self.smooth)

    def add(self, frame: np.ndarray) -> StackStatus:
        frame = np.asarray(frame)
        st = self.status
        st.frames += 1
        h, w = frame.shape[:2]
        if self._result is None or self._result.shape != frame.shape:
            self.reset()
            st = self.status
            st.frames = 1
            self._factor = max(1, math.ceil(max(h, w) / 1600))
            self._track = max(1, math.ceil(max(h, w) / 512))

        # 1. glissement latéral depuis l'image précédente : la lame a-t-elle bougé ?
        dy = dx = 0
        small = _highpass(_small(frame, self._track))
        if self._prev_hp is not None and self._prev_hp.shape == small.shape:
            sy, sx = _content_shift(self._prev_hp, small)
            if _overlap_ncc(self._prev_hp, small, sy, sx) >= MIN_MATCH:
                self._drift = (self._drift[0] + sy * self._track, self._drift[1] + sx * self._track)
        self._prev_hp = small
        st.drift_px = self._drift
        if math.hypot(*self._drift) > self.max_drift * min(h, w):
            st.accepted = False
            st.contribution = 0.0
            st.message = "La lame a bougé : remettez-la en place, ou recommencez"
            return st
        if self.align:
            dy, dx = -int(round(self._drift[0])), -int(round(self._drift[1]))
        aligned, valid = _shift_frame(frame, dy, dx)

        # 2. netteté locale, sur une image réduite
        f = self._factor
        g = _block_mean(_gray(aligned), f)
        energy = self._energy(g)
        if dy or dx:  # pas de décision sur la bande ajoutée par le décalage
            ys, xs = valid
            mask_valid = np.zeros(g.shape, bool)
            mask_valid[
                (ys.start or 0) // f + 1 : (ys.stop or h) // f - 1,
                (xs.start or 0) // f + 1 : (xs.stop or w) // f - 1,
            ] = True
            energy = np.where(mask_valid, energy, 0.0)

        # 3. ce qui est plus net qu'avant remplace l'ancien
        if self._result is None or self._best is None:
            self._result = aligned.copy()
            self._best = energy
            self._depth = np.zeros(energy.shape, np.int16)
            better_fraction = 1.0
            st.used = 1
        else:
            better = energy > self._best * (1.0 + self.margin)
            better_fraction = float(better.mean())
            if better.any():
                self._best = np.where(better, energy, self._best)
                self._depth[better] = min(st.frames - 1, np.iinfo(np.int16).max)
                full = np.repeat(np.repeat(better, f, axis=0), f, axis=1)
                if full.shape[0] < h or full.shape[1] < w:  # bord incomplet : on prolonge
                    full = np.pad(full, ((0, h - full.shape[0]), (0, w - full.shape[1])), mode="edge")
                full = full[:h, :w]
                np.copyto(self._result, aligned, where=full[..., None] if frame.ndim == 3 else full)
                st.used += 1
        st.accepted = True
        st.contribution = better_fraction
        if st.frames == 1:
            st.message = "Tournez lentement la mise au point fine à travers tout l'échantillon"
        elif better_fraction > 0.01:
            st.message = f"Cette image apporte {better_fraction:.0%} de zones plus nettes"
        else:
            st.message = "Plus rien de nouveau : vous pouvez terminer (ou explorer l'autre sens)"
        return st


# -------------------------------------------------------------------- panorama
@dataclass
class MosaicStatus:
    #: « attente » (avant le premier champ), « suivi », « ajouté », « couvert », « perdu »
    state: str = "attente"
    #: coin haut-gauche du champ actuel dans le panorama (pixels pleine résolution)
    position: tuple[float, float] | None = None
    tiles: int = 0
    moving: bool = False
    message: str = ""


@dataclass
class _Tile:
    position: tuple[float, float]  # (ligne, colonne), pixels pleine résolution
    path: Path  # image complète, sur le disque (lue à la demande)
    small_hp: np.ndarray  # pour retrouver sa place quand le suivi est perdu
    medium: np.ndarray  # détails (passe-haut), moyenne résolution, pour l'ajustement fin
    preview: np.ndarray  # image réduite pour l'aperçu
    shape: tuple[int, ...] = field(default=())


class LiveMosaic:
    """Panorama construit à la main, en direct.

    Chaque image du live est comparée à la précédente pour suivre le déplacement de la
    lame. Quand la platine s'arrête sur une zone qui apporte assez de nouveau
    (``min_new``), le champ est ajouté : sa place est d'abord ajustée finement sur les champs
    déjà posés qui le recouvrent, ce qui empêche l'erreur de s'accumuler. Si le suivi est
    perdu (déplacement trop rapide), il suffit de revenir sur une zone déjà couverte.

    Les champs sont gardés sur le disque (dossier temporaire) : un grand panorama ne remplit
    pas la mémoire. :meth:`render` assemble le résultat final.
    """

    def __init__(
        self,
        *,
        track_side: int = 480,
        preview_side: int = 640,
        min_new: float = 0.2,
        still_frames: int = 2,
        still_px: float = 0.6,
        workdir: str | Path | None = None,
    ) -> None:
        self.track_side = track_side
        self.preview_side = preview_side
        self.min_new = min_new
        self.still_frames = still_frames
        self.still_px = still_px
        self._dir = Path(tempfile.mkdtemp(prefix="fuji-panorama-", dir=workdir))
        self._counter = 0
        self.reset()

    # ------------------------------------------------------------------ état
    def reset(self) -> None:
        for tile in getattr(self, "tiles", []):
            tile.path.unlink(missing_ok=True)
        self.tiles: list[_Tile] = []
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None
        self._t = self._m = self._p = 1
        self._prev_hp: np.ndarray | None = None
        self._pos: tuple[float, float] | None = None
        self._tracking = False
        self._still = 0
        #: après « retirer le dernier champ » : rien n'est ajouté avant que la lame ait bougé
        self._hold = False
        #: position déjà ajustée à cet arrêt (une seule fois par arrêt : c'est le plus coûteux)
        self._settled = False
        self._canvas: np.ndarray | None = None
        self._covered: np.ndarray | None = None
        self._origin = (0, 0)  # coin du canevas d'aperçu, en pixels pleine résolution
        self.status = MosaicStatus()

    def close(self) -> None:
        """Libère les fichiers temporaires (à appeler une fois le panorama assemblé)."""
        shutil.rmtree(self._dir, ignore_errors=True)

    @property
    def preview(self) -> tuple[np.ndarray | None, int, tuple[int, int]]:
        """(aperçu, facteur de réduction, coin haut-gauche en pixels pleine résolution)."""
        return self._canvas, self._p, self._origin

    @property
    def frame_shape(self) -> tuple[int, ...] | None:
        return self._shape

    # ------------------------------------------------------------------ live
    def add(self, frame: np.ndarray) -> MosaicStatus:
        frame = np.asarray(frame)
        st = self.status
        if self._shape is None or frame.shape != self._shape:
            if self.tiles:
                st.message = "La taille des images du live a changé : recommencez le panorama"
                st.state = "perdu"
                return st
            self._shape, self._dtype = frame.shape, frame.dtype
            side = max(frame.shape[:2])
            self._t = max(1, math.ceil(side / self.track_side))
            self._m = max(1, math.ceil(side / 1024))
            self._p = max(1, math.ceil(side / self.preview_side))
        small = _highpass(_small(frame, self._t))

        # 1. suivre le déplacement depuis l'image précédente
        motion = None
        if self._prev_hp is not None and self._tracking and self._pos is not None:
            dy, dx = _content_shift(self._prev_hp, small)
            limit = 0.45 * min(small.shape)
            if max(abs(dy), abs(dx)) < limit and _overlap_ncc(self._prev_hp, small, dy, dx) >= MIN_MATCH:
                motion = math.hypot(dy, dx)
                # le contenu glisse à l'opposé du champ
                self._pos = (self._pos[0] - dy * self._t, self._pos[1] - dx * self._t)
            else:
                self._tracking = False
        self._prev_hp = small

        # 2. suivi perdu : retrouver sa place sur un champ déjà posé
        if self.tiles and not self._tracking:
            found = self._relocate(small)
            if found is None:
                self._still = 0
                st.state, st.moving = "perdu", True
                st.message = "Position perdue : revenez lentement sur une zone déjà couverte"
                st.position = self._pos
                return st
            self._pos, self._tracking, motion = found, True, None

        self._still = self._still + 1 if motion is not None and motion < self.still_px else 0
        st.moving = motion is None or motion >= self.still_px
        if motion is None or motion >= self.still_px:
            self._settled = False
            if motion is not None:
                self._hold = False
        st.position = self._pos

        # 3. premier champ : dès que l'image est stable
        if not self.tiles:
            if not self._tracking:  # tout début, ou image précédente méconnaissable
                self._pos, self._tracking = self._pos or (0.0, 0.0), True
                st.position = self._pos
            if self._still >= self.still_frames:
                self._add_tile(frame, small, self._pos)
                st.state, st.tiles = "ajouté", len(self.tiles)
                st.message = "Premier champ posé : déplacez la lame, puis arrêtez-vous pour ajouter un champ"
            else:
                st.state = "attente"
                st.message = "Immobilisez la lame sur le premier champ"
            return st

        if self._still < self.still_frames:
            st.state = "suivi"
            st.message = "Déplacement suivi — arrêtez-vous pour ajouter ce champ"
            return st
        if self._hold:
            st.state = "couvert"
            st.message = "Champ retiré : déplacez la lame pour continuer"
            return st

        # 4. immobile : ajuster finement la place (une fois par arrêt), puis ajouter le champ
        #    s'il apporte du neuf
        if not self._settled:
            self._pos = self._refine(frame, self._pos)
            self._settled = True
        st.position = self._pos
        new = self._new_fraction(self._pos)
        if new >= self.min_new:
            self._add_tile(frame, small, self._pos)
            self._still = 0
            st.state, st.tiles = "ajouté", len(self.tiles)
            st.message = f"Champ {len(self.tiles)} ajouté ({new:.0%} de nouvelle surface)"
        else:
            st.state = "couvert"
            st.message = "Zone déjà couverte : déplacez la lame vers une zone nouvelle"
        return st

    def undo(self) -> bool:
        """Retire le dernier champ ajouté (et refait l'aperçu)."""
        if not self.tiles:
            return False
        tile = self.tiles.pop()
        tile.path.unlink(missing_ok=True)
        self._hold, self._still = True, 0
        self._canvas, self._covered = None, None
        for t in self.tiles:
            self._paste_preview(t)
        self.status.tiles = len(self.tiles)
        if not self.tiles:
            self._pos, self._tracking = None, False
        return True

    # ------------------------------------------------------------------ calculs
    def _relocate(self, small: np.ndarray) -> tuple[float, float] | None:
        """Cherche le champ posé qui ressemble le plus à l'image actuelle (en commençant par
        les plus proches de la dernière position connue)."""
        candidates = self.tiles
        if self._pos is not None:
            candidates = sorted(
                self.tiles,
                key=lambda t: math.hypot(t.position[0] - self._pos[0], t.position[1] - self._pos[1]),
            )
        best: tuple[float, tuple[float, float]] | None = None
        for tile in candidates[:12]:
            if tile.small_hp.shape != small.shape:
                continue
            dy, dx = _content_shift(tile.small_hp, small)
            if max(abs(dy), abs(dx)) > 0.6 * min(small.shape):
                continue
            score = _overlap_ncc(tile.small_hp, small, dy, dx)
            if score >= MIN_MATCH + 0.1 and (best is None or score > best[0]):
                best = (score, (tile.position[0] - dy * self._t, tile.position[1] - dx * self._t))
        return None if best is None else best[1]

    def _refine(self, frame: np.ndarray, pos: tuple[float, float]) -> tuple[float, float]:
        """Ajuste la position sur les champs posés qui recouvrent le plus l'image actuelle."""
        m = self._m
        medium = _highpass(_block_mean(_gray(frame), m), 31)
        mh, mw = medium.shape
        estimates, weights = [], []
        overlaps = []
        for tile in self.tiles:
            r = (pos[0] - tile.position[0]) / m  # décalage du champ actuel dans la tuile
            c = (pos[1] - tile.position[1]) / m
            oh, ow = mh - abs(r), mw - abs(c)
            if oh > 48 and ow > 48 and oh * ow > 0.15 * mh * mw:
                overlaps.append((oh * ow, tile, r, c))
        for _area, tile, r, c in sorted(overlaps, key=lambda o: -o[0])[:3]:
            ir, ic = int(round(r)), int(round(c))
            t_part = tile.medium[max(0, ir) : mh + min(0, ir), max(0, ic) : mw + min(0, ic)]
            f_part = medium[max(0, -ir) : mh + min(0, -ir), max(0, -ic) : mw + min(0, -ic)]
            if t_part.shape != f_part.shape or min(t_part.shape) < 32:
                continue
            dy, dx = _content_shift(t_part, f_part, upsample=8)
            sure = _overlap_ncc(t_part, f_part, dy, dx)
            if sure < 0.5 or max(abs(dx), abs(dy)) > 0.2 * min(t_part.shape):
                continue
            # le contenu du champ actuel est décalé de (dy, dx) par rapport à la tuile
            estimates.append((tile.position[0] + ir * m - dy * m, tile.position[1] + ic * m - dx * m))
            weights.append(sure)
        if not estimates:
            return pos
        est = np.average(np.array(estimates), axis=0, weights=np.array(weights))
        return float(est[0]), float(est[1])

    def _new_fraction(self, pos: tuple[float, float]) -> float:
        """Part du champ actuel qui n'est encore couverte par aucun champ posé."""
        if self._covered is None or self._shape is None:
            return 1.0
        p = self._p
        h, w = self._shape[0] // p, self._shape[1] // p
        r0 = int(round((pos[0] - self._origin[0]) / p))
        c0 = int(round((pos[1] - self._origin[1]) / p))
        ch, cw = self._covered.shape
        rr0, cc0, rr1, cc1 = max(0, r0), max(0, c0), min(ch, r0 + h), min(cw, c0 + w)
        covered = int(self._covered[rr0:rr1, cc0:cc1].sum()) if rr1 > rr0 and cc1 > cc0 else 0
        return 1.0 - covered / float(h * w)

    def _add_tile(self, frame: np.ndarray, small: np.ndarray, pos: tuple[float, float]) -> None:
        self._counter += 1
        path = self._dir / f"champ_{self._counter:04d}.npy"
        np.save(path, frame)
        tile = _Tile(
            position=pos,
            path=path,
            small_hp=small,
            medium=_highpass(_block_mean(_gray(frame), self._m), 31),
            preview=np.ascontiguousarray(frame[:: self._p, :: self._p]),
            shape=frame.shape,
        )
        self.tiles.append(tile)
        self._paste_preview(tile)

    def _paste_preview(self, tile: _Tile) -> None:
        """Pose le champ dans l'aperçu (agrandi au besoin) ; les zones déjà couvertes restent."""
        p = self._p
        prev = tile.preview
        r0 = int(round(tile.position[0] / p))
        c0 = int(round(tile.position[1] / p))
        ph, pw = prev.shape[:2]
        if self._canvas is None:
            self._origin = (r0 * p, c0 * p)
            self._canvas = np.zeros((ph, pw, *prev.shape[2:]), prev.dtype)
            self._covered = np.zeros((ph, pw), bool)
        o_r, o_c = self._origin[0] // p, self._origin[1] // p
        ch, cw = self._canvas.shape[:2]
        top, left = min(o_r, r0), min(o_c, c0)
        bottom, right = max(o_r + ch, r0 + ph), max(o_c + cw, c0 + pw)
        if (top, left, bottom, right) != (o_r, o_c, o_r + ch, o_c + cw):
            # agrandir avec une marge d'un champ, pour ne pas tout recopier à chaque fois
            top, left = (top - ph if top < o_r else top), (left - pw if left < o_c else left)
            bottom = bottom + ph if bottom > o_r + ch else bottom
            right = right + pw if right > o_c + cw else right
            canvas = np.zeros((bottom - top, right - left, *prev.shape[2:]), prev.dtype)
            covered = np.zeros((bottom - top, right - left), bool)
            canvas[o_r - top : o_r - top + ch, o_c - left : o_c - left + cw] = self._canvas
            covered[o_r - top : o_r - top + ch, o_c - left : o_c - left + cw] = self._covered
            self._canvas, self._covered, self._origin = canvas, covered, (top * p, left * p)
            o_r, o_c = top, left
        rs, cs = r0 - o_r, c0 - o_c
        region = (slice(rs, rs + ph), slice(cs, cs + pw))
        empty = ~self._covered[region]
        target = self._canvas[region]
        target[empty] = prev[empty]
        self._covered[region] = True

    # ------------------------------------------------------------------ résultat
    def footprint(self, scale: int = 1) -> tuple[int, int]:
        """Taille (lignes, colonnes) du panorama assemblé à l'échelle 1/``scale``."""
        if not self.tiles or self._shape is None:
            return (0, 0)
        pos = np.array([t.position for t in self.tiles])
        span = pos.max(axis=0) - pos.min(axis=0)
        return (int(span[0] + self._shape[0]) // scale, int(span[1] + self._shape[1]) // scale)

    def render(self, *, max_pixels: int = 300_000_000, blend: bool = True) -> tuple[np.ndarray, int]:
        """Panorama final en pleine résolution, raccords fondus. Réduit d'un facteur entier
        (renvoyé) si le résultat dépasserait ``max_pixels`` pixels (mémoire)."""
        from .stage import stitch

        if not self.tiles:
            raise ValueError("Aucun champ dans le panorama")
        scale = 1
        while math.prod(self.footprint(scale)) > max_pixels:
            scale += 1
        tiles, offsets = [], []
        for tile in self.tiles:
            data = np.load(tile.path, mmap_mode="r")
            if scale > 1:
                data = _reduce(np.asarray(data), scale)
            tiles.append(np.asarray(data))
            offsets.append((tile.position[0] / scale, tile.position[1] / scale))
        return stitch(tiles, offsets, blend=blend), scale


def _reduce(frame: np.ndarray, factor: int) -> np.ndarray:
    """Réduction par moyenne de blocs, en gardant le type de l'image."""
    h, w = (frame.shape[0] // factor) * factor, (frame.shape[1] // factor) * factor
    shape = (h // factor, factor, w // factor, factor, *frame.shape[2:])
    out = frame[:h, :w].reshape(shape).mean(axis=(1, 3))
    return out.astype(frame.dtype) if np.issubdtype(frame.dtype, np.integer) else out
