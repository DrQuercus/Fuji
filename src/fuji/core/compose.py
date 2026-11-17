"""Images composées en direct à partir du live : le microscope peut rester entièrement manuel.

- :class:`FocusStacker` — **profondeur de champ étendue** : on tourne lentement la mise au
  point à travers l'échantillon ; chaque endroit de l'image garde l'état le plus net vu
  jusque-là. Le résultat est net partout, comme le « EDF » des logiciels de caméra.
- :class:`LiveMosaic` — **panorama** : on déplace la lame à la main, **sans s'arrêter** ; le
  logiciel suit le déplacement et pose un champ chaque fois qu'une zone nouvelle entre dans
  l'image. Aller vite donne une image un peu moins nette (flou de bougé), repasser lentement
  l'affine. Le résultat final est assemblé en pleine résolution, raccords fondus.

Les deux reçoivent les images du live une par une (méthode ``add``), dans un fil de travail :
une image qui arrive pendant un calcul est simplement sautée.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

#: en dessous de cette corrélation, deux images ne sont pas reconnues comme la même zone
MIN_MATCH = 0.3
_POOL = ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1)), thread_name_prefix="fuji-compose")


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
        return gray.astype(np.float32, copy=False)
    h, w = gray.shape[0] // factor, gray.shape[1] // factor
    # somme de tranches décalées : 7 fois plus rapide que reshape().mean() sur une vue non
    # contiguë (le canal vert d'une image couleur 4K : 8 ms au lieu de 55)
    out = np.zeros((h, w), np.float32)
    for i in range(factor):
        for j in range(factor):
            out += gray[i : h * factor : factor, j : w * factor : factor]
    out /= factor * factor
    return out


def _luma(frame: np.ndarray, factor: int) -> np.ndarray:
    """Image grise réduite d'un facteur ``factor``, rapide même en 4K : le canal vert d'une
    image couleur (le mieux résolu et le moins bruité d'un capteur à mosaïque de Bayer)."""
    plane = frame[..., 1] if frame.ndim == 3 and frame.shape[-1] >= 3 else frame
    if plane.ndim == 3:
        plane = plane[..., 0]
    return _block_mean(plane, factor)


def _small(frame: np.ndarray, factor: int) -> np.ndarray:
    """Petite image grise pour le suivi : sous-échantillonnage, moyenne par blocs (le bruit
    de la caméra est moyenné), puis léger lissage."""
    step = max(1, factor // 2)
    g = _block_mean(_gray(frame[::step, ::step]), max(1, factor // step))
    return ndi.gaussian_filter(g, 0.6)


def _highpass(g: np.ndarray, size: int = 15) -> np.ndarray:
    """Retire l'éclairage (fixe dans l'image) pour ne comparer que les détails."""
    return g - ndi.uniform_filter(g, size, mode="nearest")


@lru_cache(maxsize=8)
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
def _detail_energy(g: np.ndarray, detail: float, smooth: float) -> np.ndarray:
    band = ndi.gaussian_filter(g, detail, truncate=3.0)
    band -= ndi.gaussian_filter(g, 3.0 * detail, truncate=3.0)
    band *= band
    return ndi.gaussian_filter(band, smooth, truncate=3.0)


@lru_cache(maxsize=8)
def _noise_calibration(detail: float, smooth: float) -> tuple[float, float, float]:
    """Pour un bruit blanc d'écart type 1 : écart médian de ses plus fins détails, puis
    moyenne et écart type de l'énergie de détails qu'il produit."""
    noise = np.random.default_rng(0).standard_normal((384, 384)).astype(np.float32)
    fine = float(np.median(np.abs(noise - ndi.gaussian_filter(noise, 1.0, truncate=3.0))))
    energy = _detail_energy(noise, detail, smooth)[24:-24, 24:-24]
    return fine, float(energy.mean()), float(energy.std())


@dataclass
class StackStatus:
    """Où en est la composition, après chaque image du live."""

    frames: int = 0
    used: int = 0
    #: part de l'image améliorée par la dernière image (0 à 1) : tombe à 0 quand on a
    #: dépassé l'échantillon
    contribution: float = 0.0
    #: décalage latéral de la dernière image par rapport à l'image composée (pixels)
    drift_px: tuple[float, float] = (0.0, 0.0)
    accepted: bool = True
    #: « attente », « mieux » (l'image apporte du net), « rien » (plus rien de nouveau),
    #: « bougé » (la lame a glissé : image écartée)
    state: str = "attente"
    message: str = ""


class FocusStacker:
    """Profondeur de champ étendue, image après image.

    Pour chaque petite zone de l'image, une mesure de netteté (énergie des détails, lissée)
    est comparée à la meilleure vue jusque-là ; si la nouvelle image est nettement plus nette à
    cet endroit, ses pixels remplacent ceux du résultat. La décision se prend sur une image
    réduite (rapide même en 4K), les pixels copiés sont ceux de la pleine résolution.

    Le fond sans détail ne « scintille » pas d'une image à l'autre : un gain plus petit que
    le niveau du bruit de la caméra est ignoré, et les zones isolées de quelques pixels aussi.

    Le glissement latéral est mesuré **par rapport à l'image composée** (et non d'une image à
    la suivante : l'erreur ne s'accumule pas, on ne reste jamais bloqué). Une image trop
    décalée (plus de ``max_shift`` du champ) est écartée ; il suffit de revenir en place.
    ``align`` compense en plus les petits glissements ; désactivé par défaut, car le flou qui
    change d'une image à l'autre fausse un peu la mesure.
    """

    def __init__(
        self,
        *,
        align: bool = False,
        margin: float = 0.05,
        smooth: float = 5.0,
        detail: float = 1.5,
        max_shift: float = 0.06,
        decision_side: int = 960,
        halo: float = 3.0,
    ) -> None:
        self.align = align
        #: portée de la reprise du fond autour d'un objet devenu net (en rayons de lissage)
        self.halo = halo
        self.margin = margin
        self.smooth = smooth
        self.detail = detail
        self.max_shift = max_shift
        self.decision_side = decision_side
        self.reset()

    def reset(self) -> None:
        self._result: np.ndarray | None = None
        self._best: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._ref: np.ndarray | None = None  # image composée, grise et réduite (pour le glissement)
        self._factor = 1
        self._track = 1
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
        bruit de la caméra y imiterait de la netteté."""
        return _detail_energy(g, self.detail, self.smooth)

    def _noise_energy(self, g: np.ndarray) -> tuple[float, float]:
        """(moyenne, écart type) de la « netteté » que le seul bruit de la caméra donne à
        cette image : en dessous, une zone ne contient rien de mesurable.

        Le bruit est estimé sur les plus fins détails de l'image (écart médian, peu sensible à
        l'échantillon), puis converti grâce à un étalonnage fait une fois sur du bruit pur."""
        fine, mean, std = _noise_calibration(self.detail, self.smooth)
        sub = g[::2, ::2]
        sigma = float(np.median(np.abs(sub - ndi.gaussian_filter(sub, 1.0, truncate=3.0)))) / fine
        return mean * sigma * sigma, std * sigma * sigma

    def _shift_from_result(self, g: np.ndarray) -> tuple[float, float] | None:
        """Glissement (pixels pleine résolution) de cette image par rapport à l'image
        composée, ou None si elles ne se ressemblent pas assez pour le mesurer (image
        complètement hors du point)."""
        if self._ref is None:
            return None
        k = self._track
        ref, mov = _highpass(_block_mean(self._ref, k)), _highpass(_block_mean(g, k))
        sy, sx = _content_shift(ref, mov)
        if _overlap_ncc(ref, mov, sy, sx) < MIN_MATCH:
            return None
        unit = k * self._factor
        return sy * unit, sx * unit

    def add(self, frame: np.ndarray, t: float | None = None, exposure_s: float | None = None) -> StackStatus:
        """Ajoute une image du live (``t`` et ``exposure_s`` : ignorés, même appel que le
        panorama)."""
        frame = np.asarray(frame)
        st = self.status
        h, w = frame.shape[:2]
        if self._result is None or self._result.shape != frame.shape:
            self.reset()
            st = self.status
            self._factor = max(1, math.ceil(max(h, w) / self.decision_side))
            self._track = max(1, math.ceil(max(h, w) / self._factor / 512))
        st.frames += 1
        f = self._factor
        g = _luma(frame, f)

        # 1. la lame a-t-elle glissé ? (par rapport à l'image composée)
        dy = dx = 0
        shift = self._shift_from_result(g)
        if shift is not None:
            st.drift_px = shift
            if math.hypot(*shift) > self.max_shift * min(h, w):
                st.accepted, st.contribution, st.state = False, 0.0, "bougé"
                st.message = "La lame a bougé : remettez-la en place (ou recommencez)"
                return st
            if self.align:
                dy, dx = -int(round(shift[0])), -int(round(shift[1]))
        aligned, valid = _shift_frame(frame, dy, dx)
        if dy or dx:
            g = _luma(aligned, f)

        # 2. netteté locale, sur l'image réduite
        energy = self._energy(g)
        noise_mean, noise_std = self._noise_energy(g)
        if dy or dx:  # pas de décision sur la bande ajoutée par le décalage
            ys, xs = valid
            keep = np.zeros(g.shape, bool)
            keep[
                (ys.start or 0) // f + 1 : (ys.stop or h) // f - 1,
                (xs.start or 0) // f + 1 : (xs.stop or w) // f - 1,
            ] = True
            energy = np.where(keep, energy, 0.0)

        # 3. ce qui est nettement plus net qu'avant remplace l'ancien
        if self._result is None or self._best is None:
            self._result = aligned.copy()
            self._best = energy
            self._depth = np.zeros(energy.shape, np.int16)
            self._ref = g.copy()
            fraction = 1.0
            st.used = 1
        else:
            assert self._depth is not None and self._ref is not None
            # nettement plus net qu'avant, et pas seulement par le hasard du bruit : la zone
            # doit contenir quelque chose de mesurable (6 écarts types au-dessus du bruit :
            # avec 4, un fond vide « gagnait » encore 0,2 % de l'image à chaque passage)
            better = energy > self._best * (1.0 + self.margin) + 2.0 * noise_std
            better &= energy > noise_mean + 6.0 * noise_std
            # pas de zones isolées de quelques pixels : il faut une majorité de voisins d'accord
            better &= ndi.uniform_filter(better.astype(np.float32), 3) > 0.6
            fraction = float(better.mean())
            if fraction < 0.001:  # quelques pixels épars : du bruit, pas de la netteté
                fraction = 0.0
            if fraction > 0:
                # autour d'un objet devenu net, le fond voisin vient de la même image : sinon le
                # halo flou de cet objet, vu plus tôt hors du point, resterait autour de lui.
                # Seulement là où cette image n'est pas moins nette que ce qu'on a déjà.
                reach = 2 * int(round(self.halo * self.smooth)) + 1
                near = ndi.uniform_filter(better.astype(np.float32), reach) > 1e-6
                better |= near & (energy > 0.7 * self._best - 2.0 * noise_std)
                np.maximum(energy, self._best, out=energy)
                np.copyto(self._best, energy, where=better)
                self._depth[better] = min(st.frames - 1, np.iinfo(np.int16).max)
                np.copyto(self._ref, g, where=better)
                self._copy_where(aligned, better)
                st.used += 1
        st.accepted = True
        st.contribution = fraction
        if st.frames == 1:
            st.state = "attente"
            st.message = "Tournez lentement la mise au point fine à travers tout l'échantillon"
        elif fraction > 0.005:
            st.state = "mieux"
            st.message = f"Cette image apporte {fraction:.0%} de zones plus nettes"
        else:
            st.state = "rien"
            st.message = "Plus rien de nouveau ici : continuez, changez de sens, ou terminez"
        return st

    def _copy_where(self, frame: np.ndarray, better: np.ndarray) -> None:
        """Copie dans le résultat les pixels pleine résolution des zones ``better`` (masque de
        l'image réduite), en ne touchant que le rectangle concerné."""
        assert self._result is not None
        f = self._factor
        h, w = frame.shape[:2]
        rows, cols = np.flatnonzero(better.any(axis=1)), np.flatnonzero(better.any(axis=0))
        r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        sub = better[r0:r1, c0:c1]
        mask = np.repeat(np.repeat(sub, f, axis=0), f, axis=1) if f > 1 else sub
        y0, x0 = r0 * f, c0 * f
        y1 = h if r1 == better.shape[0] else r1 * f  # le bord incomplet suit la dernière zone
        x1 = w if c1 == better.shape[1] else c1 * f
        if mask.shape != (y1 - y0, x1 - x0):
            mask = np.pad(mask, ((0, y1 - y0 - mask.shape[0]), (0, x1 - x0 - mask.shape[1])), mode="edge")
        target, source = self._result[y0:y1, x0:x1], frame[y0:y1, x0:x1]
        where = mask[..., None] if frame.ndim == 3 else mask

        def copy(rows: slice) -> None:
            np.copyto(target[rows], source[rows], where=where[rows])

        if target.size < 2_000_000:
            copy(slice(None))
        else:  # numpy libère le verrou de Python : les bandes sont copiées en parallèle
            n = target.shape[0]
            step = max(1, n // 8)
            list(_POOL.map(copy, [slice(i, min(n, i + step)) for i in range(0, n, step)]))


# -------------------------------------------------------------------- panorama
@dataclass
class MosaicStatus:
    #: « attente » (avant le premier champ), « suivi » (en mouvement, rien à ajouter),
    #: « ajouté », « vite » (ajouté, mais un peu flou), « affiné » (zone floue remplacée par
    #: une image nette), « trop_vite » (rien n'est ajouté), « couvert », « perdu »
    state: str = "attente"
    #: coin haut-gauche du champ actuel dans le panorama (pixels pleine résolution)
    position: tuple[float, float] | None = None
    tiles: int = 0
    moving: bool = False
    #: flou de bougé estimé de la dernière image (pixels pleine résolution)
    blur_px: float = 0.0
    #: change chaque fois que l'aperçu change (champ ajouté, affiné, retiré)
    version: int = 0
    message: str = ""


@dataclass
class _Tile:
    position: tuple[float, float]  # (ligne, colonne), pixels pleine résolution
    path: Path  # image complète, sur le disque (lue à la demande)
    small_hp: np.ndarray  # pour retrouver sa place quand le suivi est perdu
    medium: np.ndarray  # détails (passe-haut), moyenne résolution, pour l'ajustement fin
    preview: np.ndarray  # image réduite pour l'aperçu
    shape: tuple[int, ...] = field(default=())
    blur: float = 0.0  # flou de bougé au moment de la prise (pixels pleine résolution)


class _Preview:
    """Aperçu réduit du panorama : l'image, et le flou du champ visible en chaque point
    (``inf`` : rien n'y est encore posé)."""

    def __init__(self, factor: int, gain: np.ndarray | None = None) -> None:
        self.factor = factor
        self.gain = gain  # compensation de l'éclairage inégal, à la taille des aperçus de champ
        self.canvas: np.ndarray | None = None
        self.blurmap: np.ndarray | None = None
        self.origin = (0, 0)  # coin du canevas, en pixels pleine résolution

    def paste(self, tile: _Tile) -> None:
        """Pose un champ (le canevas grandit au besoin) : dans les zones vides, et par-dessus
        ce qui était nettement plus flou."""
        p = self.factor
        prev = tile.preview
        r0 = int(round(tile.position[0] / p))
        c0 = int(round(tile.position[1] / p))
        ph, pw = prev.shape[:2]
        if self.canvas is None or self.blurmap is None:
            self.origin = (r0 * p, c0 * p)
            self.canvas = np.zeros((ph, pw, *prev.shape[2:]), prev.dtype)
            self.blurmap = np.full((ph, pw), np.inf, np.float32)
        o_r, o_c = self.origin[0] // p, self.origin[1] // p
        ch, cw = self.canvas.shape[:2]
        top, left = min(o_r, r0), min(o_c, c0)
        bottom, right = max(o_r + ch, r0 + ph), max(o_c + cw, c0 + pw)
        if (top, left, bottom, right) != (o_r, o_c, o_r + ch, o_c + cw):
            # agrandir avec une marge d'un champ, pour ne pas tout recopier à chaque fois
            top, left = (top - ph if top < o_r else top), (left - pw if left < o_c else left)
            bottom = bottom + ph if bottom > o_r + ch else bottom
            right = right + pw if right > o_c + cw else right
            canvas = np.zeros((bottom - top, right - left, *prev.shape[2:]), prev.dtype)
            blurmap = np.full((bottom - top, right - left), np.inf, np.float32)
            canvas[o_r - top : o_r - top + ch, o_c - left : o_c - left + cw] = self.canvas
            blurmap[o_r - top : o_r - top + ch, o_c - left : o_c - left + cw] = self.blurmap
            self.canvas, self.blurmap, self.origin = canvas, blurmap, (top * p, left * p)
            o_r, o_c = top, left
        rs, cs = r0 - o_r, c0 - o_c
        region = (slice(rs, rs + ph), slice(cs, cs + pw))
        current = self.blurmap[region]
        take = current > 1.5 * tile.blur + 0.5  # vide (inf) ou nettement plus flou
        if self.gain is not None and self.gain.shape[:2] == prev.shape[:2]:
            even = prev * self.gain
            if np.issubdtype(prev.dtype, np.integer):
                np.clip(even, 0, np.iinfo(prev.dtype).max, out=even)
            prev = even.astype(prev.dtype)
        self.canvas[region][take] = prev[take]
        current[take] = tile.blur


class LiveMosaic:
    """Panorama construit à la main, en direct et **en continu**.

    Chaque image du live est comparée à la précédente pour suivre le déplacement de la
    lame. Dès que l'image montre assez de surface nouvelle (``min_new``), elle est posée dans
    le panorama, sans attendre un arrêt : sa place est d'abord ajustée finement sur les
    champs voisins, ce qui empêche l'erreur de s'accumuler.

    La netteté dépend de la vitesse : le **flou de bougé** (vitesse × temps de pose) est
    estimé pour chaque image.

    - jusqu'à ``sharp_blur`` pixels, l'image est nette ;
    - jusqu'à ``max_blur``, elle est posée quand même (mieux vaut un champ un peu flou qu'un
      trou), et remplacée dès qu'on repasse plus lentement au même endroit ;
    - au-delà, rien n'est posé : il faut ralentir.

    Si le suivi est perdu (déplacement trop rapide, zone sans aucun détail), il suffit de
    revenir sur une zone déjà assemblée. Les défauts fixes de l'image (poussières du capteur,
    vignettage) sont appris pendant le déplacement et retirés du suivi : ils ne « retiennent »
    pas l'image.

    Les champs sont gardés sur le disque (dossier temporaire) : un grand panorama ne remplit
    pas la mémoire. :meth:`render` assemble le résultat final.
    """

    def __init__(
        self,
        *,
        track_side: int = 480,
        preview_side: int = 640,
        min_new: float = 0.15,
        sharp_blur: float | None = None,
        max_blur: float | None = None,
        workdir: str | Path | None = None,
    ) -> None:
        self.track_side = track_side
        self.preview_side = preview_side
        self.min_new = min_new
        #: None : 0,1 % du grand côté de l'image (4 pixels en 4K), au moins 2 pixels
        self.sharp_blur = sharp_blur
        #: None : 0,4 % du grand côté de l'image (15 pixels en 4K), au moins 4 pixels
        self.max_blur = max_blur
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
        self._blur_limit = 4.0
        self._sharp = 2.0
        self._prev_hp: np.ndarray | None = None
        self._prev_time: float | None = None
        self._static: np.ndarray | None = None  # défauts fixes de l'image, à l'échelle du suivi
        self._pos: tuple[float, float] | None = None
        self._tracking = False
        #: après « retirer le dernier champ » : rien n'est ajouté avant que la lame ait bougé
        self._hold: tuple[float, float] | None = None
        self._view = _Preview(1)  # l'aperçu affiché
        #: aperçu en cours de reconstruction (éclairage mieux compensé) : quelques champs à
        #: chaque image du live, pour ne jamais bloquer le suivi ; il remplace l'autre une fois fini
        self._rebuild: _Preview | None = None
        self._rebuilt = 0
        self.status = MosaicStatus()

    def close(self) -> None:
        """Libère les fichiers temporaires (à appeler une fois le panorama assemblé)."""
        shutil.rmtree(self._dir, ignore_errors=True)

    @property
    def preview(self) -> tuple[np.ndarray | None, int, tuple[int, int]]:
        """(aperçu, facteur de réduction, coin haut-gauche en pixels pleine résolution)."""
        return self._view.canvas, self._p, self._view.origin

    @property
    def frame_shape(self) -> tuple[int, ...] | None:
        return self._shape

    def bounds(self) -> tuple[float, float, float, float] | None:
        """(haut, gauche, bas, droite) de la zone assemblée, en pixels pleine résolution."""
        if not self.tiles or self._shape is None:
            return None
        pos = np.array([t.position for t in self.tiles])
        top, left = pos.min(axis=0)
        bottom, right = pos.max(axis=0) + np.array(self._shape[:2])
        return float(top), float(left), float(bottom), float(right)

    # ------------------------------------------------------------------ live
    def add(self, frame: np.ndarray, t: float | None = None, exposure_s: float | None = None) -> MosaicStatus:
        """Traite une image du live. ``t`` : instant de l'image (secondes, horloge
        quelconque ; par défaut l'horloge de l'ordinateur) ; ``exposure_s`` : temps de pose,
        pour estimer le flou de bougé."""
        now = time.monotonic() if t is None else float(t)
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
            self._blur_limit = self.max_blur if self.max_blur is not None else max(4.0, 0.004 * side)
            self._sharp = self.sharp_blur if self.sharp_blur is not None else max(2.0, 0.001 * side)
            self._view = _Preview(self._p)
        self._advance_rebuild()
        h, w = frame.shape[:2]
        raw_hp = _highpass(_small(frame, self._t))
        small = raw_hp if self._static is None else raw_hp - self._static

        # 1. suivre le déplacement depuis l'image précédente
        motion = None  # pixels de l'image de suivi
        if self._prev_hp is not None and self._tracking and self._pos is not None:
            dy, dx = _content_shift(self._prev_hp, small)
            limit = 0.45 * min(small.shape)
            if max(abs(dy), abs(dx)) < limit and _overlap_ncc(self._prev_hp, small, dy, dx) >= MIN_MATCH:
                motion = math.hypot(dy, dx)
                # le contenu glisse à l'opposé du champ
                self._pos = (self._pos[0] - dy * self._t, self._pos[1] - dx * self._t)
            else:
                self._tracking = False
        dt = None if self._prev_time is None else now - self._prev_time
        self._prev_hp, self._prev_time = small, now
        if motion is not None and motion >= 1.0:
            # la lame bouge : ce qui ne bouge pas dans l'image (poussières, vignettage) est
            # appris peu à peu, pour ne plus gêner le suivi
            if self._static is None:
                self._static = np.zeros_like(raw_hp)
            self._static += 0.04 * (raw_hp - self._static)

        # 2. suivi perdu : retrouver sa place sur un champ déjà posé
        if self.tiles and not self._tracking:
            found = self._relocate(small)
            if found is None:
                st.state, st.moving = "perdu", True
                st.message = "Position perdue : revenez lentement sur une zone déjà assemblée"
                st.position = self._pos
                return st
            self._pos, self._tracking, motion = found, True, None

        # 3. vitesse et flou de bougé
        step = None if motion is None else motion * self._t  # pixels pleine résolution
        blur = None
        if step is not None:
            if exposure_s and dt and dt > 0:
                blur = step / dt * exposure_s
            else:  # pose inconnue : on la suppose courte devant l'intervalle entre deux images
                blur = 0.25 * step
        sharp = blur is not None and blur <= self._sharp
        usable = blur is not None and blur <= self._blur_limit
        st.moving = step is None or step > 0.004 * max(h, w)
        st.blur_px = blur or 0.0
        st.position = self._pos

        # 4. premier champ : dès que l'image est assez nette
        if not self.tiles:
            if not self._tracking:  # tout début, ou image précédente méconnaissable
                self._pos, self._tracking = self._pos or (0.0, 0.0), True
                st.position = self._pos
            if usable:
                assert blur is not None
                self._add_tile(frame, small, self._pos, blur)
                st.state, st.tiles = "ajouté", len(self.tiles)
                st.message = "Premier champ posé : déplacez la lame, le panorama se construit tout seul"
            else:
                st.state = "attente"
                st.message = "Ralentissez un instant pour poser le premier champ"
            return self._done(st)

        if self._hold is not None:  # un champ vient d'être retiré : attendre un vrai déplacement
            if math.hypot(self._pos[0] - self._hold[0], self._pos[1] - self._hold[1]) < 0.08 * min(h, w):
                st.state = "couvert"
                st.message = "Champ retiré : déplacez la lame pour continuer"
                return self._done(st)
            self._hold = None

        # 5. assez de surface nouvelle : poser le champ (après ajustement sur ses voisins)
        new = self._new_fraction(self._pos)
        if new >= self.min_new:
            if not usable:
                st.state = "trop_vite"
                st.message = "Trop rapide : ralentissez, sinon cette zone restera vide"
                return self._done(st)
            assert blur is not None
            self._pos, medium = self._refine(frame, self._pos)
            st.position = self._pos
            self._add_tile(frame, small, self._pos, blur, medium)
            st.tiles = len(self.tiles)
            if sharp:
                st.state, st.message = "ajouté", f"{st.tiles} champs assemblés"
            else:
                st.state = "vite"
                st.message = (
                    f"{st.tiles} champs — un peu vite : image moins nette ici "
                    "(repassez lentement pour l'affiner)"
                )
            return self._done(st)

        # 6. zone déjà posée mais floue, et image nette : la remplacer
        if sharp and self._blurrier_fraction(self._pos, blur or 0.0) >= 0.3:
            assert blur is not None
            self._pos, medium = self._refine(frame, self._pos)
            st.position = self._pos
            self._add_tile(frame, small, self._pos, blur, medium, replace=True)
            st.state, st.tiles = "affiné", len(self.tiles)
            st.message = "Zone affinée : image plus nette"
            return self._done(st)

        if st.moving:
            st.state = "suivi"
            st.message = f"{len(self.tiles)} champs — avancez vers une zone nouvelle"
        else:
            st.state = "couvert"
            st.message = f"{len(self.tiles)} champs — zone déjà assemblée : déplacez la lame"
        return self._done(st)

    def _done(self, st: MosaicStatus) -> MosaicStatus:
        st.tiles = len(self.tiles)
        return st

    def undo(self) -> bool:
        """Retire le dernier champ ajouté (et refait l'aperçu)."""
        if not self.tiles:
            return False
        tile = self.tiles.pop()
        tile.path.unlink(missing_ok=True)
        self._hold = self._pos
        self._repaint()
        self.status.tiles = len(self.tiles)
        if not self.tiles:
            self._pos, self._tracking, self._hold = None, False, None
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

    def _refine(self, frame: np.ndarray, pos: tuple[float, float]) -> tuple[tuple[float, float], np.ndarray]:
        """Ajuste la position sur les champs posés qui recouvrent le plus l'image actuelle.
        Renvoie la position ajustée et l'image de détails (réutilisée si le champ est posé)."""
        m = self._m
        medium = _highpass(_luma(frame, m), 31)
        mh, mw = medium.shape
        estimates, weights = [], []
        overlaps = []
        for tile in self.tiles:
            r = (pos[0] - tile.position[0]) / m  # décalage du champ actuel dans la tuile
            c = (pos[1] - tile.position[1]) / m
            oh, ow = mh - abs(r), mw - abs(c)
            if oh > 48 and ow > 48 and oh * ow > 0.15 * mh * mw:
                overlaps.append((oh * ow / (1.0 + tile.blur), tile, r, c))
        for _area, tile, r, c in sorted(overlaps, key=lambda o: -o[0])[:3]:
            ir, ic = int(round(r)), int(round(c))
            t_part = tile.medium[max(0, ir) : mh + min(0, ir), max(0, ic) : mw + min(0, ic)]
            f_part = medium[max(0, -ir) : mh + min(0, -ir), max(0, -ic) : mw + min(0, -ic)]
            if t_part.shape != f_part.shape or min(t_part.shape) < 32:
                continue
            dy, dx = _content_shift(t_part, f_part, upsample=8)
            sure = _overlap_ncc(t_part, f_part, dy, dx)
            if sure < 0.4 or max(abs(dx), abs(dy)) > 0.2 * min(t_part.shape):
                continue
            # le contenu du champ actuel est décalé de (dy, dx) par rapport à la tuile
            estimates.append((tile.position[0] + ir * m - dy * m, tile.position[1] + ic * m - dx * m))
            weights.append(sure)
        if not estimates:
            return pos, medium
        est = np.average(np.array(estimates), axis=0, weights=np.array(weights))
        return (float(est[0]), float(est[1])), medium

    def _footprint(self, pos: tuple[float, float]) -> tuple[slice, slice, int] | None:
        """Zone de l'aperçu couverte par un champ placé en ``pos`` (limitée au canevas), et
        sa surface complète en pixels d'aperçu."""
        blurmap = self._view.blurmap
        if blurmap is None or self._shape is None:
            return None
        p = self._p
        h, w = len(range(0, self._shape[0], p)), len(range(0, self._shape[1], p))
        r0 = int(round((pos[0] - self._view.origin[0]) / p))
        c0 = int(round((pos[1] - self._view.origin[1]) / p))
        ch, cw = blurmap.shape
        rows, cols = slice(max(0, r0), min(ch, r0 + h)), slice(max(0, c0), min(cw, c0 + w))
        return rows, cols, h * w

    def _new_fraction(self, pos: tuple[float, float]) -> float:
        """Part du champ actuel qui n'est encore couverte par aucun champ posé."""
        area = self._footprint(pos)
        if area is None:
            return 1.0
        rows, cols, total = area
        blurmap = self._view.blurmap
        covered = int(np.isfinite(blurmap[rows, cols]).sum()) if blurmap is not None else 0
        return 1.0 - covered / float(total)

    def _blurrier_fraction(self, pos: tuple[float, float], blur: float) -> float:
        """Part du champ actuel déjà posée avec une image nettement plus floue que celle-ci."""
        area = self._footprint(pos)
        blurmap = self._view.blurmap
        if area is None or blurmap is None:
            return 0.0
        rows, cols, total = area
        part = blurmap[rows, cols]
        worse = np.isfinite(part) & (part > max(2.5 * blur, blur + 2.0))
        return float(worse.sum()) / float(total)

    def _add_tile(
        self,
        frame: np.ndarray,
        small: np.ndarray,
        pos: tuple[float, float],
        blur: float,
        medium: np.ndarray | None = None,
        replace: bool = False,
    ) -> None:
        self._counter += 1
        path = self._dir / f"champ_{self._counter:04d}.npy"
        np.save(path, frame)
        tile = _Tile(
            position=pos,
            path=path,
            small_hp=small,
            medium=medium if medium is not None else _highpass(_luma(frame, self._m), 31),
            preview=np.ascontiguousarray(frame[:: self._p, :: self._p]),
            shape=frame.shape,
            blur=float(blur),
        )
        if replace and self._shape is not None:
            # les champs flous posés presque au même endroit ne servent plus à rien
            near = 0.15 * min(self._shape[:2])
            for old in [
                t
                for t in self.tiles
                if t.blur > 2.0 * tile.blur + 1.0
                and abs(t.position[0] - pos[0]) < near
                and abs(t.position[1] - pos[1]) < near
            ]:
                old.path.unlink(missing_ok=True)
                self.tiles.remove(old)
        self.tiles.append(tile)
        self._view.paste(tile)
        self.status.version += 1
        n = len(self.tiles)
        if n >= 8 and n & (n - 1) == 0 and self._rebuild is None:
            # 8, 16, 32… champs : l'éclairage inégal est estimé de mieux en mieux ; l'aperçu
            # est refait sans les « carreaux » que laisse le vignettage, peu à peu
            gain = self._shading_gain(tile.preview.shape[:2])
            if gain is not None:
                self._rebuild, self._rebuilt = _Preview(self._p, gain), 0

    def _advance_rebuild(self, count: int = 3) -> None:
        """Avance la reconstruction de l'aperçu de quelques champs (quelques millisecondes)."""
        rebuild = self._rebuild
        if rebuild is None:
            return
        for tile in self.tiles[self._rebuilt : self._rebuilt + count]:
            rebuild.paste(tile)
        self._rebuilt = min(len(self.tiles), self._rebuilt + count)
        if self._rebuilt >= len(self.tiles):
            self._view, self._rebuild = rebuild, None
            self.status.version += 1

    def _repaint(self) -> None:
        """Refait l'aperçu d'un coup (après le retrait d'un champ)."""
        view = _Preview(self._p, self._view.gain)
        for t in self.tiles:
            view.paste(t)
        self._view, self._rebuild = view, None
        self.status.version += 1

    # ------------------------------------------------------------------ résultat
    def shading(self) -> np.ndarray | None:
        """Éclairage du champ estimé à partir des champs posés (petite image, moyenne 1 par
        canal), ou None s'il y en a trop peu.

        En chaque point de l'image, la médiane sur tous les champs donne ce qui ne dépend pas
        de l'échantillon ; seules ses variations lentes sont gardées (le vignettage), ce qui
        reste juste même sur un échantillon dense. Les poussières, elles, demandent une vraie
        image de référence."""
        if len(self.tiles) < 8:
            return None
        k = max(1, max(self.tiles[0].preview.shape[:2]) // 64)
        small = []
        for t in self.tiles[:: max(1, len(self.tiles) // 60)]:
            p = t.preview
            hh, ww = (p.shape[0] // k) * k, (p.shape[1] // k) * k
            cut = p[:hh, :ww].reshape(hh // k, k, ww // k, k, *p.shape[2:])
            small.append(cut.mean(axis=(1, 3), dtype=np.float32))
        median = np.median(np.stack(small), axis=0)
        sigma = (median.shape[0] / 6, median.shape[1] / 6) + (0,) * (median.ndim - 2)
        smooth = ndi.gaussian_filter(median, sigma, mode="nearest")
        smooth /= np.maximum(smooth.mean(axis=(0, 1), keepdims=True), 1e-6)
        return np.clip(smooth, 0.5, 2.0)

    def _shading_gain(self, shape: tuple[int, int]) -> np.ndarray | None:
        """Gain (1 / éclairage) à la taille des champs, pour l'assemblage final."""
        shading = self.shading()
        if shading is None:
            return None
        zoom = (shape[0] / shading.shape[0], shape[1] / shading.shape[1]) + (1,) * (shading.ndim - 2)
        full = ndi.zoom(shading, zoom, order=1, mode="nearest", grid_mode=True)
        full = full[: shape[0], : shape[1]]
        if full.shape[:2] != shape:  # arrondi du zoom : on complète le bord
            pad = [(0, shape[0] - full.shape[0]), (0, shape[1] - full.shape[1])] + [(0, 0)] * (full.ndim - 2)
            full = np.pad(full, pad, mode="edge")
        return (1.0 / full).astype(np.float32)

    def footprint(self, scale: int = 1) -> tuple[int, int]:
        """Taille (lignes, colonnes) du panorama assemblé à l'échelle 1/``scale``."""
        if not self.tiles or self._shape is None:
            return (0, 0)
        pos = np.array([t.position for t in self.tiles])
        span = pos.max(axis=0) - pos.min(axis=0)
        return (int(span[0] + self._shape[0]) // scale, int(span[1] + self._shape[1]) // scale)

    def render(
        self,
        *,
        max_pixels: int = 300_000_000,
        blend: bool = True,
        even: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Panorama final en pleine résolution, raccords fondus. Réduit d'un facteur entier
        (renvoyé) si le résultat dépasserait ``max_pixels`` pixels (mémoire). ``even`` :
        compense l'éclairage inégal estimé sur l'ensemble des champs (à partir de 8 champs).

        Dans les recouvrements, chaque champ compte d'autant plus qu'on est près de son centre
        et qu'il est net : un champ pris un peu vite s'efface devant un champ net. L'assemblage
        se fait par bandes : la mémoire utilisée reste celle de l'image finale."""
        if not self.tiles or self._shape is None:
            raise ValueError("Aucun champ dans le panorama")
        scale = 1
        while math.prod(self.footprint(scale)) > max_pixels:
            scale += 1
        h, w = self._shape[0] // scale, self._shape[1] // scale
        extra = self._shape[2:]
        offsets = np.round(np.array([t.position for t in self.tiles]) / scale).astype(int)
        offsets -= offsets.min(axis=0)
        height, width = int(offsets[:, 0].max()) + h, int(offsets[:, 1].max()) + w
        dtype = self._dtype if self._dtype is not None else np.dtype(np.uint16)
        out = np.zeros((height, width, *extra), dtype)
        # poids : distance au bord du champ (au cube : transition étroite, pas de « fantômes »)
        wy = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1) / (h / 2)
        wx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1) / (w / 2)
        weight = (np.minimum.outer(wy, wx) ** 3).astype(np.float32)
        quality = [1.0 / (1.0 + (t.blur / max(self._sharp, 1e-6)) ** 2) ** 2 for t in self.tiles]
        # éclairage inégal (vignettage) : estimé sur l'ensemble des champs et compensé, sinon
        # chaque rangée du panorama laisse une bande sombre
        gain = self._shading_gain((h, w)) if blend and even else None
        if gain is not None:
            weighted = gain * (weight[..., None] if extra else weight)
        else:
            weighted = None
        data = [np.load(t.path, mmap_mode="r") for t in self.tiles]
        integer = np.issubdtype(dtype, np.integer)
        band = max(32, int(24_000_000 // max(1, width * (extra[0] if extra else 1))))
        bands = range(0, height, band)
        for n, b0 in enumerate(bands):
            b1 = min(height, b0 + band)
            if not blend:
                for (r, c), tile in zip(offsets, data, strict=True):
                    lo, hi = max(b0, r), min(b1, r + h)
                    if hi > lo:
                        out[lo:hi, c : c + w] = _rows(tile, lo - r, hi - r, scale, w)
                continue
            total = np.zeros((b1 - b0, width, *extra), np.float32)
            weights = np.zeros((b1 - b0, width), np.float32)
            for (r, c), tile, q in zip(offsets, data, quality, strict=True):
                lo, hi = max(b0, r), min(b1, r + h)
                if hi <= lo:
                    continue
                part = _rows(tile, lo - r, hi - r, scale, w).astype(np.float32)
                wpart = weight[lo - r : hi - r] * np.float32(q)
                if weighted is not None:
                    part *= weighted[lo - r : hi - r] * np.float32(q)
                else:
                    part *= wpart[..., None] if extra else wpart
                total[lo - b0 : hi - b0, c : c + w] += part
                weights[lo - b0 : hi - b0, c : c + w] += wpart
            np.maximum(weights, 1e-12, out=weights)
            total /= weights[..., None] if extra else weights
            if integer:
                info = np.iinfo(dtype)
                total += 0.5
                np.clip(total, info.min, info.max, out=total)
            out[b0:b1] = total
            if progress is not None:
                progress(n + 1, len(bands))
        return out, scale


def _rows(tile: np.ndarray, r0: int, r1: int, scale: int, width: int) -> np.ndarray:
    """Lignes ``r0:r1`` (à l'échelle 1/``scale``) d'un champ enregistré, réduites au besoin."""
    if scale == 1:
        return np.asarray(tile[r0:r1, :width])
    part = np.asarray(tile[r0 * scale : r1 * scale, : width * scale])
    shape = (r1 - r0, scale, width, scale, *part.shape[2:])
    return part.reshape(shape).mean(axis=(1, 3), dtype=np.float32)
