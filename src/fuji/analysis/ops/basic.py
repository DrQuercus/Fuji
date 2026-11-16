"""Conversions, filtres, soustraction de fond, correction de champ plat."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, restoration, transform

from ..image import Image
from ..registry import Param, operation


def _per_channel(image: Image, fn) -> np.ndarray:
    d = image.data
    if image.is_rgb:
        return np.stack([fn(d[..., c]) for c in range(d.shape[-1])], axis=-1)
    return fn(d)


def _restore_dtype(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    if np.issubdtype(like.dtype, np.integer):
        info = np.iinfo(like.dtype)
        return np.clip(np.round(out), info.min, info.max).astype(like.dtype)
    return out.astype(like.dtype, copy=False)


# ----------------------------------------------------------------- conversions
@operation("Niveaux de gris", category="Conversion", imagej="Image > Type > 8-bit / 32-bit")
def to_gray(
    image: Image,
    method: Annotated[
        Literal["moyenne", "luminance"],
        Param(description="moyenne = (R+G+B)/3 comme ImageJ ; luminance = pondérée"),
    ] = "moyenne",
) -> Image:
    """Convertit une image couleur en niveaux de gris (32 bits, sans perte)."""
    return image.with_data(image.gray(method), name=f"{image.name} (gris)")


@operation("Séparer les canaux", category="Conversion", imagej="Image > Color > Split Channels")
def split_channels(image: Image) -> tuple[Image, ...]:
    """Sépare une image RGB en trois images (rouge, vert, bleu)."""
    if not image.is_rgb:
        return (image,)
    names = ("rouge", "vert", "bleu")
    return tuple(
        image.with_data(image.data[..., i].copy(), name=f"{image.name} ({names[i]})") for i in range(3)
    )


@operation("Inverser", category="Conversion", imagej="Edit > Invert")
def invert(image: Image) -> Image:
    d = image.data
    if np.issubdtype(d.dtype, np.integer):
        out = np.iinfo(d.dtype).max - d
    else:
        out = d.max() + d.min() - d
    return image.with_data(out, name=f"{image.name} (inversée)")


# ------------------------------------------------------------------- filtres
@operation("Flou gaussien", category="Filtres", imagej="Process > Filters > Gaussian Blur...")
def gaussian_blur(
    image: Image,
    sigma: Annotated[float, Param(min=0.1, max=200, step=0.5, unit="px")] = 2.0,
) -> Image:
    out = _per_channel(image, lambda a: ndi.gaussian_filter(a.astype(np.float32), sigma))
    return image.with_data(_restore_dtype(out, image.data), name=f"{image.name} (flou σ={sigma:g})")


@operation("Filtre médian", category="Filtres", imagej="Process > Filters > Median...")
def median_filter(
    image: Image,
    radius: Annotated[int, Param(min=1, max=50, unit="px")] = 2,
) -> Image:
    """Médian sur un disque de rayon donné (efficace contre le bruit impulsionnel)."""
    fp = morphology.disk(radius)
    out = _per_channel(image, lambda a: ndi.median_filter(a, footprint=fp))
    return image.with_data(out, name=f"{image.name} (médian r={radius})")


# ------------------------------------------------------------ fond, champ plat
def _shrink_factor(radius: float) -> int:
    # même stratégie qu'ImageJ (BackgroundSubtracter / RollingBall)
    if radius <= 10:
        return 1
    if radius <= 30:
        return 2
    if radius <= 100:
        return 4
    return 8


def rolling_ball_background(
    plane: np.ndarray, radius: float, *, light_background: bool = False, smooth: bool = True
) -> np.ndarray:
    """Fond estimé par « rolling ball » (algorithme de Sternberg, comme ImageJ).

    Pour la vitesse, l'image est réduite (minimum par blocs) avant de faire rouler
    la boule, puis le fond est ré-agrandi par interpolation : c'est exactement la
    stratégie d'ImageJ.
    """
    a = plane.astype(np.float32)
    if light_background:
        a = a.max() - a
    s = _shrink_factor(radius)
    small = a
    if s > 1:
        h, w = a.shape
        hp, wp = -(-h // s) * s, -(-w // s) * s
        padded = np.pad(a, ((0, hp - h), (0, wp - w)), mode="edge")
        small = padded.reshape(hp // s, s, wp // s, s).min(axis=(1, 3))
    if smooth:
        small = ndi.uniform_filter(small, 3, mode="nearest")
    bg_small = restoration.rolling_ball(small, radius=max(radius / s, 1.0))
    bg = bg_small
    if s > 1:
        bg = transform.resize(
            bg_small,
            (small.shape[0] * s, small.shape[1] * s),
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        )[: a.shape[0], : a.shape[1]]
    bg = np.minimum(bg, a)  # le fond ne dépasse jamais l'image
    if light_background:
        bg = plane.astype(np.float32).max() - bg
    return bg.astype(np.float32)


@operation("Soustraction de fond", category="Fond", imagej="Process > Subtract Background... (rolling ball)")
def subtract_background(
    image: Image,
    radius: Annotated[
        float, Param(min=1, max=1000, unit="px", description="plus grand que les objets d'intérêt")
    ] = 50.0,
    light_background: Annotated[
        bool, Param(label="fond clair", description="cocher pour le fond clair")
    ] = False,
    smooth: Annotated[bool, Param(label="lisser", advanced=True)] = True,
) -> Image:
    """Retire un fond lentement variable (éclairage inégal, autofluorescence)."""

    def one(plane: np.ndarray) -> np.ndarray:
        bg = rolling_ball_background(plane, radius, light_background=light_background, smooth=smooth)
        p = plane.astype(np.float32)
        if light_background:
            # ImageJ : objets sombres sur fond clair → le fond devient blanc
            maxval = np.iinfo(plane.dtype).max if np.issubdtype(plane.dtype, np.integer) else p.max()
            return p - bg + maxval
        return p - bg

    out = _per_channel(image, one)
    return image.with_data(_restore_dtype(out, image.data), name=f"{image.name} (fond soustrait)")


def flat_field_correct(raw: np.ndarray, flat: np.ndarray, dark: np.ndarray | None = None) -> np.ndarray:
    """(brut − noir) / (plat − noir) × moyenne(plat − noir), canal par canal."""
    raw_f = raw.astype(np.float32)
    flat_f = flat.astype(np.float32)
    if dark is not None:
        d = dark.astype(np.float32)
        raw_f = raw_f - d
        flat_f = flat_f - d
    axes = (0, 1)
    norm = flat_f.mean(axis=axes, keepdims=True)
    flat_f = np.where(flat_f <= 1e-6, np.nan, flat_f)
    out = raw_f / flat_f * norm
    return np.nan_to_num(out, nan=0.0, posinf=0.0)


@operation("Correction de champ plat", category="Fond", imagej="Process > Image Calculator (divide) / BaSiC")
def flat_field(image: Image, flat: Image, dark: Image | None = None) -> Image:
    """Corrige le vignettage et l'éclairage inégal avec une image de lame vide."""
    out = flat_field_correct(image.data, flat.data, dark.data if dark is not None else None)
    return image.with_data(_restore_dtype(out, image.data), name=f"{image.name} (champ plat)")
