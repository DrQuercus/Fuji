"""Conversions, filtres, soustraction de fond, correction de champ plat."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi

from .. import ij
from ..image import Image
from ..registry import Param, operation


def _per_channel(image: Image, fn) -> np.ndarray:  # type: ignore[no-untyped-def]
    return ij.per_channel(image.data, fn, rgb=image.is_rgb)


_restore_dtype = ij.restore_dtype
imagej_disk = ij.disk


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


@operation("Séparer les canaux", category="Couleur", imagej="Image > Color > Split Channels")
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
    """Négatif de l'image. Comme ImageJ : 255 − valeur en 8 bits (et RGB) ; sinon
    retournement entre le minimum et le maximum réels de l'image."""
    d = image.data
    if d.dtype == np.uint8:
        out = 255 - d
    else:
        lo, hi = d.min(), d.max()
        out = (hi.astype(np.float64) + lo - d).astype(d.dtype)
    return image.with_data(out, name=f"{image.name} (inversée)")


# ------------------------------------------------------------------- filtres
@operation("Flou gaussien", category="Filtres", imagej="Process > Filters > Gaussian Blur...")
def gaussian_blur(
    image: Image,
    sigma: Annotated[float, Param(min=0.1, max=200, step=0.5, unit="px")] = 2.0,
) -> Image:
    """Flou gaussien avec le noyau d'ImageJ ; hors de l'image, le pixel du bord est répété."""
    out = _per_channel(image, lambda a: ij.gaussian_blur(a, sigma))
    return image.with_data(_restore_dtype(out, image.data), name=f"{image.name} (flou σ={sigma:g})")


@operation("Filtre médian", category="Filtres", imagej="Process > Filters > Median...")
def median_filter(
    image: Image,
    radius: Annotated[float, Param(min=0.5, max=50, step=0.5, unit="px")] = 2.0,
) -> Image:
    """Médian sur le masque circulaire d'ImageJ (efficace contre le bruit impulsionnel)."""
    fp = imagej_disk(radius)
    out = _per_channel(image, lambda a: ndi.median_filter(a, footprint=fp, mode="nearest"))
    return image.with_data(out, name=f"{image.name} (médian r={radius:g})")


# ------------------------------------------------------------ fond, champ plat
def _ball(radius: float) -> tuple[int, np.ndarray]:
    """Facteur de réduction et calotte de la boule (``RollingBall`` d'ImageJ)."""
    if radius <= 10:
        shrink, trim = 1, 24
    elif radius <= 30:
        shrink, trim = 2, 24
    elif radius <= 100:
        shrink, trim = 4, 32
    else:
        shrink, trim = 8, 40
    small_radius = max(radius / shrink, 1.0)
    xtrim = int(trim * small_radius) // 100  # seule une calotte de la boule est utilisée
    half = int(np.floor(small_radius - xtrim + 0.5))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    height2 = small_radius * small_radius - xx * xx - yy * yy
    return shrink, np.sqrt(np.maximum(height2, 0.0)).astype(np.float32)


def _shrink(a: np.ndarray, s: int) -> np.ndarray:
    """Minimum par blocs s×s (blocs partiels au bord)."""
    h, w = a.shape
    hp, wp = -(-h // s) * s, -(-w // s) * s
    padded = np.pad(a, ((0, hp - h), (0, wp - w)), mode="edge")
    return padded.reshape(hp // s, s, wp // s, s).min(axis=(1, 3))


def _enlarge(small: np.ndarray, shape: tuple[int, int], s: int) -> np.ndarray:
    """Interpolation bilinéaire entre centres de blocs (``enlargeImage`` d'ImageJ)."""

    def arrays(length: int, small_length: int) -> tuple[np.ndarray, np.ndarray]:
        i = np.arange(length)
        idx = np.trunc((i - s // 2) / s).astype(np.int64)
        idx = np.clip(idx, 0, max(small_length - 2, 0))
        weight = 1.0 - ((i + 0.5) / s - (idx + 0.5))
        return idx, weight.astype(np.float32)

    if small.shape[0] < 2 or small.shape[1] < 2:  # image minuscule : pas d'interpolation
        return np.repeat(np.repeat(small, s, axis=0), s, axis=1)[: shape[0], : shape[1]]
    yi, yw = arrays(shape[0], small.shape[0])
    xi, xw = arrays(shape[1], small.shape[1])
    rows = small[:, xi] * xw + small[:, xi + 1] * (1.0 - xw)
    return rows[yi] * yw[:, None] + rows[yi + 1] * (1.0 - yw[:, None])


def _roll_ball(a: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """Surface balayée par la boule roulant sous l'image (``rollBall`` d'ImageJ).

    C'est une ouverture morphologique par la calotte : la boule est d'abord remontée
    jusqu'à toucher l'image (érosion), puis chaque pixel prend le point le plus haut
    atteint par la boule (dilatation). Les centres hors de l'image comptent aussi.
    """
    r = ball.shape[0] // 2
    padded = np.pad(a, r, mode="constant", constant_values=np.inf)
    lowest = ndi.grey_erosion(padded, structure=ball, mode="constant", cval=np.inf)
    lowest[~np.isfinite(lowest)] = -np.inf  # centres dont la calotte ne touche pas l'image
    surface = ndi.grey_dilation(lowest, structure=ball, mode="constant", cval=-np.inf)
    return surface[r:-r, r:-r] if r else surface


def rolling_ball_background(
    plane: np.ndarray, radius: float, *, light_background: bool = False, smooth: bool = True
) -> np.ndarray:
    """Fond estimé par « rolling ball » : transcription de ``BackgroundSubtracter`` d'ImageJ.

    Lissage 3×3, réduction de l'image (minimum par blocs), boule roulée sous l'image
    réduite, ré-agrandissement par interpolation. Avec lissage, le fond ne dépasse
    jamais l'image d'origine.
    """
    original = plane.astype(np.float32)
    a = -original if light_background else original.copy()
    if smooth:
        a = ndi.uniform_filter(a, 3, mode="nearest")
    shrink, ball = _ball(radius)
    small = _shrink(a, shrink) if shrink > 1 else a
    bg = _roll_ball(small, ball)
    if shrink > 1:
        bg = _enlarge(bg, a.shape, shrink)
    if light_background:
        bg = -bg
    if smooth:
        bg = np.maximum(bg, original) if light_background else np.minimum(bg, original)
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
    output: Annotated[
        Literal["image corrigée", "fond estimé"],
        Param(
            label="résultat",
            description="« fond estimé » montre ce qui serait retiré : pour vérifier le rayon",
        ),
    ] = "image corrigée",
) -> Image:
    """Retire un fond lentement variable (éclairage inégal, autofluorescence).

    Même algorithme qu'ImageJ. Les images couleur sont traitées canal par canal (option
    « Separate colors » d'ImageJ). Avec « fond clair », le fond devient blanc : 255 en
    8 bits, 65535 en 16 bits, comme dans ImageJ. « Fond estimé » correspond à la case
    « Create background » d'ImageJ.
    """

    def one(plane: np.ndarray) -> np.ndarray:
        bg = rolling_ball_background(plane, radius, light_background=light_background, smooth=smooth)
        if output == "fond estimé":
            return bg
        out = plane.astype(np.float32) - bg
        if light_background and np.issubdtype(plane.dtype, np.integer):
            out += np.iinfo(plane.dtype).max
        return out

    out = _per_channel(image, one)
    suffix = "fond estimé" if output == "fond estimé" else "fond soustrait"
    return image.with_data(_restore_dtype(out, image.data), name=f"{image.name} ({suffix})")


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
    for name, ref in (("champ plat", flat), ("noir", dark)):
        if ref is not None and ref.data.shape != image.data.shape:
            raise ValueError(
                f"L'image de {name} ({ref.data.shape}) n'a pas la taille de l'image à corriger "
                f"{image.data.shape} : même caméra, même résolution, même objectif."
            )
    out = flat_field_correct(image.data, flat.data, dark.data if dark is not None else None)
    return image.with_data(_restore_dtype(out, image.data), name=f"{image.name} (champ plat)")
