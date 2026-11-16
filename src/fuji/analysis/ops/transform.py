"""Géométrie : retourner, tourner, translater, redimensionner, regrouper, rogner."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi

from .. import ij
from ..image import Image, Rois
from ..registry import Param, operation

_Interp = Literal["bilinéaire", "bicubique", "aucune"]
_ORDER = {"aucune": 0, "bilinéaire": 1, "bicubique": 3}


def _planes(image: Image, fn) -> np.ndarray:  # type: ignore[no-untyped-def]
    return ij.per_channel(image.data, fn, rgb=image.is_rgb)


@operation("Retourner", category="Géométrie", imagej="Image > Transform > Flip Horizontally / Vertically")
def flip(
    image: Image,
    direction: Annotated[
        Literal["horizontalement", "verticalement"], Param(label="sens")
    ] = "horizontalement",
) -> Image:
    """Image miroir : gauche ↔ droite (horizontalement) ou haut ↔ bas (verticalement)."""
    out = image.data[:, ::-1] if direction == "horizontalement" else image.data[::-1]
    return image.with_data(np.ascontiguousarray(out), name=f"{image.name} (retournée)")


@operation(
    "Tourner de 90°", category="Géométrie", imagej="Image > Transform > Rotate 90 Degrees Right / Left"
)
def rotate90(
    image: Image, direction: Annotated[Literal["à droite", "à gauche"], Param(label="sens")] = "à droite"
) -> Image:
    """Quart de tour sans perte (aucune interpolation)."""
    out = np.rot90(image.data, k=-1 if direction == "à droite" else 1, axes=(0, 1))
    return image.with_data(np.ascontiguousarray(out), name=f"{image.name} (tournée)")


@operation("Tourner", category="Géométrie", imagej="Image > Transform > Rotate...")
def rotate(
    image: Image,
    angle: Annotated[
        float, Param(min=-360, max=360, step=1, unit="°", description="positif = sens horaire")
    ] = 15.0,
    interpolation: _Interp = "bilinéaire",
    enlarge: Annotated[
        bool,
        Param(label="agrandir le cadre", description="pour ne rien couper ; sinon l'image garde sa taille"),
    ] = False,
) -> Image:
    """Rotation autour du centre de l'image ; les coins vides sont mis à zéro."""
    order = _ORDER[interpolation]

    def one(a: np.ndarray) -> np.ndarray:
        out = ndi.rotate(
            a.astype(np.float64), -angle, reshape=enlarge, order=order, mode="constant", cval=0.0
        )
        return ij.restore_dtype(out, a)

    return image.with_data(_planes(image, one), name=f"{image.name} (tournée {angle:g}°)")


@operation("Translater", category="Géométrie", imagej="Image > Transform > Translate...")
def translate(
    image: Image,
    dx: Annotated[
        float, Param(label="décalage X", min=-1e5, unit="px", description="positif = vers la droite")
    ] = 10.0,
    dy: Annotated[
        float, Param(label="décalage Y", min=-1e5, unit="px", description="positif = vers le bas")
    ] = 0.0,
    interpolation: _Interp = "aucune",
) -> Image:
    """Décale l'image ; la zone découverte est mise à zéro."""
    whole = float(dx).is_integer() and float(dy).is_integer()
    order = 0 if whole else _ORDER[interpolation]

    def one(a: np.ndarray) -> np.ndarray:
        out = ndi.shift(a.astype(np.float64), (dy, dx), order=order, mode="constant", cval=0.0)
        return ij.restore_dtype(out, a)

    return image.with_data(_planes(image, one), name=f"{image.name} (translatée)")


@operation("Redimensionner", category="Géométrie", imagej="Image > Scale... / Image > Adjust > Size...")
def resize(
    image: Image,
    factor: Annotated[float, Param(label="facteur", min=0.01, max=50, step=0.25)] = 0.5,
    interpolation: _Interp = "bilinéaire",
    average: Annotated[
        bool,
        Param(label="moyenner en réduisant", description="évite le crénelage quand l'image est réduite"),
    ] = True,
) -> Image:
    """Agrandit ou réduit l'image. La taille du pixel est mise à jour : les mesures en µm
    restent justes."""
    from skimage import transform

    h, w = image.plane_shape
    shape = (max(1, int(h * factor + 0.5)), max(1, int(w * factor + 0.5)))

    def one(a: np.ndarray) -> np.ndarray:
        out = transform.resize(
            a.astype(np.float64),
            shape,
            order=_ORDER[interpolation],
            mode="edge",
            anti_aliasing=average and factor < 1,
            preserve_range=True,
        )
        return ij.restore_dtype(out, a)

    px = image.pixel_size_um * h / shape[0] if image.pixel_size_um else None
    return replace(image, data=_planes(image, one), pixel_size_um=px, name=f"{image.name} (×{factor:g})")


@operation("Regrouper les pixels (bin)", category="Géométrie", imagej="Image > Transform > Bin...")
def bin_pixels(
    image: Image,
    x: Annotated[int, Param(label="facteur X", min=1, max=64)] = 2,
    y: Annotated[int, Param(label="facteur Y", min=1, max=64)] = 2,
    method: Annotated[
        Literal["moyenne", "somme", "médiane", "maximum", "minimum"], Param(label="méthode")
    ] = "moyenne",
) -> Image:
    """Réduit l'image en regroupant les pixels par blocs (les pixels en trop au bord sont
    ignorés, comme dans ImageJ). La somme augmente le signal au prix de la résolution."""
    fn = {"moyenne": np.mean, "somme": np.sum, "médiane": np.median, "maximum": np.max, "minimum": np.min}[
        method
    ]

    def one(a: np.ndarray) -> np.ndarray:
        h, w = (a.shape[0] // y) * y, (a.shape[1] // x) * x
        if h == 0 or w == 0:
            raise ValueError("Le facteur de regroupement dépasse la taille de l'image")
        blocks = a[:h, :w].astype(np.float64).reshape(h // y, y, w // x, x)
        return ij.restore_dtype(fn(blocks, axis=(1, 3)), a)

    px = image.pixel_size_um * x if image.pixel_size_um and x == y else image.pixel_size_um
    return replace(image, data=_planes(image, one), pixel_size_um=px, name=f"{image.name} (bin {x}×{y})")


def roi_bounds(rois: Rois, index: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Rectangle englobant (ligne, colonne, hauteur, largeur) d'une ROI, limité à l'image."""
    if rois.is_line(index):
        v = rois.vertices(index)
        r0, c0 = int(np.floor(v[:, 0].min() + 0.5)), int(np.floor(v[:, 1].min() + 0.5))
        r1, c1 = int(np.floor(v[:, 0].max() + 0.5)), int(np.floor(v[:, 1].max() + 0.5))
    else:
        mask = rois.mask(index, shape)
        if not mask.any():
            raise ValueError("La ROI est en dehors de l'image")
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0] - 1), min(c1, shape[1] - 1)
    if r1 < r0 or c1 < c0:
        raise ValueError("La ROI est en dehors de l'image")
    return r0, c0, r1 - r0 + 1, c1 - c0 + 1


@operation("Rogner", category="Géométrie", imagej="Image > Crop")
def crop(
    image: Image,
    roi: Rois | None = None,
    x: Annotated[int, Param(label="X", min=0, unit="px", description="utilisé s'il n'y a pas de ROI")] = 0,
    y: Annotated[int, Param(label="Y", min=0, unit="px")] = 0,
    width: Annotated[int, Param(label="largeur", min=0, unit="px", description="0 = jusqu'au bord")] = 0,
    height: Annotated[int, Param(label="hauteur", min=0, unit="px", description="0 = jusqu'au bord")] = 0,
) -> Image:
    """Garde le rectangle englobant de la première ROI, ou le rectangle donné en pixels."""
    h, w = image.plane_shape
    if roi is not None and len(roi):
        r0, c0, rh, cw = roi_bounds(roi, 0, (h, w))
    else:
        r0, c0 = min(y, h - 1), min(x, w - 1)
        rh = (height or h) if r0 + (height or h) <= h else h - r0
        cw = (width or w) if c0 + (width or w) <= w else w - c0
    out = np.ascontiguousarray(image.data[r0 : r0 + rh, c0 : c0 + cw])
    return image.with_data(out, name=f"{image.name} (rognée)", crop_origin=(int(c0), int(r0)))


@operation("Taille du support", category="Géométrie", imagej="Image > Adjust > Canvas Size...")
def canvas_size(
    image: Image,
    width: Annotated[int, Param(label="largeur", min=1, max=100000, unit="px")] = 1024,
    height: Annotated[int, Param(label="hauteur", min=1, max=100000, unit="px")] = 1024,
    position: Annotated[Literal["centre", "haut-gauche"], Param(label="position de l'image")] = "centre",
) -> Image:
    """Agrandit (bords à zéro) ou réduit le cadre sans changer l'échelle de l'image."""
    h, w = image.plane_shape
    top = (height - h) // 2 if position == "centre" else 0
    left = (width - w) // 2 if position == "centre" else 0
    out = np.zeros((height, width, *image.data.shape[2:]), image.data.dtype)
    src_r, dst_r = max(-top, 0), max(top, 0)
    src_c, dst_c = max(-left, 0), max(left, 0)
    nh, nw = min(h - src_r, height - dst_r), min(w - src_c, width - dst_c)
    if nh > 0 and nw > 0:
        out[dst_r : dst_r + nh, dst_c : dst_c + nw] = image.data[src_r : src_r + nh, src_c : src_c + nw]
    return image.with_data(out, name=f"{image.name} (support {width}×{height})")
