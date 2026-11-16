"""Piles d'images (temps, Z...) : projection, sous-pile, montage, recalage.

Ces opérations reçoivent la pile entière. Les axes de la pile sont les curseurs affichés
sous l'image ; « dernier curseur » est en général Z, « premier curseur » le temps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .. import ij
from ..image import Image, Table
from ..registry import Param, operation

_Axis = Annotated[
    Literal["dernier curseur", "premier curseur"],
    Param(label="axe", description="quel curseur de la pile parcourir (le dernier est en général Z)"),
]
_First = Annotated[int, Param(label="premier plan", min=1, max=100000)]
_Last = Annotated[int, Param(label="dernier plan", min=0, max=100000, description="0 = jusqu'au bout")]


def lead_dims(image: Image) -> int:
    """Nombre d'axes de pile (avant les deux axes de l'image et l'éventuel axe couleur)."""
    return image.data.ndim - (3 if image.is_rgb else 2)


def _axis(image: Image, axis: str) -> int:
    n = lead_dims(image)
    if n < 1:
        raise ValueError("Cette opération demande une pile (plusieurs plans)")
    return n - 1 if axis == "dernier curseur" else 0


def _range(n: int, first: int, last: int, step: int = 1) -> slice:
    last = n if last <= 0 else min(last, n)
    first = min(max(first, 1), last)
    return slice(first - 1, last, max(step, 1))


@operation("Projection", category="Piles", imagej="Image > Stacks > Z Project...", stack=True)
def project(
    image: Image,
    method: Annotated[
        Literal["maximum", "minimum", "moyenne", "somme", "écart-type", "médiane"], Param(label="méthode")
    ] = "maximum",
    axis: _Axis = "dernier curseur",
    first: _First = 1,
    last: _Last = 0,
) -> Image:
    """Résume une pile en une image : intensité maximale (le plus courant), moyenne, somme...

    Comme ImageJ : maximum, minimum et médiane gardent le type de l'image ; moyenne, somme
    et écart-type (n − 1) donnent une image 32 bits."""
    ax = _axis(image, axis)
    sub = np.take(
        image.data, np.arange(image.data.shape[ax])[_range(image.data.shape[ax], first, last)], axis=ax
    )
    if method == "maximum":
        out = sub.max(axis=ax)
    elif method == "minimum":
        out = sub.min(axis=ax)
    elif method == "médiane":
        out = ij.restore_dtype(np.median(sub, axis=ax), sub)
    elif method == "moyenne":
        out = sub.mean(axis=ax, dtype=np.float64).astype(np.float32)
    elif method == "somme":
        out = sub.sum(axis=ax, dtype=np.float64).astype(np.float32)
    else:
        out = (
            sub.std(axis=ax, ddof=1, dtype=np.float64)
            if sub.shape[ax] > 1
            else np.zeros(np.delete(sub.shape, ax))
        )
        out = np.asarray(out, dtype=np.float32)
    if image.is_rgb and out.dtype != image.data.dtype and method in ("moyenne", "somme", "écart-type"):
        out = ij.restore_dtype(out, image.data)  # une image couleur reste en 8 bits par canal
    return image.with_data(out, name=f"{image.name} (projection {method})")


@operation(
    "Sous-pile", category="Piles", imagej="Image > Stacks > Tools > Make Substack... / Reduce...", stack=True
)
def substack(
    image: Image,
    first: _First = 1,
    last: _Last = 0,
    step: Annotated[int, Param(label="pas", min=1, max=10000, description="1 plan sur n")] = 1,
    axis: _Axis = "dernier curseur",
) -> Image:
    """Garde une partie des plans : du premier au dernier, un sur ``pas``."""
    ax = _axis(image, axis)
    keep = np.arange(image.data.shape[ax])[_range(image.data.shape[ax], first, last, step)]
    return image.with_data(np.take(image.data, keep, axis=ax), name=f"{image.name} (sous-pile)")


@operation(
    "Inverser l'ordre de la pile", category="Piles", imagej="Image > Stacks > Tools > Reverse", stack=True
)
def reverse(image: Image, axis: _Axis = "dernier curseur") -> Image:
    """Dernier plan en premier."""
    return image.with_data(
        np.flip(image.data, axis=_axis(image, axis)).copy(), name=f"{image.name} (inversée)"
    )


@operation(
    "Empiler deux images", category="Piles", imagej="Image > Stacks > Tools > Concatenate...", stack=True
)
def concatenate(image1: Image, image2: Image) -> Image:
    """Met bout à bout deux images ou deux piles de même taille (répéter pour en ajouter)."""
    if image1.plane_shape != image2.plane_shape or image1.is_rgb != image2.is_rgb:
        raise ValueError("Les deux images n'ont pas la même taille ou le même type de couleur")
    plane = 3 if image1.is_rgb else 2

    def as_stack(image: Image) -> np.ndarray:
        d = image.data
        lead = d.ndim - plane
        if lead > 1:
            raise ValueError("Empiler demande des images simples ou des piles à un seul curseur")
        return d if lead == 1 else d[None]

    a, b = as_stack(image1), as_stack(image2)
    dtype = np.result_type(a.dtype, b.dtype)
    return image1.with_data(
        np.concatenate([a.astype(dtype), b.astype(dtype)]), name=f"{image1.name} + {image2.name}"
    )


@operation("Montage", category="Piles", imagej="Image > Stacks > Make Montage...", stack=True)
def montage(
    image: Image,
    columns: Annotated[int, Param(label="colonnes", min=0, max=100, description="0 = automatique")] = 0,
    scale: Annotated[float, Param(label="échelle", min=0.05, max=1, step=0.05)] = 1.0,
    border: Annotated[int, Param(label="bordure", min=0, max=50, unit="px")] = 0,
    first: _First = 1,
    last: _Last = 0,
    step: Annotated[int, Param(label="pas", min=1, max=10000)] = 1,
) -> Image:
    """Dispose les plans d'une pile en mosaïque dans une seule image (planche-contact)."""
    lead = lead_dims(image)
    if lead < 1:
        raise ValueError("Le montage demande une pile")
    plane_shape = image.data.shape[lead:]
    planes = image.data.reshape(-1, *plane_shape)[
        _range(int(np.prod(image.data.shape[:lead])), first, last, step)
    ]
    if scale < 1:
        from skimage import transform

        h, w = max(1, int(plane_shape[0] * scale + 0.5)), max(1, int(plane_shape[1] * scale + 0.5))
        planes = np.stack(
            [
                ij.restore_dtype(
                    transform.resize(
                        p.astype(np.float64),
                        (h, w, *plane_shape[2:]),
                        anti_aliasing=True,
                        preserve_range=True,
                    ),
                    image.data,
                )
                for p in planes
            ]
        )
    n, h, w = len(planes), planes.shape[1], planes.shape[2]
    cols = columns or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    out = np.zeros(
        (rows * h + (rows - 1) * border, cols * w + (cols - 1) * border, *planes.shape[3:]), planes.dtype
    )
    if border:
        out[...] = planes.max()
    for i, plane in enumerate(planes):
        r, c = divmod(i, cols)
        out[r * (h + border) : r * (h + border) + h, c * (w + border) : c * (w + border) + w] = plane
    px = image.pixel_size_um / (h / plane_shape[0]) if image.pixel_size_um else None
    return replace(image, data=out, pixel_size_um=px, name=f"{image.name} (montage)")


@operation(
    "Recaler la pile", category="Piles", imagej="Plugins > Registration > StackReg (Translation)", stack=True
)
def register_stack(
    image: Image,
    reference: Annotated[
        Literal["premier plan", "plan précédent"],
        Param(label="référence", description="plan précédent : suit mieux une dérive lente"),
    ] = "plan précédent",
    subpixel: Annotated[bool, Param(label="précision sous-pixel")] = True,
) -> tuple[Image, Table]:
    """Corrige la dérive d'un time-lapse : chaque plan est décalé (translation seule) pour
    se superposer à la référence. Renvoie la pile recalée et le tableau des décalages."""
    from skimage.registration import phase_cross_correlation

    if lead_dims(image) != 1:
        raise ValueError("Le recalage demande une pile à un seul curseur (temps ou Z)")
    stack = image.data
    gray = [ij.intensity_plane(p, image.is_rgb).astype(np.float64) for p in stack]
    out = np.empty_like(stack)
    out[0] = stack[0]
    shifts = [(0.0, 0.0)]
    total = np.zeros(2)
    for i in range(1, len(stack)):
        ref = gray[0] if reference == "premier plan" else gray[i - 1]
        shift, _, _ = phase_cross_correlation(ref, gray[i], upsample_factor=10 if subpixel else 1)
        total = np.asarray(shift, float) if reference == "premier plan" else total + shift
        order = 1 if subpixel else 0
        moved = ij.per_channel(
            stack[i],
            lambda a, t=tuple(total), o=order: ndi.shift(a.astype(np.float64), t, order=o, mode="nearest"),
            rgb=image.is_rgb,
        )
        out[i] = ij.restore_dtype(moved, stack)
        shifts.append((-float(total[1]), -float(total[0])))  # dérive = opposé de la correction
    df = pd.DataFrame(shifts, columns=["dX", "dY"], index=pd.RangeIndex(1, len(shifts) + 1))
    df["dX"] *= image.px
    df["dY"] *= image.px
    table = Table(df, name=f"Dérive — {image.name}", units={"dX": image.unit, "dY": image.unit})
    return image.with_data(out, name=f"{image.name} (recalée)"), table
