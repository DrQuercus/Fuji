"""Calculs : opérations mathématiques sur une image, calculs entre deux images."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np

from .. import ij
from ..image import Image
from ..registry import Param, operation

_MathOp = Literal[
    "ajouter",
    "soustraire",
    "multiplier",
    "diviser",
    "minimum",
    "maximum",
    "gamma",
    "fixer",
    "logarithme",
    "exponentielle",
    "carré",
    "racine carrée",
    "valeur absolue",
    "inverse (1/x)",
]


def math_plane(a: np.ndarray, op: str, value: float) -> np.ndarray:
    """``Process > Math`` sur un plan, avec les règles d'ImageJ pour chaque type.

    8 bits : logarithme, exponentielle et gamma sont étalés sur 0–255 (``log(v)·255/log 255``,
    ``exp(v·log 255/255)``, ``255·(v/255)^γ``). 16 bits : même principe avec le minimum et
    le maximum de l'image. 32 bits : les fonctions mathématiques telles quelles.
    """
    v = a.astype(np.float64)
    simple = {
        "ajouter": lambda: v + value,
        "soustraire": lambda: v - value,
        "multiplier": lambda: v * value,
        "minimum": lambda: np.maximum(v, value),  # « Min... » : rien en dessous de la valeur
        "maximum": lambda: np.minimum(v, value),  # « Max... » : rien au-dessus
        "fixer": lambda: np.full_like(v, value),
        "carré": lambda: v * v,
        "valeur absolue": lambda: np.abs(v),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        if op in simple:
            out = simple[op]()
        elif op == "diviser":
            if value == 0 and a.dtype.kind != "f":
                return a.copy()  # ImageJ ignore la division par zéro des images entières
            out = v / value
        elif op == "racine carrée":
            out = np.sqrt(np.maximum(v, 0.0)) if a.dtype.kind != "f" else np.where(v <= 0, 0.0, np.sqrt(v))
        elif op == "inverse (1/x)":
            out = 1.0 / v
        elif a.dtype == np.uint8:
            if op == "gamma":
                out = np.exp(np.log(v / 255.0) * value) * 255.0
            elif op == "logarithme":
                out = np.log(v) * (255.0 / np.log(255.0))
            else:
                out = np.exp(v * (np.log(255.0) / 255.0))
            out = np.where(np.isfinite(out), out, 0.0)
        elif a.dtype.kind in "ui":
            lo, hi = float(a.min()), float(a.max())
            if op == "gamma":
                span = hi - lo
                safe = np.where(v > lo, (v - lo) / span if span > 0 else 1.0, 1.0)
                out = np.where((span <= 0) | (v == lo), v, np.exp(value * np.log(safe)) * span + lo)
            elif op == "logarithme":
                out = np.where(
                    v <= 0, 0.0, np.log(np.maximum(v, 1e-300)) * (hi / np.log(hi) if hi > 1 else 0.0)
                )
            else:
                out = np.exp(v * (np.log(hi) / hi)) if hi > 1 else v
        else:
            if op == "gamma":
                out = np.where(v <= 0, 0.0, np.exp(value * np.log(np.where(v > 0, v, 1.0))))
            elif op == "logarithme":
                out = np.log(v)
            else:
                out = np.exp(v)
    return ij.restore_dtype(out, a)


@operation("Mathématiques", category="Calcul", imagej="Process > Math > Add / Multiply / Log / Gamma...")
def math_op(
    image: Image,
    operation_: Annotated[_MathOp, Param(label="opération")] = "multiplier",
    value: Annotated[
        float, Param(label="valeur", min=-1e12, description="inutile pour log, exp, carré...")
    ] = 1.0,
) -> Image:
    """Applique une opération à chaque pixel. Le résultat est arrondi et borné au type de
    l'image (0–255 en 8 bits) : pour garder les valeurs exactes, convertir d'abord en 32 bits."""
    out = ij.per_channel(image.data, lambda a: math_plane(a, operation_, value), rgb=image.is_rgb)
    return image.with_data(out, name=f"{image.name} ({operation_})")


_CalcOp = Literal[
    "additionner",
    "soustraire",
    "multiplier",
    "diviser",
    "moyenne",
    "différence",
    "minimum",
    "maximum",
    "ET",
    "OU",
    "OU exclusif",
]


def calculate(a: np.ndarray, b: np.ndarray, op: str, float_result: bool) -> np.ndarray:
    """``Process > Image Calculator`` sur deux plans (résultat du type de la première image)."""
    if op in ("ET", "OU", "OU exclusif"):
        if a.dtype.kind == "f" or float_result:
            raise ValueError("Les opérations logiques demandent des images entières")
        other = ij.restore_dtype(b.astype(np.float64), a)
        fn = {"ET": np.bitwise_and, "OU": np.bitwise_or, "OU exclusif": np.bitwise_xor}[op]
        return fn(a, other)
    x, y = a.astype(np.float64), b.astype(np.float64)
    integer = a.dtype.kind in "ui" and not float_result
    with np.errstate(divide="ignore", invalid="ignore"):
        if op == "additionner":
            out = x + y
        elif op == "soustraire":
            out = x - y
        elif op == "multiplier":
            out = x * y
        elif op == "moyenne":
            out = np.floor((x + y) / 2.0) if integer else (x + y) / 2.0
        elif op == "différence":
            out = np.abs(x - y)
        elif op == "minimum":
            out = np.minimum(x, y)
        elif op == "maximum":
            out = np.maximum(x, y)
        elif integer:  # division entière ; par zéro : valeur maximale du type
            out = np.where(y == 0, np.iinfo(a.dtype).max, np.floor(x / np.where(y == 0, 1.0, y)))
        else:
            out = x / y  # par zéro : infini (ou NaN pour 0/0), réglage par défaut d'ImageJ
    if float_result or a.dtype.kind == "f":
        return out.astype(np.float32)
    info = np.iinfo(a.dtype)
    return np.clip(out, info.min, info.max).astype(a.dtype)


@operation("Calculatrice d'images", category="Calcul", imagej="Process > Image Calculator...")
def image_calculator(
    image1: Image,
    image2: Image,
    operation_: Annotated[_CalcOp, Param(label="opération")] = "soustraire",
    float_result: Annotated[
        bool,
        Param(label="résultat 32 bits", description="évite la saturation et garde les valeurs négatives"),
    ] = False,
) -> Image:
    """Combine deux images de même taille, pixel par pixel (image 1 ∘ image 2).

    Sans « résultat 32 bits », le résultat a le type de la première image et sature aux
    bornes de ce type (une soustraction ne descend pas sous 0), comme dans ImageJ."""
    a, b = image1.data, image2.data
    if image1.plane_shape != image2.plane_shape:
        raise ValueError(f"Les deux images n'ont pas la même taille : {a.shape} et {b.shape}")
    if image1.is_rgb != image2.is_rgb:  # gris ∘ couleur : le gris s'applique à chaque canal
        if image1.is_rgb:
            b = np.repeat(b[..., None], a.shape[-1], axis=-1)
        else:
            a = np.repeat(a[..., None], b.shape[-1], axis=-1)
    out = calculate(a, b, operation_, float_result)
    return image1.with_data(out, name=f"{image1.name} {operation_} {image2.name}")
