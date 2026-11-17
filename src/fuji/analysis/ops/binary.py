"""Opérations binaires d'ImageJ : érosion, dilatation, contour, squelette, distances."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi

from ..image import Image, Labels
from ..maxima import find_maxima
from ..registry import Param, operation

_EIGHT = np.ones((3, 3), bool)
_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int32)
_Iterations = Annotated[int, Param(label="nombre de passes", min=1, max=100)]
_Count = Annotated[
    int,
    Param(
        label="voisins requis",
        min=1,
        max=8,
        description="nombre de voisins de l'autre sorte pour changer un pixel (« count » d'ImageJ)",
        advanced=True,
    ),
]


def _mask(labels: Labels) -> np.ndarray:
    return labels.data > 0


def _out(mask: np.ndarray, like: Labels, suffix: str) -> Labels:
    return Labels(mask.astype(np.uint8), like.pixel_size_um, f"{like.name} ({suffix})")


def erode(mask: np.ndarray, iterations: int = 1, count: int = 1, pad_edges: bool = False) -> np.ndarray:
    """Érosion 3×3 d'ImageJ : un pixel d'objet disparaît s'il a au moins ``count`` voisins
    de fond. Hors de l'image c'est du fond, sauf avec ``pad_edges``."""
    m = mask.copy()
    for _ in range(iterations):
        background = ndi.correlate(
            (~m).astype(np.int32), _NEIGHBOURS, mode="constant", cval=0 if pad_edges else 1
        )
        m &= background < count
    return m


def dilate(mask: np.ndarray, iterations: int = 1, count: int = 1) -> np.ndarray:
    """Dilatation 3×3 d'ImageJ : un pixel de fond devient objet s'il a au moins ``count``
    voisins d'objet."""
    m = mask.copy()
    for _ in range(iterations):
        m |= ndi.correlate(m.astype(np.int32), _NEIGHBOURS, mode="constant", cval=0) >= count
    return m


@operation("Érosion", category="Binaire", imagej="Process > Binary > Erode")
def erode_op(
    mask: Labels,
    iterations: _Iterations = 1,
    count: _Count = 1,
    pad_edges: Annotated[
        bool,
        Param(
            label="protéger les bords", description="ne pas éroder depuis le bord de l'image", advanced=True
        ),
    ] = False,
) -> Labels:
    """Retire une couche de pixels autour des objets (voisinage 3×3)."""
    return _out(erode(_mask(mask), iterations, count, pad_edges), mask, "érodé")


@operation("Dilatation", category="Binaire", imagej="Process > Binary > Dilate")
def dilate_op(mask: Labels, iterations: _Iterations = 1, count: _Count = 1) -> Labels:
    """Ajoute une couche de pixels autour des objets (voisinage 3×3)."""
    return _out(dilate(_mask(mask), iterations, count), mask, "dilaté")


@operation("Contour des objets", category="Binaire", imagej="Process > Binary > Outline")
def outline(mask: Labels) -> Labels:
    """Ne garde que le bord des objets, épais d'un pixel."""
    m = _mask(mask)
    return _out(m & ~ndi.binary_erosion(m, structure=_EIGHT, border_value=0), mask, "contour")


@operation("Squelette", category="Binaire", imagej="Process > Binary > Skeletonize")
def skeletonize(mask: Labels) -> Labels:
    """Amincit les objets jusqu'à une ligne d'un pixel d'épaisseur.

    L'algorithme (Zhang–Suen) n'est pas la table d'amincissement d'ImageJ : les deux
    squelettes ont la même forme mais peuvent différer de quelques pixels."""
    from skimage import morphology

    return _out(morphology.skeletonize(_mask(mask)), mask, "squelette")


@operation("Carte des distances", category="Binaire", imagej="Process > Binary > Distance Map")
def distance_map(
    mask: Labels,
    output: Annotated[Literal["32 bits", "8 bits (comme ImageJ)"], Param(label="sortie")] = "32 bits",
    calibrated: Annotated[
        bool, Param(label="en µm", description="distances en µm si l'image est calibrée (32 bits seulement)")
    ] = False,
) -> Image:
    """Distance de chaque pixel d'objet au fond le plus proche (carte euclidienne, EDM).

    En 8 bits, les distances sont arrondies et plafonnées à 255 pixels, comme le réglage
    par défaut d'ImageJ."""
    edm = ndi.distance_transform_edt(_mask(mask))
    if output.startswith("8"):
        data: np.ndarray = np.minimum(np.floor(edm + 0.5), 255).astype(np.uint8)
    else:
        data = (edm * (mask.px if calibrated else 1.0)).astype(np.float32)
    return Image(data, mask.pixel_size_um, f"{mask.name} (distances)")


@operation("Points ultimes", category="Binaire", imagej="Process > Binary > Ultimate Points")
def ultimate_points(mask: Labels) -> Image:
    """Centre de chaque objet (ou de chaque lobe d'objets accolés) : maxima de la carte des
    distances. La valeur du point est le rayon du plus grand cercle inscrit, en pixels."""
    edm = ndi.distance_transform_edt(_mask(mask))
    positions, values, _ = find_maxima(edm, 0.5)
    out = np.zeros(edm.shape, np.uint8)
    if len(positions):
        out[positions[:, 0], positions[:, 1]] = np.minimum(np.floor(values + 0.5), 255).astype(np.uint8)
    return Image(out, mask.pixel_size_um, f"{mask.name} (points ultimes)")


@operation("Voronoï", category="Binaire", imagej="Process > Binary > Voronoi")
def voronoi(mask: Labels) -> Labels:
    """Lignes à égale distance des deux objets les plus proches : découpe l'image en zones
    d'influence, une par objet."""
    from skimage import segmentation

    m = _mask(mask)
    markers, n = ndi.label(m, structure=_EIGHT)
    if n < 2:
        return _out(np.zeros(m.shape, bool), mask, "Voronoï")
    zones = segmentation.watershed(ndi.distance_transform_edt(~m), markers, watershed_line=True)
    return _out(zones == 0, mask, "Voronoï")


@operation("Inverser le masque", category="Binaire", imagej="Edit > Invert (image binaire)")
def invert_mask(mask: Labels) -> Labels:
    """Échange objets et fond."""
    return _out(~_mask(mask), mask, "inversé")


@operation("Combiner deux masques", category="Binaire", imagej="Process > Image Calculator (AND / OR / XOR)")
def combine_masks(
    mask1: Labels,
    mask2: Labels,
    operation_: Annotated[
        Literal["ET (intersection)", "OU (union)", "OU exclusif", "1 sans 2"], Param(label="opération")
    ] = "ET (intersection)",
) -> Labels:
    """Intersection, union ou différence de deux masques de même taille."""
    a, b = _mask(mask1), _mask(mask2)
    if a.shape != b.shape:
        raise ValueError("Les deux masques n'ont pas la même taille")
    out = {
        "ET (intersection)": a & b,
        "OU (union)": a | b,
        "OU exclusif": a ^ b,
        "1 sans 2": a & ~b,
    }[operation_]
    return _out(out, mask1, operation_.split(" ")[0])
