"""Seuillage, opérations binaires, séparation des objets."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation

from ..image import Image, Labels
from ..registry import Param, operation
from ..thresholds import METHODS, threshold_value

_Method = Literal["Default", "Otsu", "Li", "Yen", "IsoData", "Mean", "Minimum", "Triangle"]
assert set(_Method.__args__) == set(METHODS)  # type: ignore[attr-defined]


@operation("Seuillage automatique", category="Segmentation", imagej="Image > Adjust > Threshold... (Auto)")
def auto_threshold(
    image: Image,
    method: Annotated[_Method, Param(label="méthode")] = "Default",
    objects: Annotated[
        Literal["claires", "sombres"],
        Param(label="objets", description="sombres : cellules sur fond clair (fond clair)"),
    ] = "claires",
) -> Labels:
    """Masque binaire des objets (mêmes méthodes et mêmes noms que dans ImageJ)."""
    values = image.gray()
    if image.data.dtype == np.uint8:
        # comme ImageJ : une image 8 bits (ou RGB convertie en 8 bits) garde ses 256 niveaux
        values = image.data if not image.is_rgb else np.round(values).astype(np.uint8)
    t = threshold_value(values, method)
    mask = values >= t if objects == "claires" else values < t
    return Labels(
        mask.astype(np.uint8),
        image.pixel_size_um,
        f"{image.name} (masque {method})",
        {"threshold": float(t), "method": method, "objects": objects},
    )


@operation("Seuil manuel", category="Segmentation", imagej="Image > Adjust > Threshold...")
def manual_threshold(
    image: Image,
    lower: Annotated[float, Param(label="minimum")] = 0.0,
    upper: Annotated[float, Param(label="maximum")] = 255.0,
) -> Labels:
    gray = image.gray()
    mask = (gray >= lower) & (gray <= upper)
    return Labels(mask.astype(np.uint8), image.pixel_size_um, f"{image.name} (masque)")


def _as_mask(labels: Labels) -> np.ndarray:
    return labels.data > 0


@operation("Remplir les trous", category="Binaire", imagej="Process > Binary > Fill Holes")
def fill_holes(mask: Labels) -> Labels:
    out = ndi.binary_fill_holes(_as_mask(mask))
    return Labels(out.astype(np.uint8), mask.pixel_size_um, f"{mask.name} (rempli)")


_SQUARE = np.ones((3, 3), bool)  # voisinage 3×3 des opérations binaires d'ImageJ


@operation("Ouverture", category="Binaire", imagej="Process > Binary > Open")
def binary_open(
    mask: Labels, iterations: Annotated[int, Param(label="itérations", min=1, max=50)] = 1
) -> Labels:
    """Érosion puis dilatation (3×3, répétées) : supprime les petits détails clairs.
    Comme ImageJ par défaut, le bord de l'image compte comme du fond."""
    out = ndi.binary_opening(_as_mask(mask), structure=_SQUARE, iterations=iterations)
    return Labels(out.astype(np.uint8), mask.pixel_size_um, f"{mask.name} (ouvert)")


@operation("Fermeture", category="Binaire", imagej="Process > Binary > Close-")
def binary_close(
    mask: Labels, iterations: Annotated[int, Param(label="itérations", min=1, max=50)] = 1
) -> Labels:
    """Dilatation puis érosion (3×3, répétées) : bouche les petits trous et fissures."""
    out = ndi.binary_closing(_as_mask(mask), structure=_SQUARE, iterations=iterations)
    return Labels(out.astype(np.uint8), mask.pixel_size_um, f"{mask.name} (fermé)")


@operation("Retirer les petits objets", category="Binaire")
def remove_small(
    mask: Labels,
    min_area: Annotated[float, Param(min=0, unit="µm² (ou px²)")] = 10.0,
) -> Labels:
    """Supprime les objets dont la surface est inférieure à ``min_area``."""
    lab, _ = ndi.label(_as_mask(mask), structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel()) * mask.px**2
    keep = areas >= min_area
    keep[0] = False
    out = keep[lab]
    return Labels(out.astype(np.uint8), mask.pixel_size_um, f"{mask.name} (nettoyé)")


@operation(
    "Étiqueter les objets", category="Segmentation", imagej="Analyze > Analyze Particles... (Count Masks)"
)
def label_objects(
    mask: Labels,
    connectivity: Annotated[Literal["8", "4"], Param(label="connexité", advanced=True)] = "8",
) -> Labels:
    """Donne un numéro à chaque objet connexe (8-connexité comme ImageJ)."""
    structure = np.ones((3, 3)) if connectivity == "8" else None
    lab, _ = ndi.label(_as_mask(mask), structure=structure)
    return Labels(lab.astype(np.int32), mask.pixel_size_um, f"{mask.name} (étiquettes)")


@operation("Séparer les objets (watershed)", category="Segmentation", imagej="Process > Binary > Watershed")
def watershed_split(
    mask: Labels,
    tolerance: Annotated[
        float, Param(min=0.1, max=20, step=0.1, unit="px", description="ImageJ utilise 0.5")
    ] = 0.5,
) -> Labels:
    """Sépare les objets qui se touchent, à partir de la carte des distances (EDM),
    comme le « Watershed » binaire d'ImageJ. Renvoie une image d'étiquettes."""
    m = _as_mask(mask)
    dist = ndi.distance_transform_edt(m)
    peaks = morphology.h_maxima(dist, tolerance) & m
    markers, _ = ndi.label(peaks, structure=np.ones((3, 3)))
    lab = segmentation.watershed(-dist, markers, mask=m, watershed_line=True)
    return Labels(lab.astype(np.int32), mask.pixel_size_um, f"{mask.name} (séparé)")
