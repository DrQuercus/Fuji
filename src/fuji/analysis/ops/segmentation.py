"""Seuillage, opérations binaires, séparation des objets."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation

from .. import ij
from ..image import Image, Labels
from ..registry import Param, operation
from ..thresholds import LOCAL_METHODS, METHODS, local_threshold, threshold_range

_Method = Literal[
    "Default",
    "Huang",
    "Intermodes",
    "IsoData",
    "IJ_IsoData",
    "Li",
    "MaxEntropy",
    "Mean",
    "MinError",
    "Minimum",
    "Moments",
    "Otsu",
    "Percentile",
    "RenyiEntropy",
    "Shanbhag",
    "Triangle",
    "Yen",
]
assert _Method.__args__ == METHODS  # type: ignore[attr-defined]
_LocalMethod = Literal["Bernsen", "Contrast", "Mean", "Median", "MidGrey", "Niblack", "Phansalkar", "Sauvola"]
assert _LocalMethod.__args__ == LOCAL_METHODS  # type: ignore[attr-defined]


def _gray_values(image: Image) -> np.ndarray:
    """Valeurs seuillées : l'image, ou son gris 8 bits si elle est en couleur (comme ImageJ)."""
    return ij.intensity_plane(image.data, image.is_rgb)


@operation("Seuillage automatique", category="Segmentation", imagej="Image > Adjust > Threshold... (Auto)")
def auto_threshold(
    image: Image,
    method: Annotated[_Method, Param(label="méthode")] = "Default",
    objects: Annotated[
        Literal["clairs", "sombres"],
        Param(label="objets", description="sombres : cellules sur fond clair (fond clair)"),
    ] = "clairs",
) -> Labels:
    """Masque binaire des objets, avec les 17 méthodes d'ImageJ (mêmes noms, mêmes niveaux).

    « clairs » correspond à la case « Dark background » d'ImageJ."""
    values = _gray_values(image)
    lower, upper = threshold_range(values, method, dark_background=objects == "clairs")
    mask = (values >= lower) & (values <= upper)
    return Labels(
        mask.astype(np.uint8),
        image.pixel_size_um,
        f"{image.name} (masque {method})",
        {"lower": float(lower), "upper": float(upper), "method": method, "objects": objects},
    )


@operation("Seuil manuel", category="Segmentation", imagej="Image > Adjust > Threshold...")
def manual_threshold(
    image: Image,
    lower: Annotated[float, Param(label="minimum", min=-1e9)] = 0.0,
    upper: Annotated[float, Param(label="maximum", min=-1e9)] = 255.0,
) -> Labels:
    """Garde les pixels dont la valeur est entre le minimum et le maximum (inclus)."""
    gray = _gray_values(image)
    mask = (gray >= lower) & (gray <= upper)
    return Labels(mask.astype(np.uint8), image.pixel_size_um, f"{image.name} (masque)")


@operation("Seuillage local", category="Segmentation", imagej="Image > Adjust > Auto Local Threshold...")
def local_threshold_op(
    image: Image,
    method: Annotated[_LocalMethod, Param(label="méthode")] = "Phansalkar",
    radius: Annotated[float, Param(label="rayon", min=1, max=200, unit="px")] = 15.0,
    objects: Annotated[Literal["clairs", "sombres"], Param(label="objets")] = "clairs",
    param1: Annotated[
        float,
        Param(
            label="paramètre 1", min=-1e6, description="0 = valeur par défaut de la méthode", advanced=True
        ),
    ] = 0.0,
    param2: Annotated[
        float,
        Param(
            label="paramètre 2", min=-1e6, description="0 = valeur par défaut de la méthode", advanced=True
        ),
    ] = 0.0,
) -> Labels:
    """Seuil calculé pour chaque pixel d'après son voisinage : utile quand l'éclairage est
    inégal ou le contraste variable. L'image est d'abord ramenée à 8 bits, comme dans Fiji."""
    mask = local_threshold(
        _gray_values(image), method, radius, param1, param2, bright_objects=objects == "clairs"
    )
    return Labels(mask.astype(np.uint8), image.pixel_size_um, f"{image.name} (masque local {method})")


def _as_mask(labels: Labels) -> np.ndarray:
    return labels.data > 0


@operation("Remplir les trous", category="Binaire", imagej="Process > Binary > Fill Holes")
def fill_holes(mask: Labels) -> Labels:
    """Remplit les zones de fond entièrement entourées par un objet (comme ImageJ, un
    trou relié au fond par un coin seulement est rempli)."""
    out = ndi.binary_fill_holes(_as_mask(mask))
    return Labels(out.astype(np.uint8), mask.pixel_size_um, f"{mask.name} (rempli)")


_SQUARE = np.ones((3, 3), bool)  # voisinage 3×3 des opérations binaires d'ImageJ


@operation("Ouverture", category="Binaire", imagej="Process > Binary > Open")
def binary_open(
    mask: Labels, iterations: Annotated[int, Param(label="nombre de passes", min=1, max=50)] = 1
) -> Labels:
    """Érosion puis dilatation (3×3, répétées) : supprime les petits détails clairs.
    Comme ImageJ par défaut, le bord de l'image compte comme du fond."""
    out = ndi.binary_opening(_as_mask(mask), structure=_SQUARE, iterations=iterations)
    return Labels(out.astype(np.uint8), mask.pixel_size_um, f"{mask.name} (ouvert)")


@operation("Fermeture", category="Binaire", imagej="Process > Binary > Close-")
def binary_close(
    mask: Labels, iterations: Annotated[int, Param(label="nombre de passes", min=1, max=50)] = 1
) -> Labels:
    """Dilatation puis érosion (3×3, répétées) : bouche les petits trous et fissures."""
    out = ndi.binary_closing(_as_mask(mask), structure=_SQUARE, iterations=iterations)
    return Labels(out.astype(np.uint8), mask.pixel_size_um, f"{mask.name} (fermé)")


@operation("Retirer les petits objets", category="Binaire")
def remove_small(
    mask: Labels,
    min_area: Annotated[float, Param(min=0, unit="µm² ou px²")] = 10.0,
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
