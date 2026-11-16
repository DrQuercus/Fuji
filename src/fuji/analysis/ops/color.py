"""Types d'image et couleur : conversions, canaux, espaces de couleur, colorants."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np

from .. import ij
from ..image import Image, Labels, Rois
from ..registry import Param, operation


# ------------------------------------------------------------------------ types
def convert_plane(data: np.ndarray, target: str, scale: bool, rgb: bool) -> np.ndarray:
    """Conversion de type d'ImageJ (``TypeConverter``) sur un plan."""
    if rgb:
        data = ij.gray8(data)  # ImageJ passe toujours par le gris 8 bits
    if target == "8 bits":
        if data.dtype == np.uint8:
            return data
        return ij.to_byte(data) if scale else ij.restore_dtype(data.astype(np.float64), np.uint8)
    if target == "16 bits":
        if data.dtype.kind == "f" and scale:
            finite = data[np.isfinite(data)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
            factor = 65535.0 / (hi - lo) if hi > lo else 1.0
            return ij.restore_dtype((np.nan_to_num(data, nan=lo) - lo) * factor, np.uint16)
        return ij.restore_dtype(data.astype(np.float64), np.uint16)
    return data.astype(np.float32)


@operation("Convertir le type", category="Conversion", imagej="Image > Type > 8-bit / 16-bit / 32-bit")
def convert_type(
    image: Image,
    type: Annotated[Literal["8 bits", "16 bits", "32 bits"], Param(label="type")] = "8 bits",
    scale: Annotated[
        bool,
        Param(
            label="mettre à l'échelle",
            description="étale min–max sur toute la plage du nouveau type (réglage par défaut d'ImageJ)",
        ),
    ] = True,
) -> Image:
    """Change le type des pixels, avec les règles d'ImageJ.

    Vers 8 bits avec mise à l'échelle : le minimum devient 0 et le maximum 255. Sans mise
    à l'échelle : les valeurs sont arrondies et bornées. Une image couleur est d'abord
    convertie en gris, ``(R+V+B)/3``.
    """
    out = convert_plane(image.data, type, scale, image.is_rgb)
    return image.with_data(out, name=f"{image.name} ({type})")


# ---------------------------------------------------------------------- canaux
@operation("Fusionner en RGB", category="Couleur", imagej="Image > Color > Merge Channels...")
def merge_rgb(red: Image | None = None, green: Image | None = None, blue: Image | None = None) -> Image:
    """Assemble jusqu'à trois images en niveaux de gris en une image couleur 8 bits.

    Chaque image est ramenée à 8 bits sur sa propre plage (min–max), comme ImageJ."""
    given = [c for c in (red, green, blue) if c is not None]
    if not given:
        raise ValueError("Choisissez au moins une image")
    shape = given[0].plane_shape
    planes = []
    for c in (red, green, blue):
        if c is None:
            planes.append(np.zeros(shape, np.uint8))
            continue
        if c.plane_shape != shape:
            raise ValueError("Les images à fusionner n'ont pas la même taille")
        planes.append(ij.to_byte(ij.intensity_plane(c.data, c.is_rgb)))
    first = given[0]
    return Image(np.stack(planes, axis=-1), first.pixel_size_um, "RGB", dict(first.metadata), rgb=True)


def _require_rgb(image: Image) -> np.ndarray:
    if not image.is_rgb:
        raise ValueError("Cette opération demande une image couleur (RGB)")
    return image.data[..., :3]


@operation("Séparer en HSB", category="Couleur", imagej="Image > Type > HSB Stack")
def split_hsb(image: Image) -> tuple[Image, Image, Image]:
    """Teinte, saturation et luminosité (0–255 chacune, mêmes valeurs qu'ImageJ)."""
    hsb = ij.rgb_to_hsb(ij.to_byte_rgb(_require_rgb(image)))
    names = ("teinte", "saturation", "luminosité")
    return tuple(  # type: ignore[return-value]
        image.with_data(hsb[..., i].copy(), name=f"{image.name} ({names[i]})") for i in range(3)
    )


@operation("Séparer en Lab", category="Couleur", imagej="Image > Type > Lab Stack")
def split_lab(image: Image) -> tuple[Image, Image, Image]:
    """Clarté L* (0–100) et axes a*, b* (CIE L*a*b*, illuminant D65), en 32 bits."""
    from skimage import color

    lab = color.rgb2lab(ij.to_byte_rgb(_require_rgb(image))).astype(np.float32)
    return tuple(  # type: ignore[return-value]
        image.with_data(lab[..., i].copy(), name=f"{image.name} ({n})") for i, n in enumerate(("L", "a", "b"))
    )


# ------------------------------------------------------------------- colorants
#: Vecteurs d'absorption (R, V, B) des colorants : ceux du plugin « Colour Deconvolution »
#: de G. Landini (méthode de Ruifrok et Johnston). Un vecteur nul est déduit des deux autres.
STAINS: dict[str, tuple[tuple[str, tuple[float, float, float]], ...]] = {
    "H&E": (
        ("hématoxyline", (0.644211, 0.716556, 0.266844)),
        ("éosine", (0.092789, 0.954111, 0.283111)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "H&E 2": (
        ("hématoxyline", (0.49015734, 0.76897085, 0.41040173)),
        ("éosine", (0.04615336, 0.8420684, 0.5373925)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "H DAB": (
        ("hématoxyline", (0.650, 0.704, 0.286)),
        ("DAB", (0.268, 0.570, 0.776)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "H&E DAB": (
        ("hématoxyline", (0.650, 0.704, 0.286)),
        ("éosine", (0.072, 0.990, 0.105)),
        ("DAB", (0.268, 0.570, 0.776)),
    ),
    "H AEC": (
        ("hématoxyline", (0.650, 0.704, 0.286)),
        ("AEC", (0.2743, 0.6796, 0.6803)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "H PAS": (
        ("hématoxyline", (0.644211, 0.716556, 0.266844)),
        ("PAS", (0.175411, 0.972178, 0.154589)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "Vert de méthyle DAB": (
        ("vert de méthyle", (0.98003, 0.144316, 0.133146)),
        ("DAB", (0.268, 0.570, 0.776)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "FastRed FastBlue DAB": (
        ("fast red", (0.21393921, 0.85112669, 0.47794022)),
        ("fast blue", (0.74890292, 0.60624161, 0.26731082)),
        ("DAB", (0.268, 0.570, 0.776)),
    ),
    "Trichrome de Masson": (
        ("bleu de méthyle", (0.7995107, 0.5913521, 0.10528667)),
        ("ponceau fuchsine", (0.09997159, 0.73738605, 0.6680326)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "Azan-Mallory": (
        ("bleu d'aniline", (0.853033, 0.508733, 0.112656)),
        ("azocarmin", (0.09289875, 0.8662008, 0.49098468)),
        ("orange G", (0.10732849, 0.36765403, 0.9237484)),
    ),
    "Bleu alcian & H": (
        ("bleu alcian", (0.874622, 0.457711, 0.158256)),
        ("hématoxyline", (0.552556, 0.7544, 0.353744)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "Feulgen vert lumière": (
        ("Feulgen", (0.46420921, 0.83008335, 0.30827187)),
        ("vert lumière", (0.94705542, 0.25373821, 0.19650764)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
    "Giemsa": (
        ("bleu de méthylène", (0.834750233, 0.513556283, 0.196330403)),
        ("éosine", (0.092789, 0.954111, 0.283111)),
        ("résidu", (0.0, 0.0, 0.0)),
    ),
}
_StainSet = Literal[
    "H&E",
    "H&E 2",
    "H DAB",
    "H&E DAB",
    "H AEC",
    "H PAS",
    "Vert de méthyle DAB",
    "FastRed FastBlue DAB",
    "Trichrome de Masson",
    "Azan-Mallory",
    "Bleu alcian & H",
    "Feulgen vert lumière",
    "Giemsa",
]
assert _StainSet.__args__ == tuple(STAINS)  # type: ignore[attr-defined]


def stain_matrix(vectors: np.ndarray) -> np.ndarray:
    """Matrice 3×3 des colorants (une ligne par colorant), complétée et normalisée comme
    dans le plugin de Fiji."""
    m = np.asarray(vectors, dtype=np.float64).copy()
    norms = np.sqrt((m * m).sum(axis=1))
    for i in range(3):
        if norms[i] != 0:
            m[i] /= norms[i]
    if not m[1].any():  # deuxième colorant absent : permutation du premier
        m[1] = m[0][[2, 0, 1]]
    if not m[2].any():  # troisième absent : complément des deux premiers
        squares = m[0] ** 2 + m[1] ** 2
        m[2] = np.where(squares > 1.0, 0.0, np.sqrt(np.maximum(1.0 - squares, 0.0)))
        m[2] /= np.sqrt((m[2] ** 2).sum())
    m[m == 0.0] = 0.001
    return m


def colour_deconvolution(rgb: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sépare une image RGB 8 bits en trois colorants.

    Renvoie ``(images, densités)`` : les images 8 bits du plugin de Fiji (255 = pas de
    colorant), et les densités optiques correspondantes (0 = pas de colorant), plus
    adaptées à une quantification.
    """
    m = stain_matrix(vectors)
    q = np.linalg.inv(m)  # densités des colorants = densités RVB · inverse
    log255 = np.log(255.0)
    od = -((255.0 * np.log((rgb[..., :3].astype(np.float64) + 1.0) / 255.0)) / log255)
    scaled = od @ q
    out = np.exp(-(scaled - 255.0) * log255 / 255.0)
    images = np.floor(np.minimum(out, 255.0) + 0.5).astype(np.uint8)
    return images, (scaled / 255.0 * np.log10(255.0)).astype(np.float32)


@operation("Déconvolution des couleurs", category="Couleur", imagej="Image > Color > Colour Deconvolution")
def colour_deconvolution_op(
    image: Image,
    stains: Annotated[_StainSet, Param(label="colorants")] = "H&E",
    output: Annotated[
        Literal["images 8 bits (comme Fiji)", "densités optiques"],
        Param(
            label="sortie",
            description="les densités optiques sont proportionnelles à la quantité de colorant",
        ),
    ] = "images 8 bits (comme Fiji)",
) -> tuple[Image, Image, Image]:
    """Sépare les colorants d'une lame histologique en fond clair (méthode de Ruifrok et
    Johnston, mêmes vecteurs et mêmes calculs que le plugin de Fiji).

    Une image par colorant. En sortie 8 bits, 255 signifie « pas de colorant » ; en
    densité optique, 0 signifie « pas de colorant » et la valeur croît avec sa quantité.
    Conditions : image RGB, fond blanc bien exposé (balance des blancs faite).
    """
    rgb = ij.to_byte_rgb(_require_rgb(image))
    names = [n for n, _ in STAINS[stains]]
    images, densities = colour_deconvolution(rgb, np.array([v for _, v in STAINS[stains]]))
    data = images if output.startswith("images") else densities
    return tuple(  # type: ignore[return-value]
        image.with_data(data[..., i].copy(), name=f"{image.name} ({names[i]})") for i in range(3)
    )


# -------------------------------------------------------------- balance, seuil
@operation("Balance des blancs", category="Couleur", imagej="Image > Adjust > Color Balance...")
def white_balance(image: Image, background: Rois | None = None) -> Image:
    """Rend le fond neutre (gris ou blanc) en ajustant le gain de chaque canal.

    Le fond est la moyenne des ROI fournies (zone vide de la lame), ou à défaut des 5 %
    de pixels les plus clairs. Le canal le plus fort n'est pas modifié."""
    rgb = _require_rgb(image).astype(np.float64)
    if background is not None and len(background):
        sel = np.zeros(rgb.shape[:2], bool)
        for i in range(len(background)):
            if not background.is_line(i):
                sel |= background.mask(i, rgb.shape[:2])
        if not sel.any():
            raise ValueError("La ROI de fond ne recouvre pas l'image")
    else:
        gray = rgb.mean(axis=-1)
        sel = gray >= np.percentile(gray, 95)
    means = rgb[sel].mean(axis=0)
    if (means <= 0).any():
        raise ValueError("Le fond est noir dans un des canaux : balance impossible")
    out = image.data.astype(np.float64).copy()
    out[..., :3] = rgb * (means.max() / means)
    return image.with_data(ij.restore_dtype(out, image.data), name=f"{image.name} (balance)")


@operation("Seuillage couleur", category="Segmentation", imagej="Image > Adjust > Color Threshold...")
def colour_threshold(
    image: Image,
    space: Annotated[Literal["HSB", "RGB", "Lab"], Param(label="espace")] = "HSB",
    min1: Annotated[float, Param(label="canal 1 min", min=-200, max=255)] = 0.0,
    max1: Annotated[float, Param(label="canal 1 max", min=-200, max=255)] = 255.0,
    min2: Annotated[float, Param(label="canal 2 min", min=-200, max=255)] = 0.0,
    max2: Annotated[float, Param(label="canal 2 max", min=-200, max=255)] = 255.0,
    min3: Annotated[float, Param(label="canal 3 min", min=-200, max=255)] = 0.0,
    max3: Annotated[float, Param(label="canal 3 max", min=-200, max=255)] = 255.0,
) -> Labels:
    """Garde les pixels dont les trois composantes sont dans les plages données.

    HSB : teinte, saturation, luminosité (0–255, comme ImageJ) ; RGB : rouge, vert, bleu ;
    Lab : L* (0–100), a*, b*. Pour la teinte, un minimum supérieur au maximum sélectionne
    la plage qui passe par zéro (les rouges)."""
    rgb = ij.to_byte_rgb(_require_rgb(image))
    if space == "HSB":
        comps = ij.rgb_to_hsb(rgb).astype(np.float64)
    elif space == "RGB":
        comps = rgb.astype(np.float64)
    else:
        from skimage import color

        comps = color.rgb2lab(rgb)
    mask = np.ones(rgb.shape[:2], bool)
    for i, (lo, hi) in enumerate(((min1, max1), (min2, max2), (min3, max3))):
        c = comps[..., i]
        if lo <= hi:
            mask &= (c >= lo) & (c <= hi)
        else:  # plage qui passe par zéro
            mask &= (c >= lo) | (c <= hi)
    return Labels(mask.astype(np.uint8), image.pixel_size_um, f"{image.name} (masque couleur)")
