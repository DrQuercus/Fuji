"""Réglages d'image : luminosité, contraste, couleurs, niveaux, bruit.

Ce sont les réglages d'un logiciel de photo ou de caméra, appliqués aux pixels (le résultat
est une nouvelle image, l'originale reste intacte). Ils servent à préparer une image **à
montrer** ; pour mesurer des intensités, on mesure l'image d'origine.

Toutes ces opérations sont propres à Fuji (pas de formule d'ImageJ à suivre) ; elles
travaillent sur une image grise ou couleur, 8 ou 16 bits, et rendent le même type.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi

from .. import ij
from ..image import Image
from ..registry import Param, operation

#: poids de la luminosité perçue (Rec. 601), comme « Niveaux de gris : luminance »
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


#: clé des métadonnées d'une image où l'on peut fixer son « blanc » (le même pour tous les
#: plans d'une pile, par exemple)
WHITE_KEY = "white_level"


def white_level(data: np.ndarray) -> float:
    """Valeur du « blanc » d'une image : 255 en 8 bits ; en 16 bits, la plus petite
    profondeur usuelle qui contient **toutes** les valeurs (4095 pour une caméra 12 bits rangée
    dans du 16 bits) ; 1 ou le maximum pour des nombres réels. Les NaN ne comptent pas."""
    if data.dtype == np.uint8:
        return 255.0
    if not data.size:
        return 1.0
    if np.issubdtype(data.dtype, np.integer):
        top = float(data.max())
        for bits in (8, 10, 12, 14, 16):
            if top <= 2**bits - 1:
                return float(2**bits - 1)
        return float(np.iinfo(data.dtype).max)
    finite = data[np.isfinite(data)]
    top = float(finite.max()) if finite.size else 1.0
    return 1.0 if top <= 1.0 else top


def white_of(image: Image) -> float:
    """Le blanc à utiliser pour cette image : celui que l'on a fixé (pile entière), sinon
    celui que ses valeurs indiquent."""
    fixed = image.metadata.get(WHITE_KEY)
    if fixed:
        return float(fixed)
    return white_level(image.data[..., :3] if image.is_rgb else image.data)


def _unit(image: Image) -> tuple[np.ndarray, float, bool]:
    """Pixels entre 0 et 1 (float32), le blanc d'origine, et « image couleur ? »."""
    white = white_of(image)
    rgb = image.is_rgb
    data = image.data[..., :3] if rgb else image.data
    return data.astype(np.float32) / np.float32(white), white, rgb


def _untouched(image: Image, suffix: str) -> Image:
    """Aucun réglage n'est actif : l'image est rendue telle quelle, au pixel près."""
    return image.with_data(image.data.copy(), name=f"{image.name} ({suffix})")


def stretch(image: Image, black: float, white: float, gamma: float = 1.0, invert: bool = False) -> Image:
    """Écrit dans les pixels ce que montre un affichage réglé entre ``black`` et ``white``
    avec ce gamma (et, avec ``invert``, une palette à l'envers) : c'est « Apply » de la
    fenêtre Luminosité / Contraste d'ImageJ. Les bornes sont prises telles quelles (un noir
    négatif, des bornes non entières sont permis)."""
    if not white > black:
        raise ValueError("Le blanc doit être plus clair que le noir")
    data = image.data[..., :3] if image.is_rgb else image.data
    full = white_of(image)
    unit = np.clip((data.astype(np.float32) - np.float32(black)) / np.float32(white - black), 0.0, 1.0)
    if gamma != 1.0:
        unit = np.power(unit, np.float32(gamma))
    if invert:
        unit = 1.0 - unit
    return _finish(image, unit, full, "niveaux")


def _finish(image: Image, unit: np.ndarray, white: float, suffix: str) -> Image:
    out = np.clip(unit, 0.0, 1.0) * np.float32(white)
    if image.is_rgb and image.data.shape[-1] == 4:  # la transparence est gardée telle quelle
        out = np.concatenate([out, image.data[..., 3:].astype(np.float32)], axis=-1)
    return image.with_data(ij.restore_dtype(out, image.data), name=f"{image.name} ({suffix})")


def _luma(unit: np.ndarray) -> np.ndarray:
    return unit @ _LUMA


def _blur(unit: np.ndarray, sigma: float) -> np.ndarray:
    return ndi.gaussian_filter(unit, (sigma, sigma) + (0,) * (unit.ndim - 2), mode="nearest")


def noise_level(unit: np.ndarray) -> float:
    """Écart type du bruit d'une image, estimé sur ses plus fins détails (écart médian, peu
    sensible au contenu de l'image)."""
    plane = unit.mean(axis=-1) if unit.ndim == 3 else unit
    step = max(1, int(np.sqrt(plane.size / 400_000)))
    plane = plane[::step, ::step]
    fine = float(np.median(np.abs(plane - ndi.gaussian_filter(plane, 1.0, mode="nearest"))))
    scale = (1.0 / np.sqrt(3.0)) if unit.ndim == 3 else 1.0  # la moyenne de 3 canaux réduit le bruit
    return fine / (_FINE_OF_UNIT_NOISE * scale)


def _fine_of_unit_noise() -> float:
    noise = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    return float(np.median(np.abs(noise - ndi.gaussian_filter(noise, 1.0, mode="nearest"))))


#: écart médian des plus fins détails d'un bruit blanc d'écart type 1 (étalonnage)
_FINE_OF_UNIT_NOISE = _fine_of_unit_noise()


# ------------------------------------------------------------ réglages de l'image
@operation("Réglages de l'image", category="Réglages")
def adjust_image(
    image: Image,
    brightness: Annotated[
        float, Param(label="luminosité", min=-100, max=100, step=1, description="plus sombre ↔ plus clair")
    ] = 0.0,
    contrast: Annotated[
        float, Param(label="contraste", min=-100, max=100, step=1, description="plus doux ↔ plus marqué")
    ] = 0.0,
    gamma: Annotated[
        float,
        Param(
            label="gamma",
            min=0.2,
            max=5.0,
            step=0.01,
            description="moins de 1 : éclaircit les zones sombres ; plus de 1 : les assombrit",
        ),
    ] = 1.0,
    saturation: Annotated[
        float,
        Param(label="saturation", min=-100, max=100, step=1, description="couleurs plus ternes ↔ plus vives"),
    ] = 0.0,
    hue: Annotated[
        float,
        Param(label="teinte", min=-180, max=180, step=1, unit="°", description="fait tourner les couleurs"),
    ] = 0.0,
    temperature: Annotated[
        float,
        Param(
            label="température",
            min=-100,
            max=100,
            step=1,
            description="plus froid (bleu) ↔ plus chaud (jaune)",
        ),
    ] = 0.0,
    tint: Annotated[
        float,
        Param(label="dominante", min=-100, max=100, step=1, description="vers le vert ↔ vers le magenta"),
    ] = 0.0,
    sharpness: Annotated[
        float,
        Param(label="netteté", min=0, max=100, step=1, description="renforce les contours (et le bruit)"),
    ] = 0.0,
    denoise: Annotated[
        float,
        Param(
            label="réduction du bruit",
            min=0,
            max=100,
            step=1,
            description="lisse le grain, en gardant les bords",
        ),
    ] = 0.0,
) -> Image:
    """Les réglages d'un logiciel de photo, réunis : luminosité, contraste, gamma, couleurs,
    netteté et bruit. Tout à zéro (gamma à 1) : l'image n'est pas modifiée.

    Dans l'ordre : réduction du bruit, température et dominante, luminosité et contraste,
    gamma, teinte et saturation, netteté. Les réglages de couleur sont sans effet sur une
    image en niveaux de gris. Le résultat sert à montrer l'image, pas à mesurer des
    intensités."""
    neutral = not any((brightness, contrast, saturation, hue, temperature, tint, sharpness, denoise))
    if neutral and gamma == 1.0:
        return _untouched(image, "réglée")
    unit, white, rgb = _unit(image)
    if denoise > 0:
        unit = _edge_preserving(unit, 0.5 + denoise / 100 * 2.5, 0.02 + denoise / 100 * 0.10)
    if rgb and (temperature or tint):
        t, d = temperature / 100 * 0.35, tint / 100 * 0.35
        gains = np.array([1 + t + d / 2, 1 - d, 1 - t + d / 2], dtype=np.float32)
        unit = unit * (gains / (gains @ _LUMA))  # la luminosité d'un gris neutre est gardée
    if brightness:
        unit = unit + np.float32(brightness / 100 * 0.5)
    if contrast:
        # −100 : tout gris ; 0 : inchangé ; +100 : presque du noir et blanc
        factor = np.float32(np.tan((contrast / 100 + 1) * np.pi / 4)) if contrast < 100 else np.float32(1e3)
        unit = (unit - 0.5) * factor + 0.5
    if gamma != 1.0:
        unit = np.power(np.clip(unit, 0.0, None), np.float32(gamma))
    if rgb and (hue or saturation):
        unit = _hue_saturation(unit, hue, saturation / 100)
    if sharpness > 0:
        unit = unit + np.float32(sharpness / 100 * 1.5) * (unit - _blur(unit, 1.5))
    return _finish(image, unit, white, "réglée")


def _hue_saturation(unit: np.ndarray, hue_deg: float, saturation: float) -> np.ndarray:
    """Rotation de la teinte et gain de saturation autour de l'axe des gris (espace YIQ)."""
    to_yiq = np.array(
        [[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]], dtype=np.float64
    )
    a = np.deg2rad(hue_deg)
    gain = max(0.0, 1.0 + saturation)
    turn = np.array(
        [[1, 0, 0], [0, gain * np.cos(a), -gain * np.sin(a)], [0, gain * np.sin(a), gain * np.cos(a)]]
    )
    matrix = (np.linalg.inv(to_yiq) @ turn @ to_yiq).astype(np.float32)
    return unit @ matrix.T


def _edge_preserving(unit: np.ndarray, sigma: float, tolerance: float) -> np.ndarray:
    """Lissage qui respecte les bords : chaque pixel va vers la moyenne de ses voisins
    d'autant plus qu'il leur ressemble (écart inférieur à ``tolerance``)."""
    smooth = _blur(unit, sigma)
    diff = unit - smooth
    if unit.ndim == 3:
        distance = np.sqrt((diff * diff).mean(axis=-1, keepdims=True))
    else:
        distance = np.abs(diff)
    keep = 1.0 - np.exp(-((distance / np.float32(tolerance)) ** 2))  # 0 : lisser, 1 : garder
    keep = _blur(keep, 1.0)
    return smooth + keep * diff


# -------------------------------------------------------------------- niveaux
@operation("Niveaux", category="Réglages", imagej="Image > Adjust > Brightness/Contrast... (Apply)")
def levels(
    image: Image,
    black: Annotated[
        float,
        Param(
            label="noir",
            min=-1e9,
            span="image",
            start="min",
            description="cette valeur et les plus sombres deviennent du noir",
        ),
    ] = 0.0,
    white: Annotated[
        float,
        Param(
            label="blanc",
            min=-1e9,
            span="image",
            start="max",
            description="cette valeur et les plus claires deviennent du blanc (0 : le maximum de l'image)",
        ),
    ] = 0.0,
    gamma: Annotated[
        float,
        Param(
            label="tons moyens (gamma)",
            min=0.1,
            max=10.0,
            step=0.01,
            description="moins de 1 : éclaircit les tons moyens ; plus de 1 : les assombrit",
        ),
    ] = 1.0,
    channel: Annotated[
        Literal["tous", "rouge", "vert", "bleu"],
        Param(label="canal", description="image couleur : régler un seul canal change la couleur"),
    ] = "tous",
    out_black: Annotated[float, Param(label="noir en sortie", min=0, max=100, unit="%", advanced=True)] = 0.0,
    out_white: Annotated[
        float, Param(label="blanc en sortie", min=0, max=100, unit="%", advanced=True)
    ] = 100.0,
) -> Image:
    """Étale les valeurs entre un noir et un blanc choisis, avec un réglage des tons moyens.

    C'est le « Apply » de la fenêtre Luminosité / Contraste d'ImageJ : ce que l'on voit avec
    les bornes d'affichage est écrit dans les pixels. Le résultat occupe toute la dynamique
    du type de l'image."""
    data = image.data[..., :3] if image.is_rgb else image.data
    top = white if white != 0 else float(np.nanmax(data))  # 0 : le maximum de l'image
    if top <= black:
        raise ValueError("Le blanc doit être plus clair que le noir")
    full = white_of(image)
    work = data.astype(np.float32)
    index = {"rouge": 0, "vert": 1, "bleu": 2}.get(channel)
    if index is not None and not image.is_rgb:
        raise ValueError("Le réglage d'un seul canal demande une image couleur")
    part = work[..., index] if index is not None else work
    unit = np.clip((part - np.float32(black)) / np.float32(top - black), 0.0, 1.0)
    if gamma != 1.0:
        unit = np.power(unit, np.float32(gamma))
    lo, hi = out_black / 100, out_white / 100
    unit = lo + unit * (hi - lo)
    if index is not None:
        out = work / np.float32(full)
        out[..., index] = unit
    else:
        out = unit
    return _finish(image, out, full, "niveaux")


# ---------------------------------------------------------- balance des couleurs
@operation("Balance des couleurs", category="Réglages", imagej="Image > Adjust > Color Balance...")
def colour_balance(
    image: Image,
    red: Annotated[float, Param(label="rouge", min=0, max=300, step=1, unit="%")] = 100.0,
    green: Annotated[float, Param(label="vert", min=0, max=300, step=1, unit="%")] = 100.0,
    blue: Annotated[float, Param(label="bleu", min=0, max=300, step=1, unit="%")] = 100.0,
    keep_brightness: Annotated[
        bool,
        Param(
            label="garder la luminosité",
            description="un gris neutre garde sa luminosité : seule la couleur change",
        ),
    ] = False,
) -> Image:
    """Renforce ou atténue chaque couleur (100 % : inchangée).

    Pour retirer une dominante à la main quand la balance des blancs automatique ne suffit
    pas, ou pour équilibrer les canaux d'une image de fluorescence fusionnée."""
    if not image.is_rgb:
        raise ValueError("La balance des couleurs demande une image couleur")
    if red == green == blue == 100:
        return _untouched(image, "couleurs")
    unit, white, _ = _unit(image)
    gains = np.array([red, green, blue], dtype=np.float32) / 100
    if keep_brightness:
        total = float(gains @ _LUMA)
        if total <= 0:
            raise ValueError("Les trois couleurs ne peuvent pas être à zéro")
        gains = gains / total
    return _finish(image, unit * gains, white, "couleurs")


# ------------------------------------------------------------- réduire le bruit
@operation("Réduire le bruit", category="Réglages", imagej="Process > Noise / Filters")
def denoise(
    image: Image,
    method: Annotated[
        Literal[
            "en gardant les bords (rapide)",
            "flou gaussien",
            "médian",
            "bilatéral (lent)",
            "moyennes non locales (très lent, le plus soigné)",
        ],
        Param(label="méthode"),
    ] = "en gardant les bords (rapide)",
    strength: Annotated[
        float,
        Param(
            label="force",
            min=1,
            max=100,
            step=1,
            description="à augmenter jusqu'à ce que le grain disparaisse",
        ),
    ] = 30.0,
    detail: Annotated[
        float,
        Param(
            label="détails gardés",
            min=0,
            max=100,
            step=1,
            description="0 : tout est lissé ; 100 : les contours et les fins détails sont protégés",
        ),
    ] = 50.0,
) -> Image:
    """Atténue le grain de la caméra (images sombres, poses courtes, fort gain).

    « En gardant les bords » convient presque toujours. Les deux méthodes lentes donnent un
    résultat plus propre sur une image très bruitée ; comptez plusieurs secondes sur une
    image de caméra 4K."""
    unit, white, rgb = _unit(image)
    s = strength / 100
    if method == "flou gaussien":
        out = _blur(unit, 0.3 + 3.0 * s)
    elif method == "médian":
        size = 2 * (1 + int(round(3 * s))) + 1
        out = ndi.median_filter(unit, size=(size, size) + (1,) * (unit.ndim - 2), mode="nearest")
    elif method.startswith("bilatéral"):
        from skimage.restoration import denoise_bilateral

        out = denoise_bilateral(
            np.clip(unit, 0, 1),
            sigma_color=0.02 + 0.2 * s * (1.2 - detail / 100),
            sigma_spatial=1.0 + 4.0 * s,
            channel_axis=-1 if rgb else None,
        ).astype(np.float32)
    elif method.startswith("moyennes non locales"):
        from skimage.restoration import denoise_nl_means

        axis = -1 if rgb else None
        sigma = noise_level(unit)
        out = denoise_nl_means(
            unit,
            h=max(1e-4, sigma * (0.4 + 1.2 * s) * (1.3 - detail / 100)),
            sigma=sigma,
            patch_size=5,
            patch_distance=6,
            fast_mode=True,
            channel_axis=axis,
        ).astype(np.float32)
    else:
        out = _edge_preserving(unit, 0.5 + 3.0 * s, (0.01 + 0.15 * s) * (1.5 - detail / 100))
    return _finish(image, out, white, "débruitée")
