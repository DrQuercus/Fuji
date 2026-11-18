"""Filtres d'ImageJ : voisinages circulaires, 3×3, convolution, Fourier, contraste."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from scipy import ndimage as ndi

from .. import ij
from ..image import Image
from ..registry import Param, operation

_Radius = Annotated[float, Param(label="rayon", min=0.5, max=200, step=0.5, unit="px")]


def _apply(image: Image, fn, suffix: str, dtype: np.dtype | None = None) -> Image:  # type: ignore[no-untyped-def]
    """Applique ``fn`` (plan → flottants) canal par canal, puis revient au type de l'image."""
    out = ij.per_channel(image.data, fn, rgb=image.is_rgb)
    return image.with_data(ij.restore_dtype(out, dtype or image.data.dtype), name=f"{image.name} ({suffix})")


def _mean(a: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return ndi.correlate(a.astype(np.float64), fp.astype(np.float64), mode="nearest") / fp.sum()


# ------------------------------------------------------- voisinage circulaire
@operation("Filtre moyenne", category="Filtres", imagej="Process > Filters > Mean...")
def mean_filter(image: Image, radius: _Radius = 2.0) -> Image:
    """Moyenne sur le voisinage circulaire d'ImageJ."""
    fp = ij.disk(radius)
    return _apply(image, lambda a: _mean(a, fp), f"moyenne r={radius:g}")


@operation("Filtre minimum", category="Filtres", imagej="Process > Filters > Minimum...")
def minimum_filter(image: Image, radius: _Radius = 2.0) -> Image:
    """Plus petite valeur du voisinage (érosion en niveaux de gris)."""
    fp = ij.disk(radius)
    return _apply(image, lambda a: ndi.minimum_filter(a, footprint=fp, mode="nearest"), f"min r={radius:g}")


@operation("Filtre maximum", category="Filtres", imagej="Process > Filters > Maximum...")
def maximum_filter(image: Image, radius: _Radius = 2.0) -> Image:
    """Plus grande valeur du voisinage (dilatation en niveaux de gris)."""
    fp = ij.disk(radius)
    return _apply(image, lambda a: ndi.maximum_filter(a, footprint=fp, mode="nearest"), f"max r={radius:g}")


@operation("Filtre variance", category="Filtres", imagej="Process > Filters > Variance...")
def variance_filter(image: Image, radius: _Radius = 2.0) -> Image:
    """Variance locale : fait ressortir les textures et les contours."""
    fp = ij.disk(radius)

    def variance(a: np.ndarray) -> np.ndarray:
        mean = _mean(a, fp)
        return np.maximum(_mean(a.astype(np.float64) ** 2, fp) - mean * mean, 0.0)

    return _apply(image, variance, f"variance r={radius:g}")


@operation("Chapeau haut-de-forme", category="Filtres", imagej="Process > Filters > Top Hat...")
def top_hat(
    image: Image,
    radius: _Radius = 10.0,
    light_background: Annotated[bool, Param(label="fond clair")] = False,
    keep_background: Annotated[
        bool, Param(label="garder le fond", description="renvoie le fond estimé au lieu de le soustraire")
    ] = False,
) -> Image:
    """Garde les détails plus petits que le rayon et retire le fond (ouverture soustraite).

    Avec « fond clair », le fond devient blanc, comme pour la soustraction de fond."""
    fp = ij.disk(radius)

    def one(a: np.ndarray) -> np.ndarray:
        first, second = (
            (ndi.maximum_filter, ndi.minimum_filter)
            if light_background
            else (ndi.minimum_filter, ndi.maximum_filter)
        )
        background = second(first(a, footprint=fp, mode="nearest"), footprint=fp, mode="nearest")
        if keep_background:
            return background.astype(np.float64)
        out = a.astype(np.float64) - background
        if light_background and a.dtype.kind in "ui":
            out += np.iinfo(a.dtype).max
        return out

    return _apply(image, one, f"top-hat r={radius:g}")


@operation(
    "Retirer les valeurs aberrantes", category="Filtres", imagej="Process > Noise > Remove Outliers..."
)
def remove_outliers(
    image: Image,
    radius: _Radius = 2.0,
    threshold: Annotated[float, Param(label="seuil", min=0, description="écart à la médiane toléré")] = 50.0,
    which: Annotated[Literal["claires", "sombres"], Param(label="valeurs")] = "claires",
) -> Image:
    """Remplace par la médiane locale les pixels qui s'en écartent de plus que le seuil
    (pixels chauds, poussières)."""
    fp = ij.disk(radius)

    def one(a: np.ndarray) -> np.ndarray:
        median = ndi.median_filter(a, footprint=fp, mode="nearest")
        diff = a.astype(np.float64) - median
        bad = diff > threshold if which == "claires" else -diff > threshold
        return np.where(bad, median, a)

    out = ij.per_channel(image.data, one, rgb=image.is_rgb)
    return image.with_data(out.astype(image.data.dtype), name=f"{image.name} (sans aberrants)")


@operation("Débruiter (despeckle)", category="Filtres", imagej="Process > Noise > Despeckle")
def despeckle(image: Image) -> Image:
    """Médian 3×3 : retire le bruit « poivre et sel »."""
    fp = ij.disk(1)
    out = ij.per_channel(
        image.data, lambda a: ndi.median_filter(a, footprint=fp, mode="nearest"), rgb=image.is_rgb
    )
    return image.with_data(out, name=f"{image.name} (débruité)")


# ------------------------------------------------------------------ gaussiens
@operation("Masque flou (netteté)", category="Filtres", imagej="Process > Filters > Unsharp Mask...")
def unsharp_mask(
    image: Image,
    sigma: Annotated[float, Param(label="rayon (sigma)", min=0.1, max=100, step=0.5, unit="px")] = 1.0,
    weight: Annotated[float, Param(label="poids", min=0.1, max=0.9, step=0.05)] = 0.6,
) -> Image:
    """Renforce la netteté : ``(image − poids × flou) / (1 − poids)``, comme ImageJ."""
    return _apply(
        image,
        lambda a: (a.astype(np.float64) - weight * ij.gaussian_blur(a, sigma, 0.01)) / (1.0 - weight),
        f"netteté σ={sigma:g}",
    )


# ------------------------------------------------------------------- filtres 3×3
def _correlate3(a: np.ndarray, kernel: list[list[float]]) -> np.ndarray:
    return ndi.correlate(a.astype(np.float64), np.array(kernel, np.float64), mode="nearest")


@operation("Lisser", category="Filtres", imagej="Process > Smooth")
def smooth(
    image: Image,
    passes: Annotated[
        int,
        Param(
            label="nombre de passes",
            min=1,
            max=20,
            description="lisser plusieurs fois de suite adoucit davantage (ImageJ : une passe)",
        ),
    ] = 1,
) -> Image:
    """Moyenne 3×3, répétée autant de fois que demandé."""

    def one(a: np.ndarray) -> np.ndarray:
        out = a.astype(np.float64)
        for _ in range(max(1, int(passes))):
            out = _correlate3(out, [[1, 1, 1]] * 3) / 9.0
        return out

    return _apply(image, one, "lissée")


@operation("Accentuer", category="Filtres", imagej="Process > Sharpen")
def sharpen(
    image: Image,
    strength: Annotated[
        float,
        Param(
            label="force",
            min=0,
            max=300,
            step=5,
            unit="%",
            description="100 % : l'accentuation d'ImageJ ; moins : plus discret ; plus : plus marqué",
        ),
    ] = 100.0,
) -> Image:
    """Accentuation 3×3 d'ImageJ (centre 12, voisins −1, divisé par 4), dosée par « force »."""
    kernel = [[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]]
    k = float(strength) / 100.0

    def one(a: np.ndarray) -> np.ndarray:
        sharp = _correlate3(a, kernel) / 4.0
        return sharp if k == 1.0 else a + (sharp - a) * k

    return _apply(image, one, "accentuée")


def _edges_other(image: Image, method: str, sigma: float, low: float, high: float) -> Image:
    """Autres détecteurs de bords que le Sobel d'ImageJ (ou Sobel après un lissage)."""
    from skimage import feature
    from skimage import filters as skf

    def one(a: np.ndarray) -> np.ndarray:
        g = a.astype(np.float64)
        if sigma > 0:
            g = ndi.gaussian_filter(g, sigma, mode="nearest")
        if method.startswith("Canny"):
            top = float(g.max() - g.min()) or 1.0
            edges = feature.canny(
                g,
                sigma=0.0,
                low_threshold=low / 100 * top * 0.5,
                high_threshold=max(high, low) / 100 * top * 0.5,
            )
            from .adjust import white_level

            return edges.astype(np.float64) * white_level(a)
        if method == "Laplacien":
            return np.abs(ndi.laplace(g, mode="nearest"))
        fn = {"Scharr": skf.scharr, "Prewitt": skf.prewitt}.get(method, skf.sobel)
        # même échelle que le Sobel d'ImageJ (noyau entier, sans division)
        return fn(g, mode="nearest") * 4.0 * np.sqrt(2)

    return _apply(image, one, "contours")


@operation("Détecter les contours", category="Filtres", imagej="Process > Find Edges")
def find_edges(
    image: Image,
    method: Annotated[
        Literal["Sobel (ImageJ)", "Scharr", "Prewitt", "Laplacien", "Canny (contours fins)"],
        Param(
            label="méthode", description="Canny donne des contours d'un pixel, les autres une force de bord"
        ),
    ] = "Sobel (ImageJ)",
    sigma: Annotated[
        float,
        Param(
            label="lissage avant",
            min=0,
            max=20,
            step=0.1,
            unit="px",
            description="flou appliqué d'abord, pour ne pas suivre le bruit (Canny : 1 à 3)",
        ),
    ] = 0.0,
    low: Annotated[
        float, Param(label="seuil bas (Canny)", min=0, max=100, step=1, unit="%", advanced=True)
    ] = 10.0,
    high: Annotated[
        float, Param(label="seuil haut (Canny)", min=0, max=100, step=1, unit="%", advanced=True)
    ] = 20.0,
) -> Image:
    """Fait ressortir les bords. Par défaut : filtre de Sobel 3×3 d'ImageJ (racine de la somme
    des carrés des deux dérivées).

    En 8 bits la valeur est tronquée (comme ImageJ) ; en 16 bits elle est arrondie."""
    if method != "Sobel (ImageJ)" or sigma > 0:
        return _edges_other(image, method, sigma, low, high)

    def one(a: np.ndarray) -> np.ndarray:
        s1 = _correlate3(a, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        s2 = _correlate3(a, [[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        out = np.sqrt(s1 * s1 + s2 * s2)
        return np.floor(out) if a.dtype == np.uint8 else out

    return _apply(image, one, "contours")


def parse_kernel(text: str) -> np.ndarray:
    """Noyau écrit comme dans ImageJ : une ligne par rangée, valeurs séparées par des espaces."""
    rows = [r.replace(",", " ").split() for r in text.strip().splitlines() if r.strip()]
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError("Le noyau doit être rectangulaire : une ligne de texte par rangée")
    kernel = np.array([[float(x) for x in r] for r in rows])
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Le noyau doit avoir une largeur et une hauteur impaires")
    return kernel


@operation("Convolution", category="Filtres", imagej="Process > Filters > Convolve...")
def convolve(
    image: Image,
    kernel: Annotated[str, Param(label="noyau", multiline=True, description="une rangée par ligne")] = (
        "-1 -1 -1\n-1  8 -1\n-1 -1 -1"
    ),
    normalize: Annotated[
        bool,
        Param(
            label="normaliser", description="divise par la somme des coefficients (si elle n'est pas nulle)"
        ),
    ] = True,
) -> Image:
    """Filtre linéaire avec un noyau libre. Comme dans ImageJ, le noyau n'est pas retourné
    (corrélation) et le pixel du bord est répété hors de l'image."""
    k = parse_kernel(kernel)
    total = k.sum()
    if normalize and total != 0:
        k = k / total
    return _apply(image, lambda a: ndi.correlate(a.astype(np.float64), k, mode="nearest"), "convolution")


# ---------------------------------------------------------------------- Fourier
def fft_bandpass_plane(
    plane: np.ndarray, large: float, small: float, stripes: str = "aucune", tolerance: float = 5.0
) -> np.ndarray:
    """Filtre passe-bande d'ImageJ (``FFTFilter``) sur un plan, résultat en flottants.

    L'image est placée au centre d'un carré dont le côté est une puissance de 2 (au moins
    1,5 fois sa taille), complété par des copies en miroir. Le filtre est gaussien dans
    l'espace de Fourier ; la composante continue (la moyenne) est conservée.
    """
    h, w = plane.shape
    n = 2
    while n < 1.5 * max(h, w):
        n *= 2
    top, left = int(np.floor((n - h) / 2.0 + 0.5)), int(np.floor((n - w) / 2.0 + 0.5))
    padded = np.pad(plane.astype(np.float64), ((top, n - h - top), (left, n - w - left)), mode="symmetric")
    freq = np.abs(np.fft.fftfreq(n, d=1.0 / n))  # indices de fréquence 0..n/2
    row2, col2 = (freq * freq)[:, None], (freq * freq)[None, :]
    scale_large = (2.0 * large / n) ** 2
    scale_small = (2.0 * small / n) ** 2
    factor = (1.0 - np.exp(-(row2 + col2) * scale_large)) * np.exp(-(row2 + col2) * scale_small)
    if stripes != "aucune":
        scale_stripes = ((100.0 - tolerance) / 100.0) ** 2
        # bandes horizontales : énergie le long de l'axe vertical des fréquences (colonne 0)
        along = col2 if stripes == "horizontales" else row2
        factor = factor * (1.0 - np.exp(-along * scale_stripes))
    factor[0, 0] = 1.0
    out = np.fft.ifft2(np.fft.fft2(padded) * factor).real
    return out[top : top + h, left : left + w]


def _stretch_limits(values: np.ndarray, saturated: float) -> tuple[float, float]:
    """Bornes de ``ContrastEnhancer.getMinAndMax`` : ``saturated`` % de pixels écrêtés en tout."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    if values.dtype == np.uint8:
        hist, lo, width = np.bincount(values.ravel(), minlength=256), 0.0, 1.0
    else:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            return lo, hi
        hist, _ = np.histogram(finite, bins=256, range=(lo, hi))
        width = (hi - lo) / 256.0
    threshold = int(finite.size * saturated / 200.0) if saturated > 0 else 0
    hmin = int(np.argmax(np.cumsum(hist) > threshold))
    hmax = 255 - int(np.argmax(np.cumsum(hist[::-1]) > threshold))
    return lo + hmin * width, lo + hmax * width


@operation("Filtre passe-bande (FFT)", category="Filtres", imagej="Process > FFT > Bandpass Filter...")
def fft_bandpass(
    image: Image,
    large: Annotated[
        float,
        Param(
            label="grandes structures",
            min=0,
            max=10000,
            unit="px",
            description="supprimées au-delà de cette taille",
        ),
    ] = 40.0,
    small: Annotated[
        float,
        Param(
            label="petites structures",
            min=0,
            max=1000,
            unit="px",
            description="supprimées en deçà de cette taille",
        ),
    ] = 3.0,
    stripes: Annotated[
        Literal["aucune", "horizontales", "verticales"], Param(label="bandes à supprimer")
    ] = "aucune",
    autoscale: Annotated[
        bool, Param(label="étaler le résultat", description="ramène le résultat sur toute la plage du type")
    ] = True,
    saturate: Annotated[bool, Param(label="saturer 1 %", advanced=True)] = True,
    tolerance: Annotated[
        float, Param(label="tolérance de direction", min=0, max=100, unit="%", advanced=True)
    ] = 5.0,
) -> Image:
    """Retire l'éclairage inégal (grandes structures) et le bruit fin (petites structures).

    Même principe et mêmes réglages que le filtre d'ImageJ. Pour une image couleur, seule
    la luminosité est filtrée (les teintes sont conservées)."""

    def one(a: np.ndarray) -> np.ndarray:
        out = fft_bandpass_plane(a, large, small, stripes, tolerance)
        if not autoscale:
            return ij.restore_dtype(out, a)
        lo, hi = _stretch_limits(out.astype(np.float32), 1.0 if saturate else 0.0)
        if a.dtype.kind == "f":
            return out.astype(a.dtype)
        if hi <= lo:
            return ij.restore_dtype(out, a)
        if a.dtype == np.uint8:  # conversion 32 → 8 bits d'ImageJ sur la plage étirée
            return ij.to_byte(out.astype(np.float32), lo, hi)
        return ij.restore_dtype((out - lo) / (hi - lo) * float(np.iinfo(a.dtype).max), a)

    if image.is_rgb:
        hsb = ij.rgb_to_hsb(ij.to_byte_rgb(image.data))
        hsb[..., 2] = one(hsb[..., 2])
        return image.with_data(ij.hsb_to_rgb(hsb), name=f"{image.name} (passe-bande)")
    return image.with_data(one(image.data), name=f"{image.name} (passe-bande)")


# --------------------------------------------------------------------- contraste
def _equalize_lut(hist: np.ndarray, classic: bool) -> np.ndarray:
    """Table d'égalisation d'ImageJ : racine carrée de l'histogramme, sauf mode classique."""
    top = len(hist) - 1
    h = hist.astype(np.float64)
    weight = h if classic else np.where(h < 2, h, np.sqrt(h))
    total = weight[0] + 2.0 * weight[1:top].sum() + weight[top]
    if total <= 0:
        return np.arange(top + 1)
    scale = top / total
    # somme cumulée « au milieu de la classe » : tout ce qui précède, plus la moitié
    before = weight[0] + 2.0 * np.concatenate([[0.0], np.cumsum(weight[1:top])[:-1]])
    lut = np.zeros(top + 1, np.int64)
    lut[1:top] = np.floor((before + weight[1:top]) * scale + 0.5)
    lut[top] = top
    return lut


@operation("Rehausser le contraste", category="Contraste", imagej="Process > Enhance Contrast...")
def enhance_contrast(
    image: Image,
    mode: Annotated[Literal["étirer", "égaliser"], Param(label="méthode")] = "étirer",
    saturated: Annotated[
        float,
        Param(label="pixels saturés", min=0, max=50, step=0.05, unit="%", description="pour « étirer »"),
    ] = 0.35,
    classic: Annotated[
        bool,
        Param(label="égalisation classique", description="sans la racine carrée d'ImageJ", advanced=True),
    ] = False,
) -> Image:
    """Étale l'histogramme (option « Normalize » d'ImageJ) ou l'égalise.

    Les valeurs des pixels sont modifiées : à réserver à l'affichage et à la segmentation,
    pas aux mesures d'intensité. Pour seulement mieux voir l'image, régler le contraste du
    calque suffit."""
    d = image.data
    if mode == "étirer":
        gray = ij.intensity_plane(d, image.is_rgb)
        lo, hi = _stretch_limits(gray, saturated)
        if hi <= lo:
            return image.with_data(d.copy(), name=f"{image.name} (contraste)")
        v = d.astype(np.float64)
        if image.is_rgb or d.dtype == np.uint8:
            mult = 256.0 if image.is_rgb else 255.0
            out = np.clip(np.trunc((v - lo) / (hi - lo) * mult), 0, 255).astype(np.uint8)
            if image.is_rgb and d.shape[-1] == 4:
                out[..., 3] = d[..., 3]
        elif d.dtype.kind in "ui":
            top = float(np.iinfo(d.dtype).max)
            out = np.clip(np.trunc((v - lo) / (hi - lo) * top), 0, top).astype(d.dtype)
        else:
            out = ((v - lo) / (hi - lo)).astype(d.dtype)
        return image.with_data(out, name=f"{image.name} (contraste)")
    if image.is_rgb:  # égalisation de la luminosité seule : les teintes ne bougent pas
        hsb = ij.rgb_to_hsb(ij.to_byte_rgb(d))
        lut = _equalize_lut(np.bincount(hsb[..., 2].ravel(), minlength=256), classic)
        hsb[..., 2] = lut[hsb[..., 2]]
        return image.with_data(ij.hsb_to_rgb(hsb), name=f"{image.name} (égalisée)")
    if d.dtype.kind not in "ui":
        raise ValueError("L'égalisation demande une image 8 ou 16 bits (comme dans ImageJ)")
    top = int(np.iinfo(d.dtype).max)
    lut = _equalize_lut(np.bincount(d.ravel(), minlength=top + 1), classic)
    return image.with_data(lut[d].astype(d.dtype), name=f"{image.name} (égalisée)")


@operation("Contraste local (CLAHE)", category="Contraste", imagej="Process > Enhance Local Contrast (CLAHE)")
def clahe(
    image: Image,
    block: Annotated[int, Param(label="taille des blocs", min=8, max=2048, unit="px")] = 127,
    bins: Annotated[int, Param(label="classes", min=16, max=4096, advanced=True)] = 256,
    slope: Annotated[float, Param(label="pente maximale", min=1, max=20, step=0.5)] = 3.0,
) -> Image:
    """Égalisation adaptative à contraste limité : révèle les détails dans les zones
    sombres et claires à la fois. Mêmes réglages que le plugin de Fiji ; le résultat peut
    en différer légèrement (autre découpage en blocs)."""
    from skimage import exposure

    def one(a: np.ndarray) -> np.ndarray:
        # 8 bits : les 256 niveaux, résultat étalé sur 0–255 (comme le plugin) ; sinon la
        # plage de l'image est conservée
        lo, hi = (0.0, 255.0) if a.dtype == np.uint8 else (float(a.min()), float(a.max()))
        if hi <= lo:
            return a.astype(np.float64)
        norm = (a.astype(np.float64) - lo) / (hi - lo)
        size = (min(block, a.shape[0]), min(block, a.shape[1]))
        return (
            exposure.equalize_adapthist(norm, kernel_size=size, clip_limit=slope / bins, nbins=bins)
            * (hi - lo)
            + lo
        )

    if image.is_rgb:
        hsb = ij.rgb_to_hsb(ij.to_byte_rgb(image.data))
        hsb[..., 2] = ij.restore_dtype(one(hsb[..., 2]), np.uint8)
        return image.with_data(ij.hsb_to_rgb(hsb), name=f"{image.name} (CLAHE)")
    return image.with_data(ij.restore_dtype(one(image.data), image.data), name=f"{image.name} (CLAHE)")
