"""Les fonctions d'ImageJ recodées dans Fuji.

Chaque résultat est comparé à une valeur calculée indépendamment : la formule d'ImageJ
écrite « à la main » (boucles, arithmétique entière de Java), un cas dont on connaît la
réponse exacte, ou une autre bibliothèque. La comparaison avec ImageJ lui-même est dans
``test_imagej_parity.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage as ndi

from fuji.analysis import Image, Labels, Points, Rois, get_operation, ij, list_operations, run
from fuji.analysis.maxima import find_maxima
from fuji.analysis.ops import analyze, arith, binary, color, filters
from fuji.analysis.registry import combine_planes
from fuji.analysis.thresholds import (
    LOCAL_METHODS,
    METHODS,
    histogram256,
    local_threshold,
    threshold_level,
    threshold_range,
)

RNG = np.random.default_rng(42)
A8 = RNG.integers(0, 256, (48, 64), dtype=np.uint8)
A16 = RNG.integers(100, 4000, (48, 64), dtype=np.uint16)
RGB = RNG.integers(0, 256, (40, 50, 3), dtype=np.uint8)


def _neighbours(a: np.ndarray) -> list[np.ndarray]:
    """Les 9 voisins de chaque pixel (p1..p9 d'ImageJ), bord répété."""
    h, w = a.shape
    pad = np.pad(a.astype(np.int64), 1, mode="edge")
    return [pad[i : i + h, j : j + w] for i in range(3) for j in range(3)]


def _bimodal() -> np.ndarray:
    """Image 8 bits à deux populations bien séparées (fond ≈ 60, objets ≈ 180)."""
    values = np.concatenate([RNG.normal(60, 12, 30000), RNG.normal(180, 15, 12000)])
    return np.clip(np.round(values), 0, 255).astype(np.uint8).reshape(140, 300)


# ============================================================ conventions d'ImageJ
def test_type_conversion_follows_imagej_scaling():
    assert ij.to_byte(np.array([[100, 200, 300, 355]], np.uint16)).tolist() == [[0, 100, 200, 255]]
    assert ij.to_byte(np.array([[0.0, 0.5, 1.0]], np.float32)).tolist() == [[0, 128, 255]]
    assert ij.gray8(np.array([[[10, 11, 12], [255, 255, 254]]], np.uint8)).tolist() == [[11, 254]]
    img16 = Image(A16)
    lo, hi = int(A16.min()), int(A16.max())
    expected = np.minimum(((A16.astype(np.int64) - lo) * (256.0 / (hi - lo + 1)) + 0.5).astype(np.int64), 255)
    assert (run("Convertir le type", img16, type="8 bits").data == expected).all()
    clipped = run("Convertir le type", img16, type="8 bits", scale=False).data
    assert clipped.max() == 255 and clipped.dtype == np.uint8
    assert run("Convertir le type", Image(A8), type="32 bits").data.dtype == np.float32
    assert (run("Convertir le type", Image(RGB), type="8 bits").data == RGB.astype(int).sum(-1) // 3).all()


def test_hsb_matches_java_single_precision():
    f = np.float32

    def java(r, g, b):  # java.awt.Color.RGBtoHSB puis (int)(x * 255.0)
        cmax, cmin = max(r, g, b), min(r, g, b)
        bri = f(cmax) / f(255.0)
        sat = f(cmax - cmin) / f(cmax) if cmax else f(0)
        hue = f(0)
        if sat != 0:
            d = f(cmax - cmin)
            rc, gc, bc = f(cmax - r) / d, f(cmax - g) / d, f(cmax - b) / d
            hue = bc - gc if r == cmax else (f(2.0) + rc - bc if g == cmax else f(4.0) + gc - rc)
            hue = hue / f(6.0)
            hue = hue + f(1.0) if hue < 0 else hue
        return [int(float(x) * 255.0) for x in (hue, sat, bri)]

    expected = np.array([[java(*map(int, px)) for px in row] for row in RGB], np.uint8)
    assert (ij.rgb_to_hsb(RGB) == expected).all()
    pure = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]], np.uint8)
    assert ij.rgb_to_hsb(pure)[0].tolist() == [[0, 255, 255], [85, 255, 255], [170, 255, 255], [0, 0, 128]]
    assert (
        np.abs(ij.hsb_to_rgb(ij.rgb_to_hsb(RGB)).astype(int) - RGB).max() <= 8
    )  # perte du passage en 8 bits


def test_gaussian_kernel_is_the_imagej_one():
    k = ij.gaussian_kernel(2.0, 0.002)
    radius = math.ceil(2.0 * math.sqrt(-2 * math.log(0.002))) + 1
    assert len(k) == 2 * radius - 1 and k.sum() == pytest.approx(1.0) and (k == k[::-1]).all()
    assert k[0] < 1e-3 * k.max()  # la queue arrive à zéro sans marche
    flat = ij.gaussian_blur(np.full((20, 30), 7, np.uint8), 3.0)
    assert np.allclose(flat, 7.0)  # une image uniforme le reste (bord répété, noyau normalisé)
    blurred = run("Flou gaussien", Image(A8), sigma=1.5).data
    reference = ndi.gaussian_filter(A8.astype(float), 1.5, mode="nearest", truncate=6)
    assert np.abs(blurred - reference).max() <= 1  # même flou qu'une gaussienne exacte, à l'arrondi près


# ==================================================================== seuillages
def test_seventeen_imagej_threshold_methods():
    assert len(METHODS) == 17 and METHODS[0] == "Default"
    hist, _, _ = histogram256(_bimodal())
    levels = {method: threshold_level(hist, method) for method in METHODS}
    for method, level in levels.items():
        assert 60 <= level <= 180, method  # toutes coupent entre les deux populations
    for method in ("Default", "IJ_IsoData", "IsoData", "Otsu", "Intermodes", "Minimum", "Huang", "MinError"):
        assert 105 <= levels[method] <= 135, method  # celles-ci visent le creux de l'histogramme
    assert levels["Default"] == levels["IJ_IsoData"]
    assert threshold_level(np.zeros(256, int), "Otsu") == 0  # image vide : pas d'erreur


def test_threshold_methods_agree_with_independent_implementations():
    from skimage import filters as skf

    img = _bimodal()
    hist, _, _ = histogram256(img)
    assert threshold_level(hist, "Otsu") == pytest.approx(skf.threshold_otsu(img), abs=1)
    assert threshold_level(hist, "Yen") == pytest.approx(skf.threshold_yen(img), abs=1)
    assert threshold_level(hist, "IsoData") == pytest.approx(skf.threshold_isodata(img), abs=1)
    assert threshold_level(hist, "Mean") == math.floor(img.mean())
    assert threshold_level(hist, "Triangle") == pytest.approx(skf.threshold_triangle(img), abs=2)
    assert threshold_level(hist, "Minimum") == pytest.approx(skf.threshold_minimum(img), abs=3)
    cumulative = np.cumsum(hist) / hist.sum()
    assert threshold_level(hist, "Percentile") == int(np.argmin(np.abs(cumulative - 0.5)))
    # Li : émulation littérale du code Java (seuil entier à chaque itération)
    data = [int(x) for x in hist]
    new = sum(i * data[i] for i in range(1, 256)) / sum(data)
    while True:
        old = new
        thr = int(old + 0.5)
        nb, no = sum(data[: thr + 1]), sum(data[thr + 1 :])
        mb = sum(i * data[i] for i in range(thr + 1)) / nb
        mo = sum(i * data[i] for i in range(thr + 1, 256)) / no
        new = int((mb - mo) / (math.log(mb) - math.log(mo)) + 0.5)
        if abs(new - old) <= 0.5:
            break
    assert threshold_level(hist, "Li") == thr


def test_threshold_is_applied_like_imagej():
    img = _bimodal()
    hist, _, _ = histogram256(img)
    level = threshold_level(hist, "Otsu")
    assert threshold_range(img, "Otsu", dark_background=True) == (level + 1, 255)
    assert threshold_range(img, "Otsu", dark_background=False) == (0, level)
    bright = run("Seuillage automatique", Image(img), method="Otsu", objects="clairs").data
    dark = run("Seuillage automatique", Image(img), method="Otsu", objects="sombres").data
    assert (bright == (img > level)).all() and (dark == (img <= level)).all()
    # 16 bits : passage par 8 bits, puis min + niveau / 255 × (max − min)
    img16 = (img.astype(np.uint16) * 13 + 500).astype(np.uint16)
    lo, hi = float(img16.min()), float(img16.max())
    level16 = threshold_level(histogram256(img16)[0], "Otsu")
    lower, upper = threshold_range(img16, "Otsu", dark_background=True)
    assert lower == pytest.approx(lo + (level16 + 1) / 255 * (hi - lo)) and upper == pytest.approx(hi)
    # une image couleur est seuillée sur son gris 8 bits (R+V+B)/3
    rgb = np.stack([img, img, img], -1)
    assert (run("Seuillage automatique", Image(rgb), method="Otsu").data == bright).all()


@pytest.mark.parametrize("method", LOCAL_METHODS)
def test_local_threshold_finds_dark_spots_on_uneven_background(method):
    yy, xx = np.mgrid[0:120, 0:160]
    background = 90 + 0.8 * xx  # éclairage inégal : aucun seuil global ne convient
    img = background.copy()
    spots = [(30, 30), (30, 130), (90, 40), (90, 120), (60, 80)]
    for r, c in spots:
        img[(yy - r) ** 2 + (xx - c) ** 2 <= 16] *= 0.3
    img = np.clip(img + RNG.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    mask = local_threshold(img, method, radius=15, bright_objects=False)
    assert all(mask[r, c] for r, c in spots), method
    far_from_spots = np.ones(img.shape, bool)
    for r, c in spots:
        far_from_spots &= (yy - r) ** 2 + (xx - c) ** 2 > 22**2
    if method in ("Phansalkar", "Sauvola"):  # les autres coupent près de la moyenne locale : fond bruité
        assert mask[far_from_spots].mean() < 0.02, method
    assert not local_threshold(img, method, radius=15, bright_objects=True)[30, 30], method


def test_local_mean_threshold_formula():
    img = RNG.integers(0, 256, (20, 24), dtype=np.uint8)
    fp = ij.disk(3)
    k = fp.shape[0] // 2
    pad = np.pad(img.astype(float), k, mode="edge")
    expected = np.zeros(img.shape, bool)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            mean = pad[r : r + 2 * k + 1, c : c + 2 * k + 1][fp].mean()
            expected[r, c] = img[r, c] > int(mean - 5)
    assert (local_threshold(img, "Mean", radius=3, param1=5) == expected).all()


def test_colour_threshold_hue_wraps_around_red():
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0], img[0, 1], img[0, 2] = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    img[1, 0], img[1, 1], img[1, 2] = (255, 0, 40), (255, 40, 0), (128, 128, 128)
    reds = run("Seuillage couleur", Image(img, rgb=True), space="HSB", min1=240, max1=15, min2=100).data
    assert reds.tolist() == [[1, 0, 0], [1, 1, 0]]
    greens = run("Seuillage couleur", Image(img, rgb=True), space="RGB", min2=200, max1=50).data
    assert greens.tolist() == [[0, 1, 0], [0, 0, 0]]


# ======================================================================= filtres
def test_3x3_filters_use_imagej_integer_arithmetic():
    p = _neighbours(A8)
    assert (run("Lisser", Image(A8)).data == (sum(p) + 4) // 9).all()
    kernel = [-1, -1, -1, -1, 12, -1, -1, -1, -1]
    total = sum(k * v for k, v in zip(kernel, p, strict=True))
    assert (run("Accentuer", Image(A8)).data == np.clip(np.trunc((total + 2) / 4.0), 0, 255)).all()
    s1 = p[0] + 2 * p[1] + p[2] - p[6] - 2 * p[7] - p[8]
    s2 = p[0] + 2 * p[3] + p[6] - p[2] - 2 * p[5] - p[8]
    assert (
        run("Détecter les contours", Image(A8)).data
        == np.minimum(np.sqrt(s1 * s1 + s2 * s2).astype(int), 255)
    ).all()
    out = run("Lisser", Image(RGB)).data  # couleur : canal par canal
    assert (out[..., 1] == (sum(_neighbours(RGB[..., 1])) + 4) // 9).all()


def test_rank_filters_use_the_imagej_circular_kernel():
    fp = ij.disk(2)
    assert fp.sum() == 21 and ij.disk(1).sum() == 9 and ij.disk(3).sum() == 37
    k = fp.shape[0] // 2
    pad = np.pad(A8.astype(float), k, mode="edge")
    windows = np.array(
        [
            [pad[r : r + 2 * k + 1, c : c + 2 * k + 1][fp] for c in range(A8.shape[1])]
            for r in range(A8.shape[0])
        ]
    )
    assert (run("Filtre moyenne", Image(A8), radius=2).data == np.floor(windows.mean(-1) + 0.5)).all()
    assert (run("Filtre minimum", Image(A8), radius=2).data == windows.min(-1)).all()
    assert (run("Filtre maximum", Image(A8), radius=2).data == windows.max(-1)).all()
    assert (run("Filtre médian", Image(A8), radius=2).data == np.median(windows, axis=-1)).all()
    variance = np.clip(np.floor(windows.var(-1) + 0.5), 0, 255)
    assert (run("Filtre variance", Image(A8), radius=2).data == variance).all()
    opened = ndi.maximum_filter(
        ndi.minimum_filter(A8, footprint=fp, mode="nearest"), footprint=fp, mode="nearest"
    )
    assert (run("Chapeau haut-de-forme", Image(A8), radius=2).data == A8 - opened).all()
    assert (run("Chapeau haut-de-forme", Image(A8), radius=2, keep_background=True).data == opened).all()
    assert (
        run("Débruiter (despeckle)", Image(A8)).data == run("Filtre médian", Image(A8), radius=1).data
    ).all()


def test_unsharp_mask_convolution_and_outliers():
    blurred = ij.gaussian_blur(A8, 1.0, 0.01)
    expected = np.clip(np.floor((A8 - 0.6 * blurred) / 0.4 + 0.5), 0, 255)
    assert (run("Masque flou (netteté)", Image(A8), sigma=1.0, weight=0.6).data == expected).all()
    box = run("Convolution", Image(A8), kernel="1 1 1\n1 1 1\n1 1 1", normalize=True).data
    assert (box == run("Lisser", Image(A8)).data).all()
    shifted = run("Convolution", Image(A8), kernel="0 0 0\n1 0 0\n0 0 0", normalize=False).data
    assert (shifted[:, 1:] == A8[:, :-1]).all()  # corrélation : le noyau n'est pas retourné
    with pytest.raises(ValueError):
        filters.parse_kernel("1 2\n3 4")
    flat = np.full((20, 20), 100, np.uint8)
    flat[5, 5], flat[12, 12] = 250, 3
    assert run("Retirer les valeurs aberrantes", Image(flat), threshold=50, which="claires").data[5, 5] == 100
    assert run("Retirer les valeurs aberrantes", Image(flat), threshold=50, which="claires").data[12, 12] == 3
    assert (
        run("Retirer les valeurs aberrantes", Image(flat), threshold=50, which="sombres").data[12, 12] == 100
    )


def test_fft_bandpass_removes_slow_variations_and_keeps_the_mean():
    yy, xx = np.mgrid[0:100, 0:140]
    img = (120 + 0.6 * xx + 25 * np.sin(xx / 2.5) * np.sin(yy / 2.5)).astype(np.float32)
    out = filters.fft_bandpass_plane(img, 40, 0)
    assert abs(np.polyfit(np.arange(140), out.mean(0), 1)[0]) < 0.08  # la pente a disparu
    assert out.mean() == pytest.approx(img.mean(), abs=1.5)  # la composante continue est gardée
    assert out.std() > 8  # les détails fins sont conservés
    assert filters.fft_bandpass_plane(img, 40, 12).std() < 0.2 * out.std()  # ... sauf si on les filtre
    stripes = (100 + 30 * np.sin(yy / 2.0)).astype(np.float32)  # bandes horizontales
    assert filters.fft_bandpass_plane(stripes, 1000, 0, "horizontales").std() < 0.2 * stripes.std()
    assert filters.fft_bandpass_plane(stripes, 1000, 0, "verticales").std() > 0.8 * stripes.std()
    as8 = run("Filtre passe-bande (FFT)", Image(ij.to_byte(img))).data
    assert as8.dtype == np.uint8 and as8.max() == 255 and as8.min() == 0  # résultat étalé


def test_enhance_contrast_equalization_table_matches_imagej_loop():
    hist = np.bincount(A8.ravel(), minlength=256)

    def weight(i):  # getWeightedValue : racine carrée de l'effectif
        h = int(hist[i])
        return float(h) if h < 2 else math.sqrt(h)

    total = weight(0) + sum(2 * weight(i) for i in range(1, 255)) + weight(255)
    scale = 255 / total
    lut = [0] * 256
    acc = weight(0)
    for i in range(1, 255):
        delta = weight(i)
        acc += delta
        lut[i] = math.floor(acc * scale + 0.5)
        acc += delta
    lut[255] = 255
    assert (run("Rehausser le contraste", Image(A8), mode="égaliser").data == np.array(lut)[A8]).all()
    narrow = (A8 // 4 + 100).astype(np.uint8)  # histogramme étroit : 100..163
    stretched = run("Rehausser le contraste", Image(narrow), mode="étirer", saturated=0.0).data
    assert stretched.min() == 0 and stretched.max() == 255
    expected = np.clip(np.trunc((narrow.astype(float) - 100) / 63 * 255), 0, 255)
    assert (stretched == expected).all()
    local = run("Contraste local (CLAHE)", Image(narrow), block=16).data
    assert local.dtype == np.uint8 and local.std() > 1.5 * narrow.std()  # étalé sur 0–255


# ======================================================================== calculs
def test_math_uses_imagej_scaling_per_type():
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    v = ramp.astype(float)

    def lut(values):
        return np.clip(np.floor(np.where(np.isfinite(values), values, 0) + 0.5), 0, 255)

    with np.errstate(divide="ignore", invalid="ignore"):
        assert (arith.math_plane(ramp, "gamma", 0.5) == lut(np.exp(np.log(v / 255) * 0.5) * 255)).all()
        assert (arith.math_plane(ramp, "logarithme", 0) == lut(np.log(v) * 255 / np.log(255))).all()
        assert (arith.math_plane(ramp, "exponentielle", 0) == lut(np.exp(v * np.log(255) / 255))).all()
    assert (arith.math_plane(ramp, "multiplier", 1.5) == lut(v * 1.5)).all()
    assert (arith.math_plane(ramp, "ajouter", 100) == np.minimum(v + 100, 255)).all()
    assert (arith.math_plane(ramp, "minimum", 50) == np.maximum(v, 50)).all()  # « Min... » relève les valeurs
    assert (arith.math_plane(ramp, "maximum", 50) == np.minimum(v, 50)).all()
    assert (arith.math_plane(ramp, "racine carrée", 0) == lut(np.sqrt(v))).all()
    f = np.array([[-4.0, 0.0, 4.0]], np.float32)
    assert arith.math_plane(f, "carré", 0).tolist() == [[16.0, 0.0, 16.0]]
    assert arith.math_plane(f, "valeur absolue", 0).tolist() == [[4.0, 0.0, 4.0]]
    assert arith.math_plane(f, "racine carrée", 0).tolist() == [[0.0, 0.0, 2.0]]
    assert arith.math_plane(f, "soustraire", 1).tolist() == [[-5.0, -1.0, 3.0]]  # 32 bits : pas de borne
    sixteen = np.array([[100, 1100, 2100]], np.uint16)
    assert arith.math_plane(sixteen, "gamma", 0.5).tolist() == [
        [100, round(math.sqrt(0.5) * 2000 + 100), 2100]
    ]


def test_image_calculator_saturates_like_imagej():
    a = np.array([[100, 200, 7, 0]], np.uint8)
    b = np.array([[150, 100, 2, 0]], np.uint8)
    calc = arith.calculate
    assert calc(a, b, "soustraire", False).tolist() == [[0, 100, 5, 0]]
    assert calc(a, b, "additionner", False).tolist() == [[250, 255, 9, 0]]
    assert calc(a, b, "diviser", False).tolist() == [[0, 2, 3, 255]]  # division entière ; /0 → 255
    assert calc(a, b, "moyenne", False).tolist() == [[125, 150, 4, 0]]
    assert calc(a, b, "différence", False).tolist() == [[50, 100, 5, 0]]
    assert calc(a, b, "minimum", False).tolist() == [[100, 100, 2, 0]]
    assert calc(a, b, "ET", False).tolist() == [[100 & 150, 200 & 100, 7 & 2, 0]]
    signed = calc(a, b, "soustraire", True)
    assert signed.dtype == np.float32 and signed.tolist() == [[-50.0, 100.0, 5.0, 0.0]]
    out = run("Calculatrice d'images", Image(RGB), Image(RGB[..., 0]), operation_="soustraire").data
    assert out.shape == RGB.shape and (out[..., 0] == 0).all()  # gris appliqué à chaque canal
    with pytest.raises(ValueError):
        run("Calculatrice d'images", Image(A8), Image(A8[:10]))


# ======================================================================== couleur
def test_colour_deconvolution_matches_the_fiji_plugin():
    he = np.array([v for _, v in color.STAINS["H&E"]])
    m = color.stain_matrix(he)
    # troisième vecteur déduit : valeurs affichées par le plugin de Fiji pour « H&E »
    assert m[2] == pytest.approx([0.6359544, 0.001, 0.7717266], abs=2e-6)
    assert np.allclose(np.sqrt((m * m).sum(1)), 1.0, atol=1e-5)
    # formule du plugin, pixel par pixel
    q = np.linalg.inv(m)
    log255 = math.log(255.0)
    for r, g, b in ((255, 255, 255), (120, 60, 140), (200, 100, 180), (10, 20, 30)):
        od = [-((255.0 * math.log((c + 1) / 255.0)) / log255) for c in (r, g, b)]
        expected = []
        for i in range(3):
            scaled = sum(od[c] * q[c, i] for c in range(3))
            expected.append(math.floor(min(math.exp(-(scaled - 255.0) * log255 / 255.0), 255.0) + 0.5))
        got = color.colour_deconvolution(np.array([[[r, g, b]]], np.uint8), he)[0][0, 0]
        assert got.tolist() == expected
    assert (
        color.colour_deconvolution(np.full((2, 2, 3), 255, np.uint8), he)[0].min() == 255
    )  # blanc : aucun colorant


def test_colour_deconvolution_recovers_known_stain_amounts():
    vectors = np.array([v for _, v in color.STAINS["H DAB"]])
    m = color.stain_matrix(vectors)
    amounts = RNG.uniform(0, 1.0, (30, 40, 3))
    amounts[..., 2] = 0  # hématoxyline et DAB seulement
    rgb = np.clip(np.round(255 * 10 ** (-(amounts @ m))), 0, 255).astype(np.uint8)
    h, dab, residual = run(
        "Déconvolution des couleurs", Image(rgb), stains="H DAB", output="densités optiques"
    )
    assert np.abs(h.data - amounts[..., 0]).mean() < 0.03
    assert np.abs(dab.data - amounts[..., 1]).mean() < 0.03
    assert np.abs(residual.data).mean() < 0.03
    assert "hématoxyline" in h.name and "DAB" in dab.name
    with pytest.raises(ValueError):
        run("Déconvolution des couleurs", Image(A8))


def test_channels_merge_split_and_white_balance():
    merged = run("Fusionner en RGB", Image(RGB[..., 0]), None, Image(RGB[..., 2]))
    assert merged.is_rgb and (merged.data[..., 0] == RGB[..., 0]).all() and (merged.data[..., 1] == 0).all()
    h, s, b = run("Séparer en HSB", Image(RGB))
    assert (np.stack([h.data, s.data, b.data], -1) == ij.rgb_to_hsb(RGB)).all()
    lum, _a, _b = run("Séparer en Lab", Image(np.array([[[255, 255, 255], [0, 0, 0]]] * 5, np.uint8)))
    assert lum.data[0].tolist() == pytest.approx([100.0, 0.0], abs=0.01)
    tinted = np.full((20, 20, 3), (200, 180, 150), np.uint8)
    tinted[5:10, 5:10] = (100, 60, 30)
    balanced = run("Balance des blancs", Image(tinted), Rois.rectangle(12, 12, 6, 6)).data
    assert balanced[15, 15].tolist() == [200, 200, 200] and balanced[7, 7].tolist() == [100, 67, 40]
    assert run("Balance des blancs", Image(tinted)).data[15, 15].tolist() == [200, 200, 200]  # fond deviné


# ====================================================================== géométrie
def test_geometry_operations():
    img = Image(np.arange(24, dtype=np.uint8).reshape(4, 6), 0.5)
    assert (run("Retourner", img).data == img.data[:, ::-1]).all()
    assert (run("Retourner", img, direction="verticalement").data == img.data[::-1]).all()
    assert (run("Tourner de 90°", img).data == np.rot90(img.data, -1)).all()
    assert (run("Tourner de 90°", img, direction="à gauche").data == np.rot90(img.data)).all()
    diagonal = Image(np.eye(7, dtype=np.uint8) * 200)
    quarter = run("Tourner", diagonal, angle=90, interpolation="aucune").data
    assert ((quarter > 0) == np.rot90(np.eye(7) > 0, -1)).all()  # angle positif = sens horaire
    assert run("Tourner", img, angle=30, enlarge=True).data.shape > img.data.shape
    moved = run("Translater", img, dx=2, dy=1).data
    assert (moved[1:, 2:] == img.data[:-1, :-2]).all() and (moved[0] == 0).all()
    double = run("Redimensionner", img, factor=2.0)
    assert double.data.shape == (8, 12) and double.pixel_size_um == pytest.approx(0.25)
    binned = run("Regrouper les pixels (bin)", img)
    assert binned.data.tolist() == [[4, 6, 8], [16, 18, 20]] and binned.pixel_size_um == 1.0
    assert run("Regrouper les pixels (bin)", img, method="somme").data.tolist() == [
        [14, 22, 30],
        [62, 70, 78],
    ]
    assert run("Rogner", img, Rois.rectangle(1, 1, 3, 2)).data.tolist() == [[7, 8, 9], [13, 14, 15]]
    assert run("Rogner", img, None, x=4, y=2, width=0, height=0).data.tolist() == [[16, 17], [22, 23]]
    larger = run("Taille du support", img, width=10, height=6).data
    assert larger.shape == (6, 10) and (larger[1:5, 2:8] == img.data).all() and larger.sum() == img.data.sum()
    assert run("Taille du support", img, width=2, height=2, position="haut-gauche").data.tolist() == [
        [0, 1],
        [6, 7],
    ]
    assert run("Retourner", Image(RGB)).data.shape == RGB.shape and run("Tourner de 90°", Image(RGB)).is_rgb


# ======================================================================== binaire
def test_binary_morphology_follows_imagej_rules():
    square = np.zeros((9, 9), bool)
    square[2:7, 2:7] = True
    assert binary.erode(square).sum() == 9 and binary.dilate(square).sum() == 49
    assert binary.erode(square, count=3).sum() == 9  # bords (3 voisins de fond) et coins (5) enlevés
    assert binary.erode(square, count=4).sum() == 21  # seulement les 4 coins
    assert binary.erode(square, iterations=3).sum() == 0
    edge = np.zeros((5, 5), bool)
    edge[:, :3] = True
    assert binary.erode(edge).sum() == 3  # le bord de l'image compte comme du fond
    assert binary.erode(edge, pad_edges=True).sum() == 10
    mask = Labels(square.astype(np.uint8))
    assert run("Contour des objets", mask).data.sum() == 16
    assert run("Inverser le masque", mask).data.sum() == 81 - 25
    assert (run("Érosion", mask).data == binary.erode(square)).all()
    assert run("Dilatation", mask, iterations=2).data.sum() == 81
    other = np.zeros((9, 9), np.uint8)
    other[4:9, 4:9] = 1
    assert run("Combiner deux masques", mask, Labels(other)).data.sum() == 9
    assert run("Combiner deux masques", mask, Labels(other), operation_="1 sans 2").data.sum() == 16
    line = np.zeros((15, 30), np.uint8)
    line[5:10, 3:27] = 1
    skeleton = run("Squelette", Labels(line)).data
    assert skeleton.sum() < 30 and skeleton[7].sum() >= 14 and (skeleton <= line).all()


def test_distance_map_ultimate_points_and_voronoi():
    rr, cc = np.ogrid[:40, :80]
    two = np.zeros((40, 80), np.uint8)
    two[(rr - 20) ** 2 + (cc - 20) ** 2 <= 144] = 1
    two[(rr - 20) ** 2 + (cc - 55) ** 2 <= 100] = 1
    edm = run("Carte des distances", Labels(two, 0.5)).data
    assert edm.dtype == np.float32 and edm[20, 20] == pytest.approx(math.sqrt(145)) and edm[0, 0] == 0
    assert run("Carte des distances", Labels(two, 0.5), calibrated=True).data[20, 20] == pytest.approx(
        math.sqrt(145) / 2
    )
    assert run("Carte des distances", Labels(two), output="8 bits (comme ImageJ)").data[20, 20] == 12
    points = run("Points ultimes", Labels(two)).data
    assert np.argwhere(points > 0).tolist() == [[20, 20], [20, 55]] and points[20, 20] == 12
    lines = run("Voronoï", Labels(two)).data
    assert lines.sum() > 30 and set(np.flatnonzero(lines.any(0))) <= {
        37,
        38,
        39,
        40,
    }  # à mi-distance des bords
    assert run("Voronoï", Labels(np.zeros((5, 5), np.uint8))).data.sum() == 0


# ========================================================================= piles
def test_stack_operations():
    data = RNG.integers(0, 200, (5, 3, 20, 30), dtype=np.uint8)
    stack = Image(data, 0.4)
    assert get_operation("Projection").stack and not get_operation("Flou gaussien").stack
    assert (run("Projection", stack).data == data.max(1)).all()
    assert (run("Projection", stack, axis="premier curseur").data == data.max(0)).all()
    assert (run("Projection", stack, method="minimum", first=2, last=3).data == data[:, 1:3].min(1)).all()
    mean = run("Projection", stack, method="moyenne").data
    assert mean.dtype == np.float32 and np.allclose(mean, data.mean(1))
    assert np.allclose(run("Projection", stack, method="somme").data, data.sum(1))
    assert np.allclose(run("Projection", stack, method="écart-type").data, data.std(1, ddof=1), atol=1e-4)
    assert (run("Projection", stack, method="médiane").data == np.median(data, 1)).all()
    assert run("Sous-pile", stack, first=2, last=4, step=2, axis="premier curseur").data.shape == (
        2,
        3,
        20,
        30,
    )
    assert (run("Inverser l'ordre de la pile", stack).data == data[:, ::-1]).all()
    sheet = run("Montage", Image(data[0]), columns=3)
    assert sheet.data.shape == (20, 90) and (sheet.data[:, 30:60] == data[0, 1]).all()
    assert run("Montage", Image(data[0]), columns=2, scale=0.5, border=2).data.shape == (22, 32)
    both = run("Empiler deux images", Image(data[0, 0]), Image(data[0]))
    assert both.data.shape == (4, 20, 30) and (both.data[0] == data[0, 0]).all()
    colour = Image(RNG.integers(0, 255, (4, 10, 12, 3), dtype=np.uint8), rgb=True)
    assert run("Projection", colour).data.shape == (10, 12, 3) and run("Projection", colour).is_rgb
    with pytest.raises(ValueError):
        run("Projection", Image(A8))


def test_stack_registration_removes_a_known_drift():
    base = np.zeros((64, 64))
    base[20:40, 25:45] = 100
    base = ndi.gaussian_filter(base, 2)
    drift = np.stack([ndi.shift(base, (i * 1.5, -i * 2.0)) for i in range(5)]).astype(np.float32)
    registered, table = run("Recaler la pile", Image(drift, 0.5))
    assert np.abs(registered.data[4] - registered.data[0]).max() < 2.0 < np.abs(drift[4] - drift[0]).max()
    assert table.data["dX"].iloc[-1] == pytest.approx(-8.0 * 0.5, abs=0.1)  # dérive en µm
    assert table.data["dY"].iloc[-1] == pytest.approx(6.0 * 0.5, abs=0.1)


def test_planes_are_recombined_into_stacks_tables_and_points():
    op = get_operation("Trouver les maxima")
    planes = []
    for k in range(3):
        plane = np.zeros((12, 12), np.uint8)
        plane[3 + k, 4] = 200
        planes.append(Image(plane, 0.5, "pile"))
    indices = [(0,), (1,), (2,)]
    points, table, _ = combine_planes([op(p, prominence=10) for p in planes], indices, (3,))
    assert points.data.tolist() == [[0, 3, 4], [1, 4, 4], [2, 5, 4]]
    assert table.data["Plan"].tolist() == ["1", "2", "3"] and len(table) == 3
    blurred = combine_planes([run("Flou gaussien", p) for p in planes], indices, (3,))
    assert blurred.data.shape == (3, 12, 12)


# ================================================================= ROI et mesures
def test_roi_masks_use_pixel_centres():
    rect = Rois.rectangle(10, 20, 30, 15)
    mask = rect.mask(0, (100, 100))
    assert mask.sum() == 450 and mask[20:35, 10:40].all() and rect.length(0) == 90
    oval = Rois.oval(10, 10, 40, 40)
    count = oval.mask(0, (100, 100)).sum()
    yy, xx = np.mgrid[0:100, 0:100]
    assert count == ((yy + 0.5 - 30) ** 2 + (xx + 0.5 - 30) ** 2 <= 400).sum()
    assert oval.length(0) == pytest.approx(2 * math.pi * 20)
    flat_oval = Rois.oval(0, 0, 40, 20)
    a, b = 20, 10
    assert flat_oval.length(0) == pytest.approx(
        math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    )
    poly = Rois.polygon([10, 50, 50, 10], [10, 10, 30, 30])
    assert poly.mask(0, (60, 60)).sum() == 800 and poly.length(0) == 120
    assert Rois.rectangle(-5, -5, 10, 10).mask(0, (20, 20)).sum() == 25  # coupée par le bord
    assert Rois.line(0, 0, 3, 4).length(0) == 5 and Rois.line(0, 0, 3, 4).is_line(0)
    with pytest.raises(ValueError):
        Rois.line(0, 0, 3, 4).mask(0, (10, 10))


def test_roi_measurements_match_hand_computation():
    plane = (np.arange(100, dtype=np.uint8).reshape(10, 10) * 2).astype(np.uint8)
    row = analyze.measure_rois(plane, Rois.rectangle(2, 3, 4, 2), 0.5).iloc[0]
    sub = plane[3:5, 2:6]
    assert row["Area"] == 8 * 0.25 and row["Mean"] == sub.mean() and row["Min"] == sub.min()
    assert row["StdDev"] == pytest.approx(sub.std(ddof=1)) and row["RawIntDen"] == sub.sum()
    assert row["IntDen"] == pytest.approx(row["Area"] * row["Mean"])
    assert (row["X"], row["Y"]) == (2.0, 2.0)  # centre (4, 4) px × 0,5 µm
    assert (row["BX"], row["BY"], row["Width"], row["Height"]) == (1.0, 1.5, 2.0, 1.0)
    assert row["Perim."] == 6.0 and row["Feret"] == pytest.approx(math.hypot(4, 2) * 0.5)
    assert row["Median"] == np.sort(sub.ravel())[4]  # médiane « haute » des images entières
    assert row["Solidity"] == pytest.approx(1.0) and row["AR"] == pytest.approx(2.0, rel=0.02)
    weights = sub.astype(float)
    xs = np.arange(2, 6) + 0.5
    assert row["XM"] == pytest.approx((weights.sum(0) * xs).sum() / weights.sum() * 0.5)
    disc = analyze.measure_rois(np.full((80, 80), 9, np.uint8), Rois.oval(10, 10, 50, 50)).iloc[0]
    assert disc["Circ."] == pytest.approx(1.0, abs=0.01) and disc["Round"] == pytest.approx(1.0, abs=0.01)
    assert disc["Major"] == pytest.approx(50, abs=0.3) and disc["Feret"] == pytest.approx(50, abs=1.5)
    line = analyze.measure_rois(plane, Rois.line(0, 0, 3, 4), 1.0).iloc[0]
    assert line["Length"] == 5.0 and line["Angle"] == pytest.approx(-53.13, abs=0.01)
    table = run("Mesurer les ROI", Image(plane, 0.5), Rois.rectangle(2, 3, 4, 2))
    assert table.units["Area"] == "µm²" and table.data["Type"].iloc[0] == "rectangle"


def test_intensity_statistics_follow_imagej_definitions():
    values = np.array([1, 2, 2, 3, 4, 10], np.uint8)
    stats = analyze.intensity_stats(values)
    assert stats["Median"] == 3 and stats["Mode"] == 2  # médiane haute, valeur la plus fréquente
    mean = values.mean()
    m2, m3, m4 = (((values - mean) ** k).mean() for k in (2, 3, 4))
    assert stats["Skew"] == pytest.approx(m3 / m2**1.5) and stats["Kurt"] == pytest.approx(m4 / m2**2 - 3)
    assert stats["StdDev"] == pytest.approx(values.std(ddof=1))
    assert (
        analyze.intensity_stats(np.array([1.0, 2.0, 3.0, 4.0]))["Median"] == 2.5
    )  # 32 bits : médiane classique
    whole = run("Mesurer", Image(A8, 0.5))
    assert whole.data["Area"].iloc[0] == A8.size * 0.25 and whole.data["Mean"].iloc[0] == pytest.approx(
        A8.mean()
    )
    masked = run("Mesurer", Image(A8), Labels((A8 > 128).astype(np.uint8)))
    assert masked.data["Min"].iloc[0] > 128 and masked.data["%Area"].iloc[0] == pytest.approx(
        100 * (A8 > 128).mean()
    )
    grey = run("Mesurer", Image(RGB))  # couleur : gris 8 bits (R+V+B)/3
    assert grey.data["Mean"].iloc[0] == pytest.approx((RGB.astype(int).sum(-1) // 3).mean())


def test_line_profile_like_imagej_getline():
    plane = (np.arange(100).reshape(10, 10) * 2).astype(np.float64)
    assert analyze.sample_line(plane, (0, 0), (0, 9)).tolist() == list(range(0, 20, 2))
    diagonal = analyze.sample_line(plane, (0, 0), (3, 4))  # longueur 5 → 6 points
    assert len(diagonal) == 6 and diagonal[1] == pytest.approx(2 * (0.6 * 10 + 0.8))
    wide = analyze.sample_line(plane, (2, 0), (2, 9), width=3)
    assert wide.tolist() == plane[2].tolist()  # moyenne des lignes 1, 2, 3 d'un dégradé linéaire
    table = run("Profil d'intensité", Image(plane, 0.5), Rois.line(0, 2, 9, 2))
    assert table.data["Valeur"].tolist() == plane[2].tolist()
    assert table.data["Distance"].tolist() == [i * 0.5 for i in range(10)] and table.plot.kind == "ligne"
    columns = run("Profil d'intensité", Image(plane), Rois.rectangle(2, 1, 5, 4)).data["Valeur"]
    assert columns.tolist() == plane[1:5, 2:7].mean(0).tolist()
    bent = Rois([("path", np.array([[0.0, 0.0], [0.0, 4.0], [3.0, 4.0]]))])
    assert len(run("Profil d'intensité", Image(plane), bent).data) == 8
    with pytest.raises(ValueError):
        run("Profil d'intensité", Image(plane), Rois())


def test_histogram_table_and_plot():
    table = run("Histogramme", Image(A8))
    assert len(table) == 256 and table.data["Nombre"].sum() == A8.size and table.plot.kind == "barres"
    assert (table.data["Nombre"].to_numpy() == np.bincount(A8.ravel(), minlength=256)).all()
    assert table.summary["Mean"] == pytest.approx(A8.mean()) and table.summary["N"] == A8.size
    part = run("Histogramme", Image(A8), Rois.rectangle(0, 0, 10, 5))
    assert part.data["Nombre"].sum() == 50
    sixteen = run("Histogramme", Image(A16), None, bins=64)
    assert len(sixteen) == 64 and sixteen.data["Nombre"].sum() == A16.size


def test_find_maxima_uses_the_imagej_prominence_rule():
    a = np.zeros((40, 60), np.float32)
    a[10, 10], a[10, 14] = 100, 90  # deux pics séparés par un creux profond
    a[25, 30:33] = 50  # plateau de trois pixels : un seul maximum, au milieu
    a[30, 50], a[30, 51], a[30, 52] = 80, 75, 78  # épaule : ne dépasse le creux que de 3
    positions, values, zones = find_maxima(a, prominence=10)
    assert positions.tolist() == [[10, 10], [10, 14], [30, 50], [25, 31]]
    assert values.tolist() == [100, 90, 80, 50] and zones.max() == 4
    assert len(find_maxima(a, prominence=2)[0]) == 5  # l'épaule compte quand la tolérance baisse
    assert len(find_maxima(a, prominence=3)[0]) == 4  # « plus que » la proéminence, pas « au moins »
    assert find_maxima(a, prominence=500)[0].tolist() == [[10, 10]]  # le maximum global est toujours gardé
    assert len(find_maxima(a, prominence=500, strict=True)[0]) == 0
    assert len(find_maxima(np.full((5, 5), 3.0), 1)[0]) == 0  # image uniforme : aucun maximum
    b = np.zeros((20, 20))
    b[0, 5], b[10, 10] = 9, 5
    assert find_maxima(b, 1, exclude_edges=True)[0].tolist() == [[10, 10]]
    assert find_maxima(-b, 1, light_background=True)[0].tolist() == [[0, 5], [10, 10]]
    points, table, labels = run("Trouver les maxima", Image(a, 0.5), prominence=10, segment=True)
    assert len(points) == 4 and table.summary["Count"] == 4 and labels.data.max() == 4
    assert table.data["X"].iloc[0] == 5.0 and table.data["Valeur"].iloc[0] == 100


def test_count_points_and_roi_mask_conversions():
    pts = Points(np.array([[2.0, 3.0], [5.0, 5.0]]), 0.5, "noyaux")
    table = run("Compter les points", pts, Image(A8, 0.5))
    assert table.summary["Count"] == 2 and table.data["X"].tolist() == [1.5, 2.5]
    assert table.data["Valeur"].tolist() == [float(A8[2, 3]), float(A8[5, 5])]
    mask = np.zeros((12, 12), np.uint8)
    mask[2:6, 3:9] = 1
    mask[8:11, 1:4] = 1
    rois = run("Masque vers ROI", Labels(mask, 0.5))
    assert len(rois) == 2 and rois.kind(0) == "polygon"
    back = run("ROI vers masque", Image(mask), rois)
    assert ((back.data > 0) == (mask > 0)).all() and back.data.max() == 2
    kept = run("Effacer avec les ROI", Image(A8[:12, :12].copy()), rois).data
    assert (kept[mask == 0] == 0).all() and (kept[mask > 0] == A8[:12, :12][mask > 0]).all()
    cleared = run("Effacer avec les ROI", Image(A8[:12, :12].copy()), rois, where="intérieur", value=7).data
    assert (cleared[mask > 0] == 7).all()


def test_colocalization_coefficients():
    signal = RNG.gamma(2.0, 20.0, 20000)
    background1, background2 = RNG.normal(5, 2, 20000).clip(0), RNG.normal(5, 2, 20000).clip(0)
    c1 = np.concatenate([signal, background1])
    c2 = np.concatenate([0.7 * signal + RNG.normal(0, 3, 20000), background2]).clip(0)
    res = analyze.colocalization(c1, c2)
    assert res["Pearson"] == pytest.approx(np.corrcoef(c1, c2)[0, 1])
    assert res["Pente"] == pytest.approx(0.7, abs=0.05)  # droite de régression orthogonale
    assert res["Pearson (sous les seuils)"] <= 0.01  # définition des seuils de Costes
    assert 3 < res["Seuil 1"] < 20 and res["tM1"] > 0.8 and res["tM2"] > 0.8
    independent = analyze.colocalization(RNG.random(8000), RNG.random(8000), 0.5, 0.5)
    assert abs(independent["Pearson"]) < 0.05 and independent["tM1"] == pytest.approx(0.5, abs=0.05)
    assert abs(independent["ICQ (Li)"]) < 0.03
    same = analyze.colocalization(c1, c1, 0.0, 0.0)
    assert same["Pearson"] == pytest.approx(1.0) and same["Recouvrement"] == pytest.approx(1.0)
    assert same["ICQ (Li)"] == pytest.approx(0.5, abs=0.01)
    exclusive = analyze.colocalization(np.array([5.0, 0, 9, 0, 4]), np.array([0, 7.0, 0, 3, 0]), 0, 0)
    assert exclusive["M1"] == 0 and exclusive["M2"] == 0
    table = run("Colocalisation", Image(c1.reshape(200, 200)), Image(c2.reshape(200, 200)))
    assert len(table) == 1 and table.data["N"].iloc[0] == 40000
    half = np.zeros((200, 200), np.uint8)
    half[:100] = 1
    assert (
        run("Colocalisation", Image(c1.reshape(200, 200)), Image(c2.reshape(200, 200)), Labels(half))
        .data["N"]
        .iloc[0]
        == 20000
    )


# ====================================================================== catalogue
def test_catalogue_covers_the_essential_imagej_commands():
    ops = list_operations()
    names = {o.name for o in ops}
    assert len(ops) >= 75
    documented = [o for o in ops if o.imagej]
    assert len(documented) >= 70  # chaque opération indique son équivalent dans les menus d'ImageJ
    for name in (
        "Convertir le type",
        "Flou gaussien",
        "Filtre médian",
        "Soustraction de fond",
        "Calculatrice d'images",
        "Seuillage automatique",
        "Analyse de particules",
        "Mesurer les ROI",
        "Profil d'intensité",
        "Histogramme",
        "Trouver les maxima",
        "Projection",
        "Déconvolution des couleurs",
        "Colocalisation",
        "Rogner",
        "Carte des distances",
    ):
        assert name in names, name
    for op in ops:
        assert op.description, op.name
