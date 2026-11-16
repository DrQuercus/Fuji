"""Comparaison directe avec ImageJ (le vrai, via PyImageJ).

Ces tests garantissent que les mesures de Fuji sont identiques à celles de Fiji.
Ils nécessitent Java et le téléchargement d'ImageJ (ou un Fiji local) :

    FUJI_TEST_IMAGEJ=1 pytest tests/test_imagej_parity.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from skimage.draw import disk, ellipse

pytestmark = pytest.mark.imagej

if not os.environ.get("FUJI_TEST_IMAGEJ"):
    pytest.skip("définir FUJI_TEST_IMAGEJ=1 pour comparer avec ImageJ", allow_module_level=True)
pytest.importorskip("imagej")

from fuji.analysis import Image, Labels, run  # noqa: E402
from fuji.analysis.imagej import ImageJBridge  # noqa: E402

COLUMNS = [
    "Area",
    "Perim.",
    "Circ.",
    "Major",
    "Minor",
    "Angle",
    "Feret",
    "MinFeret",
    "AR",
    "Round",
    "Solidity",
    "X",
    "Y",
]


@pytest.fixture(scope="module")
def bridge():
    return ImageJBridge.get()


def test_analyze_particles_parity(bridge):
    m = np.zeros((240, 320), np.uint8)
    rr, cc = disk((60, 60), 25)
    m[rr, cc] = 255
    rr, cc = ellipse(150, 170, 18, 55, rotation=math.radians(35))
    m[rr, cc] = 255
    m[30:70, 200:290] = 255
    m[190:220, 30:45] = 255
    ours, _ = run("Analyse de particules", Labels((m > 0).astype(np.uint8), 0.5))
    macro = """
setThreshold(128, 255);
run("Set Scale...", "distance=2 known=1 unit=micron");
run("Set Measurements...", "area centroid perimeter fit shape feret's redirect=None decimal=6");
run("Analyze Particles...", "display clear");
"""
    res = bridge.run_macro(macro, Image(m, 0.5, "parite"))
    theirs = res.table.data
    assert len(theirs) == len(ours)
    a = ours.data.sort_values(["Y", "X"]).reset_index(drop=True)
    b = theirs.sort_values(["Y", "X"]).reset_index(drop=True)
    for col in COLUMNS:
        np.testing.assert_allclose(a[col], b[col], rtol=2e-3, atol=2e-3, err_msg=col)


# ====================================================================================
# Fonctions d'ImageJ recodées dans Fuji : comparaison pixel à pixel avec ImageJ lui-même
# ====================================================================================
from fuji.analysis import Rois, ij  # noqa: E402
from fuji.analysis.ops import analyze  # noqa: E402
from fuji.analysis.thresholds import METHODS, threshold_range  # noqa: E402

_RNG = np.random.default_rng(7)


def _test_image(dtype=np.uint8) -> np.ndarray:
    """Image de test : objets flous sur fond en pente, avec du bruit."""
    yy, xx = np.mgrid[0:160, 0:220].astype(np.float64)
    img = 40 + 0.25 * xx
    for cy, cx, r, a in ((40, 50, 14, 120), (100, 150, 20, 90), (120, 60, 9, 150), (50, 170, 6, 110)):
        img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    img += _RNG.normal(0, 4, img.shape)
    if dtype == np.uint8:
        return np.clip(np.round(img), 0, 255).astype(np.uint8)
    if dtype == np.uint16:
        return np.clip(np.round(img * 40 + 300), 0, 65535).astype(np.uint16)
    return (img / 3.0).astype(np.float32)


def _ij(bridge, data: np.ndarray, macro: str) -> np.ndarray:
    return np.asarray(bridge.run_macro(macro, Image(data, None, "parite")).image.data)


#: (opération Fuji, réglages, macro ImageJ, écart maximal toléré)
_EXACT_8BIT = [
    ("Lisser", {}, 'run("Smooth");', 0),
    ("Accentuer", {}, 'run("Sharpen");', 0),
    ("Détecter les contours", {}, 'run("Find Edges");', 0),
    ("Inverser", {}, 'run("Invert");', 0),
    ("Flou gaussien", {"sigma": 2.0}, 'run("Gaussian Blur...", "sigma=2");', 0),
    ("Flou gaussien", {"sigma": 12.0}, 'run("Gaussian Blur...", "sigma=12");', 1),
    ("Filtre médian", {"radius": 2.0}, 'run("Median...", "radius=2");', 0),
    ("Filtre moyenne", {"radius": 3.0}, 'run("Mean...", "radius=3");', 0),
    ("Filtre minimum", {"radius": 2.0}, 'run("Minimum...", "radius=2");', 0),
    ("Filtre maximum", {"radius": 2.0}, 'run("Maximum...", "radius=2");', 0),
    ("Filtre variance", {"radius": 2.0}, 'run("Variance...", "radius=2");', 0),
    ("Chapeau haut-de-forme", {"radius": 10.0}, 'run("Top Hat...", "radius=10");', 0),
    (
        "Masque flou (netteté)",
        {"sigma": 1.0, "weight": 0.6},
        'run("Unsharp Mask...", "radius=1 mask=0.60");',
        0,
    ),
    ("Débruiter (despeckle)", {}, 'run("Despeckle");', 0),
    (
        "Retirer les valeurs aberrantes",
        {"radius": 2.0, "threshold": 20.0, "which": "claires"},
        'run("Remove Outliers...", "radius=2 threshold=20 which=Bright");',
        0,
    ),
    (
        "Convolution",
        {"kernel": "-1 -1 -1\n-1 12 -1\n-1 -1 -1", "normalize": True},
        'run("Convolve...", "text1=[-1 -1 -1\\n-1 12 -1\\n-1 -1 -1] normalize");',
        0,
    ),
    ("Soustraction de fond", {"radius": 25.0}, 'run("Subtract Background...", "rolling=25");', 0),
    ("Mathématiques", {"operation_": "gamma", "value": 0.5}, 'run("Gamma...", "value=0.5");', 0),
    ("Mathématiques", {"operation_": "logarithme"}, 'run("Log");', 0),
    ("Mathématiques", {"operation_": "multiplier", "value": 1.5}, 'run("Multiply...", "value=1.5");', 0),
    ("Mathématiques", {"operation_": "racine carrée"}, 'run("Square Root");', 0),
    (
        "Rehausser le contraste",
        {"mode": "étirer", "saturated": 0.35},
        'run("Enhance Contrast...", "saturated=0.35 normalize");',
        0,
    ),
    (
        "Rehausser le contraste",
        {"mode": "égaliser"},
        'run("Enhance Contrast...", "saturated=0.35 equalize");',
        0,
    ),
    ("Retourner", {"direction": "horizontalement"}, 'run("Flip Horizontally");', 0),
    ("Tourner de 90°", {"direction": "à droite"}, 'run("Rotate 90 Degrees Right");', 0),
    ("Regrouper les pixels (bin)", {"x": 2, "y": 2}, 'run("Bin...", "x=2 y=2 bin=average");', 0),
    (
        "Filtre passe-bande (FFT)",
        {"large": 40.0, "small": 3.0},
        'run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 '
        'autoscale saturate");',
        3,
    ),
]


@pytest.mark.parametrize(("name", "params", "macro", "tolerance"), _EXACT_8BIT, ids=lambda v: str(v)[:40])
def test_filter_parity_8bit(bridge, name, params, macro, tolerance):
    data = _test_image(np.uint8)
    ours = run(name, Image(data), **params).data.astype(np.int64)
    theirs = _ij(bridge, data, macro).astype(np.int64)
    assert ours.shape == theirs.shape, name
    diff = np.abs(ours - theirs)
    assert diff.max() <= tolerance, f"{name} : écart max {diff.max()}, {int((diff > tolerance).sum())} pixels"


@pytest.mark.parametrize(
    ("name", "params", "macro", "tolerance"),
    [
        ("Lisser", {}, 'run("Smooth");', 0),
        ("Détecter les contours", {}, 'run("Find Edges");', 0),
        ("Flou gaussien", {"sigma": 2.0}, 'run("Gaussian Blur...", "sigma=2");', 1),
        ("Filtre médian", {"radius": 2.0}, 'run("Median...", "radius=2");', 0),
        ("Filtre moyenne", {"radius": 3.0}, 'run("Mean...", "radius=3");', 0),
        ("Convertir le type", {"type": "8 bits"}, 'run("8-bit");', 0),
        ("Mathématiques", {"operation_": "gamma", "value": 0.5}, 'run("Gamma...", "value=0.5");', 1),
        ("Mathématiques", {"operation_": "logarithme"}, 'run("Log");', 1),
        ("Soustraction de fond", {"radius": 25.0}, 'run("Subtract Background...", "rolling=25");', 1),
    ],
    ids=lambda v: str(v)[:40],
)
def test_filter_parity_16bit(bridge, name, params, macro, tolerance):
    data = _test_image(np.uint16)
    ours = run(name, Image(data), **params).data.astype(np.int64)
    theirs = _ij(bridge, data, macro).astype(np.int64)
    diff = np.abs(ours - theirs)
    assert diff.max() <= tolerance, f"{name} : écart max {diff.max()}, {int((diff > tolerance).sum())} pixels"


def test_float_to_byte_conversion_parity(bridge):
    data = _test_image(np.float32)
    ours = run("Convertir le type", Image(data), type="8 bits").data
    assert np.abs(ours.astype(int) - _ij(bridge, data, 'run("8-bit");').astype(int)).max() == 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_auto_threshold_parity(bridge, method, dtype):
    data = _test_image(dtype)
    macro = f'setAutoThreshold("{method} dark"); getThreshold(lower, upper); print(lower + " " + upper);'
    log = bridge.run_macro(macro, Image(data, None, "parite")).log.strip().splitlines()[-1]
    lower, upper = (float(x) for x in log.split())
    ours = threshold_range(data, method, dark_background=True)
    assert ours == pytest.approx((lower, upper), abs=1e-6 if dtype == np.uint8 else 0.01), method


@pytest.mark.parametrize(
    ("name", "params", "macro"),
    [
        ("Érosion", {}, 'run("Erode");'),
        ("Dilatation", {}, 'run("Dilate");'),
        ("Ouverture", {}, 'run("Open");'),
        ("Fermeture", {}, 'run("Close-");'),
        ("Contour des objets", {}, 'run("Outline");'),
        ("Remplir les trous", {}, 'run("Fill Holes");'),
    ],
)
def test_binary_parity(bridge, name, params, macro):
    data = _test_image(np.uint8)
    mask = ((data > 95) & ~((data > 150) & (data < 160))).astype(np.uint8)  # objets avec des trous
    ours = run(name, Labels(mask), **params).data > 0
    theirs = _ij(bridge, mask * 255, 'setOption("BlackBackground", true);\n' + macro) > 0
    assert int((ours != theirs).sum()) == 0, name


def test_distance_map_parity(bridge):
    mask = (_test_image(np.uint8) > 95).astype(np.uint8)
    ours = run("Carte des distances", Labels(mask), output="8 bits (comme ImageJ)").data
    theirs = _ij(bridge, mask * 255, 'setOption("BlackBackground", true);\nrun("Distance Map");')
    assert np.abs(ours.astype(int) - theirs.astype(int)).max() == 0


@pytest.mark.parametrize("prominence", [5, 20, 60])
def test_find_maxima_parity(bridge, prominence):
    data = _test_image(np.uint8)
    points, _table, _ = run("Trouver les maxima", Image(data), prominence=float(prominence))
    theirs = _ij(bridge, data, f'run("Find Maxima...", "prominence={prominence} output=[Single Points]");')
    expected = {tuple(p) for p in np.argwhere(theirs > 0).tolist()}
    assert {tuple(int(v) for v in p) for p in points.data} == expected


_MEASURE = (
    "area mean standard modal min centroid center perimeter bounding fit shape feret's "
    "integrated median skewness kurtosis area_fraction redirect=None decimal=9"
)
_ROI_COLUMNS = [
    "Area",
    "Mean",
    "StdDev",
    "Mode",
    "Min",
    "Max",
    "X",
    "Y",
    "XM",
    "YM",
    "Perim.",
    "BX",
    "BY",
    "Width",
    "Height",
    "Major",
    "Minor",
    "Angle",
    "Circ.",
    "Feret",
    "IntDen",
    "Median",
    "Skew",
    "Kurt",
    "RawIntDen",
    "FeretAngle",
    "MinFeret",
    "AR",
    "Round",
    "Solidity",
]


@pytest.mark.parametrize(
    ("roi", "make"),
    [
        (Rois.rectangle(20, 30, 60, 40), "makeRectangle(20, 30, 60, 40);"),
        (Rois.oval(100, 40, 70, 50), "makeOval(100, 40, 70, 50);"),
        (
            Rois.polygon([30, 120, 90, 40], [20, 35, 110, 90]),
            "makePolygon(30, 20, 120, 35, 90, 110, 40, 90);",
        ),
    ],
    ids=["rectangle", "ovale", "polygone"],
)
def test_roi_measurement_parity(bridge, roi, make):
    data = _test_image(np.uint8)
    ours = analyze.measure_rois(data, roi, 0.5).iloc[0]
    macro = f'run("Set Scale...", "distance=2 known=1 unit=micron");\n{make}\n'
    macro += f'run("Set Measurements...", "{_MEASURE}");\nrun("Measure");'
    theirs = bridge.run_macro(macro, Image(data, 0.5, "parite")).table.data.iloc[0]
    for column in _ROI_COLUMNS:
        assert ours[column] == pytest.approx(theirs[column], rel=2e-3, abs=2e-3), column


def test_line_profile_parity(bridge):
    data = _test_image(np.uint8)
    macro = "makeLine(10, 20, 150, 90);\nprofile = getProfile();\n"
    macro += "for (i = 0; i < profile.length; i++) print(profile[i]);"
    theirs = [float(x) for x in bridge.run_macro(macro, Image(data, None, "parite")).log.split()]
    ours = analyze.roi_profile(data, Rois.line(10, 20, 150, 90), 0)
    assert len(ours) == len(theirs)
    np.testing.assert_allclose(ours, theirs, atol=1e-3)


def test_hsb_parity(bridge):
    rgb = _RNG.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    hsb = ij.rgb_to_hsb(rgb)
    stack = np.asarray(bridge.run_macro('run("HSB Stack");', Image(rgb, None, "parite")).image.data)
    stack = stack if stack.shape[-1] == 3 else np.moveaxis(stack, 0, -1)
    assert int((hsb != stack).sum()) == 0


def test_colour_deconvolution_parity(bridge):
    rgb = _RNG.integers(60, 256, (60, 80, 3), dtype=np.uint8)
    ours = run("Déconvolution des couleurs", Image(rgb), stains="H&E")
    for index, image in enumerate(ours, start=1):
        macro = 'title = getTitle();\nrun("Colour Deconvolution", "vectors=[H&E] hide");\n'
        macro += f'selectImage(title + "-(Colour_{index})");'
        try:
            theirs = _ij(bridge, rgb, macro)
        except Exception as exc:  # plugin absent (ImageJ sans Fiji)
            pytest.skip(f"plugin Colour Deconvolution indisponible : {exc}")
        assert np.abs(image.data.astype(int) - theirs.astype(int)).max() <= 1
