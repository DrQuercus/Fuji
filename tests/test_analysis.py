"""Analyse : valeurs de référence (géométrie exacte et définitions d'ImageJ)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.draw import disk, ellipse

from fuji.analysis import Image, Labels, get_operation, list_operations, run
from fuji.analysis.shape import feret, fit_ellipse, trace_outline, traced_perimeter
from fuji.analysis.thresholds import ij_default_level, threshold_value

SQ2 = 2 - math.sqrt(2)


# ------------------------------------------------------------------ contours
def test_traced_perimeter_matches_imagej_reference():
    # valeurs données dans le code source d'ImageJ (PolygonRoi.getTracedPerimeter)
    assert traced_perimeter(trace_outline(np.ones((1, 1), bool))) == pytest.approx(4 - 2 * SQ2)
    assert traced_perimeter(trace_outline(np.ones((2, 2), bool))) == pytest.approx(8 - 4 * SQ2)


def test_rectangle_outline_and_feret():
    m = np.zeros((20, 30), bool)
    m[5:15, 4:24] = True  # 10 x 20 pixels
    verts = trace_outline(m)
    assert sorted(map(tuple, verts)) == sorted([(4, 5), (24, 5), (24, 15), (4, 15)])
    assert traced_perimeter(verts) == pytest.approx(60 - 4 * SQ2)
    f = feret(verts)
    assert f.max == pytest.approx(math.hypot(10, 20))
    assert f.min == pytest.approx(10)


def test_diagonal_pixels_are_one_object_like_imagej():
    m = np.zeros((5, 5), bool)
    m[1, 1] = m[2, 2] = True  # 8-connexes
    verts = trace_outline(m)
    xs, ys = verts[:, 0], verts[:, 1]
    assert xs.min() == 1 and xs.max() == 3 and ys.min() == 1 and ys.max() == 3


def test_ellipse_fit_axes_and_angle():
    m = np.zeros((200, 200), bool)
    rr, cc = ellipse(100, 100, 20, 50, rotation=math.radians(30))
    m[rr, cc] = True
    rr, cc = np.nonzero(m)
    e = fit_ellipse(rr, cc)
    assert e.major == pytest.approx(100, rel=0.02)
    assert e.minor == pytest.approx(40, rel=0.03)
    # rotation skimage (sens anti-horaire, y vers le bas) = angle ImageJ (y vers le haut)
    assert e.angle == pytest.approx(30, abs=1.0)
    assert math.pi / 4 * e.major * e.minor == pytest.approx(m.sum(), rel=1e-6)


# ------------------------------------------------------------------ seuils
def test_ij_default_on_bimodal_histogram():
    hist = np.zeros(256, int)
    hist[50] = 1000
    hist[200] = 1000
    level = ij_default_level(hist)
    assert 100 <= level <= 150


@pytest.mark.parametrize("method", ["Default", "Otsu", "Li", "Yen", "IsoData", "Mean", "Triangle"])
def test_threshold_methods_separate_two_populations(method):
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(1000, 50, 5000), rng.normal(3000, 80, 3000)])
    values = np.clip(values, 0, 4095).astype(np.uint16)
    t = threshold_value(values, method)
    misclassified = np.mean(values[:5000] >= t) + np.mean(values[5000:] < t)
    assert misclassified < 0.05


# ------------------------------------------------------------------ particules
@pytest.fixture
def shapes_image():
    m = np.zeros((200, 300), np.uint8)
    rr, cc = disk((50, 50), 20)
    m[rr, cc] = 1
    rr, cc = ellipse(120, 150, 15, 40, rotation=math.radians(30))
    m[rr, cc] = 1
    m[150:180, 220:280] = 1  # rectangle 30 x 60 px
    m[0:10, 100:110] = 1  # touche le bord
    m[100:102, 10:12] = 1  # minuscule
    return m


def test_analyze_particles_measurements(shapes_image):
    gray = Image(np.where(shapes_image > 0, 200.0, 50.0).astype(np.float32), 0.5, "g")
    table, objects = run("Analyse de particules", Labels(shapes_image, 0.5), intensity=gray)
    df = table.data
    assert len(df) == 5
    # objets numérotés dans l'ordre de balayage (haut → bas), comme ImageJ
    disk_row = df.loc[((df["X"] - 25.25) ** 2 + (df["Y"] - 25.25) ** 2).idxmin()]
    assert disk_row["Area"] == pytest.approx(disk((50, 50), 20)[0].size * 0.25)
    assert disk_row["X"] == pytest.approx(50.5 * 0.5, abs=0.01)
    assert disk_row["Mean"] == pytest.approx(200)
    assert disk_row["IntDen"] == pytest.approx(disk_row["Area"] * 200)
    assert 0.85 < disk_row["Circ."] <= 1.0
    rect = df[(df["Width"] == 30) & (df["Height"] == 15)].iloc[0]
    assert rect["Perim."] == pytest.approx((180 - 4 * SQ2) * 0.5)
    assert rect["Feret"] == pytest.approx(math.hypot(30, 15))
    assert rect["MinFeret"] == pytest.approx(15)
    assert rect["Solidity"] == pytest.approx(1.0)
    assert table.units["Area"] == "µm²"
    assert objects.count == 5


def test_particle_filters(shapes_image):
    mask = Labels(shapes_image, 0.5)
    t, _ = run("Analyse de particules", mask, exclude_edges=True)
    assert len(t) == 4
    t, objs = run("Analyse de particules", mask, min_area_um2=10, exclude_edges=True)
    assert len(t) == 3 and objs.count == 3
    # carré 10 px : Circ. ImageJ = 0.886 (correction des coins) ; disque : 0.93
    t, _ = run("Analyse de particules", mask, min_area_um2=10, min_circularity=0.9)
    assert len(t) == 1
    assert t.summary["Count"] == 1


def test_watershed_separates_touching_disks():
    m = np.zeros((100, 160), np.uint8)
    for c in (50, 100):
        rr, cc = disk((50, c), 28)
        m[rr, cc] = 1
    lab = run("Séparer les objets (watershed)", Labels(m))
    assert lab.count == 2


def test_background_subtraction_flattens_gradient():
    _, xx = np.mgrid[0:256, 0:256]
    bg = 100 + 0.5 * xx
    img = bg.copy()
    rr, cc = disk((128, 128), 6)
    img[rr, cc] += 300
    out = run("Soustraction de fond", Image(img.astype(np.uint16)), radius=30)
    d = out.data.astype(float)
    assert d[rr, cc].mean() > 250
    assert np.percentile(d, 50) < 15


def test_flat_field_correction():
    yy, xx = np.mgrid[0:64, 0:64]
    flat = 1000 * (1 - 0.3 * ((xx - 32) ** 2 + (yy - 32) ** 2) / 32**2)
    raw = 0.5 * flat
    out = run("Correction de champ plat", Image(raw.astype(np.float32)), Image(flat.astype(np.float32)))
    assert out.data.std() / out.data.mean() < 1e-3


def test_rgb_threshold_on_dark_objects():
    img = np.full((60, 60, 3), 220, np.uint8)
    rr, cc = disk((30, 30), 10)
    img[rr, cc] = (120, 40, 140)
    mask = run("Seuillage automatique", Image(img), method="Otsu", objects="sombres")
    assert mask.data.sum() == rr.size


def test_registry_lists_operations_and_rejects_unknown_params():
    names = {op.name for op in list_operations()}
    assert {"Flou gaussien", "Analyse de particules", "Macro ImageJ"} <= names
    op = get_operation("Flou gaussien")
    assert [p.name for p in op.params] == ["sigma"]
    with pytest.raises(TypeError):
        run("Flou gaussien", Image(np.zeros((4, 4))), rayon=3)


def test_measure_with_mask():
    img = Image(np.arange(100, dtype=np.float32).reshape(10, 10), 2.0)
    mask = np.zeros((10, 10), np.uint8)
    mask[:5] = 1
    t = run("Mesurer", img, Labels(mask, 2.0))
    row = t.data.iloc[0]
    assert row["Area"] == pytest.approx(50 * 4)
    assert row["Mean"] == pytest.approx(24.5)
    assert row["%Area"] == pytest.approx(50)


# --------------------------------------------------------------------- fidélité à ImageJ
def _polygon(mask):
    return [tuple(int(c) for c in v) for v in trace_outline(np.asarray(mask, bool))]


def test_outline_starts_and_turns_like_the_imagej_wand():
    """Sommets attendus en déroulant Wand.traceEdge à la main (sens horaire, départ sous
    le bord droit du premier segment de la ligne du haut)."""
    assert _polygon([[1]]) == [(1, 1), (0, 1), (0, 0), (1, 0)]
    # .X.
    # XXX
    assert _polygon([[0, 1, 0], [1, 1, 1]]) == [
        (2, 1), (3, 1), (3, 2), (0, 2), (0, 1), (1, 1), (1, 0), (2, 0),
    ]  # fmt: skip
    # pixels reliés par un coin : un seul contour qui repasse par le sommet commun
    assert _polygon([[1, 0], [0, 1]]) == [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1), (0, 1), (0, 0), (1, 0)]


def test_perimeter_depends_on_the_starting_vertex_as_in_imagej():
    """Dans les « escaliers », le nombre de coins comptés dépend du premier sommet : il
    faut partir du même qu'ImageJ. Valeurs attendues obtenues en déroulant à la main
    Wand.traceEdge puis PolygonRoi.getTracedPerimeter."""
    # XXX
    # ..X      contour de longueur 10
    shape = np.array([[1, 1, 1], [0, 0, 1]], bool)
    verts = trace_outline(shape)
    assert [tuple(int(c) for c in v) for v in verts] == [(3, 2), (2, 2), (2, 1), (0, 1), (0, 0), (3, 0)]
    assert traced_perimeter(verts) == pytest.approx(10 - 4 * SQ2)  # 4 coins comptés
    # en partant du coin haut-gauche, l'algorithme en compterait 5
    assert traced_perimeter(np.roll(verts, 2, axis=0)) == pytest.approx(10 - 5 * SQ2)
    # escalier : longueur 12, 6 coins comptés
    stairs = trace_outline(np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], bool))
    assert traced_perimeter(stairs) == pytest.approx(12 - 6 * SQ2)


def test_feret_angle_follows_imagej_tie_breaking():
    m = np.zeros((20, 30), bool)
    m[5:15, 4:24] = True  # rectangle : ses deux diagonales sont égales
    f = feret(trace_outline(m))
    assert f.angle == pytest.approx(180 - math.degrees(math.atan2(10, 20)))  # diagonale haut-gauche
    assert (f.x, f.y) == (4.0, 5.0)
    square = np.ones((6, 6), bool)
    assert feret(trace_outline(square)).angle == pytest.approx(135.0)


def test_ellipse_angle_conventions():
    def angle(mask):
        rr, cc = np.nonzero(np.asarray(mask, bool))
        return fit_ellipse(rr, cc).angle

    assert angle(np.ones((3, 9))) == 0.0  # horizontal : 0, jamais 180
    assert angle(np.ones((9, 3))) == 90.0
    assert angle(np.ones((5, 5))) == 0.0  # isotrope
    assert angle(np.eye(9)[::-1]) == pytest.approx(45.0)  # monte vers la droite
    assert angle(np.eye(9)) == pytest.approx(135.0)


def test_ij_default_rounds_half_up_like_java():
    hist = np.zeros(256, int)
    hist[50] = hist[151] = 1000  # moyenne des deux classes : 100,5
    assert ij_default_level(hist) == 101


def test_rolling_ball_is_an_opening_by_the_imagej_ball():
    from fuji.analysis.ops.basic import _ball, imagej_disk, rolling_ball_background

    assert [(_ball(r)[0], _ball(r)[1].shape[0]) for r in (5, 10, 20, 50, 150)] == [
        (1, 9), (1, 17), (2, 17), (4, 19), (8, 25),
    ]  # fmt: skip
    assert imagej_disk(1).sum() == 9 and imagej_disk(2).sum() == 21
    rng = np.random.default_rng(3)
    img = rng.normal(500, 30, (64, 80)).astype(np.float32)
    bg = rolling_ball_background(img, 5, smooth=False)
    assert (bg <= img + 1e-3).all()  # la boule reste sous l'image
    # idempotence d'une ouverture : le fond du fond est le fond
    np.testing.assert_allclose(rolling_ball_background(bg, 5, smooth=False), bg, atol=1e-3)
    # un fond plan incliné est retrouvé, l'objet posé dessus est ignoré
    _, xx = np.mgrid[0:128, 0:128]
    plane = (100 + 0.5 * xx).astype(np.float32)
    spot = plane.copy()
    spot[60:66, 60:66] += 300
    np.testing.assert_allclose(rolling_ball_background(spot, 30), plane, atol=1.0)


def test_light_background_matches_imagej_convention():
    img = np.full((64, 64), 200, np.uint8)
    img[30:34, 30:34] = 80
    out = run("Soustraction de fond", Image(img), radius=10, light_background=True).data
    assert out.dtype == np.uint8 and out[0, 0] == 255 and out[31, 31] == 255 - 120


def test_binary_masks_stored_as_0_255_are_masks():
    m = np.zeros((40, 40), np.uint8)
    m[5:10, 5:10] = 255
    m[20:30, 20:30] = 255
    labels = Labels(m)
    assert labels.is_binary and labels.count == 2
    table, _ = run("Analyse de particules", labels)
    assert len(table) == 2
    assert not Labels(np.array([[0, 1, 2]])).is_binary


def test_binary_open_close_use_imagej_neighbourhood():
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 1
    m[2, 2] = 1  # pixel isolé
    m[9, 9] = 0  # petit trou
    opened = run("Ouverture", Labels(m)).data
    assert opened[2, 2] == 0 and opened[5:15, 5:15].sum() == 99
    closed = run("Fermeture", Labels(m)).data
    assert closed[9, 9] == 1


def test_invert_like_imagej():
    assert run("Inverser", Image(np.array([[0, 10, 255]], np.uint8))).data.tolist() == [[255, 245, 0]]
    out = run("Inverser", Image(np.array([[100, 1000, 4000]], np.uint16))).data
    assert out.tolist() == [[4000, 3100, 100]] and out.dtype == np.uint16


def test_file_calibration_units(tmp_path):
    import tifffile

    data = np.zeros((8, 8), np.uint8)
    cases = {
        "um.tif": ({"imagej": True, "resolution": (2.0, 2.0), "metadata": {"unit": "um"}}, 0.5),
        "micron.tif": ({"imagej": True, "resolution": (4.0, 4.0), "metadata": {"unit": "micron"}}, 0.25),
        "nm.tif": ({"imagej": True, "resolution": (0.01, 0.01), "metadata": {"unit": "nm"}}, 0.1),
        "pixel.tif": ({"imagej": True, "resolution": (1.0, 1.0), "metadata": {"unit": "pixel"}}, None),
        "photo.tif": ({"resolution": (72, 72)}, None),  # 72 dpi : pas une calibration
    }
    for name, (kwargs, expected) in cases.items():
        tifffile.imwrite(tmp_path / name, data, **kwargs)
        px = Image.from_file(tmp_path / name).pixel_size_um
        assert px == (pytest.approx(expected) if expected else None), name
    tifffile.imwrite(
        tmp_path / "ome.ome.tif", data, ome=True,
        metadata={"axes": "YX", "PhysicalSizeX": 250, "PhysicalSizeXUnit": "nm"},
    )  # fmt: skip
    img = Image.from_file(tmp_path / "ome.ome.tif")
    assert img.pixel_size_um == pytest.approx(0.25) and img.name == "ome"


def test_csv_opens_cleanly_in_excel(tmp_path):
    table, _ = run("Analyse de particules", Labels(np.ones((4, 4), np.uint8), 0.5))
    path = table.to_csv(tmp_path / "r.csv")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and "Area (µm²)" in raw.decode("utf-8-sig")


def test_operations_are_sorted_like_a_french_dictionary():
    """« Érosion » se range à E, pas après Z (ordre des codes Unicode)."""
    from fuji.analysis import list_operations
    from fuji.analysis.registry import _sort_key

    assert _sort_key("Érosion") < _sort_key("Fermeture") < _sort_key("Ouverture")
    for category in {o.category for o in list_operations()}:
        names = [o.name for o in list_operations(category)]
        assert names == sorted(names, key=_sort_key)
