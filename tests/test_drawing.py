"""Dessin sur l'image : réglages du style, numéros et mesures affichés, outils en plus."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from fuji.analysis import Rois


def _rois(px=None):
    return Rois(
        [
            ("rectangle", np.array([[10, 10], [10, 50], [40, 50], [40, 10]], float)),
            ("line", np.array([[0, 0], [30, 40]], float)),
            ("path", np.array([[0, 10], [0, 0], [10, 0]], float)),
            ("ellipse", np.array([[10, 10], [10, 50], [40, 50], [40, 10]], float)),
            ("polygon", np.array([[0, 0], [0, 10], [10, 0]], float)),
            ("path", np.array([[0, 0], [0, 10], [10, 10], [10, 30]], float)),
        ],
        px,
    )


def test_geometric_area_and_angle():
    rois = _rois()
    assert rois.area(0) == 30 * 40 and rois.area(1) == 0 and rois.area(2) == 0
    assert rois.area(3) == pytest.approx(np.pi * 15 * 20) and rois.area(4) == 50
    assert rois.angle(2) == pytest.approx(90)  # l'angle au sommet du milieu
    assert rois.angle(1) == pytest.approx(-36.87, abs=0.01)  # ligne qui descend : négatif, comme ImageJ
    flat = Rois([("path", np.array([[5, 0], [5, 10], [5, 30]], float))])
    assert flat.angle(0) == pytest.approx(180)
    with pytest.raises(ValueError, match="trois points"):
        rois.angle(5)


def test_measure_gives_the_vertex_angle_of_a_three_point_path():
    from fuji.analysis.ops.analyze import measure_rois

    df = measure_rois(np.zeros((60, 60)), _rois())
    assert df.loc[3, "Angle"] == pytest.approx(90) and df.loc[2, "Angle"] == pytest.approx(-36.87, abs=0.01)
    assert df.loc[6, "Length"] == 40 and df.loc[6, "Angle"] == pytest.approx(-18.43, abs=0.01)


def test_labels_shown_next_to_the_shapes():
    pytest.importorskip("napari")
    from fuji.ui.drawing import number_text, shape_text

    rois = _rois(0.5)
    assert shape_text(rois, 0) == "1 : 300 µm²" and shape_text(rois, 1) == "2 : 25 µm"
    assert shape_text(rois, 2) == "3 : 90°" and shape_text(rois, 3) == "4 : 235,6 µm²"
    assert shape_text(rois, 5) == "6 : 20 µm"  # un trajet plus long : sa longueur
    assert shape_text(_rois(), 0) == "1 : 1200 px²"  # sans échelle : en pixels
    assert shape_text(rois, 0, measures=False) == "1" and shape_text(rois, 0, numbers=False) == "300 µm²"
    assert shape_text(rois, 0, numbers=False, measures=False) == ""
    assert number_text(12.3456) == "12,35" and number_text(123456.7) == "123457" and number_text(0.5) == "0,5"


# ------------------------------------------------------------------ interface


@pytest.fixture
def window(qtbot, tmp_path):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("pas d'affichage (lancer avec xvfb-run)")
    pytest.importorskip("napari")
    from fuji.ui.app import create_window, isolate_settings

    isolate_settings(str(tmp_path))
    _, win = create_window("demo")
    win.m.camera._timing = False
    qtbot.addWidget(win)
    return win


def _settle(qtbot, window):
    """Attend que les étiquettes du dessin soient à jour : le délai dépend de la machine (plus
    long quand d'autres fenêtres ont été ouvertes avant dans le même programme)."""
    styler = window.analysis.drawing
    qtbot.wait(10)
    qtbot.waitUntil(lambda: not styler._timer.isActive(), timeout=5000)
    qtbot.wait(10)


def _image(window, value=200, name="lame"):
    layer = window.viewer.add_image(np.full((200, 300), value, np.uint8), name=name)
    layer.contrast_limits = (0, 255)
    window.set_pixel_size(layer, 0.5)
    return layer


@pytest.mark.gui
def test_shapes_show_numbers_and_measures_and_are_counted(window, qtbot):
    panel = window.analysis
    window.set_mode("analyse", "analysis")
    _image(window)
    assert panel.roi_count.isHidden()
    panel.set_tool("shapes", "add_rectangle")
    roi = window.viewer.layers["ROI"]
    roi.add_rectangles(np.array([[10, 10], [10, 50], [40, 50], [40, 10]], float))
    roi.add_lines(np.array([[0, 0], [30, 40]], float))
    roi.add_paths(np.array([[0, 10], [0, 0], [10, 0]], float))
    _settle(qtbot, window)
    assert roi.features["texte"].tolist() == ["1 : 300 µm²", "2 : 25 µm", "3 : 90°"]
    assert list(roi.text.values) == ["1 : 300 µm²", "2 : 25 µm", "3 : 90°"] and roi.text.visible
    assert panel.roi_count.text() == "3 formes" and not panel.roi_count.isHidden()
    # une forme modifiée : sa mesure suit
    data = list(roi.data)
    data[1] = np.array([[0, 0], [60, 80]], float)
    roi.data = data
    _settle(qtbot, window)
    assert roi.features["texte"].tolist()[1] == "2 : 50 µm"

    panel.set_tool("points", "add")
    points = window.viewer.layers["Points"]
    for y, x in ((20, 30), (50, 60)):
        points.add(np.array([[float(y), float(x)]]))
    _settle(qtbot, window)
    assert list(points.text.values) == ["1", "2"] and panel.roi_count.text() == "3 formes · 2 points"
    panel.clear_rois()
    _settle(qtbot, window)
    assert panel.roi_count.isHidden() and len(roi.features) == 0


@pytest.mark.gui
def test_drawing_settings_restyle_what_is_drawn_and_persist(window, qtbot):
    from fuji.ui.drawing import ON_DARK, ON_LIGHT, DrawingStyle

    panel = window.analysis
    styler = panel.drawing
    window.set_mode("analyse", "analysis")
    image = _image(window)
    panel.set_tool("shapes", "add_rectangle")
    roi = window.viewer.layers["ROI"]
    roi.add_rectangles(np.array([[10, 10], [10, 50], [40, 50], [40, 10]], float))
    panel.set_tool("points", "add")
    points = window.viewer.layers["Points"]
    points.add(np.array([[20.0, 30.0]]))
    _settle(qtbot, window)

    def rgb(color):
        return tuple(round(float(c), 2) for c in color[:3])

    # couleur automatique : vert foncé sur une image claire, jaune sur une image sombre
    assert styler.style.color == "auto" and styler.color() == ON_LIGHT
    assert rgb(roi.edge_color[0]) == (0.0, 0.52, 0.24)
    image.data = np.full((200, 300), 20, np.uint8)
    styler.refresh()
    assert styler.color() == ON_DARK and rgb(roi.edge_color[0]) == (1.0, 0.83, 0.0)

    styler.choices[("color", "#ff4d4d")].trigger()
    styler.choices[("fill", 0.25)].trigger()
    assert rgb(roi.edge_color[0]) == (1.0, 0.3, 0.3) and float(roi.face_color[0][3]) == pytest.approx(0.25)
    assert rgb(points.face_color[0]) == (1.0, 0.3, 0.3) and rgb(roi.text.color.constant) == (1.0, 0.3, 0.3)
    # épaisseur et taille en pixels de l'écran : elles suivent le zoom
    window.viewer.camera.zoom = 4.0  # 4 px d'écran par µm, soit 2 par pixel de l'image
    styler.choices[("width", 4.0)].trigger()
    styler.choices[("point_size", 18.0)].trigger()
    assert roi.edge_width[0] == pytest.approx(2.0) and float(points.size[0]) == pytest.approx(9.0)
    assert roi.current_edge_width == pytest.approx(2.0)  # les formes suivantes aussi
    window.viewer.camera.zoom = 8.0
    qtbot.wait(200)
    assert roi.edge_width[0] == pytest.approx(1.0) and float(points.size[0]) == pytest.approx(4.5)

    styler.checks["measures"].trigger()
    assert roi.features["texte"].tolist() == ["1"] and roi.text.visible
    styler.checks["numbers"].trigger()
    assert roi.features["texte"].tolist() == [""] and not roi.text.visible and not points.text.visible
    styler.choices[("text_size", 13.0)].trigger()
    assert DrawingStyle.load() == DrawingStyle("#ff4d4d", 4.0, 0.25, 18.0, False, False, 13.0)
    with pytest.raises(KeyError):
        styler.set_style(couleur="rouge")


@pytest.mark.gui
def test_extra_drawing_tools(window):
    panel = window.analysis
    window.set_mode("analyse", "analysis")
    _image(window)
    expected = {
        "roi_free": "add_polygon_lasso",
        "roi_path": "add_polyline",
        "roi_angle": "add_polyline",
        "roi_edit": "direct",
        "roi_rect": "add_rectangle",
    }
    for name, mode in expected.items():
        panel.tool_buttons[name].click()
        assert window.viewer.layers["ROI"].mode == mode
    panel.tool_buttons["move"].click()
    assert window.viewer.layers["ROI"].mode == "pan_zoom"
    assert "Angle" in panel.tool_buttons["roi_angle"].toolTip()
