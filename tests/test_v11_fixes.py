"""Défauts corrigés en 1.1 : dessin (étiquettes, collage, échelle), histogramme du contraste,
« Appliquer le contraste aux pixels », curseurs au clavier, volet des résultats."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gui


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
    win.set_mode("analyse", "analysis")
    return win


def _settle(qtbot, window):
    """Attend que les étiquettes du dessin soient à jour : le délai dépend de la machine (plus
    long quand d'autres fenêtres ont été ouvertes avant dans le même programme)."""
    styler = window.analysis.drawing
    qtbot.wait(10)
    qtbot.waitUntil(lambda: not styler._timer.isActive(), timeout=5000)
    qtbot.wait(10)


def _image(window, name="lame", px=None, value=100):
    layer = window.viewer.add_image(np.full((200, 300), value, np.uint8), name=name)
    layer.contrast_limits = (0, 255)
    if px:
        window.set_pixel_size(layer, px)
    return layer


# ------------------------------------------------------------------ dessin
def test_setting_the_scale_keeps_the_shapes_on_their_pixels(window, qtbot):
    """« Définir l'échelle » d'après une ligne tracée : la ligne quittait la barre mesurée, son
    étiquette restait en pixels et « Mesurer les ROI » donnait quatre fois trop."""
    from fuji.ui.panels.analysis_panel import layer_to_rois

    panel = window.analysis
    layer = _image(window)
    panel.set_tool("shapes", "add_line")
    roi = window.viewer.layers["ROI"]
    roi.add_lines(np.array([[20, 10], [20, 210]], float))
    panel.set_tool("points", "add")
    points = window.viewer.layers["Points"]
    points.add(np.array([[50.0, 60.0]]))
    _settle(qtbot, window)
    assert roi.features["texte"].tolist() == ["1 : 200 px"]
    window.set_pixel_size(layer, 0.5)
    _settle(qtbot, window)
    assert tuple(roi.scale) == (0.5, 0.5) and tuple(points.scale) == (0.5, 0.5)
    np.testing.assert_allclose(roi.data[0], [[20, 10], [20, 210]])  # les mêmes pixels de l'image
    np.testing.assert_allclose(points.data, [[50, 60]])
    assert roi.features["texte"].tolist() == ["1 : 100 µm"]
    rois = layer_to_rois(roi, layer)
    assert rois.length(0) * rois.px == pytest.approx(100)
    assert "0,5 µm/px" in window.display_bar.info_label.text()  # et non plus « sans échelle »


def test_labels_use_the_scale_of_the_displayed_image(window, qtbot):
    """Le calque de dessin est commun aux images ouvertes : l'unité affichée à côté d'une forme
    est celle de l'image que l'on regarde, comme dans le tableau de mesures."""
    panel = window.analysis
    first = _image(window, "calibrée", px=0.5)
    panel.set_tool("shapes", "add_line")
    roi = window.viewer.layers["ROI"]
    roi.add_lines(np.array([[20, 10], [20, 210]], float))
    _settle(qtbot, window)
    assert roi.features["texte"].tolist() == ["1 : 100 µm"]
    first.visible = False
    _image(window, "sans échelle")
    _settle(qtbot, window)
    assert roi.features["texte"].tolist() == ["1 : 100 px"]  # 100 µm du monde = 100 pixels de cette image


def test_pasted_shapes_and_points_are_numbered(window, qtbot):
    """Ctrl+C, Ctrl+V : la copie portait le numéro de l'original et n'était pas comptée."""
    panel = window.analysis
    _image(window, px=0.5)
    panel.set_tool("shapes", "add_rectangle")
    roi = window.viewer.layers["ROI"]
    roi.add_rectangles(np.array([[10, 10], [10, 50], [40, 50], [40, 10]], float))
    roi.add_rectangles(np.array([[60, 10], [60, 50], [90, 50], [90, 10]], float))
    _settle(qtbot, window)
    assert roi.feature_defaults["texte"].tolist() == [""]  # une forme en cours de tracé : pas de « None »
    roi.selected_data = {1}
    roi._copy_data()
    roi._paste_data()
    qtbot.waitUntil(lambda: [t.split(" ")[0] for t in roi.features["texte"]] == ["1", "2", "3"], timeout=3000)
    assert panel.roi_count.text() == "3 formes"
    panel.set_tool("points", "add")
    points = window.viewer.layers["Points"]
    points.add(np.array([[20.0, 30.0]]))
    points.add(np.array([[50.0, 60.0]]))
    _settle(qtbot, window)
    points.selected_data = {1}
    points._copy_data()
    points._paste_data()
    qtbot.waitUntil(lambda: list(points.text.values) == ["1", "2", "3"], timeout=3000)
    assert panel.roi_count.text() == "3 formes · 3 points"


def test_measure_follows_the_mouse_and_a_click_draws_nothing(window, qtbot):
    panel = window.analysis
    _image(window, px=0.5)
    panel.set_tool("shapes", "add_rectangle")
    roi = window.viewer.layers["ROI"]
    styler = panel.drawing

    class Move:
        type = "mouse_move"

    # pendant un glisser, la mise à jour n'est plus repoussée à chaque mouvement de souris
    styler._timer.stop()
    starts, start = [], styler._timer.start
    styler._timer.start = lambda *a: (starts.append(1), start(*a))
    drag = styler._follow_drag(roi, Move)
    next(drag)
    for _ in range(5):
        next(drag)
    assert len(starts) == 1 and styler._timer.isActive()  # lancé une fois, il expire à son heure
    drag.close()
    del styler._timer.start

    # un simple clic laissait une forme de deux pixels, invisible mais comptée et mesurée
    from fuji.ui.documents import camera_of

    zoom = float(camera_of(window.viewer).zoom)
    tiny = 1.0 / zoom / 0.5  # un pixel d'écran, en pixels de l'image
    roi.add_rectangles(np.array([[10, 10], [10, 10 + tiny], [10 + tiny, 10 + tiny], [10 + tiny, 10]]))
    styler._drop_click(roi, 0)
    assert len(roi.data) == 0
    roi.add_rectangles(np.array([[10, 10], [10, 50], [40, 50], [40, 10]], float))
    styler._drop_click(roi, 0)
    assert len(roi.data) == 1  # une vraie forme reste


def test_drawing_layers_keep_their_name_and_broken_settings_are_bounded(window, qtbot):
    from fuji.ui.drawing import DrawingStyle
    from fuji.ui.prefs import settings

    panel = window.analysis
    _image(window)
    panel.set_tool("shapes", "add_rectangle")
    roi = window.viewer.layers["ROI"]
    layers = window.layers_panel
    layers.rebuild()
    layers.select(roi)
    assert not layers.rename_btn.isEnabled() and "garde son nom" in layers.rename_btn.toolTip()
    layers.rename(roi)  # double-clic sur la ligne : refusé aussi
    assert roi.name == "ROI"
    roi.name = "zones"  # par la liste de napari (mode expert) ou un script : le nom revient
    assert roi.name == "ROI"

    # points effacés pendant qu'ils étaient sélectionnés, puis zoom : napari levait une erreur
    panel.set_tool("points", "add")
    points = window.viewer.layers["Points"]
    points.add(np.array([[20.0, 30.0], [50.0, 60.0], [70.0, 80.0]]))
    points.selected_data = {2}
    points.data = np.zeros((0, 2))
    panel.drawing._rescale()
    points.add(np.array([[20.0, 30.0]]))
    points.selected_data = {0}
    panel.clear_rois()
    panel.drawing._rescale()
    assert len(points.data) == 0 and not points.selected_data

    s = settings()
    s.setValue("drawing/width", 500.0)
    s.setValue("drawing/fill", -3.0)
    s.setValue("drawing/text_size", 0.0)
    style = DrawingStyle.load()
    assert style.width == 20.0 and style.fill == 0.0 and style.text_size == 5.0


# ------------------------------------------------------------------ affichage
def test_histogram_limit_stays_in_the_frame_and_a_click_moves_nothing(window, qtbot):
    from qtpy.QtCore import QPoint, Qt

    window.show()
    rng = np.random.default_rng(0)
    window.viewer.add_image(rng.integers(600, 3400, (120, 160)).astype(np.uint16), name="image")
    bar = window.display_bar
    qtbot.wait(150)
    hist = bar.histogram
    layer = bar.layer
    before = tuple(float(v) for v in layer.contrast_limits)
    y = hist.height() // 2
    middle = QPoint(hist.width() // 2, y)
    # un clic (ou le premier appui d'un double-clic) ne déplace pas de borne
    qtbot.mouseClick(hist, Qt.MouseButton.LeftButton, pos=middle)
    assert tuple(float(v) for v in layer.contrast_limits) == before
    log = hist.log_scale
    qtbot.mouseDClick(hist, Qt.MouseButton.LeftButton, pos=middle)
    assert hist.log_scale != log and tuple(float(v) for v in layer.contrast_limits) == before
    # borne basse tirée hors du cadre, puis petits mouvements : elle reste au bord, sans s'emballer
    x0 = int(hist._x_of(before[0]))
    qtbot.mousePress(hist, Qt.MouseButton.LeftButton, pos=QPoint(x0 + 1, y))
    values = []
    for dx in (-10, -25, -26, -27, -28, -30, -29, -31):
        qtbot.mouseMove(hist, QPoint(x0 + dx, y))
        values.append(float(layer.contrast_limits[0]))
    qtbot.mouseRelease(hist, Qt.MouseButton.LeftButton, pos=QPoint(x0 - 31, y))
    assert len(set(values[2:])) == 1 and values[-1] >= before[0] - 0.05 * (before[1] - before[0])
    # la plage permise de napari ne reste pas immense après un contraste très adouci
    bar.contrast.setValue(-100)
    bar.full_contrast()
    lo_r, hi_r = (float(v) for v in layer.contrast_limits_range)
    lo, hi = (float(v) for v in layer.contrast_limits)
    assert hi_r - lo_r <= 4 * (hi - lo)


def test_transparent_png_and_real_images_in_the_display_bar(window, qtbot):
    from fuji.ui.display_bar import _float_decimals

    rgba = np.zeros((60, 80, 4), np.uint8)
    rgba[..., :3] = np.random.default_rng(1).integers(20, 90, (60, 80, 3))
    rgba[..., 3] = 255  # la transparence ne compte pas dans l'histogramme
    window.viewer.add_image(rgba, name="logo", rgb=True)
    bar = window.display_bar
    qtbot.wait(150)
    assert bar._plane.shape[-1] == 3 and bar._data_range[1] < 100 and len(bar.histogram._hists) == 3
    assert _float_decimals(0.0, 0.002) == 6 and _float_decimals(0.0, 1.0) == 3
    assert _float_decimals(0.0, 60000.0) == 0 and _float_decimals(1.0, 1.0) == 3


def test_apply_contrast_writes_what_is_shown(window, qtbot):
    """Un noir négatif (luminosité montée) devenait 0, des bornes non entières étaient arrondies."""
    data = np.linspace(0, 2000, 100 * 120).reshape(100, 120).astype(np.uint16)
    layer = window.viewer.add_image(data, name="image")
    layer.metadata["bit_depth"] = 12
    layer.contrast_limits_range = (-1000, 4095)
    layer.contrast_limits = (-800.0, 2500.5)
    before = len(window.viewer.layers)
    window._apply_display("image", -800.0, 2500.5, 1.0)
    qtbot.waitUntil(lambda: len(window.viewer.layers) == before + 1, timeout=10000)
    result = window.viewer.layers[-1]
    expected = np.round((data.astype(np.float64) + 800.0) / 3300.5 * 4095)
    assert np.abs(np.asarray(result.data, dtype=np.float64) - expected).max() <= 1
    assert tuple(float(v) for v in result.contrast_limits) == (0.0, 4095.0) and result.gamma == 1.0


# ------------------------------------------------------------------ réglages et résultats
def test_slider_keys_step_like_the_field(qtbot):
    from qtpy.QtCore import Qt

    from fuji.ui.panels.param_form import SliderSpin

    control = SliderSpin(False, 0.1, 50.0, 0.5)
    qtbot.addWidget(control)
    control.show()
    control.set_value(2.0)
    emitted = []
    control.changed.connect(lambda: emitted.append(control.value()))
    control.slider.setFocus()
    qtbot.keyClick(control.slider, Qt.Key.Key_Right)
    assert control.value() == pytest.approx(2.5)  # un appui, un pas (il en fallait vingt-quatre)
    qtbot.keyClick(control.slider, Qt.Key.Key_Left)
    qtbot.keyClick(control.slider, Qt.Key.Key_Left)
    assert control.value() == pytest.approx(1.5) and len(emitted) == 3
    # un mouvement de curseur qui ne change pas la valeur ne relance pas l'aperçu
    control.slider.setValue(control.slider.value() + 1)
    assert control.value() == pytest.approx(1.5) and len(emitted) == 3


def test_results_pane_closes_with_its_last_table(window):
    import pandas as pd

    from fuji.analysis.image import Table

    window.show()
    window.show_table(Table(pd.DataFrame({"Area": [1.0, 2.0]}), name="Mesure — a"))
    window.show_table(Table(pd.DataFrame({"Area": [3.0]}), name="Particules — a"))
    assert not window.results_dock.isHidden() and window.results_tabs.count() == 2
    window._close_results_tab(0)
    assert not window.results_dock.isHidden()
    window._close_results_tab(0)
    assert window.results_dock.isHidden()  # un volet vide ne prend plus la place de l'image


def test_failures_stay_in_the_status_bar(window):
    """Un échec du matériel s'effaçait en six secondes, comme un message ordinaire."""
    window.notify("Image enregistrée")
    assert window.statusBar().currentMessage() == "Image enregistrée"
    window.notify("Déplacement impossible : le contrôleur ne répond pas")
    assert window.statusBar().currentMessage().startswith("⚠ Déplacement impossible")


def test_shortcuts_are_written_for_this_computer():
    from fuji.text import file_manager
    from fuji.ui.components import shortcut

    assert shortcut("Ctrl+K") == ("⌘K" if sys.platform == "darwin" else "Ctrl+K")
    assert file_manager() in ("le Finder", "l'Explorateur", "le gestionnaire de fichiers")
