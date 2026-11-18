"""Barre d'affichage sous l'image (Analyse) : zoom, luminosité / contraste, options."""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("pas d'affichage (lancer avec xvfb-run)", allow_module_level=True)
pytest.importorskip("napari")


@pytest.fixture
def window(qtbot, tmp_path):
    from fuji.ui.app import create_window, isolate_settings

    isolate_settings(str(tmp_path))
    _, win = create_window("demo")
    win.m.camera._timing = False
    qtbot.addWidget(win)
    return win


def _ramp(window, name="rampe"):
    """Une image 16 bits dont les valeurs vont de 1000 à 3000."""
    import numpy as np

    data = np.tile(np.linspace(1000, 3000, 200), (100, 1)).astype(np.uint16)
    return window.viewer.add_image(data, name=name)


def test_brightness_and_contrast_sliders_move_the_same_limits():
    from fuji.ui.display_bar import sliders_to_window, window_to_sliders

    span = (1000.0, 3000.0)
    assert window_to_sliders(1000, 3000, span) == (0, 0)  # la fenêtre couvre l'image
    assert sliders_to_window(0, 0, span) == (1000.0, 3000.0)
    lo, hi = sliders_to_window(0, 50, span)  # contraste +50 : fenêtre dix fois plus étroite
    assert hi - lo == pytest.approx(200) and (lo + hi) / 2 == pytest.approx(2000)
    lo, hi = sliders_to_window(40, 0, span)  # luminosité +40 : fenêtre vers les valeurs sombres
    assert (lo + hi) / 2 == pytest.approx(1600) and hi - lo == pytest.approx(2000)
    for b, c in ((-60, 20), (35, -40), (100, 100)):
        assert window_to_sliders(*sliders_to_window(b, c, span), span) == (b, c)
    assert window_to_sliders(0, 1e9, span) == (-100, -100)  # hors course : en butée, sans erreur


def test_brightness_contrast_and_zoom_in_the_bar(window, qtbot):
    bar = window.display_bar
    window.set_mode("analyse")
    layer = _ramp(window)
    qtbot.wait(250)
    bar.full_contrast()
    assert tuple(layer.contrast_limits) == (1000.0, 3000.0)
    assert (bar.brightness.value(), bar.contrast.value()) == (0, 0)
    assert bar.brightness_label.text() == "0" and bar.contrast_label.text() == "0"

    bar.contrast.setValue(50)  # dix fois plus de contraste, autour du même gris moyen
    lo, hi = layer.contrast_limits
    assert hi - lo == pytest.approx(200) and (lo + hi) / 2 == pytest.approx(2000)
    assert bar.contrast_label.text() == "+50" and bar.min_box.value() == pytest.approx(1900)
    bar.brightness.setValue(-40)  # plus sombre : la fenêtre monte vers les valeurs claires
    lo, hi = layer.contrast_limits
    assert (lo + hi) / 2 == pytest.approx(2400) and hi - lo == pytest.approx(200)
    assert bar.contrast.value() == 50  # le contraste n'a pas bougé
    # les bornes saisies à la main déplacent les curseurs
    bar.max_box.setValue(3000)
    assert bar.brightness.value() < -40
    bar.brightness_label.click()
    bar.contrast_label.click()
    assert tuple(layer.contrast_limits) == pytest.approx((1000.0, 3000.0))

    # zoom : 100 % = un pixel de l'image pour un pixel de l'écran, quelle que soit l'échelle
    window.set_pixel_size(layer, 0.5)
    bar.set_zoom(100)
    assert window.viewer.camera.zoom == pytest.approx(2.0) and bar.zoom_btn.text() == "100 %"
    bar.zoom_in_btn.click()
    assert bar.zoom_percent() == pytest.approx(125) and bar.zoom_btn.text() == "125 %"
    bar.zoom_out_btn.click()
    bar.zoom_out_btn.click()
    assert bar.zoom_percent() == pytest.approx(80)
    entries = {a.text(): a for a in bar.zoom_menu.actions()}
    entries["400 %"].trigger()
    assert bar.zoom_percent() == pytest.approx(400)
    entries["Ajuster à la fenêtre"].trigger()
    assert bar.zoom_percent() != pytest.approx(400)


def test_settings_drawer_folds_and_is_remembered(window, qtbot):
    from fuji.ui.prefs import settings

    bar = window.display_bar
    window.set_mode("analyse")
    _ramp(window)
    assert bar.more_btn.isChecked() and not bar.brightness_group.isHidden()
    assert not bar.colormap_group.isHidden() and bar.mask_group.isHidden()  # pas de masque
    bar.more_btn.click()
    assert bar.brightness_group.isHidden() and bar.gamma_group.isHidden()
    assert not bar.histogram.isHidden() and not bar.view_row.isHidden()  # l'essentiel reste
    assert settings().value("display/more", True, type=bool) is False
    bar.more_btn.click()
    assert not bar.contrast_group.isHidden()


def test_auto_saturation_smoothing_and_inverted_display(window, qtbot):
    import numpy as np

    from fuji.ui.prefs import settings

    bar = window.display_bar
    window.set_mode("analyse")
    layer = _ramp(window)
    qtbot.wait(250)
    bar.auto_contrast()
    tight = tuple(layer.contrast_limits)
    bar.saturation_actions[5.0].trigger()  # 5 % de pixels saturés de chaque côté : appliqué aussitôt
    wide = tuple(layer.contrast_limits)
    assert wide[0] > tight[0] + 50 and wide[1] < tight[1] - 50
    assert settings().value("display/auto_saturation", type=float) == 5.0
    assert "5 %" in bar.auto_btn.toolTip() and bar.saturation_actions[5.0].isChecked()
    bar.saturation_actions[0.0].trigger()
    assert tuple(layer.contrast_limits) == (1000.0, 3000.0)

    assert layer.interpolation2d == "nearest"
    bar.smooth_action.trigger()
    assert layer.interpolation2d == "linear"
    assert _ramp(window, "autre").interpolation2d == "linear"  # les suivantes aussi
    bar.smooth_action.trigger()
    assert layer.interpolation2d == "nearest"

    # négatif à l'écran : la palette est retournée, la liste garde son nom
    window.viewer.layers.selection.active = layer
    assert bar.layer is layer and not bar.invert.isChecked()
    bar.invert.setChecked(True)
    assert layer.colormap.name == "gray (inversé)"
    assert np.allclose(layer.colormap.map(np.array([0.0, 1.0]))[:, :3], [[1, 1, 1], [0, 0, 0]])
    assert bar.colormap.currentText() == "Gris" and bar.invert.isChecked()
    bar.colormap.setCurrentIndex(bar.colormap.findData("green"))
    assert layer.colormap.name == "green (inversé)"
    layer.gamma = 2.0
    bar.reset_action.trigger()
    assert layer.colormap.name == "green" and layer.gamma == 1.0 and not bar.invert.isChecked()
    assert tuple(layer.contrast_limits) == (1000.0, 3000.0)


def test_scale_bar_settings_apply_to_both_views_and_persist(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QInputDialog

    from fuji.ui.display_bar import DisplayBar, scale_bar_of, scale_bar_prefs

    bar = window.display_bar
    bars = [scale_bar_of(window.viewer), scale_bar_of(window.scope_viewer)]
    assert all(b.visible for b in bars)
    bar.scale_choices[("position", "top_left")].trigger()
    bar.scale_choices[("color", "yellow")].trigger()
    bar.scale_choices[("font_size", 14.0)].trigger()
    bar.scale_choices[("length", 50.0)].trigger()
    bar.scale_actions["box"].trigger()
    for b in bars:
        assert b.position == "top_left" and b.colored and b.font_size == 14
        assert tuple(b.color[:3]) == (1.0, 1.0, 0.0) and b.length == 50 and not b.box
    bar.scale_choices[("color", "auto")].trigger()
    bar.scale_choices[("length", 0.0)].trigger()
    assert not bars[0].colored and bars[0].length is None
    # une longueur au choix
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (37.5, True))
    bar.scale_other.trigger()
    assert bars[1].length == 37.5 and bar.scale_other.text() == "Autre… (37,5)"
    assert not any(a.isChecked() for (k, _), a in bar.scale_choices.items() if k == "length")
    bar.scale_actions["visible"].trigger()
    assert not any(b.visible for b in bars)
    # retenu : une nouvelle barre retrouve les réglages
    assert scale_bar_prefs() == {
        "visible": False,
        "position": "top_left",
        "color": "auto",
        "font_size": 14.0,
        "length": 37.5,
        "box": False,
        "ticks": True,
    }
    again = DisplayBar(window.viewer)
    qtbot.addWidget(again)
    assert not again.scale_actions["visible"].isChecked()
    assert again.scale_choices[("position", "top_left")].isChecked()
    with pytest.raises(KeyError):
        bar.set_scale_bar(couleur="rouge")


def test_mask_transparency_and_outlines(window, qtbot):
    import numpy as np

    bar = window.display_bar
    window.set_mode("analyse")
    _ramp(window)
    mask = np.zeros((100, 200), np.uint8)
    mask[20:60, 30:90] = 1
    first = window.viewer.add_labels(mask, name="masque")
    qtbot.wait(50)
    assert not bar.mask_group.isHidden()
    assert bar.mask_opacity.value() == round(first.opacity * 100)
    bar.mask_opacity.setValue(30)
    bar.mask_contour.setChecked(True)
    assert first.opacity == pytest.approx(0.3) and first.contour == 2
    second = window.viewer.add_labels(mask, name="masque 2")  # un nouveau masque garde le style
    assert second.opacity == pytest.approx(0.3) and second.contour == 2
    first.visible = second.visible = False
    qtbot.wait(50)
    assert bar.mask_group.isHidden()
    first.visible = True
    qtbot.wait(50)
    assert not bar.mask_group.isHidden() and bar.mask_contour.isChecked()


def test_apply_contrast_to_pixels_makes_a_new_image(window, qtbot):
    import numpy as np

    from fuji.analysis.ops.adjust import white_level

    bar = window.display_bar
    window.set_mode("analyse")
    layer = _ramp(window)
    qtbot.wait(250)
    bar.min_box.setValue(1500)
    bar.max_box.setValue(2500)
    layer.gamma = 2.0
    bar.apply_action.trigger()
    panel = window.analysis
    qtbot.waitUntil(lambda: not panel.busy and "rampe (niveaux)" in window.viewer.layers, timeout=20000)
    assert window._section == "analysis" and panel._op.name == "Niveaux"
    out = np.asarray(window.viewer.layers["rampe (niveaux)"].data)
    source = np.asarray(layer.data).astype(np.float64)
    full = white_level(np.asarray(layer.data))  # dynamique de l'image (ici 12 bits dans 16)
    expected = np.clip((source - 1500) / 1000, 0, 1) ** 2 * full
    assert out.dtype == np.uint16 and np.abs(out - expected).max() <= 1
    assert np.array_equal(np.asarray(layer.data), source)  # l'image d'origine n'est pas touchée
