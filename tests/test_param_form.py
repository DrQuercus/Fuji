"""Formulaire des réglages d'un outil : curseurs, valeurs par défaut, mémoire, contexte."""

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


def _form(qtbot, name):
    from fuji.analysis import get_operation
    from fuji.ui.panels.param_form import ParamForm

    form = ParamForm()
    qtbot.addWidget(form)
    form.build(list(get_operation(name).params))
    return form


def test_every_setting_gets_a_fitting_control(qtbot):
    """Nombre borné : curseur + champ ; choix : liste ; oui / non : case ; texte : ligne ou bloc."""
    from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit

    from fuji.analysis import list_operations
    from fuji.ui.panels.param_form import ParamForm, SliderSpin

    form = ParamForm()
    qtbot.addWidget(form)
    kinds = set()
    for op in list_operations():
        form.build(list(op.params))
        assert [c.name for c in form] == [p.name for p in op.params]
        assert form.values() == {p.name: p.default for p in op.params}, op.name
        assert not form.reset_btn.isEnabled()  # tout est à sa valeur par défaut
        for control in form:
            kinds.add(type(control.editor))
            if isinstance(control.editor, SliderSpin):  # le curseur ne déplace pas la valeur par défaut
                before = control.value
                control.editor._to_slider(float(before))
                assert control.value == before
    assert {SliderSpin, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit} <= kinds


def test_slider_and_field_move_together_and_stay_handy(qtbot):
    form = _form(qtbot, "Flou gaussien")
    sigma = next(c for c in form)
    editor = sigma.editor
    seen = []
    form.changed.connect(lambda: seen.append(sigma.value))
    # le rayon va jusqu'à 200 dans le champ, mais le curseur reste précis autour de l'usage courant
    assert editor.spin.maximum() == 200 and editor._hi < 30
    editor.slider.setValue(editor.slider.maximum())
    assert sigma.value == pytest.approx(editor._hi, abs=0.5) and seen
    editor.spin.setValue(120.0)  # au-delà de la course du curseur : accepté, curseur en butée
    assert sigma.value == 120.0 and editor.slider.value() == editor.slider.maximum()
    assert form.reset_btn.isEnabled()
    form.reset_btn.click()
    assert sigma.value == 2.0 and not form.reset_btn.isEnabled()
    # un pas de réglage est respecté par le curseur
    sharpen = _form(qtbot, "Masque flou (netteté)")
    weight = next(c for c in sharpen if c.name == "weight")
    weight.editor.slider.setValue(333)
    assert round(weight.value * 100) % 1 == 0


def test_advanced_settings_are_reachable_in_every_tool(qtbot):
    """Les réglages avancés ne demandent plus le mode expert : un bloc à déplier."""
    form = _form(qtbot, "Filtre passe-bande (FFT)")
    assert not form.advanced_btn.isHidden() and form.advanced_btn.text() == "Réglages avancés (2)"
    assert form.advanced_box.isHidden() and {"saturate", "tolerance"} <= set(form.values())
    form.advanced_btn.click()
    assert not form.advanced_box.isHidden()
    form.build([], expert=True)  # un outil sans réglage
    assert form.advanced_btn.isHidden() and form.reset_btn.isHidden()


def test_wrong_saved_values_are_ignored(qtbot):
    form = _form(qtbot, "Seuillage automatique")
    form.set_values({"method": "Otsu", "objects": "violets", "inconnu": 3})
    assert form.values() == {"method": "Otsu", "objects": "clairs"}


def test_lengths_show_their_size_in_micrometres(window):
    import numpy as np

    panel = window.analysis
    layer = window.viewer.add_image(np.zeros((40, 50), np.uint8), name="image")
    panel.select_operation("Flou gaussien")
    sigma = next(c for c in panel._params)
    assert sigma.equivalent.text() == ""  # image sans échelle
    window.set_pixel_size(layer, 0.5)
    panel._refresh_inputs()
    assert sigma.equivalent.text() == "≈ 1 µm"
    sigma.value = 5.0
    assert sigma.equivalent.text() == "≈ 2,5 µm"


def test_threshold_sliders_span_the_values_of_the_image(window, qtbot):
    """Un seuil va du pixel le plus sombre au plus clair de l'image choisie (12 bits ici),
    et part d'une valeur utile au lieu de 0–255."""
    import numpy as np

    rng = np.random.default_rng(0)
    data = rng.integers(300, 3000, (60, 80)).astype(np.uint16)
    window.viewer.add_image(data, name="douze bits")
    panel = window.analysis
    panel.select_operation("Seuil manuel")
    lower, upper = (next(c for c in panel._params if c.name == n) for n in ("lower", "upper"))
    assert (lower.editor._lo, lower.editor._hi) == (0.0, float(data.max()))
    assert upper.value == float(data.max()) and lower.value == pytest.approx((300 + data.max()) / 2, abs=2)
    assert not panel._params.reset_btn.isEnabled()  # ce sont les valeurs de départ pour cette image
    lower.value = 1000.0
    panel._run()
    qtbot.waitUntil(lambda: not panel.busy, timeout=10000)
    mask = np.asarray(window.viewer.layers[-1].data).astype(bool)
    assert mask.sum() == int((data >= 1000).sum())
    # une autre image : les bornes suivent, le réglage fait à la main est gardé s'il reste possible
    window.viewer.add_image((data // 16).astype(np.uint8), name="huit bits")
    panel._inputs["image"].setCurrentText("huit bits")
    assert lower.editor._hi == float((data // 16).max())


def test_tools_remember_their_last_settings(window, qtbot):
    """Comme ImageJ : rouvrir un outil retrouve les réglages de sa dernière exécution."""
    import numpy as np

    window.viewer.add_image(np.random.default_rng(1).integers(0, 255, (40, 50), dtype=np.uint8), name="image")
    panel = window.analysis
    panel.select_operation("Flou gaussien")
    next(c for c in panel._params).value = 6.5
    panel._run()
    qtbot.waitUntil(lambda: not panel.busy, timeout=10000)
    panel.select_operation("Filtre médian")
    next(c for c in panel._params).value = 9.0  # modifié mais jamais exécuté : pas retenu
    panel.select_operation("Flou gaussien")
    assert panel._params.values() == {"sigma": 6.5} and panel._params.reset_btn.isEnabled()
    panel._params.reset_btn.click()
    assert panel._params.values() == {"sigma": 2.0}
    panel.select_operation("Filtre médian")
    assert panel._params.values() == {"radius": 2.0}


def test_preview_can_be_compared_and_made_transparent(window, qtbot):
    import numpy as np

    data = np.zeros((60, 80), np.uint8)
    data[20:40, 30:60] = 200
    window.viewer.add_image(data, name="image")
    panel = window.analysis
    window.set_mode("analyse", "analysis")
    panel.select_operation("Seuillage automatique")
    assert panel.compare_btn.isHidden() and panel.opacity_box.isHidden()
    panel.preview.setChecked(True)
    assert not panel.compare_btn.isHidden()
    qtbot.waitUntil(lambda: bool(panel._preview_layers()), timeout=10000)
    preview = panel._preview_layers()[0]
    panel.preview_opacity.setValue(40)
    assert preview.opacity == pytest.approx(0.4)
    panel.compare_btn.pressed.emit()  # bouton maintenu : on revoit l'original
    assert not preview.visible
    panel.compare_btn.released.emit()
    assert preview.visible
    panel.preview.setChecked(False)
    assert not panel._preview_layers() and panel.compare_btn.isHidden()
