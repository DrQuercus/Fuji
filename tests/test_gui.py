"""Tests de l'interface (nécessitent un affichage, ex. ``xvfb-run pytest``)."""

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
    from qtpy.QtCore import QSettings

    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    from fuji.ui.app import create_window

    _, win = create_window("demo")
    win.m.camera._timing = False
    qtbot.addWidget(win)  # qtbot ferme et détruit la fenêtre en fin de test
    return win


def test_window_live_snap_and_analysis(window, qtbot):
    from fuji.analysis import get_operation

    window.live.start()
    qtbot.waitUntil(lambda: "Live" in window.viewer.layers, timeout=5000)
    assert window.topbar.live.isChecked()
    window.live.stop()
    window._on_snap(window.m.snap())
    assert "Capture 1" in window.viewer.layers
    assert window.quickbar.histogram._hists

    panel = window.analysis
    op = get_operation("Niveaux de gris")
    panel._op = op
    panel._build_form(op)
    panel._inputs["image"].setCurrentText("Capture 1")
    inputs, params = panel._collect()
    panel._show_outputs(op, op.func(*inputs, **params))
    assert any("gris" in layer.name for layer in window.viewer.layers)


def test_sections_and_expert_mode(window):
    from fuji.ui.chrome import SECTIONS

    for key, *_ in SECTIONS:
        window.sidebar.select(key)
        assert window.stack.currentWidget() is window._pages[key]
    window.sidebar._clicked(window.sidebar.current())  # recliquer : replie le panneau
    assert not window.side.isVisible() or window.side.isHidden()
    window.set_expert(True)
    assert window.control_tabs.isTabVisible(window._props_index)
    assert window.topbar.expert.isChecked() and window.settings_panel.expert.isChecked()
    window.set_expert(False)
    assert not window.control_tabs.isTabVisible(window._props_index)


def test_quickbar_controls_microscope(window):
    qb = window.quickbar
    qb.exposure.setValue(25.0)
    assert window.m.camera.get_exposure() == pytest.approx(25.0)
    qb._objective_clicked("40x")
    assert window.m.objective.label == "40x/0.65"
    qb._channel_clicked("GFP")
    assert window.m.channel == "GFP"


def test_theme_switch(window):
    from qtpy.QtWidgets import QApplication

    from fuji.ui.theme import ThemeManager

    tm = ThemeManager.get()
    tm.set(mode="clair", accent="Turquoise")
    assert "#14b8a6" in QApplication.instance().styleSheet()
    assert window.viewer.theme.startswith("fuji-light")
    tm.set(mode="sombre", accent="Bleu")
    assert window.viewer.theme.startswith("fuji-dark")


def test_editor_roundtrip_is_lossless(qtbot):
    from fuji.core.profile import MicroscopeProfile, builtin_profiles
    from fuji.ui.config.editor import ConfigEditor

    for name, path in builtin_profiles().items():
        original = MicroscopeProfile.load(path)
        editor = ConfigEditor(original.model_dump(mode="json"))
        qtbot.addWidget(editor)
        for i in range(len(editor.pages)):
            editor.nav.setCurrentRow(i)
        devices = editor.pages[2]
        editor.nav.setCurrentRow(2)
        for r in range(devices.list.count()):
            devices.list.setCurrentRow(r)
        assert editor.collect() == original, name


def test_editor_builds_new_microscope_and_saves(qtbot, tmp_path):
    from fuji.core.profile import MicroscopeProfile
    from fuji.ui.config.editor import ConfigEditor
    from fuji.ui.config.wizard import template_data

    editor = ConfigEditor(template_data("vide"))
    qtbot.addWidget(editor)
    editor.go_to("Périphériques")
    devices = editor.pages[2]
    devices._add("demo.camera")
    devices._add("manual.state")
    editor.go_to("Objectifs")
    objectives = editor.pages[4]
    objectives._append({"label": "10x/0.25", "magnification": 10, "na": 0.25})
    objectives._append({"label": "40x/0.65", "magnification": 40, "na": 0.65})
    editor.go_to("Rôles")
    roles = editor.pages[3]
    roles.combos["objective_changer"].setCurrentIndex(roles.combos["objective_changer"].findData("tourelle"))
    editor.go_to("Général")
    editor.pages[0].name.setText("Mon microscope de test")
    saved = []
    editor.saved.connect(saved.append)
    editor.save()
    profile = MicroscopeProfile.load(saved[0])
    assert profile.name == "Mon microscope de test"
    assert [o.label for o in profile.objectives] == ["10x/0.25", "40x/0.65"]
    assert profile.roles.objective_changer == "tourelle"
    assert profile.check() == []


def test_hot_swap_microscope(window, tmp_path):
    from fuji.core.profile import MicroscopeProfile

    path = tmp_path / "mini.yaml"
    MicroscopeProfile.model_validate(
        {
            "name": "Mini",
            "devices": [
                {
                    "name": "cam",
                    "driver": "demo.camera",
                    "options": {"width": 64, "height": 48, "simulate_timing": False},
                }
            ],
        }
    ).save(path)
    window.switch_profile(str(path))
    assert window.m.profile.name == "Mini"
    assert window.windowTitle().endswith("Mini")
    assert window.quickbar is not None and window.m.camera.sensor_shape() == (48, 64)
