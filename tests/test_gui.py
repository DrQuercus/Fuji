"""Test de fumée de l'interface (nécessite un affichage, ex. ``xvfb-run pytest``)."""

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
    qtbot.addWidget(win)
    yield win
    win.close()


def test_window_live_snap_and_analysis(window, qtbot):
    from fuji.analysis import get_operation

    window.live.start()
    qtbot.waitUntil(lambda: "Live" in window.viewer.layers, timeout=5000)
    window.live.stop()
    window._on_snap(window.m.snap())
    assert "Capture 1" in window.viewer.layers
    assert window.viewer.layers["Capture 1"].metadata["pixel_size_um"] is not None

    panel = window.analysis
    op = get_operation("Niveaux de gris")
    panel._op = op
    panel._build_form(op)
    panel._inputs["image"].setCurrentText("Capture 1")
    inputs, params = panel._collect()
    panel._show_outputs(op, op.func(*inputs, **params))
    assert any("gris" in layer.name for layer in window.viewer.layers)


def test_expert_mode_toggles_properties(window):
    window.set_expert(True)
    assert window.tools.isTabVisible(window._props_index)
    window.set_expert(False)
    assert not window.tools.isTabVisible(window._props_index)


def test_acquisition_panel_plan(window):
    panel = window.acquisition
    panel.use_time.setChecked(True)
    panel.loops.setValue(5)
    panel.update_plan()
    assert "5" in panel.plan_label.text()
