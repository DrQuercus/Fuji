"""Panneau « Calques » : la liste de Fuji, en français, à la place des composants napari."""

from __future__ import annotations

import os
import sys

import numpy as np
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


def _names(panel):
    return [row.layer.name for row in panel._rows]


def test_layers_are_listed_top_first_and_follow_the_view(window, qtbot):
    panel = window.layers_panel
    window.set_mode("analyse", "layers")
    assert not panel.empty.isHidden() and panel.card.isHidden() and panel.actions_row.isHidden()
    image = window.viewer.add_image(np.zeros((40, 60), np.uint16), name="lame")
    mask = window.viewer.add_labels(np.ones((40, 60), np.uint8), name="lame (masque)")
    window.viewer.add_shapes(ndim=2, name="ROI")
    qtbot.wait(50)
    assert _names(panel) == ["ROI", "lame (masque)", "lame"] and panel.empty.isHidden()
    assert [row.icon.toolTip() for row in panel._rows] == ["Formes dessinées", "Masque", "Image"]
    # l'œil montre et cache ; la ligne grisée le rappelle
    row = panel._rows[1]
    row.eye.click()
    assert not mask.visible and not row.name.isEnabled()
    mask.visible = True
    qtbot.wait(50)
    assert panel._rows[1].eye.isChecked() and panel._rows[1].name.isEnabled()
    # un clic choisit le calque, dans la vue aussi
    panel._rows[2].clicked.emit(image)
    assert window.viewer.layers.selection.active is image and panel._rows[2].property("current") is True
    mask.name = "noyaux"
    qtbot.wait(50)
    assert _names(panel)[1] == "noyaux"


def test_selected_layer_settings_are_in_french(window, qtbot):
    panel = window.layers_panel
    window.set_mode("analyse", "layers")
    image = window.viewer.add_image(np.zeros((40, 60), np.uint16), name="lame")
    mask = window.viewer.add_labels(np.ones((40, 60), np.uint8), name="masque")
    qtbot.wait(50)
    panel.select(mask)
    assert not panel.card.isHidden() and not panel.contour.isHidden() and panel.blending_row.isHidden()
    panel.opacity.setValue(40)
    assert mask.opacity == pytest.approx(0.4) and panel.opacity_label.text() == "40 %"
    panel.contour.setChecked(True)
    assert mask.contour == 2
    mask.opacity = 0.9  # changé ailleurs (barre sous l'image) : le panneau suit
    assert panel.opacity.value() == 90
    panel.select(image)
    assert panel.contour.isHidden() and not panel.blending_row.isHidden()
    assert panel.blending.currentText().startswith("Normale")
    panel.blending.setCurrentIndex(panel.blending.findData("additive"))
    assert str(image.blending) == "additive"
    assert "barre sous l'image" in panel.tip.text()
    # les réglages complets de napari (en anglais) : seulement en mode expert
    assert panel.expert_box.isHidden()
    window.set_expert(True)
    assert not panel.expert_box.isHidden()
    window.set_expert(False)
    assert panel.expert_box.isHidden()


def test_layer_actions_move_rename_save_delete(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QFileDialog, QInputDialog

    panel = window.layers_panel
    window.set_mode("analyse", "layers")
    image = window.viewer.add_image(np.zeros((40, 60), np.uint16), name="lame")
    first = window.viewer.add_labels(np.ones((40, 60), np.uint8), name="masque A")
    second = window.viewer.add_labels(np.ones((40, 60), np.uint8), name="masque B")
    qtbot.wait(50)
    panel.select(first)
    assert panel.up_btn.isEnabled() and panel.down_btn.isEnabled()
    panel.up_btn.click()
    qtbot.wait(50)
    assert _names(panel) == ["masque A", "masque B", "lame"] and not panel.up_btn.isEnabled()
    panel.down_btn.click()
    panel.down_btn.click()
    qtbot.wait(50)
    assert _names(panel) == ["masque B", "lame", "masque A"] and not panel.down_btn.isEnabled()
    # renommer un résultat ; une image se renomme dans la liste « Images »
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("noyaux", True))
    panel.rename_btn.click()
    assert first.name == "noyaux"
    panel.select(image)
    assert not panel.rename_btn.isEnabled() and "liste « Images »" in panel.rename_btn.toolTip()
    saved = []
    panel.save_requested.connect(lambda: saved.append(window.viewer.layers.selection.active.name))
    # la fenêtre « Enregistrer » de l'application s'ouvrirait pour de bon : on l'annule
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    panel.save_btn.click()
    assert saved == ["lame"]
    # supprimer un résultat l'enlève ; supprimer l'image passe par la fermeture (et sa question)
    panel.select(second)
    panel.delete_btn.click()
    qtbot.wait(50)
    assert "masque B" not in window.viewer.layers and _names(panel) == ["lame", "noyaux"]
    closed = []
    monkeypatch.setattr(window, "close_image", lambda doc=None: closed.append(doc))
    panel.select(image)
    panel.delete_btn.click()
    assert closed == ["lame"]


def test_each_open_image_has_its_own_list(window, qtbot):
    """Plusieurs images ouvertes : la liste montre celle qui est affichée et les ROI communes."""
    from fuji.ui import documents

    panel = window.layers_panel
    a = window.viewer.add_image(np.zeros((40, 60), np.uint16), name="A")
    mask = window.viewer.add_labels(np.ones((40, 60), np.uint8), name="A (masque)")
    documents.tag(mask, "A")
    b = window.viewer.add_image(np.zeros((40, 60), np.uint16), name="B")
    window.viewer.add_shapes(ndim=2, name="ROI")
    documents.show_only(window.viewer, "B")
    qtbot.wait(50)
    assert _names(panel) == ["ROI", "B"]
    documents.show_only(window.viewer, "A")
    qtbot.wait(50)
    assert _names(panel) == ["ROI", "A (masque)", "A"]
    a.visible = False  # cachée par son œil : sa liste reste
    qtbot.wait(50)
    assert _names(panel) == ["ROI", "A (masque)", "A"]
    del b
