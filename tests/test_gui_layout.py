"""Tenue de l'interface : rien ne déborde, les panneaux ont le même style, les nombres
s'écrivent à la française."""

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


def _content(page):
    from qtpy.QtWidgets import QScrollArea

    widget = page.layout().itemAt(0).widget()
    return widget.widget() if isinstance(widget, QScrollArea) else widget


def _place(window, key):
    """Le contenu de la page d'une section et la largeur dont il dispose, là où il est affiché
    (« Contrôle » vit dans le panneau de gauche ; « Images » y passe dans l'espace Analyse)."""
    for pages, panel in ((window._left_pages, window.left), (window._pages, window.side)):
        holder = pages.get(key)
        if holder is not None and holder.layout().count():
            return _content(holder), panel.minimumWidth() - 32
    raise AssertionError(f"page « {key} » vide")


def test_no_panel_is_wider_than_its_place(window, qtbot):
    """Assistants et Préférences étaient plus larges que le panneau : coupés au bord droit."""
    from fuji.ui.chrome import sections_of

    window.show()
    side_room = window.side.minimumWidth() - 32  # largeur minimale du panneau, moins ses marges
    for mode in ("microscope", "analyse"):
        window.set_mode(mode)
        for key in sections_of(mode):
            window.sidebar.select(key)
            qtbot.wait(20)
            content, room = _place(window, key)
            need = content.minimumSizeHint().width()
            assert need <= room, (mode, key, need, room)
    # les assistants, un par un (leurs pages ne sont pas visibles depuis la liste)
    window.set_mode("microscope", "assistants")
    for key in window.assistants.assistants:
        window.assistants.open(key)
        qtbot.wait(20)
        need = window.assistants.minimumSizeHint().width()
        assert need <= side_room, (key, need, side_room)
    # la barre sous l'image tient dans la zone de l'image la plus étroite
    window.set_mode("analyse")
    window.viewer.add_image(np.zeros((40, 60), np.uint16), name="image")
    window.viewer.add_labels(np.ones((40, 60), np.uint8), name="masque")
    qtbot.wait(50)
    assert window.display_bar.minimumSizeHint().width() <= window.canvas_stack.minimumWidth()


def test_assistants_are_a_list_then_a_page(window):
    panel = window.assistants
    window.set_mode("microscope", "assistants")
    assert panel.current() is None and not panel.menu_page.isHidden() and panel.page.isHidden()
    assert list(panel.rows) == ["kohler", "focus", "calibration", "illumination", "stage"]  # démo motorisée
    assert panel.rows["focus"].summary.text().startswith("Une jauge de netteté")
    panel.rows["calibration"].clicked.emit()
    assert panel.current() == "calibration" and panel.page_title.text() == "Taille du pixel"
    assert panel.stack.currentWidget() is panel.calibration and panel.menu_page.isHidden()
    panel.back_btn.click()
    assert panel.current() is None and not panel.menu_page.isHidden()
    # la mesure de netteté a des noms compréhensibles (plus de « tenengrad »)
    assert panel.focus.metric.itemText(0) == "Contours (conseillée)"
    assert panel.focus.metric.itemData(0) == "tenengrad"
    panel.focus.metric.setCurrentIndex(1)
    assert panel.focus.tracker.method == "laplacien"


def test_acquisition_options_unfold_with_their_switch(window):
    panel = window.acquisition
    window.set_mode("microscope", "acquisition")
    card = panel.use_time.parent()
    assert not panel.use_time.isChecked() and card.content.isHidden()
    panel.use_time.setChecked(True)
    assert not card.content.isHidden() and not panel.interval.isHidden()
    panel.use_time.setChecked(False)
    assert card.content.isHidden()


def test_numbers_are_french_and_a_typed_dot_is_a_decimal_point(window, qtbot):
    """Sur un Mac en français, taper « 12.5 » dans un champ donnait 125."""
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QDoubleSpinBox

    window.show()
    box = window.control.findChildren(QDoubleSpinBox)[0]
    assert "," in box.text()  # « 10,00 ms »
    for typed, expected in (("12.5", 12.5), ("7,25", 7.25)):
        box.setFocus()
        box.selectAll()
        QTest.keyClicks(box, typed)
        QTest.keyClick(box, Qt.Key.Key_Return)
        assert box.value() == pytest.approx(expected), typed


def test_napari_style_stays_on_napari_widgets(window):
    """La feuille de style de napari élargissait tous les champs de nombres de Fuji."""
    from qtpy.QtWidgets import QApplication

    assert "QtLayerList" not in QApplication.instance().styleSheet()
    assert "QtLayerList" in window.qt_viewer.styleSheet()
    window.show()
    assert window.display_bar.min_box.minimumWidth() < 70


def test_empty_analysis_invites_to_open_an_image(window, qtbot):
    from fuji.ui.display_bar import scale_bar_of

    window.show()
    window.set_mode("analyse")
    qtbot.wait(50)
    assert not window.empty_hint.isHidden() and not scale_bar_of(window.viewer).visible
    assert window.empty_hint.geometry() == window.qt_viewer.geometry()
    layer = window.viewer.add_image(np.zeros((40, 60), np.uint16), name="image")
    qtbot.wait(50)
    assert window.empty_hint.isHidden() and scale_bar_of(window.viewer).visible
    window.viewer.layers.remove(layer)
    qtbot.wait(50)
    assert not window.empty_hint.isHidden() and not scale_bar_of(window.viewer).visible


def test_status_bar_gives_the_pixel_under_the_mouse(window):
    data = np.arange(40 * 60, dtype=np.uint16).reshape(40, 60)
    layer = window.viewer.add_image(data, name="image")
    window.set_pixel_size(layer, 0.5)
    window.set_mode("analyse")
    window.viewer.cursor.position = (10 * 0.5, 25 * 0.5)  # en µm : ligne 10, colonne 25
    assert window._cursor_text() == f"x 25 · y 10 — {data[10, 25]}"
    window.viewer.cursor.position = (500.0, 500.0)  # hors de l'image
    assert window._cursor_text() == ""
    rgb = window.viewer.add_image(np.full((40, 60, 3), 7, np.uint8), name="couleur", rgb=True)
    window.viewer.layers.selection.active = rgb
    window.viewer.cursor.position = (3.0, 4.0)
    assert window._cursor_text() == "x 4 · y 3 — 7, 7, 7"


def test_results_dock_has_its_own_title_bar(window):
    from pandas import DataFrame

    from fuji.analysis.image import Table

    bar = window.results_dock.titleBarWidget()
    assert bar is not None and window.accumulate_box.parent() is bar
    window.show_table(Table(DataFrame({"A": [1.5, 2.5]}), name="Mesure — essai"))
    assert not window.results_dock.isHidden()
    widget = window.results_tabs.widget(0)
    assert widget.title.text() == "<b>Mesure — essai</b> — 2 lignes"
