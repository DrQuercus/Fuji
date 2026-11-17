"""Espace Analyse : menus des outils, fiches, démarches guidées et barre d'affichage."""

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


def _idle(qtbot, panel):
    qtbot.wait(120)
    qtbot.waitUntil(lambda: not panel._busy, timeout=20000)
    qtbot.wait(120)


def _cells(window):
    """Une image de fond clair : des « cellules » sombres sur un fond clair, à l'échelle."""
    import numpy as np

    yy, xx = np.mgrid[:120, :160]
    img = np.full((120, 160), 200, np.uint8)
    for cy, cx in ((30, 30), (30, 90), (80, 50), (85, 120), (40, 135)):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 12**2] = 60
    img[30, 30] = 200  # un trou dans une cellule
    layer = window.viewer.add_image(img, name="cellules")
    window.set_pixel_size(layer, 0.5)
    return layer


def test_tools_are_grouped_by_task_with_explanations(window):
    from fuji.analysis.guide import MENUS

    panel = window.analysis
    assert panel.current_page() == "menu"
    menu = panel.menu
    assert [s.menu.key for s in menu.sections.values()] == [m.key for m in MENUS]
    # tous les menus sont fermés : seuls les titres et leur explication sont visibles
    assert not any(s.is_open() for s in menu.sections.values())
    section = menu.sections["isoler"]
    row = section.rows["Seuillage automatique"]
    assert row.summary.text() == "Sépare les objets du fond automatiquement"
    section.header.clicked.emit()
    assert section.is_open()

    # recherche sans accents ni majuscules, dans les noms, les explications et ImageJ
    panel.search.setText("COMPTER")
    found = panel.visible_operations()
    assert "Analyse de particules" in found and "Trouver les maxima" in found and "Flou gaussien" not in found
    assert menu.sections["compter"].is_open()  # le menu qui contient un résultat s'ouvre
    assert not menu.recipe_rows["compter"].isHidden()  # les démarches sont cherchées aussi
    panel.search.setText("gaussian blur")
    assert panel.visible_operations() == ["Flou gaussien"]
    panel.search.setText("zzzz")
    assert panel.visible_operations() == [] and not menu.no_match.isHidden()
    panel.search.clear()
    assert len(panel.visible_operations()) == sum(len(m.operations) for m in MENUS)
    assert menu.sections["isoler"].is_open() and not menu.sections["compter"].is_open()


def test_tool_page_explains_then_runs(window, qtbot):
    panel = window.analysis
    layer = _cells(window)
    window.viewer.layers.selection.active = layer
    panel.menu.sections["isoler"].rows["Seuillage automatique"].clicked.emit()
    assert panel.current_page() == "operation"
    assert panel.op_title.text() == "Seuillage automatique"
    assert "Quand l'utiliser" in panel.when.text() and "Threshold" in panel.imagej_label.text()
    assert panel.details.isHidden()
    panel.details_btn.click()
    assert not panel.details.isHidden() and panel.details.text()
    assert panel._inputs["image"].currentText() == "cellules"
    assert next(w for w in panel._params if w.name == "method").label == "Méthode"  # majuscule
    panel.preview.setChecked(True)
    qtbot.waitUntil(lambda: any(ly.name.startswith("Aperçu") for ly in window.viewer.layers), timeout=10000)
    # retour aux menus : l'aperçu s'arrête et le menu de l'outil reste ouvert
    panel.back_btn.click()
    assert panel.current_page() == "menu" and not panel.preview.isChecked()
    assert not any(ly.name.startswith("Aperçu") for ly in window.viewer.layers)
    assert panel.menu.sections["isoler"].is_open()


def test_guided_cell_count_from_start_to_table(window, qtbot):
    import numpy as np

    panel = window.analysis
    window.set_mode("analyse", "analysis")
    layer = _cells(window)
    window.viewer.layers.selection.active = layer
    panel.menu.recipe_rows["compter"].clicked.emit()
    assert panel.current_page() == "recipe"
    steps = panel.recipe_page.step_buttons
    assert len(steps) == 6

    # étape 3 : le seuillage devine que les objets sont sombres (le fond est clair)
    steps[2].click()
    assert panel.current_page() == "operation" and panel._op.name == "Seuillage automatique"
    assert not panel.banner.isHidden() and "Étape 3 sur 6" in panel.banner_text.text()
    assert next(w for w in panel._params if w.name == "objects").value == "sombres"
    panel._run()
    _idle(qtbot, panel)
    assert "Fait" in panel.banner_text.text()
    mask = window.viewer.layers[-1]
    assert int(np.asarray(mask.data).astype(bool).sum()) > 1500

    # étape suivante : « Remplir les trous » prend le masque qui vient d'être créé
    panel.next_btn.click()
    assert panel._op.name == "Remplir les trous" and "Étape 4 sur 6" in panel.banner_text.text()
    assert panel._inputs["mask"].currentText() == mask.name
    panel._run()
    _idle(qtbot, panel)
    panel.next_btn.click()  # étape 5 (si besoin) : séparer les objets collés
    assert panel._op.name == "Séparer les objets (watershed)"
    panel.next_btn.click()  # on la saute
    assert panel._op.name == "Analyse de particules"
    assert panel.next_btn.text() == "Terminer la démarche"
    tabs = window.results_tabs.count()
    panel._run()
    _idle(qtbot, panel)
    assert window.results_tabs.count() == tabs + 1
    table = window.results_tabs.widget(window.results_tabs.count() - 1).table
    assert len(table.data) == 5  # les cinq cellules

    # la page des étapes montre ce qui est fait ; le retour ramène aux étapes
    panel.back_btn.click()
    assert panel.current_page() == "recipe"
    panel.recipe_page.back.emit()
    assert panel.current_page() == "menu" and panel._recipe is None


def test_guided_measure_starts_drawing(window):
    panel = window.analysis
    _cells(window)
    panel.open_recipe("mesurer")
    panel.recipe_page.step_buttons[0].click()  # « Dessiner sur l'image »
    assert window.viewer.layers["ROI"].mode == "add_ellipse"
    assert panel.current_page() == "recipe" and panel._step == 1
    assert panel.recipe_page._rows[0].findChild(type(panel.op_title), "StepNumberDone") is not None


def test_analysis_menu_bar_opens_tools_and_recipes(window):
    from fuji.analysis.guide import MENUS

    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&Analyse")
    titles = [a.text() for a in menu.actions() if a.menu() is not None]
    assert titles == ["Démarches guidées", *(m.title for m in MENUS)]
    sub = next(a.menu() for a in menu.actions() if a.text() == "Isoler les objets")
    act = next(a for a in sub.actions() if a.text() == "Seuil manuel")
    assert act.statusTip() == "Garde les pixels entre deux valeurs choisies"
    act.trigger()
    assert window.mode == "analyse" and window._section == "analysis"
    assert window.analysis._op.name == "Seuil manuel"
    window.open_recipe("profil")
    assert window.analysis.current_page() == "recipe"


def test_display_bar_follows_the_image_and_sets_the_contrast(window, qtbot):
    import numpy as np

    bar = window.display_bar
    window.set_mode("analyse")
    assert bar.isVisibleTo(window.canvas_stack.widget(1))
    assert bar.name_label.text() == "Aucune image" and bar.histogram.isHidden()

    rng = np.random.default_rng(1)
    stack = rng.normal(1000, 30, (3, 64, 80)).astype(np.uint16)
    stack[2] += 2000  # le 3e plan est bien plus clair
    layer = window.viewer.add_image(stack, name="pile")
    assert bar.layer is layer and bar.name_label.text() == "pile"
    assert "80×64" in bar.info_label.text() and "pile 3" in bar.info_label.text()
    assert "16 bits" in bar.info_label.text()

    # Auto : bornes autour des valeurs du plan affiché (0,35 % saturés de chaque côté)
    window.viewer.dims.set_current_step(0, 0)
    qtbot.wait(250)
    bar.auto_contrast()
    lo, hi = layer.contrast_limits
    assert 880 < lo < 960 and 1040 < hi < 1120
    assert bar.min_box.value() == pytest.approx(lo) and bar.max_box.value() == pytest.approx(hi)
    # le plan affiché change : l'histogramme suit, et Auto aussi
    window.viewer.dims.set_current_step(0, 2)
    qtbot.wait(250)
    bar.auto_contrast()
    assert layer.contrast_limits[0] > 2800
    # bornes saisies, tirées sur l'histogramme, valeurs impossibles refusées
    bar.max_box.setValue(5000)
    assert layer.contrast_limits[1] == 5000
    bar.histogram.limits_dragged.emit(100.0, 4000.0)
    assert tuple(layer.contrast_limits) == (100.0, 4000.0)
    bar.min_box.setValue(4500)
    assert tuple(layer.contrast_limits) == (100.0, 4000.0) and bar.min_box.value() == 100
    bar.full_contrast()
    assert layer.contrast_limits[1] == pytest.approx(float(stack[2].max()))

    # gamma et couleur ; ce n'est que de l'affichage
    bar.gamma.setValue(bar.gamma.maximum())
    assert layer.gamma == pytest.approx(5.0)
    bar.gamma_label.click()
    assert layer.gamma == pytest.approx(1.0)
    bar.colormap.setCurrentIndex(bar.colormap.findData("green"))
    assert layer.colormap.name == "green"
    layer.colormap = "magenta"  # changé ailleurs (panneau des calques) : la barre suit
    assert bar.colormap.currentData() == "magenta"
    assert np.array_equal(layer.data, stack)

    # un masque sélectionné : la barre garde l'image du dessous
    labels = window.viewer.add_labels(np.zeros((64, 80), np.uint8), name="masque")
    window.viewer.layers.selection.active = labels
    assert bar.layer is layer
    rgb = window.viewer.add_image(np.zeros((20, 30, 3), np.uint8), name="couleur", rgb=True)
    assert bar.layer is rgb and bar.colormap_group.isHidden() and "couleur 8 bits" in bar.info_label.text()
    window.viewer.layers.remove(rgb)
    assert bar.layer is layer
    window.viewer.layers.clear()
    assert bar.layer is None and bar.name_label.text() == "Aucune image" and bar.gamma_group.isHidden()


def test_window_stays_laptop_height_with_tall_layer_controls(window):
    """Les réglages d'un masque ou de formes (calques) défilent au lieu d'agrandir la fenêtre."""
    import numpy as np

    window.set_mode("analyse", "layers")
    window.viewer.add_image(np.zeros((50, 60), np.uint8), name="image")
    window.viewer.add_labels(np.zeros((50, 60), np.uint8), name="masque")
    window.viewer.add_shapes(name="ROI")
    for name in ("masque", "ROI"):
        window.viewer.layers.selection.active = window.viewer.layers[name]
        assert window.side.minimumSizeHint().height() < 460, name


# ------------------------------------------------- ouvrir, fermer, changer d'image
def _three_captures(window, qtbot):
    for pos in ((0, 0), (300, 150), (-200, 80)):
        window.m.xy_stage.move_to(*pos)
        window._on_snap(window.m.snap())
        qtbot.waitUntil(lambda: not window.gallery._reserved, timeout=10000)
    return window.gallery


def _layers(window):
    return [(ly.name, ly.visible) for ly in window.viewer.layers]


def test_opening_an_image_replaces_the_open_one(window, qtbot):
    gallery = _three_captures(window, qtbot)
    asked = []
    window._ask_close = lambda docs, lost, opening: asked.append(docs) or "close"
    gallery.list.setCurrentRow(2)
    gallery.open_selected()
    assert _layers(window) == [("capture_001", True)] and window.mode == "analyse"
    gallery.list.setCurrentRow(1)
    gallery.open_selected()  # remplace, sans question : pas de résultat à perdre
    assert _layers(window) == [("capture_002", True)] and asked == []

    # avec un résultat (masque) et une ROI : on demande ; « Annuler » ne change rien
    panel = window.analysis
    panel.select_operation("Seuillage automatique")
    panel._run()
    qtbot.waitUntil(lambda: not panel._busy, timeout=20000)
    panel.set_tool("shapes", "add_rectangle")
    window.viewer.layers["ROI"].add_rectangles([[10, 10], [40, 60]])
    window._ask_close = lambda docs, lost, opening: asked.append((docs, opening)) or "cancel"
    gallery.list.setCurrentRow(0)
    gallery.open_selected()
    assert asked == [(["capture_002"], True)]
    assert [n for n, _ in _layers(window)] == ["capture_002", "capture_002 (masque Default)", "ROI"]

    # « Garder ouverte » : la nouvelle image est seule visible, on passe de l'une à l'autre
    window._ask_close = lambda docs, lost, opening: "keep"
    gallery.open_selected()
    assert _layers(window)[-1] == ("capture_003", True)
    assert not window.viewer.layers["capture_002"].visible
    assert not window.viewer.layers["capture_002 (masque Default)"].visible
    assert window.viewer.layers["ROI"].visible  # les ROI sont communes, comme le ROI Manager
    bar = window.display_bar
    assert not bar.doc_combo.isHidden() and bar.name_label.isHidden()
    assert [bar.doc_combo.itemData(i) for i in range(bar.doc_combo.count())] == ["capture_002", "capture_003"]
    bar.doc_combo.activated.emit(0)
    assert window.viewer.layers["capture_002"].visible and not window.viewer.layers["capture_003"].visible
    assert bar.layer.name == "capture_002"
    # rouvrir une image déjà ouverte la remontre, sans la dupliquer
    gallery.list.setCurrentRow(0)
    gallery.open_selected()
    assert window.viewer.layers["capture_003"].visible and len(window.viewer.layers) == 4

    # « Fermer » : l'image affichée et ses résultats ; la dernière emporte les ROI
    window._ask_close = lambda docs, lost, opening: "close"
    window.show_document("capture_002")
    bar.close_btn.click()
    assert [n for n, _ in _layers(window)] == ["ROI", "capture_003"]
    assert window.viewer.layers["capture_003"].visible and bar.name_label.text() == "capture_003"
    window.close_image()  # Ctrl+W
    assert len(window.viewer.layers) == 0 and bar.name_label.text() == "Aucune image"


def test_close_asks_only_when_results_would_be_lost(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    gallery = _three_captures(window, qtbot)
    gallery.list.setCurrentRow(0)
    gallery.open_selected()
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()) or 0)
    window.close_image()  # aucune perte : pas de question
    assert shown == [] and len(window.viewer.layers) == 0
    gallery.open_selected()
    window.analysis.set_tool("shapes", "add_ellipse")
    window.viewer.layers["ROI"].add_ellipses([[10, 10], [40, 60]])
    window.close_image()  # la question s'affiche ; sans réponse (fenêtre fermée) : on annule
    assert shown == ["Fermer capture_003 et ses résultats ?"] and len(window.viewer.layers) == 2


def test_deleted_or_renamed_images_follow_in_analysis(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QInputDialog, QMessageBox

    gallery = _three_captures(window, qtbot)
    gallery.list.setCurrentRow(0)
    gallery.open_selected()
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("lame A", True))
    gallery.rename_selected()
    assert [n for n, _ in _layers(window)] == ["lame A"]
    assert window.viewer.layers["lame A"].metadata["source"].endswith("lame A.ome.tif")
    gallery.open_selected()  # rouvrir après le renommage : pas de doublon
    assert len(window.viewer.layers) == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    gallery.delete_selected()  # supprimée de la liste : elle se ferme aussi dans Analyse
    assert len(window.viewer.layers) == 0


def test_napari_layer_and_view_buttons_work(window):
    """Ces boutons passaient par les « actions » de napari, sans effet dans Fuji."""
    import numpy as np

    viewer, qt = window.viewer, window.qt_viewer
    viewer.add_image(np.zeros((4, 30, 40), np.uint8), name="a")
    viewer.add_image(np.zeros((30, 40), np.uint8), name="b")
    viewer.layers.selection = {viewer.layers["b"]}
    qt.layerButtons.deleteButton.click()
    assert [ly.name for ly in viewer.layers] == ["a"]
    viewer.camera.zoom = 20
    qt.viewerButtons.resetViewButton.click()
    assert viewer.camera.zoom < 20
    qt.viewerButtons.ndisplayButton.click()
    assert viewer.dims.ndisplay == 3
    qt.viewerButtons.ndisplayButton.click()
    assert viewer.dims.ndisplay == 2
    grid = getattr(viewer, "grid", None) or viewer.canvas.grid
    qt.viewerButtons.gridViewButton.click()
    assert grid.enabled
    assert qt.viewerButtons.consoleButton.isHidden()
