"""Non-régressions du tour du code (version 0.9) : ce qui ne marchait pas, et marche."""

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


def test_napari_layer_controls_and_context_menu_work_without_a_napari_window(window, qtbot):
    """Les boutons de mode des réglages d'un calque (dessiner, peindre, sélectionner…) et le
    clic droit sur la liste des calques levaient une erreur à chaque clic."""
    import numpy as np
    from napari._app_model.constants import MenuId
    from napari._app_model.context import get_context
    from napari._qt._qapp_model import build_qmodel_menu
    from qtpy.QtWidgets import QAbstractButton

    window.set_mode("analyse", "layers")
    viewer, qt = window.viewer, window.qt_viewer
    image = viewer.add_image(np.zeros((4, 40, 50), np.uint16), name="pile")
    shapes = viewer.add_shapes(
        [np.array([[5, 5], [5, 20], [20, 20], [20, 5]])], shape_type="rectangle", name="ROI", ndim=2
    )
    points = viewer.add_points(np.array([[10.0, 10.0]]), name="Points", ndim=2)
    labels = viewer.add_labels(np.zeros((40, 50), np.uint8), name="masque")
    clicked = 0
    for layer in (shapes, points, labels, image):
        viewer.layers.selection.active = layer
        qtbot.wait(20)
        for b in qt.controls.widgets[layer].findChildren(QAbstractButton):
            if b.property("mode") and b.isEnabled():
                b.click()  # une exception ici fait échouer le test (pytest-qt)
                clicked += 1
        if hasattr(layer, "mode"):
            layer.mode = "pan_zoom"
    assert clicked >= 15
    viewer.layers.selection = {image}
    menu = build_qmodel_menu(MenuId.LAYERLIST_CONTEXT, parent=qt.layers)
    menu.update_from_context(get_context(viewer.layers))
    entries = [a for a in menu.actions() if not a.isSeparator()]
    assert len(entries) > 10
    count = len(viewer.layers)
    next(a for a in entries if a.text() == "Duplicate Layer").trigger()
    assert len(viewer.layers) == count + 1


def test_a_file_dropped_on_the_window_opens_in_analysis(window, tmp_path):
    import numpy as np
    import tifffile
    from qtpy.QtCore import QMimeData, QPointF, Qt, QUrl
    from qtpy.QtGui import QDropEvent

    assert window.acceptDrops() and not window.qt_viewer.acceptDrops() and not window.scope_qt.acceptDrops()
    first, second = tmp_path / "lame A.tif", tmp_path / "lame B.tif"
    for path in (first, second):
        tifffile.imwrite(path, np.random.default_rng(0).integers(0, 255, (40, 50), dtype=np.uint8))
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(first)), QUrl.fromLocalFile(str(second))])
    event = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dropEvent(event)
    assert window.mode == "analyse"
    # le premier remplace, le second s'ouvre à côté : le dernier est affiché
    assert [ly.name for ly in window.viewer.layers] == ["lame A", "lame B"]
    assert window.viewer.layers["lame B"].visible and not window.viewer.layers["lame A"].visible


def test_acquired_frames_are_shown_in_front_of_the_frozen_live(window, qtbot):
    from fuji.ui.main_window import ACQUISITION_LAYER

    window.live.start()
    qtbot.waitUntil(lambda: "Live" in window.scope_viewer.layers, timeout=5000)

    class Summary:
        files, error, acquired = [], None, 1

    layers = window.scope_viewer.layers
    window._on_acq_starting()
    try:
        window._on_acq_frame(None, window.m.snap())
        during = [(ly.name, ly.visible) for ly in layers]
    finally:
        window._on_acq_done(Summary())  # sinon la fermeture demanderait « arrêter l'acquisition ? »
    assert during == [("Live", False), (ACQUISITION_LAYER, True)]  # avant : cachée sous le live figé
    assert ACQUISITION_LAYER not in layers and layers["Live"].visible
    window.live.stop()


def test_run_during_a_preview_is_not_lost(window, qtbot):
    """« Exécuter » pendant le calcul d'un aperçu était ignoré, sans un mot."""
    import numpy as np

    panel = window.analysis
    window.set_mode("analyse", "analysis")
    window.viewer.add_image(np.random.default_rng(1).integers(0, 255, (60, 80), dtype=np.uint8), name="image")
    panel.select_operation("Flou gaussien")
    panel._busy = True  # un aperçu est en cours de calcul
    panel.run_btn.click()
    assert panel._run_pending and not panel.run_btn.isEnabled()
    panel._finished()  # l'aperçu se termine : l'exécution demandée part
    qtbot.waitUntil(lambda: not panel.busy, timeout=10000)
    assert any("flou" in ly.name.lower() for ly in window.viewer.layers)


def test_renamed_layers_stay_usable_as_inputs(window):
    import numpy as np

    panel = window.analysis
    layer = window.viewer.add_image(np.zeros((30, 40), np.uint8), name="avant")
    panel.select_operation("Flou gaussien")
    assert panel._inputs["image"].currentText() == "avant"
    layer.name = "après"
    assert panel._inputs["image"].currentText() == "après"  # avant : ancien nom, puis KeyError


def test_missing_inputs_say_what_to_do(window, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    panel = window.analysis
    panel.select_operation("Remplir les trous")
    panel._run()
    assert "Il manque un masque" in shown[0] and "Isoler les objets" in shown[0]
    panel.select_operation("Flou gaussien")
    panel._run()
    assert "ouvrez-en une" in shown[1]


def test_cursor_coordinates_no_longer_erase_messages(window):
    class Event:
        value = "[12, 34]: 1200"

    window.notify("capture_001 enregistrée")
    window._on_status(Event())
    assert window.statusBar().currentMessage() == "capture_001 enregistrée"
    assert window.cursor_label.text() == "[12, 34]: 1200"
    Event.value = "Ready"  # le « Ready » de napari n'est pas affiché
    window._on_status(Event())
    assert window.cursor_label.text() == ""
    window.warn("La caméra n'envoie plus d'images")
    assert window.statusBar().currentMessage().startswith("⚠")


def test_a_silent_camera_is_reported(window, qtbot, monkeypatch):
    """Caméra débranchée : le live se figeait sans un mot."""
    import time

    said = []
    window.live.stalled.connect(said.append)
    window.live.start()
    qtbot.waitUntil(lambda: "Live" in window.scope_viewer.layers, timeout=5000)
    cam = window.m.camera
    cam.stop_live()  # la caméra cesse d'envoyer des images, le live de Fuji attend toujours

    def lost() -> None:
        raise RuntimeError("caméra débranchée")

    monkeypatch.setattr(cam, "_check_alive", lost)
    window.live._poll()  # la dernière image déjà arrivée
    window.live._last_frame_time = time.monotonic() - 60  # … puis plus rien depuis une minute
    window.live._poll()
    window.live._poll()
    assert said == ["La caméra n'envoie plus d'images : caméra débranchée"]  # dit une seule fois
    assert "n'envoie plus d'images" in window.statusBar().currentMessage()
    window.live.stop()


def test_standard_buttons_are_in_french(window):
    from qtpy.QtWidgets import QMessageBox

    box = QMessageBox(
        QMessageBox.Icon.Question,
        "Fermer",
        "?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
    )
    texts = {b.text().replace("&", "") for b in box.buttons()}
    assert texts == {"Oui", "Non", "Annuler"}


def test_microscope_menu_starts_live_compositions_and_reference(window, qtbot, monkeypatch):
    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&Microscope")
    actions = {a.text(): a for a in menu.actions()}
    window.set_mode("analyse")
    actions["Panorama en direct"].trigger()
    assert window.mode == "microscope" and window._section == "acquisition"
    assert window.compose.running and window.compose.kind == "panorama"
    actions["Profondeur de champ étendue en direct"].trigger()  # une seule à la fois
    assert window.compose.kind == "panorama" and "déjà en cours" in window.statusBar().currentMessage()
    window.compose.cancel()
    taken = []
    monkeypatch.setattr(window.control.flatfield, "take", taken.append)
    actions["Prendre l'image de référence"].trigger()
    assert taken == ["flat"]
    window.live.stop()


def test_tool_settings_have_french_names(window):
    """Plus de « Sigma », « Radius », « Method », « Tolerance » dans les formulaires."""
    from fuji.analysis import list_operations

    english = {"sigma", "radius", "method", "tolerance", "min area", "command", "interpolation", "macro"}
    for op in list_operations():
        for p in op.params:
            assert p.label.lower() not in english, (op.name, p.name, p.label)
    panel = window.analysis
    panel.select_operation("Flou gaussien")
    assert next(w for w in panel._params if w.name == "sigma").label == "Rayon du flou (px)"
