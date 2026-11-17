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
    from fuji.ui.app import create_window, isolate_settings

    isolate_settings(str(tmp_path))
    _, win = create_window("demo")
    win.m.camera._timing = False
    qtbot.addWidget(win)  # qtbot ferme et détruit la fenêtre en fin de test
    return win


def _capture(window, qtbot):
    """Prend une image : elle rejoint la liste « Images », puis on l'ouvre dans Analyse."""
    window._on_snap(window.m.snap())
    qtbot.waitUntil(lambda: not window.gallery._reserved, timeout=10000)
    info = window.gallery.entries()[0]
    window.open_capture(info, None)
    return window.viewer.layers[info.name]


def test_window_live_snap_and_analysis(window, qtbot):
    from fuji.analysis import get_operation

    window.live.start()
    qtbot.waitUntil(lambda: "Live" in window.scope_viewer.layers, timeout=5000)
    assert window.topbar.live.isChecked() and "Live" not in window.viewer.layers
    window.live.stop()
    window._on_snap(window.m.snap())
    assert window.quickbar.histogram._hists
    assert len(window.viewer.layers) == 0  # la capture va dans la liste, pas dans Analyse
    qtbot.waitUntil(lambda: not window.gallery._reserved, timeout=10000)
    (info,) = window.gallery.entries()
    assert info.name == "capture_001" and info.path.exists() and info.thumbnail_path.exists()
    assert info.metadata["objective"] == "10x/0.25" and info.pixel_size_um == window.m.pixel_size_um()
    assert window.sidebar.buttons["images"].text() == "Images (1)"
    window.gallery.list.setCurrentRow(0)
    window.gallery.open_selected()  # double-clic : ouvrir dans Analyse
    assert window.mode == "analyse" and "capture_001" in window.viewer.layers

    panel = window.analysis
    op = get_operation("Niveaux de gris")
    panel._op = op
    panel._build_form(op)
    panel._inputs["image"].setCurrentText("capture_001")
    inputs, params = panel._collect()
    panel._show_outputs(op, op.func(*inputs, **params))
    assert any("gris" in layer.name for layer in window.viewer.layers)


def test_sections_and_expert_mode(window):
    from fuji.ui.chrome import SECTIONS

    for key, *_ in SECTIONS:
        window.sidebar.select(key)
        if key == "control":  # toujours ouvert, dans le panneau de gauche
            assert not window.left.isHidden() and window.left_stack.currentWidget() is window._left_pages[key]
        else:
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


# ------------------------------------------------------------------ non-régressions
def _mini_profile(path, name="Mini", driver="demo.camera", **options):
    from fuji.core.profile import MicroscopeProfile

    opts = {"width": 64, "height": 48, "simulate_timing": False} if driver == "demo.camera" else {}
    MicroscopeProfile.model_validate(
        {"name": name, "devices": [{"name": "cam", "driver": driver, "options": {**opts, **options}}]}
    ).save(path)
    return path


@pytest.fixture
def exclusive_camera_driver():
    """Caméra qui, comme une vraie, ne peut être ouverte qu'une fois à la fois."""
    import fuji.drivers as drivers
    from fuji.drivers.demo import DemoCamera

    state = {"open": 0}

    class ExclusiveCamera(DemoCamera):
        def initialize(self) -> None:
            if state["open"]:
                raise RuntimeError("caméra déjà utilisée")
            state["open"] += 1
            super().initialize()

        def shutdown(self) -> None:
            if self.initialized:
                state["open"] -= 1
            super().shutdown()

    @drivers.register_driver("test.exclusive_camera", kind="camera")
    def _factory(spec, ctx):  # type: ignore[no-untyped-def]
        return ExclusiveCamera(spec.name, width=64, height=48, simulate_timing=False)

    yield state
    drivers._REGISTRY.pop("test.exclusive_camera", None)


def test_switch_between_profiles_sharing_one_camera(window, tmp_path, exclusive_camera_driver):
    """L'ancien microscope est fermé avant d'ouvrir le nouveau (même caméra physique)."""
    a = _mini_profile(tmp_path / "a.yaml", "A", driver="test.exclusive_camera")
    b = _mini_profile(tmp_path / "b.yaml", "B", driver="test.exclusive_camera")
    window.switch_profile(str(a))
    assert window.m.profile.name == "A"
    window.switch_profile(str(b))
    assert window.m.profile.name == "B" and exclusive_camera_driver["open"] == 1
    window.reload_microscope()
    assert window.m.profile.name == "B" and window.m.connected


def test_failed_switch_returns_to_previous_microscope(window, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    good = _mini_profile(tmp_path / "bon.yaml", "Bon")
    bad = _mini_profile(tmp_path / "mauvais.yaml", "Mauvais", driver="pilote.inexistant")
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    window.switch_profile(str(good))
    window.switch_profile(str(bad))
    assert shown and "Bon" in shown[0]
    assert window.m.profile.name == "Bon" and window.m.connected
    assert window.m.snap().data.shape[:2] == (48, 64)  # l'interface reste utilisable


def test_manual_prompt_from_worker_thread_can_be_aborted(window, qtbot):
    """Une consigne manuelle en attente ne bloque ni l'interface ni la fermeture."""
    import threading

    from fuji.core.manual import ManualAction

    handler = window.manual_handler
    handler._ask = lambda request: None  # la boîte de dialogue ne s'ouvre jamais
    handler._request.disconnect()
    answers = []
    worker = threading.Thread(target=lambda: answers.append(handler(ManualAction("obj", "Placez le 40x"))))
    worker.start()
    qtbot.wait(150)
    assert worker.is_alive()  # il attend la réponse
    handler.abort_pending()
    worker.join(timeout=2)
    assert not worker.is_alive() and answers == [False]


def test_manual_prompt_answer_reaches_worker_thread(window, qtbot, monkeypatch):
    import threading

    from qtpy.QtWidgets import QMessageBox

    from fuji.core.manual import ManualAction

    shown_in = []

    def fake_exec(box):  # type: ignore[no-untyped-def]
        shown_in.append(threading.current_thread())
        box._clicked = box.buttons()[0]  # « Continuer »
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._clicked)
    answers = []
    action = ManualAction("obj", "Placez le 40x")
    worker = threading.Thread(target=lambda: answers.append(window.manual_handler(action)))
    worker.start()
    qtbot.waitUntil(lambda: bool(answers), timeout=3000)
    worker.join(timeout=2)
    assert answers == [True]
    assert shown_in == [threading.main_thread()]  # la boîte s'ouvre dans le thread de l'interface
    assert window.manual_handler(action) is True  # appel direct depuis le thread de l'interface


def test_controls_locked_during_acquisition(window, qtbot, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    # « Lancer quand même ? » (avertissements du plan) : oui
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    panel = window.acquisition
    panel.folder.setText(str(tmp_path))
    panel.use_time.setChecked(True)
    panel.interval.setValue(0.05)
    panel.loops.setValue(4)
    window.live.start()
    done = []
    panel.acquisition_done.connect(done.append)
    panel.start()
    assert window._acquiring and not window.live.running
    assert not window.topbar.capture.isEnabled() and not window.quickbar.isEnabled()
    window.snap()  # refusée pendant l'acquisition
    assert not window.gallery.entries()
    qtbot.waitUntil(lambda: bool(done), timeout=10000)
    assert done[0].acquired == 4 and done[0].error is None
    assert window.topbar.capture.isEnabled() and window.quickbar.isEnabled()
    (entry,) = window.gallery.entries()  # le fichier de l'acquisition rejoint la liste des images
    assert entry.kind == "acquisition" and entry.path.exists() and "4 images" in entry.summary
    assert window.live.running  # le live, arrêté pendant l'acquisition, a repris


def test_quickbar_keeps_objectives_with_same_magnification_apart(qtbot, mono_demo_profile):
    from fuji.core import Microscope
    from fuji.ui.quickbar import QuickBar

    prof = mono_demo_profile
    prof.objectives.append(type(prof.objectives[0])(label="10x/0.30 Ph", magnification=10, na=0.3))
    with Microscope(prof) as scope:
        bar = QuickBar(scope)
        qtbot.addWidget(bar)
        options = bar.objectives.options()
        assert len(set(options)) == len(options) == 3
        bar._objective_clicked("10x/0.30 Ph")
        assert scope.objective.label == "10x/0.30 Ph"
        bar._objective_clicked("40x")
        assert scope.objective.label == "40x/0.65"
        bar.shutdown()


def test_analysis_uses_the_displayed_plane_of_a_stack(window):
    import numpy as np

    from fuji.ui.panels.analysis_panel import layer_to_image

    stack = np.zeros((5, 3, 20, 30), np.uint16)
    for t in range(5):
        for z in range(3):
            stack[t, z] = 10 * t + z
    layer = window.viewer.add_image(stack, name="pile", scale=(1, 2.0, 0.5, 0.5))
    window.viewer.dims.set_current_step(0, 3)
    window.viewer.dims.set_current_step(1, 2)
    img = layer_to_image(layer, window.viewer)
    assert img.data.shape == (20, 30) and int(img.data[0, 0]) == 32
    assert img.name == "pile [4, 3]" and img.pixel_size_um == pytest.approx(0.5)
    assert layer_to_image(layer).data.shape == stack.shape  # sans viewer : la pile entière


def test_reference_image_corrects_live_and_captures(window, qtbot, monkeypatch):
    """Image de référence : prise dans Contrôle, le fond est corrigé dès le live."""
    import numpy as np
    from qtpy.QtWidgets import QMessageBox

    def corners(data):  # luminosité des coins par rapport au centre
        d = data.astype(float)
        h, w = d.shape[:2]
        return d[:40, :40].mean() / d[h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20].mean()

    m, widget = window.m, window.control.flatfield
    m.camera.set_property("Échantillon", "vide")  # une zone vide de la lame
    assert not widget.enable.isChecked() and "Aucune référence" in widget.status.text()
    assert corners(m.snap().data) < 0.9  # vignettage du simulateur

    # activée sans référence : dit quoi faire, les captures le signalent
    widget.enable.setChecked(True)
    assert m.flatfield.enabled and not m.flatfield.active
    assert "Prendre la référence" in widget.status.text()
    window._on_snap(m.snap())
    qtbot.waitUntil(lambda: not window.gallery._reserved, timeout=10000)
    assert "NON corrigé" in window.statusBar().currentMessage()

    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.Yes)
    widget.take_btn.click()
    qtbot.waitUntil(lambda: m.flatfield.active, timeout=20000)
    assert asked == [] and "Fond corrigé" in widget.status.text()
    assert window.assistants.illumination.reference.enable.isChecked()  # même état partout
    frame = m.snap()
    assert corners(frame.data) == pytest.approx(1.0, abs=0.02)
    assert frame.metadata.extra["flat_field_corrected"] is True
    window.live.start()
    qtbot.waitUntil(lambda: "Live" in window.scope_viewer.layers, timeout=5000)
    qtbot.wait(300)
    assert corners(np.asarray(window.scope_viewer.layers["Live"].data)) == pytest.approx(1.0, abs=0.02)
    window.live.stop()
    window._on_snap(m.snap())
    qtbot.waitUntil(lambda: not window.gallery._reserved, timeout=10000)
    assert "fond corrigé" in window.statusBar().currentMessage()

    # autre objectif : pas de référence pour lui, la correction se suspend et le dit
    other = next(o.label for o in m.profile.objectives if o.label != m.objective.label)
    m.set_objective(other)
    qtbot.waitUntil(lambda: "Pas de référence" in widget.status.text(), timeout=3000)
    assert not m.flatfield.active and corners(m.snap().data) < 0.9
    widget.enable.setChecked(False)
    assert "Aucune référence" in widget.status.text()


def test_doubtful_reference_images_are_refused_or_questioned(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    m, widget = window.m, window.control.flatfield
    warned, asked = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.No)
    # un échantillon dans le champ : question, et « non » n'enregistre rien
    widget.take_btn.click()
    qtbot.waitUntil(lambda: bool(asked), timeout=20000)
    qtbot.waitUntil(lambda: widget.take_btn.isEnabled(), timeout=5000)
    assert "zone vide" in asked[0] and not m.flatfield.store.exists(*m.flatfield.keys())
    # image saturée : refusée, avec la marche à suivre
    m.camera.set_property("Échantillon", "vide")
    m.camera.set_exposure(m.camera.get_exposure() * 40)
    widget.take_btn.click()
    qtbot.waitUntil(lambda: bool(warned), timeout=20000)
    assert "saturée" in warned[0] and "Réduisez l'exposition" in warned[0]
    assert not m.flatfield.store.exists(*m.flatfield.keys())


def test_focus_peaking_layer_is_scaled_to_the_image(window):
    import numpy as np

    widget = window.assistants.focus
    frame = np.random.default_rng(0).integers(0, 4000, (2160, 3840), dtype=np.uint16)
    widget.peaking.setChecked(True)
    widget.process(frame)
    layer = window.scope_viewer.layers["Zones nettes"]
    px = window.m.pixel_size_um() or 1.0
    step = round(float(layer.scale[-1]) / px)
    assert step == 4 and layer.data.shape == (540, 960)
    extent = layer.data.shape[1] * float(layer.scale[-1])
    assert extent == pytest.approx(3840 * px)  # le masque recouvre toute l'image


def test_calibration_belongs_to_the_objective_it_was_measured_with(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    cal = window.assistants.calibration
    window.m.camera.set_property("Échantillon", "micromètre")
    window.m.set_objective("10x/0.25", declare=True)
    expected = window.m.pixel_size_um()
    cal._measure()
    qtbot.waitUntil(lambda: cal._result is not None, timeout=10000)
    qtbot.waitUntil(cal.measure.isEnabled, timeout=5000)
    assert cal._result.reliable and cal.save.isEnabled()
    window.m.set_objective("40x/0.65", declare=True)  # l'utilisateur change d'objectif...
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    cal._save()  # ...la calibration va quand même à l'objectif mesuré
    by_label = {o.label: o for o in window.m.profile.objectives}
    assert by_label["10x/0.25"].pixel_size_um == pytest.approx(expected / window.m.camera.binning, rel=5e-3)
    assert by_label["40x/0.65"].pixel_size_um is None


def test_channel_page_reload_keeps_changes_made_elsewhere(qtbot):
    """Revenir sur la page « Canaux » ne doit pas réécrire l'ancien formulaire par-dessus
    ce que d'autres pages ont modifié."""
    from fuji.core.profile import MicroscopeProfile, builtin_profiles
    from fuji.ui.config.editor import ConfigEditor

    data = MicroscopeProfile.load(builtin_profiles()["demo"]).model_dump(mode="json")
    editor = ConfigEditor(data)
    qtbot.addWidget(editor)
    editor.go_to("Canaux")
    page = next(p for p in editor.pages if p.title == "Canaux")
    first = editor.data["channels"][0]
    editor.go_to("Général")
    first["exposure_ms"] = 123.0  # modifié par une autre page (ou le YAML)
    editor.go_to("Canaux")
    assert editor.data["channels"][0]["exposure_ms"] == 123.0
    page._commit()
    assert editor.data["channels"][0]["exposure_ms"] == 123.0


# ------------------------------------------------------- analyse à la manière d'ImageJ
def _idle(qtbot, panel):
    qtbot.wait(120)
    qtbot.waitUntil(lambda: not panel._busy, timeout=20000)
    qtbot.wait(120)


def _last_table(window):
    return window.results_tabs.widget(window.results_tabs.count() - 1)


def test_roi_tools_and_quick_measurements(window, qtbot):
    import numpy as np

    capture = _capture(window, qtbot)
    px = capture.metadata["pixel_size_um"]
    panel = window.analysis
    panel.set_tool("shapes", "add_rectangle")
    roi = window.viewer.layers["ROI"]
    assert roi.mode == "add_rectangle" and tuple(roi.scale) == tuple(capture.scale[-2:])
    roi.add_rectangles(np.array([[100, 150], [100, 400], [300, 400], [300, 150]], float))
    roi.add_lines(np.array([[50, 50], [50, 600]], float))
    panel.set_tool("", "pan_zoom")
    assert roi.mode == "pan_zoom"
    window.viewer.layers.selection.active = capture

    window.quick_measure()  # Ctrl+M : mesure les ROI dessinées, comme ImageJ
    _idle(qtbot, panel)
    table = _last_table(window).table
    assert table.data["Type"].tolist() == ["rectangle", "line"]
    assert table.data["Area"].iloc[0] == pytest.approx(200 * 250 * px * px)
    assert table.data["Length"].iloc[1] == pytest.approx(550 * px)

    window.quick_operation("Profil d'intensité")
    _idle(qtbot, panel)
    widget = _last_table(window)
    assert widget.plot is not None and widget.table.plot.kind == "ligne"
    widget.plot.grab()  # le graphique se dessine sans erreur
    widget.plot.mouseDoubleClickEvent(None)
    widget.plot.grab()

    window.quick_operation("Histogramme")
    _idle(qtbot, panel)
    assert _last_table(window).table.plot.kind == "barres"

    panel.clear_rois()
    assert len(roi.data) == 0
    window.quick_measure()  # plus de ROI : toute l'image
    _idle(qtbot, panel)
    assert _last_table(window).table.name.startswith("Mesure")


def test_roi_coordinates_follow_the_image_scale(window):
    """Un calque de formes n'a pas forcément l'échelle de l'image mesurée."""
    import numpy as np

    from fuji.ui.panels.analysis_panel import layer_to_points, layer_to_rois

    image = window.viewer.add_image(
        np.zeros((100, 200), np.uint8), name="img", scale=(0.5, 0.5), translate=(10, 20)
    )
    image.metadata["pixel_size_um"] = 0.5
    shapes = window.viewer.add_shapes(
        [np.array([[15.0, 30.0], [15.0, 50.0], [25.0, 50.0], [25.0, 30.0]])],
        shape_type="rectangle",
        scale=(1, 1),
    )
    rois = layer_to_rois(shapes, image, window.viewer)
    assert rois.vertices(0).tolist() == [[10, 20], [10, 60], [30, 60], [30, 20]] and rois.pixel_size_um == 0.5
    points = window.viewer.add_points(np.array([[20.0, 40.0]]), scale=(2, 2))
    assert layer_to_points(points, image, window.viewer).data.tolist() == [[60.0, 120.0]]


def test_live_preview_shows_and_removes_its_layer(window, qtbot):
    _capture(window, qtbot)
    panel = window.analysis
    window.sidebar.select("analysis")
    panel.select_operation("Flou gaussien")
    panel._inputs["image"].setCurrentText("capture_001")
    before = len(window.viewer.layers)

    def previews():
        return [layer.name for layer in window.viewer.layers if layer.name.startswith("Aperçu")]

    panel.preview.setChecked(True)
    qtbot.waitUntil(lambda: bool(previews()), timeout=10000)
    assert previews() == ["Aperçu — capture_001 (flou σ=2)"]
    next(w for w in panel._params if w.name == "sigma").value = 5.0
    qtbot.waitUntil(lambda: previews() == ["Aperçu — capture_001 (flou σ=5)"], timeout=10000)
    assert "Aperçu — capture_001 (flou σ=5)" not in [panel._inputs["image"].itemText(i) for i in range(5)]
    panel._run()
    _idle(qtbot, panel)
    assert previews() == [] and window.viewer.layers[-1].name == "capture_001 (flou σ=5)"
    assert len(window.viewer.layers) == before + 1
    panel.preview.setChecked(False)


def test_operations_apply_to_a_plane_or_to_the_whole_stack(window, qtbot):
    import numpy as np

    panel = window.analysis
    window.sidebar.select("analysis")
    stack = np.zeros((4, 30, 40), np.uint8)
    for k in range(4):
        stack[k, 5 + k, 10] = 200
    layer = window.viewer.add_image(stack, name="pile", scale=(1, 0.5, 0.5))
    layer.metadata["pixel_size_um"] = 0.5
    window.viewer.dims.set_current_step(0, 2)

    panel.select_operation("Trouver les maxima")
    panel._inputs["image"].setCurrentText("pile")
    assert not panel.whole_stack.isHidden()
    panel._run()  # le plan affiché seulement
    _idle(qtbot, panel)
    points = window.viewer.layers[-1]
    assert points.data.tolist() == [[7.0, 10.0]] and "[3]" in _last_table(window).table.name

    panel.whole_stack.setChecked(True)
    panel._run()
    _idle(qtbot, panel)
    points = window.viewer.layers[-1]
    assert points.data.tolist() == [[k, 5.0 + k, 10.0] for k in range(4)]
    assert _last_table(window).table.data["Plan"].tolist() == ["1", "2", "3", "4"]

    panel.select_operation("Flou gaussien")
    panel._inputs["image"].setCurrentText("pile")
    panel.whole_stack.setChecked(True)
    panel._run()
    _idle(qtbot, panel)
    assert window.viewer.layers[-1].data.shape == (4, 30, 40)

    panel.select_operation("Projection")  # opération de pile : reçoit toute la pile
    panel._inputs["image"].setCurrentText("pile")
    assert panel.whole_stack.isHidden()
    panel._run()
    _idle(qtbot, panel)
    assert window.viewer.layers[-1].data.shape == (30, 40) and int(window.viewer.layers[-1].data.sum()) == 800


def test_scale_and_view_export(window, tmp_path, monkeypatch):
    import numpy as np
    from qtpy.QtWidgets import QFileDialog

    layer = window.viewer.add_image(np.zeros((50, 80), np.uint8), name="sans échelle")
    window.set_pixel_size(layer, 0.25)
    assert layer.metadata["pixel_size_um"] == 0.25 and tuple(layer.scale) == (0.25, 0.25)
    out = tmp_path / "vue.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window.export_view()
    assert out.exists() and out.stat().st_size > 1000


def test_every_operation_builds_its_form(window):
    """Toutes les opérations du catalogue s'affichent dans le panneau (formulaire généré)."""
    from fuji.analysis import list_operations

    panel = window.analysis
    panel.set_expert(True)
    for op in list_operations():
        panel.select_operation(op.name)
        assert panel._op is op and set(panel._inputs) == {spec.name for spec in op.inputs}
        assert {w.name for w in panel._params} == {p.name for p in op.params}
    panel.show_menu()
    panel.search.setText("threshold")  # recherche par le nom de la commande ImageJ
    names = panel.visible_operations()
    assert "Seuillage automatique" in names and "Seuillage local" in names and "Flou gaussien" not in names


# ------------------------------------------------- espaces, liste des images, petit écran
def test_microscope_and_analysis_spaces_stay_apart(window, qtbot):
    from fuji.ui.chrome import sections_of

    assert window.mode == "microscope" and window.canvas_stack.currentIndex() == 0
    visible = [k for k, b in window.sidebar.buttons.items() if not b.isHidden()]
    assert visible == sections_of("microscope")
    window.live.start()
    qtbot.waitUntil(lambda: "Live" in window.scope_viewer.layers, timeout=5000)

    window.set_mode("analyse")
    assert window.canvas_stack.currentIndex() == 1 and not window.live.running  # live en pause
    assert window.topbar.live.isHidden() and window.topbar.capture.isHidden()
    assert [k for k, b in window.sidebar.buttons.items() if not b.isHidden()] == sections_of("analyse")
    window.sidebar.select("analysis")
    window.analysis.set_tool("shapes", "add_rectangle")
    roi = window.viewer.layers["ROI"]
    # la liste des images reste ouverte à gauche, en même temps que les outils
    shown = window.centralWidget()  # (la fenêtre de test n'est pas affichée à l'écran)
    assert window.gallery.isVisibleTo(shown) and window.analysis.isVisibleTo(shown)
    window.sidebar.select("images")  # elle est déjà là : rien ne change, l'outil reste actif
    assert window._section == "analysis" and roi.mode == "add_rectangle"
    window.sidebar.select("layers")  # changer de section à droite relâche l'outil de dessin
    assert roi.mode == "pan_zoom"

    window.set_mode("microscope")
    qtbot.waitUntil(lambda: window.live.running, timeout=5000)  # le live reprend au retour
    assert window._section == "acquisition"
    # le contrôle du microscope reste ouvert à gauche, avec n'importe quelle section à droite
    for key in ("assistants", "acquisition", "images", "config"):
        window.sidebar.select(key)
        assert window.control.isVisibleTo(shown) and window.stack.currentWidget() is window._pages[key]
    window.sidebar._clicked("control")  # son bouton le replie… et le rouvre
    assert window.left.isHidden() and window._section == "config"
    window.sidebar._clicked("control")
    assert window.control.isVisibleTo(shown)
    window.sidebar.select("layers")  # une section de l'autre espace y emmène
    assert window.mode == "analyse"
    window._toggle_live()  # Ctrl+L depuis Analyse : retour au microscope, live allumé
    assert window.mode == "microscope" and window.live.running
    window.live.stop()


def test_image_list_rename_delete_and_reload(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QInputDialog, QMessageBox

    from fuji.ui.gallery import GalleryPanel

    gallery = window.gallery
    for _ in range(3):
        window._on_snap(window.m.snap())
    qtbot.waitUntil(lambda: not gallery._reserved, timeout=10000)
    assert [i.name for i in gallery.entries()] == ["capture_003", "capture_002", "capture_001"]

    gallery.list.setCurrentRow(2)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("lame A — champ 1", True))
    gallery.rename_selected()
    renamed = gallery.entries()[2]
    assert renamed.name == "lame A — champ 1" and renamed.path.exists() and renamed.thumbnail_path.exists()

    gallery.list.setCurrentRow(0)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    gone = gallery.entries()[0].path
    gallery.delete_selected()
    assert not gone.exists() and [i.name for i in gallery.entries()] == ["capture_002", "lame A — champ 1"]

    again = GalleryPanel()  # le lendemain matin… ou après un redémarrage de Fuji
    qtbot.addWidget(again)
    names = [i.name for i in again.entries()]
    assert names == ["capture_002", "lame A — champ 1"]
    assert again.entries()[1].summary == renamed.summary and again.entries()[1].metadata["objective"]


def test_an_image_that_cannot_be_saved_is_kept_and_reported(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))

    def disk_full(*_a, **_k):  # type: ignore[no-untyped-def]
        raise OSError("disque plein")

    monkeypatch.setattr(window.gallery.store, "save", disk_full)
    for _ in range(3):  # assez pour qu'elle ne soit plus parmi les images gardées en mémoire
        window._on_snap(window.m.snap())
    qtbot.waitUntil(lambda: len(warnings) == 3, timeout=10000)
    assert "disque plein" in warnings[0]
    first = window.gallery.list.item(2)
    assert "NON enregistrée" in first.text()
    window.gallery.open_item(first)  # toujours récupérable : ouverte depuis la mémoire
    assert "capture_001" in window.viewer.layers


def test_window_fits_a_small_laptop_screen(qtbot, tmp_path):
    """Le Leitz manuel avec ses 4 objectifs : la fenêtre doit tenir sur un portable."""
    from fuji.core.profile import MicroscopeProfile, builtin_profiles
    from fuji.ui.app import create_window

    profile = MicroscopeProfile.load(builtin_profiles()["leitz-diaplan-touptek"])
    for device in profile.devices:  # la caméra ToupTek n'est pas là : caméra simulée
        if device.driver.startswith("touptek"):
            device.driver = "demo.camera"
            device.options = {"width": 640, "height": 360, "color": True, "simulate_timing": False}
    path = tmp_path / "leitz.yaml"
    profile.save(path)
    _, win = create_window(str(path))
    qtbot.addWidget(win)
    win.show()

    def shrink() -> None:
        for _ in range(3):  # comme quand on rétrécit la fenêtre à la souris : la barre se compacte
            win.resize(600, 500)
            qtbot.wait(50)

    for mode in ("microscope", "analyse"):
        win.set_mode(mode)
        shrink()  # panneau de gauche ouvert (contrôle, ou liste des images)
        # la largeur exacte dépend des polices du système
        assert win.width() <= 1100 and win.height() <= 700, (mode, win.size())
        win.toggle_left_panel()  # replié : de la place pour un très petit écran
        shrink()
        assert win.width() <= 840, (mode, win.size())
        win.toggle_left_panel()
    win.set_mode("microscope")
    win.resize(1280, 800)
    qtbot.waitUntil(lambda: win.width() <= 1280, timeout=3000)
    qtbot.wait(50)
    # contrôle ouvert à gauche : sous l'image, seulement l'histogramme (pas de réglages en double)
    assert win.quickbar.height() < 110 and win.canvas_stack.width() > 450
    win.toggle_left_panel()  # contrôle replié : la barre rapide reprend tous ses réglages
    qtbot.wait(50)
    assert win.quickbar.exposure.isVisibleTo(win.quickbar) and win.quickbar.height() < 200


def test_live_compositions_by_hand_from_the_acquisition_section(window, qtbot):
    """Profondeur de champ étendue en tournant la mise au point, panorama en poussant la
    lame sans s'arrêter : comme sur le Leitz manuel, mais avec le simulateur."""
    from fuji.ui.compose_panel import INSET

    m, compose, viewer = window.m, window.compose, window.scope_viewer
    window.sidebar.select("acquisition")

    m.camera.set_property("Épaisseur de l'échantillon", 40.0)
    compose.start("edf")
    assert compose.running and compose.kind == "edf" and window.live.running
    assert viewer.text_overlay.visible and "mise au point" in viewer.text_overlay.text
    for z in (-20, -10, 0, 10, 20):
        m.focus.move_to(z)
        qtbot.wait(500)
    qtbot.waitUntil(lambda: "Composition" in viewer.layers and compose.finish_btn.isEnabled(), timeout=10000)
    # l'image composée prend la vue ; le live reste visible, en vignette dans un coin
    live = viewer.layers["Live"]
    assert live.visible and float(live.scale[-1]) == pytest.approx(INSET * m.pixel_size_um())
    assert viewer.layers.index("Composition") < viewer.layers.index("Live")
    compose.restart()  # « Recommencer » : on repart de zéro, sans quitter
    assert compose.running and compose._engine.status.frames <= 1
    for z in (-20, 0, 20):
        m.focus.move_to(z)
        qtbot.wait(400)
    qtbot.waitUntil(lambda: compose.finish_btn.isEnabled(), timeout=10000)
    compose.finish()
    qtbot.waitUntil(lambda: bool(window.gallery.entries()) and not window.gallery._reserved, timeout=10000)
    edf = window.gallery.entries()[0]
    assert edf.kind == "edf" and edf.name == "profondeur_001" and "images fusionnées" in edf.summary
    assert "Composition" not in viewer.layers and viewer.layers["Live"].visible
    assert float(viewer.layers["Live"].scale[-1]) == pytest.approx(m.pixel_size_um())  # live en grand
    assert not viewer.text_overlay.visible

    m.camera.set_property("Épaisseur de l'échantillon", 0.0)
    m.focus.move_to(0)
    compose.start("panorama")
    field_h, field_w = m.camera.sensor_shape()
    px = m.pixel_size_um()
    # la main pousse la lame de 1,2 champ vers la droite, sans jamais s'arrêter
    steps = 40
    for i in range(1, steps + 1):
        m.xy_stage.move_to(1.2 * field_w * px * i / steps, 0.0)
        qtbot.wait(120)
    qtbot.waitUntil(lambda: len(compose._engine.tiles) >= 4, timeout=15000)
    assert "Champ actuel" in viewer.layers and "Composition" in viewer.layers
    assert "champs" in viewer.text_overlay.text and compose.undo_btn.isEnabled()
    assert viewer.layers["Live"].translate[-1] > 0.5 * field_w * px  # le live suit la lame
    qtbot.wait(800)  # la lame est arrêtée : plus aucun champ ne s'ajoute
    count = len(compose._engine.tiles)
    compose.undo()
    qtbot.wait(800)  # la lame n'a pas bougé : le champ retiré ne revient pas
    assert len(compose._engine.tiles) == count - 1
    compose.finish()
    qtbot.waitUntil(lambda: not compose.running, timeout=30000)
    qtbot.waitUntil(lambda: not window.gallery._reserved, timeout=10000)
    pano = window.gallery.entries()[0]
    assert pano.kind == "panorama" and f"{count - 1} champs" in pano.summary
    # plus large que haut : au moins 1,7 champ de large (la caméra simulée est tournée de 1,5°)
    assert pano.shape[1] > 1.7 * field_w and field_h <= pano.shape[0] < 1.1 * field_h
    assert tuple(viewer.layers["Live"].translate[-2:]) == (0.0, 0.0)

    compose.start("edf")  # annuler : rien n'est ajouté
    qtbot.wait(300)
    compose.cancel()
    assert not compose.running and len(window.gallery.entries()) == 2
    window.live.stop()
