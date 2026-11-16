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
    panel.acquisition_done.disconnect(window._on_acq_done)  # pas de question « ouvrir ? »
    panel.acquisition_done.connect(done.append)
    panel.start()
    assert window._acquiring and not window.live.running
    assert not window.topbar.capture.isEnabled() and not window.quickbar.isEnabled()
    window.snap()  # refusée pendant l'acquisition
    assert "Capture 1" not in window.viewer.layers
    qtbot.waitUntil(lambda: bool(done), timeout=10000)
    window.topbar.set_acquisition(0, 0, False)
    window._acquiring = False
    window._set_hardware_controls_enabled(True)
    assert done[0].acquired == 4 and done[0].error is None
    assert window.topbar.capture.isEnabled() and window.quickbar.isEnabled()


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


def test_flat_field_requested_but_missing_is_reported(window):
    illum = window.assistants.illumination
    frame = window.m.snap()
    data, applied = illum.correct(frame.data)
    assert applied is None and data is frame.data
    illum.apply.setChecked(True)
    assert illum.correct(frame.data)[1] is False  # demandée, mais aucune référence
    window._on_snap(frame)
    assert "NON appliqué" in window.statusBar().currentMessage()
    illum._pending = ("flat", illum._keys())
    illum._acquired(np_full_like(frame.data))
    data, applied = illum.correct(frame.data)
    assert applied is True and data.shape == frame.data.shape


def np_full_like(reference):  # type: ignore[no-untyped-def]
    import numpy as np

    return np.full(reference.shape, 1000.0, np.float32)


def test_saturated_flat_field_is_rejected(window, monkeypatch):
    import numpy as np
    from qtpy.QtWidgets import QMessageBox

    illum = window.assistants.illumination
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    frame = window.m.snap().data
    illum._pending = ("flat", illum._keys())
    illum._acquired(np.full(frame.shape, 2**window.m.camera.bit_depth - 1, np.float32))
    assert shown and "saturée" in shown[0]
    assert illum.store.load(*illum._keys()) is None


def test_focus_peaking_layer_is_scaled_to_the_image(window):
    import numpy as np

    widget = window.assistants.focus
    frame = np.random.default_rng(0).integers(0, 4000, (2160, 3840), dtype=np.uint16)
    widget.peaking.setChecked(True)
    widget.process(frame)
    layer = window.viewer.layers["Zones nettes"]
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
