"""Interface sur un microscope motorisé (simulé) et serveur pour IA intégré à la fenêtre."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import threading

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

    _, win = create_window("motorise-simulation")
    win.m.camera._timing = False
    win.m.device("moteurs").link._io.speed = 3000  # contrôleur simulé accéléré
    qtbot.addWidget(win)
    return win


@pytest.fixture
def warnings_shown(monkeypatch):
    """Les boîtes d'avertissement ne bloquent pas le test : leur texte est noté."""
    from qtpy.QtWidgets import QMessageBox

    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text, *a, **k: shown.append(text))
    return shown


def test_motor_controls_jog_origin_and_emergency_stop(window, qtbot):
    from qtpy.QtWidgets import QPushButton

    m, panel = window.m, window.control
    assert not m.xy_stage.manual and not m.focus.manual
    buttons = {b.text(): b for b in panel.findChildren(QPushButton)}
    panel.xy_step.setValue(250)
    buttons["→"].click()  # le mouvement part dans un thread : l'interface reste libre
    qtbot.waitUntil(lambda: m.xy_stage.get_position() == pytest.approx((250.0, 0.0)), timeout=5000)
    buttons["↑"].click()
    qtbot.waitUntil(lambda: m.xy_stage.get_position() == pytest.approx((250.0, -250.0)), timeout=5000)
    panel.refresh()
    assert "250.0" in panel.xy_label.text()
    buttons["Origine ici"].click()
    assert m.xy_stage.get_position() == pytest.approx((0.0, 0.0))
    panel.z_step.setValue(5)
    buttons["▲"].click()
    qtbot.waitUntil(lambda: m.focus.get_position() == pytest.approx(5.0), timeout=5000)
    buttons["Z = 0 ici"].click()
    assert m.focus.get_position() == pytest.approx(0.0)
    # hors course : refusé avec un message, la platine ne bouge pas
    messages: list[str] = []
    panel.message.connect(messages.append)
    panel.run_motion(m.focus.move_to, 900.0)
    qtbot.waitUntil(lambda: bool(messages), timeout=5000)
    assert "hors de la course" in messages[0] and m.focus.get_position() == pytest.approx(0.0)
    # arrêt d'urgence pendant un long déplacement (contrôleur à vitesse réelle)
    m.device("moteurs").link._io.speed = 1.0
    panel.run_motion(m.xy_stage.move_to, 15000.0, 0.0)
    qtbot.wait(300)
    window.quickbar.stop_motion()
    qtbot.waitUntil(
        lambda: any("arrêt" in t.lower() or "stop" in t.lower() for t in messages[1:]), timeout=8000
    )
    qtbot.wait(300)
    x, _ = m.xy_stage.get_position()
    assert 0 < x < 15000  # arrêtée en route


def test_stage_assistant_refuses_a_dark_field_then_measures_and_centres(window, qtbot, warnings_shown):
    m, stage = window.m, window.assistants.stage
    notes: list[str] = []
    window._message.connect(notes.append)
    stage._calibrate()  # lampe éteinte : image noire
    qtbot.waitUntil(lambda: bool(warnings_shown), timeout=30000)
    assert "pas assez de détails" in warnings_shown[0]
    assert "refusée" in stage.orientation_text.text() and not stage.save.isEnabled()
    qtbot.waitUntil(stage.calibrate.isEnabled, timeout=5000)

    m.apply_channel("Fond clair")
    stage._calibrate()
    qtbot.waitUntil(lambda: stage._calibration is not None, timeout=30000)
    qtbot.waitUntil(stage.calibrate.isEnabled, timeout=5000)
    assert "tournée" in stage.orientation_text.text() and "⚠" not in stage.orientation_text.text()
    assert stage.save.isEnabled()
    stage._save()
    assert m.profile.stage_orientation is not None
    stage._backlash("x")
    qtbot.waitUntil(lambda: "Jeu en X" in stage.backlash_text.text(), timeout=30000)
    assert len(warnings_shown) == 1

    # double-clic sur l'image : ce point vient au centre du champ
    from fuji.analysis.registration import image_shift

    before = m.snap().data
    h, w = before.shape[:2]
    window.center_on(h / 2 - 90, w / 2 + 140, before.shape)
    qtbot.waitUntil(lambda: any(t.startswith("Platine") for t in notes), timeout=10000)
    dx, dy = image_shift(before, m.snap().data)
    assert (dx, dy) == pytest.approx((-140, 90), abs=3)


def test_mosaic_with_autofocus_from_the_acquisition_panel(window, qtbot, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    m, panel = window.m, window.acquisition
    m.apply_channel("Fond clair")
    m.focus.move_to(6.0)  # légèrement défocalisé : l'autofocus doit rattraper
    panel.folder.setText(str(tmp_path))
    panel.name.setText("mosaique")
    panel.use_mosaic.setChecked(True)
    panel.mosaic_cols.setValue(2)
    panel.mosaic_rows.setValue(2)
    panel.use_autofocus.setChecked(True)
    panel.af_range.setValue(24)
    panel.af_step.setValue(3)
    done, focus_used = [], []
    panel.acquisition_done.disconnect(window._on_acq_done)  # pas de question « ouvrir ? »
    panel.acquisition_done.connect(done.append)
    panel.frame_ready.connect(lambda _event, frame: focus_used.append(frame.metadata.z_um))
    panel.start()
    assert window._acquiring
    qtbot.waitUntil(lambda: bool(done), timeout=120000)
    window.topbar.set_acquisition(0, 0, False)
    window._acquiring = False
    window._set_hardware_controls_enabled(True)
    assert done[0].acquired == 4 and done[0].error is None
    mosaics = [layer for layer in window.viewer.layers if layer.name.startswith("Mosaïque 2×2")]
    assert len(mosaics) == 1
    tile_h, tile_w = m.camera.sensor_shape()
    height, width = mosaics[0].data.shape[:2]
    assert 1.7 * tile_h < height < 2 * tile_h and 1.7 * tile_w < width < 2 * tile_w
    assert len(focus_used) == 4 and max(abs(z) for z in focus_used) < 2.0  # mise au point retrouvée
    assert m.focus.get_position() == pytest.approx(6.0)  # réglages de départ rétablis à la fin
    assert list(tmp_path.glob("mosaique*"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ai_session(port: int, calls: list[tuple[str, dict]]) -> dict:
    """Joue le rôle de l'IA : se connecte au serveur de la fenêtre et appelle des outils."""
    from mcp import Client

    out: dict = {"results": []}

    async def session() -> None:
        async with Client(f"http://127.0.0.1:{port}/mcp") as client:
            out["tools"] = [t.name for t in (await client.list_tools()).tools]
            for name, args in calls:
                result = await client.call_tool(name, args)
                out["results"].append(
                    (bool(result.is_error), [c.type for c in result.content], result.content[0].text)
                )

    def run() -> None:
        try:
            asyncio.run(session())
        except Exception as exc:
            out["error"] = repr(exc)
        out["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return out


def test_ai_server_toggle_shows_in_fuji_what_the_ai_does(window, qtbot, warnings_shown, monkeypatch):
    pytest.importorskip("mcp", reason='pip install "fuji-microscopy[ia]"')
    from qtpy.QtWidgets import QMessageBox

    # consigne manuelle (« Placez l'objectif… ») : affichée dans la fenêtre, refusée puis acceptée
    prompts: list[str] = []
    answer = {"index": 1}  # 0 = « Continuer », 1 = « Annuler »

    def fake_exec(box):  # type: ignore[no-untyped-def]
        prompts.append(box.text())
        box._clicked = box.buttons()[answer["index"]]
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._clicked)
    settings = window.settings_panel
    port = _free_port()
    settings.ai_port.setValue(port)
    settings.ai_enabled.setChecked(True)
    assert window._ai_server is not None and window._ai_server.running
    assert f"127.0.0.1:{port}" in settings.ai_status.text() and not settings.ai_port.isEnabled()
    assert f'"url": "http://127.0.0.1:{port}/mcp"' in settings.ai_config()

    out = _ai_session(
        port,
        [
            ("set_channel", {"name": "Fond clair"}),
            ("snap", {"name": "vue", "max_side": 128}),
            ("run_operation", {"name": "Seuillage automatique", "inputs": {"image": "vue"}, "params": {}}),
            ("move_stage", {"x_um": 40, "y_um": 0}),
            ("set_objective", {"label": "40x/0.65"}),
        ],
    )
    qtbot.waitUntil(lambda: "done" in out, timeout=60000)
    assert "error" not in out, out.get("error")
    assert {"snap", "move_stage", "run_operation", "stop_motion"} <= set(out["tools"])
    results = out["results"]
    assert [r[0] for r in results] == [False, False, False, False, True]
    assert results[1][1] == ["text", "image"]
    # la manipulation manuelle est demandée à l'écran de Fuji ; refusée, l'IA en est informée
    assert len(prompts) == 1 and "40x/0.65" in prompts[0]
    assert "annulé" in results[4][2] and window.m.objective.label == "10x/0.25"
    assert window.m.xy_stage.get_position() == pytest.approx((40.0, 0.0))  # même microscope que la fenêtre
    qtbot.waitUntil(lambda: "IA — vue" in window.viewer.layers, timeout=5000)
    assert any(layer.name.startswith("IA — ") and layer.name != "IA — vue" for layer in window.viewer.layers)

    answer["index"] = 0  # cette fois l'utilisateur tourne la tourelle et confirme
    out = _ai_session(port, [("set_objective", {"label": "40x/0.65"})])
    qtbot.waitUntil(lambda: "done" in out, timeout=60000)
    assert "error" not in out, out.get("error")
    assert out["results"][0][0] is False and window.m.objective.label == "40x/0.65"

    # « observation seulement » : l'IA regarde et analyse, mais ne règle plus rien
    settings.ai_hardware.setChecked(False)
    assert "observation" in settings.ai_status.text()
    out = _ai_session(port, [("snap", {"show": False}), ("set_exposure", {"exposure_ms": 20})])
    qtbot.waitUntil(lambda: "done" in out, timeout=60000)
    assert "error" not in out, out.get("error")
    assert [r[0] for r in out["results"]] == [False, True] and "lecture seule" in out["results"][1][2]

    settings.ai_enabled.setChecked(False)
    assert (
        window._ai_server is None and settings.ai_status.text() == "Arrêté" and settings.ai_port.isEnabled()
    )
    assert not warnings_shown
