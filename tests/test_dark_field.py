"""Fond noir : un type de canal, avec ses contraintes (ouverture de l'objectif) et ses
réglages (exposition, lumière parasite retirée, assistants en fond clair)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from fuji.core.profile import ChannelPreset, Objective

DRY = Objective(label="40x/0.65", magnification=40, na=0.65)
OIL = Objective(label="100x/1.25 huile", magnification=100, na=1.25, immersion="huile")


def test_dark_field_objective_limit():
    bright = ChannelPreset(name="Fond clair")
    assert bright.objective_limit() is None and bright.objective_issue(OIL) is None
    dark = ChannelPreset(name="Fond noir", kind="fond noir")
    assert dark.dark_field and dark.kind_name == "Fond noir"
    assert dark.objective_limit() == 0.65  # condenseur à sec
    assert dark.objective_issue(DRY) is None
    issue = dark.objective_issue(OIL)
    assert "100x/1.25 huile" in issue and "ouverture 1,25" in issue and "0,65" in issue
    oil = ChannelPreset(name="Fond noir", kind="fond noir", immersion_condenser=True)
    assert oil.objective_limit() == 1.0 and oil.objective_issue(OIL) is not None
    stopped = ChannelPreset(name="Fond noir", kind="fond noir", max_objective_na=1.3)
    assert stopped.objective_issue(OIL) is None


def test_builtin_profiles_offer_dark_field():
    from fuji.core.profile import MicroscopeProfile, builtin_profiles

    for name in ("demo", "leitz-diaplan-touptek"):
        profile = MicroscopeProfile.load(builtin_profiles()[name])
        dark = profile.channel("Fond noir")
        assert dark is not None and dark.kind == "fond noir", name
        assert "condenseur" in dark.states and profile.check() == []
        lamp = profile.roles.transmitted_light
        assert dark.lights[lamp].on and dark.lights[lamp].intensity == 100
        assert profile.channel("Fond clair").states["condenseur"] == "Fond clair"


def test_simulated_dark_field_and_too_open_objective(demo_scope):
    m = demo_scope
    m.set_objective("10x/0.25")
    m.apply_channel("Fond noir")
    assert m.channel_kind == "fond noir" and m.channel_issue() is None
    frame = m.snap()
    data = frame.data.astype(np.float64)
    assert np.median(data) < 0.05 * 4095 < np.percentile(data, 99.5)  # fond noir, objets clairs
    assert frame.metadata.contrast == "fond noir" and frame.metadata.channel == "Fond noir"
    m.set_objective("100x/1.25 huile")
    assert "ne sera pas noir" in m.channel_issue()
    assert np.median(m.snap().data) > 0.3 * 4095  # la lumière directe entre
    m.set_objective("10x/0.25")
    m.apply_channel("Fond clair")
    assert np.median(m.snap().data) > 0.3 * 4095 and m.current_metadata().contrast == "transmission"
    m.apply_channel("GFP")
    assert m.current_metadata().channel_color == "green"


def test_auto_exposure_on_a_dark_background(demo_scope):
    """Fond noir : les hautes lumières sont les tout derniers pixels, pas le fond."""
    m = demo_scope
    m.set_objective("10x/0.25")
    m.apply_channel("Fond noir")
    cam = m.camera
    cam.set_exposure(1.0)
    cam.auto_expose(target=0.8, sparse=True)
    data = m.snap().data
    assert 0.6 * 4095 < np.percentile(data, 99.95) <= 4095
    assert float(np.mean(data >= 4090)) < 0.001


def test_dark_field_reference_is_subtracted():
    from fuji.core.flatfield import SUBTRACT, FlatField, FlatFieldError, check_background

    rng = np.random.default_rng(0)
    background = np.full((60, 80), 40.0) + rng.normal(0, 1, (60, 80))
    background[10:14, 20:24] += 300  # poussière du condenseur : un point brillant
    ref = FlatField(background, info={"mode": SUBTRACT})
    frame = background.copy()
    frame[30:40, 30:40] += 1000  # un objet
    out = ref.apply(frame.astype(np.uint16))
    assert out.dtype == np.uint16
    assert out[10:14, 20:24].mean() < 60  # le reflet a disparu
    assert out[30:40, 30:40].mean() > 1000  # l'objet reste
    assert 30 < np.median(out) < 50  # le noir de la caméra est gardé
    assert check_background(background.astype(np.uint16), 4095) == []
    with_objects = background.copy()
    with_objects[:, :20] += 2000
    assert "brillent" in check_background(with_objects, 4095)[0]
    with pytest.raises(FlatFieldError, match="saturée"):
        check_background(np.full((40, 40), 4095.0), 4095)


def test_dark_field_reference_from_the_microscope(demo_scope):
    m = demo_scope
    m.set_objective("10x/0.25")
    m.camera.set_property("Échantillon", "vide")
    m.apply_channel("Fond noir")
    candidate = m.flatfield.measure("flat", 2)
    assert candidate.info["mode"] == "soustraction" and candidate.warnings == []
    m.flatfield.save(candidate)
    assert m.flatfield.active and "Lumière parasite retirée" in m.flatfield.status().text
    m.apply_channel("Fond clair")  # une autre référence (ou aucune) pour le fond clair
    assert m.flatfield.status().state == "missing"


def test_acquisition_warns_about_a_too_open_objective(demo_scope):
    import useq

    from fuji.core.acquisition import plan_acquisition

    demo_scope.set_objective("100x/1.25 huile")
    plan = plan_acquisition(demo_scope, useq.MDASequence(channels=["Fond noir"]))
    assert any("ne sera pas noir" in w for w in plan.warnings)


def test_editor_model_adds_a_ready_dark_field_channel():
    from fuji.core.profile import MicroscopeProfile, builtin_profiles
    from fuji.ui.config import model as M

    data = MicroscopeProfile.load(builtin_profiles()["leitz-diaplan-touptek"]).model_dump(mode="json")
    data["devices"] = [d for d in data["devices"] if d["name"] != "condenseur"]
    data["channels"] = [c for c in data["channels"] if c["kind"] != "fond noir"]
    for ch in data["channels"]:
        ch["states"] = {}
    done = M.setup_dark_field(data)
    assert len(done) == 2 and "Condenseur" in done[0]
    channel = next(c for c in data["channels"] if c["kind"] == "fond noir")
    assert channel["name"] == "Fond noir" and channel["states"] == {"condenseur": "Fond noir"}
    assert channel["lights"]["lampe"] == {"on": True, "intensity": 100.0}
    assert data["channels"][0]["states"] == {"condenseur": "Fond clair"}
    assert M.setup_dark_field(data) == []  # déjà là : rien de plus
    MicroscopeProfile.model_validate(data)  # la configuration reste valable


# ------------------------------------------------------------------ interface
gui = pytest.mark.gui


@pytest.fixture
def window(qtbot, tmp_path):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("pas d'affichage (lancer avec xvfb-run)")
    pytest.importorskip("napari")
    from fuji.ui.app import create_window, isolate_settings

    isolate_settings(str(tmp_path))
    _, win = create_window("demo")
    win.m.camera._timing = False
    qtbot.addWidget(win)
    return win


@gui
def test_control_panel_explains_dark_field(window, qtbot, monkeypatch):
    from fuji.ui.panels.microscope_panel import ADD_CHANNEL

    panel = window.control
    window.m.set_objective("10x/0.25")
    panel._apply_channel("Fond noir")
    panel.refresh()
    assert not panel.channel_note.isHidden() and panel.channel_note.objectName() == "hint"
    assert "diaphragme d'ouverture" in panel.channel_note.text() and "0,65" in panel.channel_note.text()
    assert not panel.wb_btn.isEnabled()  # pas de fond blanc en fond noir
    window.m.set_objective("100x/1.25 huile")
    panel.refresh()
    assert panel.channel_note.objectName() == "warning" and "ne sera pas noir" in panel.channel_note.text()
    window.m.set_objective("10x/0.25")
    panel._apply_channel("Fond clair")
    panel.refresh()
    assert panel.channel_note.isHidden() and panel.wb_btn.isEnabled()
    # « Ajouter un canal » au bas de la liste : l'éditeur, sur la page des canaux
    opened = []
    monkeypatch.setattr(window, "edit_profile", lambda path=None, page=None: opened.append(page))
    index = panel.channel.findText(ADD_CHANNEL)
    panel.channel.setCurrentIndex(index)
    panel.channel.activated.emit(index)
    assert opened == ["ChannelsPage"] and panel.channel.currentText() == "Fond clair"


@gui
def test_kohler_waits_for_bright_field(window, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: told.append(a[2]))
    window.control._apply_channel("Fond noir")
    kohler = window.assistants.kohler
    kohler._toggle()
    assert not kohler.running and "fond clair" in told[0]
    window.control._apply_channel("Fond clair")
    kohler._toggle()
    assert kohler.running
    kohler._toggle()


@gui
def test_editor_offers_dark_field(window, qtbot, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    from fuji.core.profile import MicroscopeProfile, builtin_profiles
    from fuji.ui.config.editor import ConfigEditor

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    data = MicroscopeProfile.load(builtin_profiles()["motorise-simulation"]).model_dump(mode="json")
    editor = ConfigEditor(data, None, window)
    qtbot.addWidget(editor)
    editor.show_page("ChannelsPage")
    page = next(p for p in editor.pages if type(p).__name__ == "ChannelsPage")
    page._add_dark_field()
    dark = next(c for c in editor.data["channels"] if c["kind"] == "fond noir")
    assert dark["name"] == "Fond noir"
    assert page.list.currentItem().text().startswith("Fond noir")
    page._commit()
    assert dark["kind"] == "fond noir" and dark["max_objective_na"] is None
    MicroscopeProfile.model_validate(editor.data)
