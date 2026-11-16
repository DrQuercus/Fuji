from __future__ import annotations

import numpy as np
import pytest

from fuji.core import Microscope
from fuji.core.manual import ManualAction, ManualActionDeclined


def test_demo_capabilities_and_snap(demo_scope):
    caps = demo_scope.capabilities()
    assert caps["camera"] and caps["xy_motorized"] and caps["focus_motorized"]
    assert caps["fluorescence"] and caps["color_camera"]
    frame = demo_scope.snap()
    assert frame.data.shape == (768, 1024, 3)
    assert frame.data.dtype == np.uint16
    md = frame.metadata
    assert md.objective == "10x/0.25"
    assert md.pixel_size_um == pytest.approx(0.29)
    assert md.channel == "Fond clair"
    assert md.devices["cubes"] == "Vide"


def test_pixel_size_follows_objective(demo_scope):
    demo_scope.set_objective("40x/0.65")
    assert demo_scope.pixel_size_um() == pytest.approx(2.9 / 40)
    assert demo_scope.pixel_size_source() == "estimé"
    demo_scope.set_pixel_calibration(0.0731)
    assert demo_scope.pixel_size_um() == pytest.approx(0.0731)
    assert demo_scope.pixel_size_source() == "calibré"


def test_channels_switch_light_path(demo_scope):
    demo_scope.apply_channel("GFP")
    d = demo_scope.devices
    assert d["cubes"].get_label() == "GFP"
    assert d["led_470"].is_on() and not d["lampe"].is_on()
    assert demo_scope.camera.get_exposure() == pytest.approx(50)
    img = demo_scope.snap().data.astype(float)
    assert img[..., 1].mean() > img[..., 0].mean()  # émission verte
    demo_scope.apply_channel("Fond clair")
    assert d["lampe"].is_on() and not d["led_470"].is_on()


def test_defocus_reduces_sharpness(demo_scope):
    from fuji.assistants import sharpness

    sharp = sharpness(demo_scope.snap().data)
    demo_scope.focus.move_to(25)
    blurred = sharpness(demo_scope.snap().data)
    assert blurred < 0.7 * sharp


def test_live_produces_frames(mono_demo_profile):
    import time

    with Microscope(mono_demo_profile) as scope:
        cam = scope.camera
        cam.start_live()
        time.sleep(0.3)
        n, frame = cam.latest_frame()
        cam.stop_live()
        assert n >= 2 and frame.shape == (120, 160)


def test_manual_devices_prompt_only_when_needed():
    from fuji.core import MicroscopeProfile

    prof = MicroscopeProfile.from_yaml(
        """
name: Manuel
devices:
  - {name: objectifs, driver: manual.state}
  - {name: lampe, driver: manual.light}
  - {name: cam, driver: demo.camera, options: {width: 64, height: 48, simulate_timing: false}}
roles: {camera: cam, objective_changer: objectifs, transmitted_light: lampe}
objectives:
  - {label: "10x/0.25", magnification: 10, na: 0.25}
  - {label: "40x/0.65", magnification: 40, na: 0.65}
"""
    )
    asked: list[ManualAction] = []
    with Microscope(prof, manual_handler=lambda a: asked.append(a) or True) as scope:
        assert scope.objective_changer.manual
        scope.set_objective("40x/0.65", declare=True)  # déclaré : pas de consigne
        assert not asked
        assert scope.objective.label == "40x/0.65"
        scope.set_objective("10x/0.25")  # demandé par un processus : consigne
        assert len(asked) == 1 and "10x/0.25" in asked[0].message
        scope.set_objective("10x/0.25")  # déjà en place : rien
        assert len(asked) == 1
        scope.manual_handler = lambda a: False
        with pytest.raises(ManualActionDeclined):
            scope.set_objective("40x/0.65")
        assert scope.objective.label == "10x/0.25"
