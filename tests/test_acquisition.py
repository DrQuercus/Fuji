from __future__ import annotations

import re
import threading

import numpy as np
import pytest
import tifffile
import useq

from fuji.core import Microscope
from fuji.core.acquisition import AcquisitionEngine, plan_acquisition
from fuji.io import OMETiffWriter, save_image


def test_multidimensional_acquisition_writes_ome_tiff(mono_demo_profile, tmp_path):
    with Microscope(mono_demo_profile) as scope:
        seq = useq.MDASequence(
            time_plan={"interval": 0, "loops": 2}, z_plan={"range": 4, "step": 2}, channels=["BF"]
        )
        frames = []
        engine = AcquisitionEngine(scope)
        engine.frame_acquired.connect(lambda ev, fr: frames.append(fr))
        summary = engine.run(seq, writer=OMETiffWriter(tmp_path, "essai"))
        assert summary.acquired == summary.total == 6 and not summary.error
        # Z relatif autour de la position de départ, puis retour
        assert [f.metadata.z_um for f in frames[:3]] == [-2.0, 0.0, 2.0]
        assert scope.focus.get_position() == 0.0
    path = summary.files[0]
    data = tifffile.imread(path, squeeze=False)  # canal unique conservé
    assert data.shape[:5] == (2, 1, 3, 120, 160)
    data = data.reshape(2, 1, 3, 120, 160)
    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata
    assert 'DimensionOrder="XYZCT"' in xml
    assert data[1, 0, 2].any()  # dernier plan bien écrit
    assert 'PhysicalSizeX="0.29"' in xml
    assert 'PhysicalSizeZ="2.0"' in xml
    assert len(re.findall(r"<Plane ", xml)) == 6
    assert (tmp_path / "essai_metadata.json").exists()


def test_cancel_stops_acquisition(mono_demo_profile, tmp_path):
    with Microscope(mono_demo_profile) as scope:
        engine = AcquisitionEngine(scope)
        seq = useq.MDASequence(time_plan={"interval": 0.2, "loops": 50})
        engine.start(seq, writer=OMETiffWriter(tmp_path, "stop"))
        threading.Timer(0.3, engine.cancel).start()
        engine.wait(10)
        assert not engine.is_running


def test_plan_warnings(mono_demo_profile, tmp_path):
    with Microscope(mono_demo_profile) as scope:
        scope.camera.set_exposure(500)
        seq = useq.MDASequence(time_plan={"interval": 0.1, "loops": 3}, channels=["Inconnu"])
        plan = plan_acquisition(scope, seq, tmp_path)
        assert plan.n_frames == 3
        assert any("intervalle" in w.lower() for w in plan.warnings)
        assert any("inconnu" in w.lower() for w in plan.warnings)


def test_manual_actions_are_counted_and_prompted(tmp_path):
    from fuji.core import MicroscopeProfile

    prof = MicroscopeProfile.from_yaml(
        """
name: Manuel fluo
devices:
  - {name: cubes, driver: manual.state, options: {labels: [Vide, GFP]}}
  - {name: lampe, driver: manual.light}
  - {name: hbo, driver: manual.light, options: {initial_on: false}}
  - {name: z, driver: manual.focus}
  - {name: cam, driver: demo.camera, options: {width: 32, height: 24, simulate_timing: false}}
roles: {camera: cam, focus: z}
channels:
  - {name: BF, states: {cubes: Vide}, lights: {lampe: {on: true}}}
  - {name: GFP, kind: fluorescence, states: {cubes: GFP}, lights: {hbo: {on: true}}}
default_channel: BF
"""
    )
    asked = []
    with Microscope(prof, manual_handler=lambda a: asked.append(a.message) or True) as scope:
        seq = useq.MDASequence(channels=["BF", "GFP"], z_plan={"range": 4, "step": 2})
        plan = plan_acquisition(scope, seq)
        assert plan.manual_actions >= 2
        summary = AcquisitionEngine(scope).run(seq)
        assert summary.acquired == 6
    assert any("GFP" in m for m in asked)
    assert any("mise au point" in m for m in asked)


def test_save_image_keeps_calibration(tmp_path):
    from fuji.analysis import Image

    data = (np.arange(30 * 40 * 3) % 255).astype(np.uint8).reshape(30, 40, 3)
    path = save_image(tmp_path / "rgb.ome.tif", data, pixel_size_um=0.5)
    img = Image.from_file(path)
    assert img.is_rgb and img.pixel_size_um == pytest.approx(0.5)
    np.testing.assert_array_equal(img.data, data)
