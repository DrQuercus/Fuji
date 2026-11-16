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


# --------------------------------------------------------------------- régressions
ONE_FRAME = useq.MDASequence(time_plan={"interval": 0, "loops": 1})


def _motorized(tmp_path=None):
    from fuji.core import MicroscopeProfile

    return MicroscopeProfile.from_yaml(
        """
name: Motorisé
devices:
  - {name: xy, driver: demo.xy_stage}
  - {name: z, driver: demo.focus, options: {initial_um: 1000}}
  - {name: cam, driver: demo.camera,
     options: {width: 32, height: 24, color: false, simulate_timing: false}}
roles: {camera: cam, focus: z, xy_stage: xy}
"""
    )


def test_engine_always_finishes_even_if_setup_fails(mono_demo_profile, tmp_path):
    """Un dossier inutilisable ne doit pas laisser le moteur « en cours » pour toujours."""
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    with Microscope(mono_demo_profile) as scope:
        engine = AcquisitionEngine(scope)
        done = []
        engine.finished.connect(done.append)
        engine.start(ONE_FRAME, writer=OMETiffWriter(blocker / "sous-dossier", "x"))
        engine.wait(10)
        assert not engine.is_running
        assert len(done) == 1 and done[0].error and done[0].acquired == 0
        # et le moteur reste utilisable
        assert engine.run(ONE_FRAME).acquired == 1


def test_relative_z_is_applied_per_position():
    """Une position sans Z garde la mise au point de départ ; une position avec Z l'impose."""
    with Microscope(_motorized()) as scope:
        seq = useq.MDASequence(
            stage_positions=[{"x": 0, "y": 0}, {"x": 10, "y": 0, "z": 50.0}],
            z_plan={"range": 2, "step": 1},
        )
        zs = {}
        engine = AcquisitionEngine(scope)
        engine.frame_acquired.connect(
            lambda ev, fr: zs.setdefault(ev.index["p"], []).append(fr.metadata.z_um)
        )
        summary = engine.run(seq)
        assert not summary.error
        assert zs[0] == [999.0, 1000.0, 1001.0]  # autour de la position de départ, jamais vers 0
        assert zs[1] == [49.0, 50.0, 51.0]
        assert scope.focus.get_position() == 1000.0
        assert scope.xy_stage.get_position() == (0.0, 0.0)


def test_relative_grid_is_centred_on_the_starting_position():
    with Microscope(_motorized()) as scope:
        scope.xy_stage.move_to(500, 300)
        seq = useq.MDASequence(
            grid_plan={"rows": 1, "columns": 2, "fov_width": 100, "fov_height": 100, "relative_to": "center"}
        )
        xs = []
        engine = AcquisitionEngine(scope)
        engine.frame_acquired.connect(lambda ev, fr: xs.append((fr.metadata.x_um, fr.metadata.y_um)))
        engine.run(seq)
        assert xs == [(450.0, 300.0), (550.0, 300.0)]
        assert scope.xy_stage.get_position() == (500.0, 300.0)


def test_time_lapse_per_position_waits_at_every_position():
    """Axe « position » à l'extérieur : la minuterie repart à chaque position."""
    import time as _time

    with Microscope(_motorized()) as scope:
        seq = useq.MDASequence(
            time_plan={"interval": 0.25, "loops": 2},
            stage_positions=[(0, 0), (10, 0)],
            axis_order="ptcz",
        )
        t0 = _time.monotonic()
        summary = AcquisitionEngine(scope).run(seq)
        assert summary.acquired == 4
        assert _time.monotonic() - t0 >= 0.5  # 0,25 s d'attente à chacune des deux positions
        assert plan_acquisition(scope, seq).duration_s >= 0.5


def test_runs_never_overwrite_each_other(mono_demo_profile, tmp_path):
    with Microscope(mono_demo_profile) as scope:
        engine = AcquisitionEngine(scope)
        writers = [OMETiffWriter(tmp_path, "essai") for _ in range(2)]
        first = engine.run(ONE_FRAME, writer=writers[0])
        with writers[1] as w:  # « with » + fermeture par le moteur : deux fermetures
            second = engine.run(ONE_FRAME, writer=w)
    assert not first.error and not second.error
    assert first.files[0].name == "essai.ome.tif" and second.files[0].name == "essai_1.ome.tif"
    assert writers[0].metadata_path.name == "essai_metadata.json"
    assert writers[1].metadata_path.name == "essai_1_metadata.json"
    assert writers[0].metadata_path.exists() and writers[1].metadata_path.exists()


def test_plan_matches_what_the_engine_will_do():
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
  - {name: BF, exposure_ms: 10, states: {cubes: Vide}, lights: {lampe: {on: true}}}
  - {name: GFP, exposure_ms: 400, kind: fluorescence, states: {cubes: GFP}, lights: {hbo: {on: true}}}
default_channel: BF
"""
    )
    asked = []
    with Microscope(prof, manual_handler=lambda a: asked.append(a.message) or True) as scope:
        scope.camera.set_exposure(30)  # réglage en cours, conservé tant que le canal ne change pas
        seq = useq.MDASequence(channels=["BF", "GFP"], z_plan={"range": 2, "step": 2}, axis_order="tpzc")
        plan = plan_acquisition(scope, seq, overhead_s=0.0)
        assert plan.duration_s == pytest.approx(0.03 + 0.4 + 0.01 + 0.4)
        AcquisitionEngine(scope).run(seq, restore=False)
        assert plan.manual_actions == len(asked)


def test_plan_stays_fast_for_huge_sequences(mono_demo_profile):
    import time as _time

    with Microscope(mono_demo_profile) as scope:
        seq = useq.MDASequence(time_plan={"interval": 5, "loops": 100_000}, z_plan={"range": 20, "step": 1})
        t0 = _time.monotonic()
        plan = plan_acquisition(scope, seq)
        assert _time.monotonic() - t0 < 5
        assert plan.n_frames == 2_100_000
        assert plan.duration_s == pytest.approx(500_000, rel=0.02)
