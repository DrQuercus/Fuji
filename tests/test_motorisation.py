"""Motorisation : pilote G-code (testé contre un contrôleur simulé), outils de platine,
mosaïques, autofocus pendant une acquisition."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import useq

from fuji.core import Microscope, MicroscopeProfile
from fuji.core import stage as S
from fuji.core.acquisition import AcquisitionEngine, AutofocusPlan
from fuji.core.devices import XYStage
from fuji.drivers.gcode import GCodeError, GCodeLink, MotionStopped, list_serial_ports
from fuji.drivers.gcode_sim import SimulatedController


def _scope(sim, *, camera=True, shape=(384, 512), camera_stage=None, **options):
    devices = [
        {
            "name": "moteurs",
            "driver": "gcode",
            "options": {"link": GCodeLink("simulation", io=sim), **options},
        },
        {"name": "obj", "driver": "demo.state"},
    ]
    if camera:
        cam = {"width": shape[1], "height": shape[0], "simulate_timing": False, "color": False}
        if camera_stage:
            cam["xy_stage"] = camera_stage
        devices.append({"name": "cam", "driver": "demo.camera", "options": cam})
    profile = MicroscopeProfile.model_validate(
        {
            "name": "Motorisé",
            "devices": devices,
            "roles": {"camera": "cam" if camera else None, "objective_changer": "obj"},
            "objectives": [
                {"label": "10x", "magnification": 10, "na": 0.25},
                {"label": "40x", "magnification": 40, "na": 0.65},
            ],
        }
    )
    scope = Microscope(profile)
    scope.connect()
    return scope


@pytest.fixture
def sim():
    return SimulatedController("grbl", speed=3000)


# ------------------------------------------------------------------------- pilote
def test_gcode_driver_creates_stage_focus_wheel_and_light(sim):
    scope = _scope(
        sim,
        camera=False,
        um_per_unit_z=100,
        light="broche",
        wheel_axis="A",
        wheel_labels=["DAPI", "GFP"],
        wheel_positions=[0, 90],
    )
    try:
        assert list(scope.devices)[:5] == [
            "moteurs",
            "moteurs_xy",
            "moteurs_z",
            "moteurs_roue",
            "moteurs_lampe",
        ]
        caps = scope.capabilities()
        assert caps["xy_motorized"] and caps["focus_motorized"]
        assert scope.xy_stage.name == "moteurs_xy" and scope.focus.name == "moteurs_z"
        scope.xy_stage.move_to(2000, -1000)  # µm → 2 mm, −1 mm pour le contrôleur
        assert scope.xy_stage.get_position() == pytest.approx((2000, -1000))
        assert (sim.position["X"], sim.position["Y"]) == pytest.approx((2.0, -1.0))
        scope.xy_stage.move_by(-500, 250)
        assert scope.xy_stage.get_position() == pytest.approx((1500, -750))
        scope.focus.move_to(50)  # 100 µm par unité du contrôleur
        assert scope.focus.get_position() == pytest.approx(50) and sim.position["Z"] == pytest.approx(0.5)
        scope.transmitted_light.set_intensity(40)
        scope.transmitted_light.set_on(True)
        assert sim.spindle == pytest.approx(400)
        scope.transmitted_light.set_on(False)
        assert sim.spindle == 0
        wheel = scope.device("moteurs_roue")
        wheel.set_label("GFP")
        assert wheel.get_label() == "GFP" and sim.position["A"] == pytest.approx(90)
        assert scope.device("moteurs").get_property("État") == "Idle"
    finally:
        scope.disconnect()
    assert not sim.is_open  # port fermé, lampe éteinte


def test_moves_outside_the_allowed_travel_are_refused_not_clipped(sim):
    scope = _scope(
        sim, camera=False, z_min_um=-200, z_max_um=300, x_min_um=0, x_max_um=5000, y_min_um=0, y_max_um=5000
    )
    try:
        assert scope.focus.limits() == (-200, 300)
        assert scope.xy_stage.limits() == ((0, 5000), (0, 5000))
        scope.focus.move_to(250)
        with pytest.raises(ValueError, match="hors de la course"):
            scope.focus.move_to(301)
        with pytest.raises(ValueError, match="hors de la course"):
            scope.xy_stage.move_to(100, -1)
        assert scope.focus.get_position() == pytest.approx(250)  # rien n'a bougé
        assert scope.xy_stage.get_position() == pytest.approx((0, 0))
    finally:
        scope.disconnect()


def test_backlash_compensation_always_approaches_from_the_same_side(sim):
    scope = _scope(sim, camera=False, backlash_xy_um=20)
    try:
        scope.xy_stage.move_to(2000, 0)
        sim.lines_received.clear()
        scope.xy_stage.move_to(1500, 0)  # recul : dépasser de 20 µm puis revenir
        moves = [line for line in sim.lines_received if line.startswith("G53")]
        assert [m.split()[2] for m in moves] == ["X1.48000", "X1.50000"]
        sim.lines_received.clear()
        scope.xy_stage.move_to(1800, 0)  # avance : un seul mouvement
        assert len([line for line in sim.lines_received if line.startswith("G53")]) == 1
    finally:
        scope.disconnect()


def test_origin_can_be_redefined_and_inverted_axes(sim):
    scope = _scope(sim, camera=False, invert_x=True)
    try:
        scope.xy_stage.move_to(300, 100)
        assert sim.position["X"] == pytest.approx(-0.3)  # axe inversé
        scope.xy_stage.set_origin()
        assert scope.xy_stage.get_position() == pytest.approx((0, 0))
        scope.xy_stage.move_to(100, 0)
        assert sim.position["X"] == pytest.approx(-0.4) and scope.xy_stage.get_position() == pytest.approx(
            (100, 0)
        )
        scope.focus.move_to(40)
        scope.focus.set_origin()
        assert scope.focus.get_position() == pytest.approx(0)
        with pytest.raises(NotImplementedError, match="prise d'origine"):
            scope.xy_stage.home()  # pas de fins de course sur ce contrôleur
    finally:
        scope.disconnect()


def test_homing_and_alarm_unlock():
    sim = SimulatedController("grbl", speed=3000, homing=True)
    sim.alarm = 9  # un contrôleur avec fins de course démarre verrouillé
    scope = _scope(sim, camera=False)
    try:
        assert "$X" in sim.lines_received and sim.alarm is None  # déverrouillé à la connexion
        scope.xy_stage.move_to(500, 500)
        scope.xy_stage.home()
        assert scope.xy_stage.get_position() == pytest.approx((0, 0))
    finally:
        scope.disconnect()
    locked = SimulatedController("grbl", speed=3000)
    locked.alarm = 9
    scope = _scope(locked, camera=False, unlock_on_connect=False)
    try:
        with pytest.raises(GCodeError, match="alarme"):
            scope.xy_stage.move_to(10, 10)
        scope.device("moteurs").unlock()
        scope.xy_stage.move_to(10, 10)
    finally:
        scope.disconnect()


def test_controller_errors_are_explained(sim):
    link = GCodeLink("simulation", io=sim)
    link.open()
    try:
        with pytest.raises(GCodeError, match="non reconnue"):
            link.command("G99")
        limited = SimulatedController("grbl", speed=3000, travel={"X": (0, 10)})
        link2 = GCodeLink("simulation", io=limited)
        link2.open()
        with pytest.raises(GCodeError, match="hors de la course"):
            link2.move({"X": 50.0}, 600)
        link2.close()
    finally:
        link.close()


def test_emergency_stop_interrupts_a_long_move():
    sim = SimulatedController("grbl", speed=1.0)  # temps réel : 100 mm à 600 mm/min = 10 s
    scope = _scope(sim, camera=False)
    try:
        errors: list[Exception] = []

        def long_move():
            try:
                scope.xy_stage.move_to(100_000, 0)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=long_move)
        t0 = time.monotonic()
        worker.start()
        time.sleep(0.3)
        assert scope.xy_stage.is_busy()
        x_during, _ = scope.xy_stage.get_position()  # la position se lit pendant le mouvement
        assert 0 < x_during < 100_000
        scope.stop_motion()
        worker.join(timeout=5)
        assert not worker.is_alive() and time.monotonic() - t0 < 5
        assert errors and isinstance(errors[0], MotionStopped)
        x_stop, _ = scope.xy_stage.get_position()
        assert x_during <= x_stop < 20_000 and not scope.xy_stage.is_busy()
        scope.xy_stage.move_by(100, 0)  # le contrôleur reste utilisable, la position est connue
        assert scope.xy_stage.get_position()[0] == pytest.approx(x_stop + 100, abs=0.5)
    finally:
        scope.disconnect()


def test_marlin_controllers_are_supported():
    sim = SimulatedController("marlin", speed=3000)
    profile = MicroscopeProfile.model_validate(
        {
            "name": "Marlin",
            "devices": [
                {
                    "name": "m",
                    "driver": "gcode",
                    "options": {
                        "link": GCodeLink("simulation", firmware="marlin", io=sim),
                        "light": "ventilateur",
                        "light_max": 255,
                    },
                }
            ],
        }
    )
    with Microscope(profile) as scope:
        scope.xy_stage.move_to(3000, 1000)
        assert scope.xy_stage.get_position() == pytest.approx((3000, 1000))
        scope.focus.move_by(-200)
        assert scope.focus.get_position() == pytest.approx(-200)
        scope.transmitted_light.set_on(True)
        assert sim.fan == pytest.approx(255)
        assert "M400" in sim.lines_received


def test_missing_controller_gives_a_clear_error():
    class Silent:
        timeout = 0.05
        is_open = True

        def write(self, data):
            return len(data)

        def readline(self):
            time.sleep(0.01)
            return b""

        def close(self):
            self.is_open = False

    link = GCodeLink("COM99", io=Silent())
    link._wake_up = lambda: GCodeLink._wake_up(link)  # type: ignore[method-assign]
    with pytest.raises(GCodeError, match="ne répond"):
        link.open()
    link.close()
    assert isinstance(list_serial_ports(), list)


# ---------------------------------------------------------------- outils de platine
@pytest.mark.parametrize("rotation", [0.0, 12.0, -15.0])
def test_stage_calibration_finds_the_camera_orientation(sim, rotation):
    scope = _scope(sim)
    try:
        scope.camera.set_property("Rotation caméra", rotation)
        start = scope.xy_stage.get_position()
        cal = S.calibrate_stage(scope)
        assert cal.rotation_deg == pytest.approx(rotation, abs=0.5) and not cal.mirrored
        assert cal.pixel_size_um == pytest.approx(scope.pixel_size_um(), rel=0.03)
        assert abs(cal.scale_error) < 0.03 and cal.skew < 0.02
        assert scope.xy_stage.get_position() == pytest.approx(start)  # revenu au départ
        S.save_stage_calibration(scope, cal)
        assert np.allclose(scope.profile.stage_orientation, cal.orientation)
        # centrer un point de l'image : il doit arriver au centre
        before = scope.snap().data
        S.center_on_pixel(scope, 100.0, 380.0, before.shape)
        dx, dy = S.image_shift(before, scope.snap().data)
        assert (dx, dy) == pytest.approx((511 / 2 - 380, 383 / 2 - 100), abs=2.5)
    finally:
        scope.disconnect()


def test_stage_measurements_are_refused_rather_than_wrong(sim):
    """Champ vide, image noire : la platine bouge mais rien ne la suit dans l'image. La mesure
    doit être refusée, jamais donner une orientation ou un jeu inventés."""
    scope = _scope(sim, light="broche")
    lamp = scope.device("moteurs_lampe")
    try:
        scope.xy_stage.move_to(300.0, -200.0)
        lamp.set_on(True)
        scope.camera.set_property("Échantillon", "vide")  # éclairage seul : fixe dans l'image
        with pytest.raises(RuntimeError, match=r"Mesure (impossible|incohérente)"):
            S.calibrate_stage(scope)
        with pytest.raises(RuntimeError, match=r"Mesure (impossible|incohérente)"):
            S.measure_backlash(scope, "x")
        scope.camera.set_property("Échantillon", "cellules")
        lamp.set_on(False)  # image noire : il ne reste que le bruit de lecture
        with pytest.raises(RuntimeError, match="pas assez de détails"):
            S.calibrate_stage(scope)
        with pytest.raises(RuntimeError, match="pas assez de détails"):
            S.measure_backlash(scope, "y")
        assert scope.xy_stage.get_position() == pytest.approx((300.0, -200.0))  # revenue au départ
        assert scope.profile.stage_orientation is None
        lamp.set_on(True)  # la même lame, éclairée : la mesure passe
        assert S.calibrate_stage(scope).skew < 0.02
    finally:
        scope.disconnect()


def test_stage_calibration_works_against_a_travel_limit(sim):
    """En butée d'un côté, le pas de mesure ne se fait que de l'autre : même résultat."""
    scope = _scope(sim, x_min_um=-1000, x_max_um=1000, y_min_um=-1000, y_max_um=1000)
    try:
        scope.xy_stage.move_to(1000.0, -1000.0)
        cal = S.calibrate_stage(scope)
        assert cal.rotation_deg == pytest.approx(scope.camera.get_property("Rotation caméra"), abs=0.5)
        assert cal.pixel_size_um == pytest.approx(scope.pixel_size_um(), rel=0.03)
        assert scope.xy_stage.get_position() == pytest.approx((1000.0, -1000.0))
    finally:
        scope.disconnect()


def test_simulated_noise_is_independent_between_frames(demo_scope):
    """Le bruit simulé ne doit pas être un même motif décalé d'une image à l'autre : un
    recalage d'images le « reconnaîtrait », ce qu'aucune vraie caméra ne permet."""
    from fuji.analysis.registration import MIN_CONFIDENCE, image_shift, shift_confidence

    cam = demo_scope.camera
    demo_scope.device("lampe").set_on(False)  # image noire : uniquement du bruit
    a, b = cam.snap(), cam.snap()
    assert a.std() > 0 and not np.array_equal(a, b)
    assert abs(shift_confidence(a, b, image_shift(a, b))) < MIN_CONFIDENCE / 3


def test_stage_tools_need_a_motorized_stage(demo_scope):
    profile = MicroscopeProfile.model_validate(
        {
            "name": "Manuel",
            "devices": [{"name": "cam", "driver": "demo.camera", "options": {"simulate_timing": False}}],
        }
    )
    with Microscope(profile) as scope, pytest.raises(RuntimeError, match="motorisée"):
        S.calibrate_stage(scope)


def test_backlash_is_measured_on_the_image_and_cancelled_by_the_driver():
    import fuji.drivers as drivers

    def make(compensation: float):
        sim = SimulatedController("grbl", speed=3000, backlash={"X": 0.008})  # 8 µm de jeu réel

        class Truth(XYStage):  # ce que la caméra voit : la position réelle de la platine
            def get_position(self):
                return sim.mechanical["X"] * 1000.0, sim.mechanical["Y"] * 1000.0

            def _move_to(self, x, y):
                raise NotImplementedError

        @drivers.register_driver("test.truth", kind="xy_stage")
        def _factory(spec, ctx):
            return Truth(spec.name)

        devices = [
            {"name": "verite", "driver": "test.truth"},
            {
                "name": "moteurs",
                "driver": "gcode",
                "options": {"link": GCodeLink("simulation", io=sim), "backlash_xy_um": compensation},
            },
            {"name": "obj", "driver": "demo.state"},
            {
                "name": "cam",
                "driver": "demo.camera",
                "options": {
                    "width": 512,
                    "height": 384,
                    "simulate_timing": False,
                    "color": False,
                    "xy_stage": "verite",
                },
            },
        ]
        profile = MicroscopeProfile.model_validate(
            {
                "name": "Jeu",
                "devices": devices,
                "roles": {"camera": "cam", "objective_changer": "obj", "xy_stage": "moteurs_xy"},
                "objectives": [{"label": "10x", "magnification": 10, "na": 0.25}],
            }
        )
        scope = Microscope(profile)
        scope.connect()
        return scope

    try:
        scope = make(0.0)
        assert S.measure_backlash(scope, "x") == pytest.approx(8.0, abs=0.8)
        assert S.measure_backlash(scope, "y") == pytest.approx(0.0, abs=0.5)
        scope.disconnect()
        scope = make(8.0)  # jeu reporté dans les options du pilote : il est rattrapé
        assert S.measure_backlash(scope, "x") == pytest.approx(0.0, abs=0.8)
        scope.disconnect()
    finally:
        drivers._REGISTRY.pop("test.truth", None)


# -------------------------------------------------------------------------- mosaïques
def test_mosaic_plan_follows_the_stage_orientation(sim):
    scope = _scope(sim)
    try:
        plan = S.plan_mosaic(scope, columns=3, rows=2, overlap=0.1, center=(1000, 2000))
        px = scope.pixel_size_um()
        assert len(plan.positions) == 6 and plan.tile_shape == (384, 512)
        xs = sorted({round(x, 3) for x, _ in plan.positions})
        assert np.diff(xs) == pytest.approx([512 * 0.9 * px] * 2)
        assert [round(x - 1000, 1) for x, _ in plan.positions] == pytest.approx(
            [-133.6, 0.0, 133.6, 133.6, 0.0, -133.6], abs=0.1
        )  # serpentin : la deuxième rangée revient
        assert plan.offsets_px[0] == (0, 0) and plan.offsets_px[3] == pytest.approx(
            (384 * 0.9, 2 * 512 * 0.9)
        )
        scope.profile.stage_orientation = [[0.0, 1.0], [-1.0, 0.0]]  # caméra tournée d'un quart de tour
        turned = S.plan_mosaic(scope, columns=3, rows=1, overlap=0.0, center=(0, 0))
        assert [round(x, 3) for x, _ in turned.positions] == [0, 0, 0]
        assert [round(y, 2) for _, y in turned.positions] == pytest.approx([512 * px, 0, -512 * px], abs=0.01)
        with pytest.raises(ValueError):
            S.plan_mosaic(scope, 0, 1)
    finally:
        scope.disconnect()


def test_stitching_rebuilds_the_original_image():
    rng = np.random.default_rng(3)
    from scipy import ndimage as ndi

    big = (ndi.gaussian_filter(rng.random((300, 420)), 3) * 60000).astype(np.uint16)
    h, w = 120, 160
    offsets = [(0, 0), (0, 130), (0, 260), (90, 260), (90, 130), (90, 0), (180, 0), (180, 130), (180, 260)]
    tiles = [big[r : r + h, c : c + w] for r, c in offsets]
    assert (S.stitch(tiles, offsets) == big).all()
    assert (S.stitch(tiles, offsets, blend=False) == big).all()
    # positions fausses de quelques pixels (platine imparfaite) : recalées d'après le contenu
    wrong = [
        (r + int(e[0]), c + int(e[1])) for (r, c), e in zip(offsets, rng.integers(-4, 5, (9, 2)), strict=True)
    ]
    wrong[0] = offsets[0]
    refined = S.refine_offsets(tiles, [(float(r), float(c)) for r, c in wrong])
    assert np.abs(np.array(refined) - np.array(offsets)).max() <= 1.0
    rgb = np.stack([big, big // 2, big // 3], -1)
    colour = S.stitch([rgb[r : r + h, c : c + w] for r, c in offsets], offsets)
    assert colour.shape == (300, 420, 3) and (colour == rgb).all()
    with pytest.raises(ValueError):
        S.stitch(tiles, offsets[:3])


def test_mosaic_acquired_with_the_motor_stage_matches_a_wider_view(sim):
    h, w = 288, 384
    scope = _scope(sim, shape=(h, w))
    try:
        scope.camera.set_property("Rotation caméra", 5.0)
        S.save_stage_calibration(scope, S.calibrate_stage(scope))
        plan = S.plan_mosaic(scope, columns=3, rows=3, overlap=0.2)
        sequence = useq.MDASequence(stage_positions=[{"x": x, "y": y} for x, y in plan.positions])
        tiles = {}
        engine = AcquisitionEngine(scope)
        engine.frame_acquired.connect(lambda event, frame: tiles.__setitem__(event.index["p"], frame.data))
        summary = engine.run(sequence)
        assert summary.acquired == 9 and summary.error is None
        ordered = [tiles[i] for i in range(9)]
        offsets = S.refine_offsets(ordered, plan.offsets_px)
        assert np.abs(np.array(offsets) - np.array(plan.offsets_px)).max() < 6  # la platine simulée est juste
        mosaic = S.stitch(ordered, offsets)
        assert mosaic.shape[0] > 2.4 * h and mosaic.shape[1] > 2.4 * w
        # une vue prise à cheval sur plusieurs tuiles doit se retrouver dans la mosaïque, raccords compris
        origin = np.array(offsets).min(axis=0)
        checked = 0
        for i, j in ((0, 1), (4, 5), (1, 4), (0, 4)):
            (xa, ya), (xb, yb) = plan.positions[i], plan.positions[j]
            scope.xy_stage.move_to((xa + xb) / 2, (ya + yb) / 2)
            view = scope.snap().data
            r0, c0 = np.round((np.array(offsets[i]) + np.array(offsets[j])) / 2 - origin).astype(int)
            patch = mosaic[r0 : r0 + h, c0 : c0 + w]
            dy, dx, score = S._displacement(view, patch, 12)
            if score < 0.5:  # champ presque vide : rien à comparer
                continue
            checked += 1
            assert abs(dx) <= 2 and abs(dy) <= 2, (i, j, dx, dy)
        assert checked >= 2
    finally:
        scope.disconnect()


# ------------------------------------------------------- autofocus pendant l'acquisition
def test_autofocus_at_each_position_recentres_the_z_stack(sim):
    scope = _scope(sim, shape=(240, 320))
    try:
        scope.focus.move_to(7.0)  # départ flou : la netteté est à Z = 0
        sequence = useq.MDASequence(
            stage_positions=[{"x": 0, "y": 0}, {"x": 60, "y": 0}],
            z_plan={"range": 4, "step": 2},
        )
        zs: dict[int, list[float]] = {0: [], 1: []}
        engine = AcquisitionEngine(scope)
        engine.frame_acquired.connect(lambda event, frame: zs[event.index["p"]].append(frame.metadata.z_um))
        summary = engine.run(sequence, autofocus=AutofocusPlan(range_um=20, step_um=2))
        assert summary.error is None and summary.acquired == 6
        for stack in zs.values():
            assert np.mean(stack) == pytest.approx(0.0, abs=1.0)  # pile centrée sur la netteté trouvée
            assert np.ptp(stack) == pytest.approx(4.0)
        assert scope.focus.get_position() == pytest.approx(7.0)  # état de départ restauré
        zs = {0: [], 1: []}
        without = engine.run(sequence)
        assert without.error is None and np.mean(zs[0]) == pytest.approx(7.0)  # sans autofocus : Z de départ
    finally:
        scope.disconnect()


def test_autofocus_keeps_the_focus_when_the_field_is_empty(sim):
    from fuji.assistants import FocusNotFound, autofocus

    scope = _scope(sim, shape=(120, 160))
    try:
        scope.set_objective("40x")  # champ minuscule, sans cellule : la netteté ne varie pas
        scope.focus.move_to(3.0)
        with pytest.raises(FocusNotFound):
            autofocus(scope, range_um=20, step_um=2)
        assert scope.focus.get_position() == pytest.approx(3.0)  # pas de plan choisi au hasard
        zs = []
        engine = AcquisitionEngine(scope)
        engine.frame_acquired.connect(lambda event, frame: zs.append(frame.metadata.z_um))
        summary = engine.run(
            useq.MDASequence(stage_positions=[{"x": 0, "y": 0}]),
            autofocus=AutofocusPlan(range_um=20, step_um=2),
        )
        assert summary.error is None and zs == pytest.approx([3.0])  # l'acquisition continue, Z conservé
    finally:
        scope.disconnect()
