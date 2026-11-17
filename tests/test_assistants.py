from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage as ndi

from fuji.assistants import (
    FocusTracker,
    KohlerAssistant,
    KohlerStep,
    analyze_field_diaphragm,
    autofocus,
    calibrate_micrometer,
    illumination_uniformity,
    recommended_aperture,
)


def _grating(shape=(600, 800), period=23.7, angle_deg=4.0, width=2.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    th = math.radians(angle_deg)
    s = xx * math.cos(th) + yy * math.sin(th)
    phase = (s % period) / period
    lines = np.minimum(phase, 1 - phase) * period < width / 2
    img = 3000 - 2000 * ndi.gaussian_filter(lines.astype(float), 1.0)
    return img + np.random.default_rng(0).normal(0, 20, shape)


def test_micrometer_calibration_on_rotated_grating():
    cal = calibrate_micrometer(_grating(), spacing_um=10)
    assert cal.period_px == pytest.approx(23.7, rel=2e-3)
    assert cal.pixel_size_um == pytest.approx(10 / 23.7, rel=2e-3)
    assert abs(cal.angle_deg) == pytest.approx(4.0, abs=0.5)
    assert cal.n_lines > 25


@pytest.mark.parametrize("objective", ["4x/0.10", "10x/0.25"])
def test_micrometer_calibration_on_simulator(demo_scope, objective):
    cam = demo_scope.camera
    cam.set_property("Échantillon", "micromètre")
    demo_scope.set_objective(objective)
    cal = calibrate_micrometer(demo_scope.snap().data, 10.0)
    assert cal.pixel_size_um == pytest.approx(demo_scope.pixel_size_um(), rel=3e-3)


def test_field_diaphragm_sharpness_and_centering(demo_scope):
    cam = demo_scope.camera
    cam.set_property("Échantillon", "vide")
    cam.set_property("Diaphragme de champ", 30)
    cam.set_property("Hauteur du condenseur (erreur)", 400)
    blurred = analyze_field_diaphragm(demo_scope.snap().data)
    cam.set_property("Hauteur du condenseur (erreur)", 0)
    sharp = analyze_field_diaphragm(demo_scope.snap().data)
    assert blurred.visible and sharp.visible and sharp.fully_inside
    assert sharp.edge_width_px < 0.5 * blurred.edge_width_px
    assert sharp.offset_fraction > 0.1
    dy, dx = sharp.offset_px
    assert dx > 0 and dy < 0  # décentrage X +14 %, Y −9 % dans le simulateur
    cam.set_property("Décentrage condenseur X", 0)
    cam.set_property("Décentrage condenseur Y", 0)
    centered = analyze_field_diaphragm(demo_scope.snap().data)
    assert centered.offset_fraction < 0.02


def test_kohler_assistant_walkthrough(demo_scope):
    cam = demo_scope.camera
    cam.set_property("Échantillon", "vide")
    k = KohlerAssistant(objective_na=0.25, condenser_na=0.9, stable_frames=2)
    assert k.update(demo_scope.snap().data).step is KohlerStep.CLOSE_FIELD
    cam.set_property("Diaphragme de champ", 30)
    for _ in range(3):
        k.update(demo_scope.snap().data)
    assert k.step is KohlerStep.FOCUS_CONDENSER
    for err in (400, 200, 0):
        cam.set_property("Hauteur du condenseur (erreur)", err)
        g = k.update(demo_scope.snap().data)
    assert g.can_advance and g.edge_width_px == pytest.approx(g.best_edge_width_px)
    k.advance()
    g = k.update(demo_scope.snap().data)
    assert g.step is KohlerStep.CENTER_CONDENSER and g.arrow is not None
    cam.set_property("Décentrage condenseur X", 0)
    cam.set_property("Décentrage condenseur Y", 0)
    for _ in range(3):
        k.update(demo_scope.snap().data)
    assert k.step is KohlerStep.OPEN_FIELD
    cam.set_property("Diaphragme de champ", 100)
    for _ in range(3):
        g = k.update(demo_scope.snap().data)
    assert k.step is KohlerStep.SET_APERTURE
    g = k.update(demo_scope.snap().data)
    assert "0.17" in g.message and "0.20" in g.message


def test_recommended_aperture():
    assert recommended_aperture(0.25) == pytest.approx((0.175, 0.2))
    assert recommended_aperture(1.4, 0.9) == pytest.approx((0.9, 0.9))


def test_uniformity_detects_lamp_offset(demo_scope):
    cam = demo_scope.camera
    cam.set_property("Échantillon", "vide")
    rep = illumination_uniformity(demo_scope.snap().data)
    assert rep.tilt_percent > 5
    assert 135 < rep.tilt_direction_deg < 225  # plus sombre à gauche
    assert "gauche" in rep.message


def test_focus_tracker_and_autofocus(demo_scope):
    tracker = FocusTracker()
    values = []
    for z in (-20, -10, 0, 10):
        demo_scope.focus.move_to(z)
        values.append(tracker.update(demo_scope.snap().data).value)
    assert int(np.argmax(values)) == 2
    demo_scope.focus.move_to(7)
    z = autofocus(demo_scope, range_um=24, step_um=3)
    assert abs(z) < 1.5


# ------------------------------------------------------------------ non-régressions
def _micrometer_scope(width, height):
    from fuji.core import Microscope, MicroscopeProfile

    return Microscope(
        MicroscopeProfile.from_yaml(
            f"""
name: Mire
devices:
  - {{name: obj, driver: demo.state}}
  - {{name: cam, driver: demo.camera, options: {{width: {width}, height: {height}, color: true,
      bit_depth: 12, simulate_timing: false, specimen: micromètre}}}}
roles: {{camera: cam, objective_changer: obj}}
objectives:
  - {{label: "4x/0.10", magnification: 4, na: 0.1}}
  - {{label: "40x/0.65", magnification: 40, na: 0.65}}
  - {{label: "100x/1.25", magnification: 100, na: 1.25, immersion: huile}}
"""
        )
    )


@pytest.mark.parametrize("rotation", [0.0, 0.2, -3.0, 14.0])
@pytest.mark.parametrize("objective", ["4x/0.10", "40x/0.65"])
def test_micrometer_calibration_any_orientation_and_magnification(rotation, objective):
    """Traits fins ne couvrant qu'une bande de l'image (4x), caméra tournée ou parfaitement
    alignée (0° : cas où un profil trop fin créait de faux traits)."""
    with _micrometer_scope(1024, 768) as scope:
        scope.camera.set_property("Rotation caméra", rotation)
        scope.set_objective(objective)
        cal = calibrate_micrometer(scope.snap().data, 10.0)
        assert cal.reliable
        assert cal.pixel_size_um == pytest.approx(scope.pixel_size_um(), rel=2e-3)
        # l'image tourne dans le sens opposé à la caméra : on compare les valeurs absolues
        assert abs(cal.angle_deg) == pytest.approx(abs(rotation), abs=0.3)


def test_micrometer_calibration_on_4k_low_magnification():
    with _micrometer_scope(3840, 2160) as scope:
        scope.set_objective("4x/0.10")
        cal = calibrate_micrometer(scope.snap().data, 10.0)
        assert cal.reliable and cal.n_lines > 50
        assert cal.pixel_size_um == pytest.approx(scope.pixel_size_um(), rel=1e-3)


def test_micrometer_calibration_with_too_few_lines_is_not_reliable():
    """Peu de traits visibles : le résultat est signalé comme peu fiable, jamais présenté
    comme sûr."""
    with _micrometer_scope(1024, 768) as scope:
        scope.camera.set_property("Rotation caméra", 14.0)
        scope.set_objective("100x/1.25")
        try:
            cal = calibrate_micrometer(scope.snap().data, 10.0)
        except ValueError:
            return  # refus net : acceptable aussi
        assert cal.n_lines < 5 and not cal.reliable


def test_calibration_rejects_images_without_graduations(demo_scope):
    demo_scope.camera.set_property("Échantillon", "vide")
    with pytest.raises(ValueError):
        calibrate_micrometer(demo_scope.snap().data, 10.0)
    with pytest.raises(ValueError):
        calibrate_micrometer(np.full((200, 300), 1000, np.uint16), 10.0)


def test_focus_peaking_reports_its_subsampling_step():
    from fuji.assistants import focus_peaking

    small = np.random.default_rng(0).normal(1000, 50, (300, 400))
    mask, step = focus_peaking(small)
    assert step == 1 and mask.shape == small.shape and mask.dtype == bool
    big = np.zeros((2160, 3840, 3), np.uint16)
    mask, step = focus_peaking(big)
    assert step == 4 and mask.shape == (540, 960)


@pytest.mark.parametrize("start_ms", [0.05, 5.0, 2000.0])
def test_auto_exposure_converges_from_dark_and_saturated(demo_scope, start_ms):
    cam = demo_scope.camera
    cam.set_exposure(start_ms)
    cam.auto_expose(0.7)
    peak = float(np.percentile(cam.snap()[::4, ::4], 99.5)) / (2**cam.bit_depth - 1)
    assert 0.6 < peak < 0.8


def test_autofocus_restores_focus_when_the_scan_fails(demo_scope, monkeypatch):
    demo_scope.focus.move_to(7.0)
    calls = {"n": 0}
    real_snap = demo_scope.snap

    def flaky():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("caméra débranchée")
        return real_snap()

    monkeypatch.setattr(demo_scope, "snap", flaky)
    with pytest.raises(RuntimeError):
        autofocus(demo_scope, range_um=24, step_um=3)
    assert demo_scope.focus.get_position() == pytest.approx(7.0)
