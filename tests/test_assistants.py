from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage as ndi

from fuji.assistants import (
    FlatFieldStore,
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


def test_flat_field_store_roundtrip(tmp_path):
    store = FlatFieldStore("Mon microscope", root=tmp_path)
    flat = np.full((8, 8), 1000.0, np.float32)
    flat[:, :4] = 500
    store.save("10x/0.25", "Fond clair", flat)
    raw = (flat * 0.4).astype(np.uint16)
    corrected = store.correct(raw, "10x/0.25", "Fond clair")
    assert corrected is not None and corrected.std() < 1
    assert store.correct(raw, "40x", "Fond clair") is None
