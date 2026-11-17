"""Image de référence (champ plat) : correction du fond à la sortie de la caméra."""

from __future__ import annotations

import time

import numpy as np
import pytest

from fuji.core import Microscope
from fuji.core.flatfield import FlatField, FlatFieldError, FlatFieldStore, check_reference


def _vignetted(h: int = 120, w: int = 160, channels: int = 3) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    shade = 1 - 0.3 * (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 2
    flat = 2500 * shade
    return np.stack([flat * k for k in (1.0, 0.9, 0.8)][:channels], -1) if channels else flat


def test_reference_flattens_the_background_and_keeps_the_colours():
    flat = _vignetted()
    ref = FlatField(flat)
    raw = (flat * 0.6).astype(np.uint16)  # la même scène vide, moins exposée
    out = ref.apply(raw, max_value=4095)
    assert out.dtype == raw.dtype and out.shape == raw.shape
    for c in range(3):  # chaque canal devient uniforme, à sa propre moyenne (balance des blancs intacte)
        assert out[..., c].std() < 1.0
        assert out[..., c].mean() == pytest.approx(raw[..., c].mean(), rel=0.01)
    # un objet sombre garde son contraste
    raw[40:60, 60:80] //= 2
    out = ref.apply(raw)
    assert out[50, 70, 0] == pytest.approx(out[10, 10, 0] / 2, rel=0.03)


def test_dark_frame_is_subtracted_and_values_stay_in_range():
    flat, dark = _vignetted(channels=0), np.full((120, 160), 100.0, np.float32)
    ref = FlatField(flat + dark, dark)
    out = ref.apply((flat * 0.5 + dark).astype(np.uint16))
    assert out.std() < 1.0 and out.mean() == pytest.approx(flat.mean() * 0.5, rel=0.01)
    # saturation : la correction ne dépasse pas la valeur maximale de la caméra
    bright = np.full((120, 160), 4095, np.uint16)
    assert int(ref.apply(bright, max_value=4095).max()) == 4095
    # hors du champ éclairé (référence presque noire) : le gain est borné, pas d'explosion du bruit
    holed = flat.copy()
    holed[:10] = 1.0
    assert FlatField(holed).gain.max() <= 8.0
    # image en nombres réels : pas d'arrondi
    assert FlatField(flat).apply((flat * 0.5).astype(np.float32)).dtype == np.float32


def test_reference_follows_camera_binning_but_not_another_framing():
    ref = FlatField(_vignetted(120, 160))
    assert ref.for_shape((120, 160, 3)) is ref
    binned = ref.for_shape((60, 80, 3))
    assert binned is not None and binned.gain.shape == (60, 80, 3)
    raw = (_vignetted(60, 80) * 0.7).astype(np.uint16)
    assert binned.apply(raw)[..., 0].std() < 4.0
    assert ref.for_shape((240, 320, 3)).gain.shape == (240, 320, 3)  # référence prise en binning
    assert ref.for_shape((100, 160, 3)) is None and ref.for_shape((120, 160)) is None
    with pytest.raises(ValueError, match="tailles différentes"):
        ref.apply(np.zeros((100, 160, 3), np.uint16))


def test_store_keeps_one_reference_per_objective_and_channel(tmp_path):
    store = FlatFieldStore("Mon microscope", root=tmp_path)
    flat = _vignetted()
    path = store.save("10x/0.25", "Fond clair", flat, info={"date": "2026-11-17T14:02:00", "frames": 16})
    assert path.exists() and store.exists("10x/0.25", "Fond clair") and not store.exists("40x", "Fond clair")
    loaded, dark = store.load("10x/0.25", "Fond clair")
    assert dark is None and np.allclose(loaded, flat, rtol=1e-3)  # 16 bits : précision suffisante
    assert store.info("10x/0.25", "Fond clair")["frames"] == 16 and store.info("40x", None) is None
    assert store.reference("40x", "Fond clair") is None
    # ancien format (float32 compressé, sans renseignements) : toujours lisible
    np.savez_compressed(store.path("4x", "Fond clair"), flat=flat.astype(np.float32))
    assert np.allclose(store.load("4x", "Fond clair")[0], flat) and store.info("4x", "Fond clair") == {}
    assert store.delete("4x", "Fond clair") and not store.delete("4x", "Fond clair")
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_doubtful_references_are_refused_or_flagged():
    flat = _vignetted()
    assert check_reference(flat, 4095) == []
    with pytest.raises(FlatFieldError, match="saturée"):
        check_reference(np.full_like(flat, 4095), 4095)
    with pytest.raises(FlatFieldError, match="trop sombre"):
        check_reference(flat * 0.03, 4095)
    assert "sombre" in check_reference(flat * 0.25, 4095)[0]
    busy = flat.copy()
    busy[30:80, 40:110] *= 0.5  # un échantillon dans le champ
    assert "zone vide" in check_reference(busy, 4095)[0]
    dusty = flat.copy()
    dusty[50:53, 60:63] *= 0.6  # une poussière : c'est justement ce qu'on veut corriger
    assert check_reference(dusty, 4095) == []


def test_correction_is_applied_at_the_camera_for_live_and_snaps(tmp_path, monkeypatch):
    monkeypatch.setenv("FUJI_HOME", str(tmp_path))

    def corners(data: np.ndarray) -> float:
        d = data.astype(float)
        h, w = d.shape[:2]
        return d[:40, :40].mean() / d[h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20].mean()

    with Microscope.from_file("demo") as scope:
        scope.camera._timing = False
        scope.camera.set_property("Échantillon", "vide")
        ff, changes = scope.flatfield, []
        ff.changed.connect(lambda: changes.append(ff.status().state))
        assert ff.status().state == "off" and not ff.active and corners(scope.snap().data) < 0.9
        ff.set_enabled(True)
        assert ff.status().state == "missing" and "Prendre la référence" in ff.status().text

        seen = []
        candidate = ff.measure(frames=6, progress=lambda done, total: seen.append((done, total)))
        assert seen[-1] == (6, 6) and candidate.warnings == [] and candidate.info["frames"] == 6
        assert corners(candidate.image) < 0.9  # mesurée sur les images brutes
        ff.save(candidate)
        assert ff.active and ff.status().state == "on" and "Fond corrigé" in ff.status().text
        frame = scope.snap()
        assert corners(frame.data) == pytest.approx(1.0, abs=0.02)
        assert frame.metadata.extra["flat_field_corrected"] is True

        # le live est corrigé lui aussi ; reprendre une référence pendant le live mesure du brut
        scope.camera.start_live()
        time.sleep(0.4)
        assert corners(scope.camera.latest_frame()[1]) == pytest.approx(1.0, abs=0.02)
        again = ff.measure(frames=4)
        assert corners(again.image) < 0.9
        time.sleep(0.3)
        assert corners(scope.camera.latest_frame()[1]) == pytest.approx(1.0, abs=0.02)
        scope.camera.stop_live()

        # un noir (lumière coupée) se joint à la référence ; une image claire n'est pas un noir
        with pytest.raises(FlatFieldError, match="pas noire"):
            ff.measure("dark", frames=2)
        # autre objectif : pas de référence, la correction se suspend sans rien casser
        other = next(o.label for o in scope.profile.objectives if o.label != scope.objective.label)
        scope.set_objective(other)
        assert ff.status().state == "missing" and not ff.active and corners(scope.snap().data) < 0.9
        scope.set_objective(frame.metadata.objective)
        assert ff.active
        ff.set_enabled(False)
        assert ff.status().state == "off" and "désactivée" in ff.status().text
        assert "extra" not in changes and {"missing", "on", "off"} <= set(changes)
        assert ff.clear() and ff.status().text.startswith("Aucune référence")


def test_a_failing_filter_never_stops_the_camera(tmp_path, monkeypatch):
    monkeypatch.setenv("FUJI_HOME", str(tmp_path))
    with Microscope.from_file("demo") as scope:
        scope.camera._timing = False

        def broken(_frame: np.ndarray) -> np.ndarray:
            raise RuntimeError("panne")

        scope.camera.frame_filter = broken
        assert scope.snap().data.ndim == 3  # image brute, pas d'exception


def test_a_4k_frame_is_corrected_fast_enough_for_live():
    rng = np.random.default_rng(0)
    flat = (2500 + 300 * rng.random((2160, 3840, 3))).astype(np.float32)
    ref = FlatField(flat)
    frame = (flat * 0.8).astype(np.uint16)
    ref.apply(frame, 4095)
    t0 = time.perf_counter()
    for _ in range(3):
        ref.apply(frame, 4095)
    assert (time.perf_counter() - t0) / 3 < 0.15  # ≈ 15 ms sur un portable récent
