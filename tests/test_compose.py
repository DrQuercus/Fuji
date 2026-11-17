"""Compositions en direct : profondeur de champ étendue et panorama à la main."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from fuji.core.compose import FocusStacker, LiveMosaic


def _texture(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), 2.0) + 0.5 * ndi.gaussian_filter(
        rng.normal(0, 1, (h, w)), 8
    )
    img = (base - base.min()) / (base.max() - base.min())
    return (np.stack([img, img**1.3, 1 - 0.5 * img], -1) * 3000 + 300).astype(np.uint16)


def _details(a: np.ndarray, b: np.ndarray) -> float:
    """Ressemblance des détails fins (corrélation des passe-haut)."""
    ga, gb = a[..., :3].mean(-1), b[..., :3].mean(-1)
    ha, hb = ga - ndi.gaussian_filter(ga, 3), gb - ndi.gaussian_filter(gb, 3)
    return float(np.corrcoef(ha.ravel(), hb.ravel())[0, 1])


def _noisy(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(a + rng.normal(0, 20, a.shape), 0, 4095).astype(np.uint16)


@pytest.fixture
def banded_frames():
    """Six images d'un même champ : chacune n'est nette que sur une bande verticale."""
    rng = np.random.default_rng(2)
    sharp = _texture(360, 540)
    blurred = np.stack([ndi.gaussian_filter(sharp[..., c].astype(np.float32), 4) for c in range(3)], -1)
    frames = []
    for k in range(6):
        band = np.zeros(sharp.shape[:2], np.float32)
        band[:, k * 90 : (k + 1) * 90] = 1
        band = ndi.gaussian_filter(band, 5)[..., None]
        frames.append(_noisy(band * sharp + (1 - band) * blurred, rng))
    return sharp, frames


# ------------------------------------------------------- profondeur de champ étendue
def test_focus_stacking_is_sharp_everywhere(banded_frames):
    sharp, frames = banded_frames
    stack = FocusStacker()
    contributions = [stack.add(f).contribution for f in frames]
    assert stack.result.dtype == frames[0].dtype and stack.result.shape == frames[0].shape
    assert _details(stack.result, sharp) > 0.95 > 0.7 > _details(frames[0], sharp)
    assert contributions[0] == 1.0 and all(0.05 < c < 0.6 for c in contributions[1:])
    assert stack.status.used == 6 and stack.depth_map().max() == 5
    # repasser sur des images déjà vues n'apporte plus rien
    assert stack.add(frames[2]).contribution < 0.02
    assert "Plus rien de nouveau" in stack.status.message


def test_focus_stacking_rejects_a_moved_slide_even_without_compensation(banded_frames):
    _sharp, frames = banded_frames
    stack = FocusStacker()
    for frame in frames:
        stack.add(frame)
    before = stack.result.copy()
    moved = stack.add(np.roll(frames[3], (100, 150), axis=(0, 1)))
    assert not moved.accepted and "bougé" in moved.message
    assert np.array_equal(stack.result, before)


def test_focus_stacking_follows_a_slight_drift_and_rejects_a_moved_slide(banded_frames):
    sharp, frames = banded_frames
    stack = FocusStacker(align=True)
    for k, frame in enumerate(frames):  # la lame glisse de 2 px et 3 px par image
        assert stack.add(np.roll(frame, (2 * k, -3 * k), axis=(0, 1))).accepted
    assert stack.status.drift_px == pytest.approx((10, -15), abs=1)
    inner = (slice(20, -20), slice(20, -20))
    assert _details(stack.result[inner], sharp[inner]) > 0.95
    moved = stack.add(np.roll(frames[3], (100, 150), axis=(0, 1)))
    assert not moved.accepted and "bougé" in moved.message
    assert _details(stack.result[inner], sharp[inner]) > 0.95  # l'image ignorée n'a rien abîmé


def test_focus_stacking_a_thick_specimen_in_the_simulator():
    """L'échantillon simulé est incliné : chaque endroit est net à une mise au point différente."""
    from fuji.core import Microscope, MicroscopeProfile

    profile = MicroscopeProfile.from_yaml(
        """
name: Épais
devices:
  - {name: z, driver: demo.focus}
  - {name: obj, driver: demo.state}
  - {name: cam, driver: demo.camera,
     options: {width: 512, height: 384, color: true, bit_depth: 12, simulate_timing: false}}
roles: {camera: cam, focus: z, objective_changer: obj}
objectives:
  - {label: "10x/0.25", magnification: 10, na: 0.25}
"""
    )
    with Microscope(profile) as scope:
        scope.camera.set_property("Bruit", False)
        scope.focus.move_to(0)
        truth = scope.snap().data
        scope.camera.set_property("Bruit", True)
        scope.camera.set_property("Épaisseur de l'échantillon", 60.0)
        stack, frames = FocusStacker(), []
        for z in range(-36, 37, 6):
            scope.focus.move_to(z)
            frames.append(scope.snap().data)
            stack.add(frames[-1])

    def error(img: np.ndarray) -> float:
        a, b = img[..., :3].mean(-1).astype(np.float32), truth[..., :3].mean(-1).astype(np.float32)
        return float(np.abs(ndi.gaussian_filter(a, 1.5) - ndi.gaussian_filter(b, 1.5)).mean())

    errors = [error(f) for f in frames]
    assert error(stack.result) < 0.85 * min(errors), (error(stack.result), errors)


# -------------------------------------------------------------------- panorama
class _Slide:
    """Une grande lame, vue par la caméra à l'endroit où on l'a poussée."""

    def __init__(self) -> None:
        self.world = _texture(1100, 1500, seed=5)
        self.rng = np.random.default_rng(3)

    def view(self, r: float, c: float, h: int = 300, w: int = 400) -> np.ndarray:
        return _noisy(self.world[int(r) : int(r) + h, int(c) : int(c) + w].astype(np.float32), self.rng)


def _push(mosaic: LiveMosaic, slide: _Slide, stops: list[tuple[int, int]], pause: int = 3) -> list[str]:
    """Pousse la lame d'arrêt en arrêt (quelques images pendant le mouvement, puis immobile)."""
    states, cur = [], stops[0]
    for stop in stops:
        for a in (0.25, 0.5, 0.75, 1.0):
            states.append(
                mosaic.add(slide.view(cur[0] + a * (stop[0] - cur[0]), cur[1] + a * (stop[1] - cur[1]))).state
            )
        for _ in range(pause):
            states.append(mosaic.add(slide.view(*stop)).state)
        cur = stop
    return states


def test_panorama_places_each_field_where_it_is(tmp_path):
    slide = _Slide()
    stops = [(50, 50), (50, 300), (50, 550), (260, 550), (260, 300), (260, 50)]
    mosaic = LiveMosaic(workdir=tmp_path)
    states = _push(mosaic, slide, stops)
    assert states.count("ajouté") == len(stops) == len(mosaic.tiles)
    first = np.array(mosaic.tiles[0].position)
    for tile, truth in zip(mosaic.tiles, stops, strict=True):
        assert np.array(tile.position) - first + stops[0] == pytest.approx(truth, abs=1.0)
    image, scale = mosaic.render()
    assert scale == 1 and image.shape[:2] == (510, 900)
    ref = slide.world[50 : 50 + 510, 50 : 50 + 900]
    assert _details(image, ref) > 0.97
    preview, factor, _origin = mosaic.preview  # l'aperçu couvre tout le panorama
    assert preview.shape[0] * factor >= image.shape[0] and preview.shape[1] * factor >= image.shape[1]
    mosaic.close()
    assert not list(tmp_path.glob("fuji-panorama-*/*.npy"))  # fichiers temporaires effacés


def test_panorama_only_adds_still_fields_with_new_area(tmp_path):
    slide = _Slide()
    mosaic = LiveMosaic(workdir=tmp_path)
    for k in range(12):  # la lame ne s'arrête jamais : rien de flou n'est ajouté
        mosaic.add(slide.view(50 + 15 * k, 50 + 20 * k))
    assert not mosaic.tiles and mosaic.status.state == "attente"
    for _ in range(3):
        mosaic.add(slide.view(250, 300))
    assert len(mosaic.tiles) == 1
    for _ in range(3):  # quelques pixels plus loin : rien de nouveau à ajouter
        mosaic.add(slide.view(255, 305))
    assert len(mosaic.tiles) == 1 and mosaic.status.state == "couvert"
    mosaic.close()


def test_panorama_recovers_after_losing_track_and_can_undo(tmp_path):
    slide = _Slide()
    mosaic = LiveMosaic(workdir=tmp_path)
    _push(mosaic, slide, [(50, 50), (50, 300)])
    assert len(mosaic.tiles) == 2
    lost = mosaic.add(slide.view(700, 1000))  # bien trop vite : plus rien en commun
    assert lost.state == "perdu" and "revenez" in lost.message
    back = mosaic.add(slide.view(60, 310))  # retour sur une zone connue
    assert back.state in ("suivi", "couvert")
    first = np.array(mosaic.tiles[0].position)
    assert np.array(back.position) - first + (50, 50) == pytest.approx((60, 310), abs=1.0)
    assert mosaic.undo() and len(mosaic.tiles) == 1
    for _ in range(4):  # immobile au même endroit : le champ retiré ne revient pas tout seul
        mosaic.add(slide.view(60, 310))
    assert len(mosaic.tiles) == 1 and "déplacez" in mosaic.status.message
    image, _ = mosaic.render()
    assert image.shape[:2] == (300, 400)
    mosaic.close()


def test_large_panorama_is_reduced_to_fit_in_memory(tmp_path):
    slide = _Slide()
    mosaic = LiveMosaic(workdir=tmp_path)
    _push(mosaic, slide, [(50, 50), (50, 300), (260, 300)])
    full = mosaic.footprint()
    image, scale = mosaic.render(max_pixels=full[0] * full[1] // 3)
    assert scale == 2 and image.shape[:2] == pytest.approx((full[0] // 2, full[1] // 2), abs=2)
    assert image.dtype == np.uint16
    mosaic.close()
