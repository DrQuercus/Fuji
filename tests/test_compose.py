"""Compositions en direct : profondeur de champ étendue et panorama à la main."""

from __future__ import annotations

import itertools

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


def test_focus_stacking_is_never_stuck_after_the_slide_moved(banded_frames):
    """Le glissement est mesuré par rapport à l'image composée, sans cumul : dès que la lame
    revient en place, la composition reprend (avant, elle restait bloquée)."""
    sharp, frames = banded_frames
    stack = FocusStacker()
    for frame in frames[:3]:
        stack.add(frame)
    for _ in range(5):  # la lame est déplacée un moment
        st = stack.add(np.roll(frames[3], (100, 150), axis=(0, 1)))
        assert st.state == "bougé" and not st.accepted
    for frame in frames[3:]:  # elle revient : les images suivantes comptent de nouveau
        assert stack.add(frame).accepted
    assert stack.status.state in ("mieux", "rien") and _details(stack.result, sharp) > 0.95


def test_focus_stacking_leaves_an_empty_background_alone():
    """Sur un fond sans détail, le bruit de la caméra ne fait pas « gagner » une image sur
    l'autre : rien n'est remplacé, la composition ne scintille pas."""
    rng = np.random.default_rng(5)
    stack = FocusStacker()
    first = np.clip(rng.normal(3000, 25, (360, 540, 3)), 0, 4095).astype(np.uint16)
    stack.add(first)
    gains = [
        stack.add(np.clip(rng.normal(3000, 25, first.shape), 0, 4095).astype(np.uint16)).contribution
        for _ in range(8)
    ]
    assert max(gains) == 0.0 and stack.status.state == "rien"
    assert np.array_equal(stack.result, first)


def test_focus_stacking_a_4k_frame_is_fast_enough_for_live():
    """Une image 4K couleur est fusionnée en bien moins d'une demi-seconde."""
    import time

    rng = np.random.default_rng(6)
    base = ndi.zoom(rng.random((270, 480, 3)).astype(np.float32), (8, 8, 1), order=1) * 3000
    frames = [np.clip(base + rng.normal(0, 30, base.shape), 0, 4095).astype(np.uint16) for _ in range(2)]
    stack = FocusStacker()
    stack.add(frames[0])
    t0 = time.perf_counter()
    for i in range(4):
        stack.add(frames[(i + 1) % 2])
    assert (time.perf_counter() - t0) / 4 < 0.5
    assert stack.result.shape == (2160, 3840, 3)


# -------------------------------------------------------------------- panorama
class _Rig:
    """Une grande lame sous une caméra fixe, comme sur un vrai microscope : éclairage inégal
    (vignettage), poussières fixes, bruit, et flou de bougé quand la lame se déplace."""

    def __init__(self, field: tuple[int, int] = (300, 400), exposure_s: float = 0.004) -> None:
        self.slide = _texture(1500, 2200, seed=7).astype(np.float32)
        self.field, self.exposure_s = field, exposure_s
        self.rng = np.random.default_rng(8)
        h, w = field
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        shade = 1 - 0.09 * (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
        for _ in range(8):  # poussières sur le capteur
            y, x, r = self.rng.uniform(0, h), self.rng.uniform(0, w), self.rng.uniform(2, 5)
            shade *= 1 - 0.2 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * r * r))
        self.shade = shade[..., None]

    def frame(self, pos: tuple[float, float], velocity: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        h, w = self.field
        blur = (velocity[0] * self.exposure_s, velocity[1] * self.exposure_s)
        n = max(1, int(np.ceil(max(abs(blur[0]), abs(blur[1])))) + 1)
        acc = np.zeros((h, w, 3), np.float32)
        for k in range(n):  # la lame avance pendant la pose
            f = (k + 0.5) / n - 0.5
            y, x = int(round(pos[0] + f * blur[0])), int(round(pos[1] + f * blur[1]))
            acc += self.slide[y : y + h, x : x + w]
        img = acc / n * self.shade + self.rng.normal(0, 15, acc.shape)
        return np.clip(img, 0, 4095).astype(np.uint16)


def _glide(
    mosaic: LiveMosaic,
    rig: _Rig,
    stops: list[tuple[float, float]],
    speed: float = 400.0,
    fps: float = 30.0,
    t0: float = 0.0,
) -> tuple[list[str], float, list[float]]:
    """La main pousse la lame d'un point au suivant **sans s'arrêter**, en tremblant un peu.
    Renvoie les états, l'instant final et l'erreur de position à chaque image suivie."""
    states, errors, t = [], [], t0
    for a, b in itertools.pairwise(stops):
        dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        steps = max(1, int(dist / speed * fps))
        for i in range(1, steps + 1):
            f = i / steps
            tremor = 1.2 * np.sin(31 * t)
            pos = (a[0] + f * (b[0] - a[0]) + tremor, a[1] + f * (b[1] - a[1]) - tremor)
            vel = ((b[0] - a[0]) / dist * speed, (b[1] - a[1]) / dist * speed)
            before = len(mosaic.tiles)
            st = mosaic.add(rig.frame(pos, vel), t, rig.exposure_s)
            states.append(st.state)
            if before == 0 and mosaic.tiles:  # où était vraiment la lame au premier champ posé
                first = mosaic.tiles[0].position
                mosaic.truth0 = (pos[0] - first[0], pos[1] - first[1])
            if st.position is not None and st.state != "perdu" and mosaic.tiles:
                dy = st.position[0] + mosaic.truth0[0] - pos[0]
                dx = st.position[1] + mosaic.truth0[1] - pos[1]
                errors.append(float(np.hypot(dy, dx)))
            t += 1.0 / fps
    return states, t, errors


_SERPENTINE = [
    (100.0, 100.0),
    (100.0, 1300.0),
    (350.0, 1300.0),
    (350.0, 100.0),
    (600.0, 100.0),
    (600.0, 1300.0),
]


def test_panorama_builds_without_ever_stopping(tmp_path):
    """La lame est poussée en continu, en serpentin : tout le parcours est assemblé, chaque
    champ à sa place (avant, il fallait s'arrêter net sur chaque champ)."""
    rig, mosaic = _Rig(), LiveMosaic(workdir=tmp_path)
    states, _t, errors = _glide(mosaic, rig, _SERPENTINE)
    assert "perdu" not in states and "trop_vite" not in states
    assert len(mosaic.tiles) > 15 and states.count("ajouté") == len(mosaic.tiles)
    assert np.median(errors) < 2.0 and max(errors) < 6.0  # pixels
    image, scale = mosaic.render()
    assert scale == 1 and image.dtype == np.uint16
    top, left = mosaic.bounds()[:2]
    # toute la zone parcourue est couverte, et ressemble à la lame
    assert image.shape[0] >= 780 and image.shape[1] >= 1580 and (image.sum(-1) > 0).mean() > 0.97
    r, c = int(round(top + mosaic.truth0[0])), int(round(left + mosaic.truth0[1]))
    truth = rig.slide[r : r + image.shape[0], c : c + image.shape[1]]
    inner = (slice(60, 700), slice(60, 1500))
    assert _details(image[inner], truth[inner]) > 0.9
    mosaic.close()
    assert not list(tmp_path.glob("fuji-panorama-*"))


def test_panorama_evens_out_the_lighting(tmp_path):
    """Le vignettage est estimé sur l'ensemble des champs et compensé : pas de bandes sombres
    entre les rangées, même sans image de référence."""
    rig, mosaic = _Rig(), LiveMosaic(workdir=tmp_path)
    _glide(mosaic, rig, _SERPENTINE)
    shading = mosaic.shading()
    assert shading is not None and shading.min() < 0.97 < 1.01 < shading.max()  # coins sombres
    top, left = mosaic.bounds()[:2]
    r, c = int(round(top + mosaic.truth0[0])), int(round(left + mosaic.truth0[1]))

    def unevenness(image: np.ndarray) -> float:
        """Écart de luminosité à grande échelle entre le panorama et la lame."""
        truth = rig.slide[r : r + image.shape[0], c : c + image.shape[1], 1]
        ratio = ndi.gaussian_filter(image[..., 1].astype(np.float32), 20) / ndi.gaussian_filter(truth, 20)
        ratio = ratio[80:700, 80:1500]
        return float(np.ptp(ratio) / ratio.mean())

    assert unevenness(mosaic.render()[0]) < 0.6 * unevenness(mosaic.render(even=False)[0])
    mosaic.close()


def test_panorama_too_fast_is_softer_then_refined_and_reports_speed(tmp_path):
    rig, mosaic = _Rig(), LiveMosaic(workdir=tmp_path)
    row = [(100.0, 100.0), (100.0, 1300.0)]
    states, t, _ = _glide(mosaic, rig, row, speed=800.0)  # 3,2 px de flou : posé, mais « vite »
    assert "vite" in states and "trop_vite" not in states and "perdu" not in states
    soft = max(t.blur for t in mosaic.tiles)
    assert soft > 2.5
    # on repasse lentement : les champs flous sont remplacés par des nets
    states, t, _ = _glide(mosaic, rig, row[::-1], speed=250.0, t0=t + 0.1)
    assert "affiné" in states
    assert max(t.blur for t in mosaic.tiles) < soft
    # beaucoup trop vite : rien n'est posé, on demande de ralentir
    fast = LiveMosaic(workdir=tmp_path)
    _states, t, _ = _glide(fast, rig, [(100.0, 100.0), (100.0, 130.0)], speed=100.0)  # un premier champ
    n = len(fast.tiles)
    states, _t, _ = _glide(fast, rig, [(100.0, 130.0), (100.0, 1300.0)], speed=2500.0, t0=t)
    assert n >= 1 and len(fast.tiles) == n and "trop_vite" in states
    assert "ralentissez" in fast.status.message.lower() or fast.status.state == "perdu"
    for m in (mosaic, fast):
        m.close()


def test_panorama_recovers_after_losing_track_and_can_undo(tmp_path):
    rig, mosaic = _Rig(), LiveMosaic(workdir=tmp_path)
    states, t, _ = _glide(mosaic, rig, [(100.0, 100.0), (100.0, 700.0)])
    n = len(mosaic.tiles)
    assert n >= 4 and "perdu" not in states
    # un geste brusque : la lame saute loin, le suivi est perdu, rien n'est ajouté
    for k in range(4):
        st = mosaic.add(rig.frame((900.0, 1500.0 + 3 * k)), t + 0.1 * (k + 1), rig.exposure_s)
    assert st.state == "perdu" and len(mosaic.tiles) == n and "revenez" in st.message
    # retour sur une zone assemblée : la position est retrouvée, et le panorama continue
    states, t, errors = _glide(mosaic, rig, [(100.0, 400.0), (100.0, 1000.0)], t0=t + 1.0)
    assert states[0] in ("suivi", "couvert", "ajouté") and "perdu" not in states[1:]
    assert len(mosaic.tiles) > n and max(errors[5:]) < 6.0

    # retirer le dernier champ : il ne revient pas tant que la lame n'a pas bougé
    count, version = len(mosaic.tiles), mosaic.status.version
    assert mosaic.undo() and len(mosaic.tiles) == count - 1 and mosaic.status.version > version
    for k in range(5):
        st = mosaic.add(rig.frame((100.0, 1000.0)), t + 0.05 * (k + 1), rig.exposure_s)
    assert len(mosaic.tiles) == count - 1 and "retiré" in st.message
    mosaic.close()


def test_panorama_is_not_held_back_by_dust_on_the_sensor(tmp_path):
    """Des poussières fixes très marquées ne « retiennent » pas l'image : elles sont apprises
    pendant le déplacement et retirées du suivi."""
    rig, mosaic = _Rig(), LiveMosaic(workdir=tmp_path)
    h, w = rig.field
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rng = np.random.default_rng(3)
    for _ in range(60):
        y, x = rng.uniform(0, h), rng.uniform(0, w)
        rig.shade *= (1 - 0.45 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 18.0))[..., None]
    states, _t, errors = _glide(mosaic, rig, _SERPENTINE[:4])
    assert "perdu" not in states and np.median(errors) < 2.5 and max(errors) < 8.0
    mosaic.close()


def test_large_panorama_is_reduced_to_fit_in_memory(tmp_path):
    rig, mosaic = _Rig(), LiveMosaic(workdir=tmp_path)
    _glide(mosaic, rig, [(100.0, 100.0), (100.0, 900.0)])
    full, one = mosaic.render()
    done = []
    small, scale = mosaic.render(
        max_pixels=full.shape[0] * full.shape[1] // 3, progress=lambda a, b: done.append((a, b))
    )
    assert one == 1 and scale == 2 and done and done[-1][0] == done[-1][1]
    assert abs(small.shape[0] - full.shape[0] // 2) <= 1 and abs(small.shape[1] - full.shape[1] // 2) <= 1
    assert abs(float(small.mean()) - float(full.mean())) < 0.02 * float(full.mean())
    mosaic.close()
