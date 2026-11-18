"""Réglages d'image (luminosité, couleurs, niveaux, bruit) et options ajoutées aux outils."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from fuji.analysis import Image, get_operation, run
from fuji.analysis.ops.adjust import noise_level, white_level


def _scene(noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Disques sombres sur fond clair, en 12 bits rangés dans du 16 bits : (gris, couleur)."""
    yy, xx = np.mgrid[:160, :220]
    clean = np.full((160, 220), 0.75, np.float32)
    for cy, cx, r in ((40, 50, 18), (100, 140, 30), (70, 190, 10)):
        clean[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 0.3
    if noise:
        clean = clean + np.random.default_rng(seed).normal(0, noise, clean.shape).astype(np.float32)
    gray = (np.clip(clean, 0, 1) * 4095).astype(np.uint16)
    return gray, np.stack([gray, (gray * 0.9).astype(np.uint16), (gray * 0.8).astype(np.uint16)], -1)


def test_white_level_follows_the_camera_depth():
    assert white_level(np.zeros((4, 4), np.uint8)) == 255
    assert white_level(np.full((8, 8), 4000, np.uint16)) == 4095  # 12 bits dans du 16 bits
    assert white_level(np.full((8, 8), 40000, np.uint16)) == 65535
    assert white_level(np.full((8, 8), 0.7, np.float32)) == 1.0


def test_image_settings_at_rest_change_nothing_and_keep_the_type():
    gray, rgb = _scene()
    for data in (gray, rgb, (gray // 16).astype(np.uint8)):
        out = run("Réglages de l'image", Image(data))
        assert out.data.dtype == data.dtype and np.array_equal(out.data, data)
    op = get_operation("Réglages de l'image")
    assert len(op.params) == 9 and all(p.meta.description for p in op.params)
    assert all(p.meta.min is not None and p.meta.max is not None for p in op.params)  # tout a un curseur


def test_each_image_setting_does_what_its_name_says():
    gray, rgb = _scene()
    image = Image(rgb)
    mean = rgb.astype(float).mean()

    def adjusted(**settings: float) -> np.ndarray:
        return run("Réglages de l'image", image, **settings).data.astype(float)

    assert adjusted(brightness=20).mean() > mean + 300 and adjusted(brightness=-20).mean() < mean - 300
    assert adjusted(contrast=40).std() > rgb.std() * 1.3 and adjusted(contrast=-40).std() < rgb.std() * 0.8
    assert adjusted(contrast=-100).std() < 1.0  # tout gris
    assert adjusted(gamma=0.6).mean() > mean > adjusted(gamma=1.8).mean()
    gray_out = adjusted(saturation=-100)
    assert np.abs(gray_out[..., 0] - gray_out[..., 2]).max() <= 1  # plus aucune couleur
    spread = np.ptp(adjusted(saturation=60).mean(axis=(0, 1)))
    assert spread > np.ptp(rgb.astype(float).mean(axis=(0, 1))) * 1.3
    warm, cold = adjusted(temperature=60).mean(axis=(0, 1)), adjusted(temperature=-60).mean(axis=(0, 1))
    assert warm[0] / warm[2] > rgb[..., 0].mean() / rgb[..., 2].mean() > cold[0] / cold[2]
    green, magenta = adjusted(tint=-60).mean(axis=(0, 1)), adjusted(tint=60).mean(axis=(0, 1))
    assert green[1] / green[0] > magenta[1] / magenta[0]
    assert not np.allclose(adjusted(hue=90).mean(axis=(0, 1)), rgb.astype(float).mean(axis=(0, 1)), rtol=0.02)
    # netteté : le bord devient plus raide ; bruit : le fond devient plus lisse, le bord reste
    edge = np.abs(np.diff(adjusted(sharpness=80)[40, 60:75, 0])).max()
    assert edge > np.abs(np.diff(rgb[40, 60:75, 0].astype(float))).max()
    noisy = Image(_scene(noise=0.04)[1])
    smooth = run("Réglages de l'image", noisy, denoise=60).data.astype(float)
    assert smooth[5:25, 5:25, 0].std() < 0.6 * noisy.data[5:25, 5:25, 0].std()
    assert np.abs(np.diff(smooth[40, 60:75, 0])).max() > 0.5 * 0.45 * 4095
    # une image grise ignore les réglages de couleur
    g = run("Réglages de l'image", Image(gray), saturation=80, hue=120, temperature=50, tint=50)
    assert np.array_equal(g.data, gray)


def test_levels_write_the_display_range_into_the_pixels():
    gray, rgb = _scene()
    dark, bright = float(gray.min()), float(gray.max())  # les disques, le fond
    out = run("Niveaux", Image(gray), black=dark, white=bright)
    assert out.data.dtype == np.uint16 and out.data.min() == 0 and out.data.max() == 4095
    assert run("Niveaux", Image(gray), black=1000.0).data.max() == 4095  # blanc à 0 : maximum de l'image
    out = run("Niveaux", Image(gray), black=1000.0, white=3200.0)
    darker = run("Niveaux", Image(gray), black=1000.0, white=3200.0, gamma=2.0).data
    assert darker.mean() < out.data.mean()
    one = run("Niveaux", Image(rgb), black=1000.0, white=3000.0, channel="rouge").data
    assert np.array_equal(one[..., 1:], rgb[..., 1:]) and not np.array_equal(one[..., 0], rgb[..., 0])
    limited = run("Niveaux", Image(gray), out_black=20.0, out_white=80.0).data
    assert limited.min() >= 0.19 * 4095 and limited.max() <= 0.81 * 4095
    with pytest.raises(ValueError, match="plus clair que le noir"):
        run("Niveaux", Image(gray), black=3000.0, white=1000.0)
    with pytest.raises(ValueError, match="image couleur"):
        run("Niveaux", Image(gray), channel="vert")


def test_colour_balance_and_white_balance_options():
    _gray, rgb = _scene()
    means = rgb.astype(float).mean(axis=(0, 1))
    out = run("Balance des couleurs", Image(rgb), red=50.0, blue=120.0).data.astype(float).mean(axis=(0, 1))
    assert out[0] == pytest.approx(means[0] * 0.5, rel=0.02) and out[1] == pytest.approx(means[1], rel=0.01)
    assert out[2] > means[2] * 1.1
    kept = run("Balance des couleurs", Image(rgb), red=200.0, green=200.0, blue=200.0, keep_brightness=True)
    assert np.allclose(kept.data, rgb, atol=1)  # même gain partout, luminosité gardée : rien ne change
    with pytest.raises(ValueError, match="image couleur"):
        run("Balance des couleurs", Image(rgb[..., 0]))
    # balance des blancs : valeurs par défaut inchangées, force réglable, fond amené au blanc
    full = run("Balance des blancs", Image(rgb)).data.astype(float)
    background = full[2:12, 2:12].mean(axis=(0, 1))
    assert np.ptp(background) < 0.01 * background.mean()
    half = (
        run("Balance des blancs", Image(rgb), strength=50.0).data.astype(float)[2:12, 2:12].mean(axis=(0, 1))
    )
    assert np.ptp(background) < np.ptp(half) < np.ptp(rgb[2:12, 2:12].astype(float).mean(axis=(0, 1)))
    white = run("Balance des blancs", Image(rgb), target="fond blanc").data
    assert white[2:12, 2:12].min() >= 4090 and white.max() == 4095


def test_denoise_methods_remove_grain_and_respect_edges():
    gray, _ = _scene(noise=0.05)
    truth = _scene()[0].astype(np.float32)
    noisy_error = float(np.abs(gray.astype(np.float32) - truth).mean())
    assert noise_level(gray.astype(np.float32) / 4095) == pytest.approx(0.05, rel=0.3)
    for method in get_operation("Réduire le bruit").params[0].meta.choices:
        out = run("Réduire le bruit", Image(gray), method=method, strength=40.0)
        assert out.data.dtype == gray.dtype and out.name.endswith("(débruitée)")
        assert float(np.abs(out.data.astype(np.float32) - truth).mean()) < 0.6 * noisy_error, method
    careful = run("Réduire le bruit", Image(gray), strength=40.0, detail=90.0).data.astype(float)
    blurry = run("Réduire le bruit", Image(gray), method="flou gaussien", strength=40.0).data.astype(float)
    assert np.abs(np.diff(careful[40, 60:75])).max() > 2 * np.abs(np.diff(blurry[40, 60:75])).max()
    colour = run("Réduire le bruit", Image(_scene(noise=0.05)[1]))
    assert colour.data.shape == (160, 220, 3)


def test_edge_detection_methods_and_default_is_imagej_sobel():
    gray = (_scene()[0] // 16).astype(np.uint8)
    default = run("Détecter les contours", Image(gray)).data
    assert np.array_equal(default, run("Détecter les contours", Image(gray), method="Sobel (ImageJ)").data)
    for method in ("Scharr", "Prewitt", "Laplacien"):
        out = run("Détecter les contours", Image(gray), method=method).data
        assert out.dtype == np.uint8 and out[5:15, 5:15].max() == 0 and out.max() > 50, method
    canny = run("Détecter les contours", Image(gray), method="Canny (contours fins)", sigma=1.5).data
    assert set(np.unique(canny)) == {0, 255} and 0.005 < (canny > 0).mean() < 0.05  # traits d'un pixel
    smooth = run("Détecter les contours", Image(gray), sigma=3.0).data
    assert smooth.max() < default.max()


def test_thresholds_can_clean_their_mask_in_one_go():
    """Lisser, remplir les trous, retirer les petites taches : à zéro, le masque d'ImageJ."""
    gray, _ = _scene(noise=0.06)
    gray[100, 140] = 4000  # un trou dans le grand disque
    for name, extra in (
        ("Seuillage automatique", {"method": "Otsu", "objects": "sombres"}),
        ("Seuil manuel", {"lower": 0.0, "upper": 2100.0}),
        ("Seuillage local", {"objects": "sombres"}),
    ):
        raw = run(name, Image(gray), **extra).data
        clean = run(name, Image(gray), smooth=1.5, fill_holes=True, min_size=40, **extra).data
        assert set(np.unique(clean)) <= {0, 1}
        objects = ndi.label(clean, structure=np.ones((3, 3)))[1]
        assert objects <= ndi.label(raw, structure=np.ones((3, 3)))[1], name
        if name != "Seuillage local":
            assert objects == 3 and clean[100, 140] == 1, name
    params = {p.name for p in get_operation("Seuillage automatique").params}
    assert {"smooth", "fill_holes", "min_size"} <= params
