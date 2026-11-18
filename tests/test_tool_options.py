"""Réglages ajoutés aux outils existants : sans y toucher, le résultat d'ImageJ est gardé."""

from __future__ import annotations

import numpy as np
import pytest

from fuji.analysis import Image, Rois, run

RNG = np.random.default_rng(7)
GRAY = Image(RNG.integers(0, 255, (40, 60), dtype=np.uint8), name="gris")
COLOR = Image(RNG.integers(0, 4000, (40, 60, 3), dtype=np.uint16), name="couleur")
LINE = Rois.line(5, 5, 45, 5)


def test_histogram_range_and_channels():
    plain = run("Histogramme", GRAY)  # sans réglage : les 256 classes d'ImageJ
    assert len(plain) == 256 and plain.plot.kind == "barres" and plain.plot.colors == ()
    part = run("Histogramme", GRAY, lower=50, upper=99)
    inside = int(((GRAY.data >= 50) & (GRAY.data <= 99)).sum())
    assert part.data["Valeur"].tolist() == list(range(50, 100)) and part.summary["N"] == inside
    assert part.data["Nombre"].sum() == inside and part.summary["Min"] >= 50
    with pytest.raises(ValueError, match="entre ces deux valeurs"):
        run("Histogramme", GRAY, lower=300, upper=400)

    three = run("Histogramme", COLOR, channel="les trois couleurs", bins=32)
    assert list(three.data.columns) == ["Valeur", "Rouge", "Vert", "Bleu"] and len(three) == 32
    assert three.plot.kind == "ligne" and three.plot.colors == ("#f05252", "#2fbf71", "#4f8cff")
    assert all(three.data[c].sum() == 2400 for c in ("Rouge", "Vert", "Bleu"))
    assert three.summary["Mean vert"] == pytest.approx(COLOR.data[..., 1].mean())
    green = run("Histogramme", COLOR, channel="vert", bins=16)
    assert list(green.data.columns) == ["Valeur", "Vert"] and green.plot.colors == ("#2fbf71",)
    # une image en gris n'a qu'un canal, quel que soit le choix
    assert run("Histogramme", GRAY, channel="rouge").data.equals(plain.data)


def test_profile_channels_and_smoothing():
    plain = run("Profil d'intensité", COLOR, LINE)
    assert list(plain.data.columns) == ["Distance", "Valeur"] and plain.plot.colors == ()
    three = run("Profil d'intensité", COLOR, LINE, channel="les trois couleurs")
    assert list(three.data.columns) == ["Distance", "Rouge", "Vert", "Bleu"]
    assert three.plot.colors == ("#f05252", "#2fbf71", "#4f8cff")
    assert three.data["Vert"].tolist() == COLOR.data[5, 5:46, 1].astype(float).tolist()
    smooth = run("Profil d'intensité", COLOR, LINE, smooth=9)
    assert smooth.data["Valeur"].std() < plain.data["Valeur"].std() / 2
    assert smooth.data["Valeur"].mean() == pytest.approx(plain.data["Valeur"].mean(), rel=0.05)
    two = Rois(LINE.shapes + Rois.line(5, 20, 45, 20).shapes)
    both = run("Profil d'intensité", COLOR, two, channel="rouge")
    assert list(both.data.columns) == ["Distance", "ROI 1 rouge", "ROI 2 rouge"]


def test_sharpen_strength_and_smooth_passes():
    sharp = run("Accentuer", GRAY)
    assert np.array_equal(run("Accentuer", GRAY, strength=0).data, GRAY.data)
    half = run("Accentuer", GRAY, strength=50)
    change = np.abs(sharp.data.astype(int) - GRAY.data).mean()
    assert 0 < np.abs(half.data.astype(int) - GRAY.data).mean() < change
    once, thrice = run("Lisser", GRAY), run("Lisser", GRAY, passes=3)
    assert thrice.data.std() < once.data.std() < GRAY.data.std()


def test_background_estimate_can_be_shown():
    xx = np.mgrid[:80, :80][1]
    ramp = (40 + xx).astype(np.uint8)  # fond qui monte de gauche à droite
    ramp[38:42, 38:42] = 250  # un petit objet clair
    image = Image(ramp, name="pente")
    background = run("Soustraction de fond", image, radius=15, output="fond estimé")
    corrected = run("Soustraction de fond", image, radius=15)
    assert background.name == "pente (fond estimé)" and corrected.name == "pente (fond soustrait)"
    assert background.data[40, 40] < 200  # l'objet n'est pas dans le fond
    assert int(corrected.data[40, 40]) + int(background.data[40, 40]) == pytest.approx(250, abs=2)
