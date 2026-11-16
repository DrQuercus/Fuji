"""Comparaison directe avec ImageJ (le vrai, via PyImageJ).

Ces tests garantissent que les mesures de Fuji sont identiques à celles de Fiji.
Ils nécessitent Java et le téléchargement d'ImageJ (ou un Fiji local) :

    FUJI_TEST_IMAGEJ=1 pytest tests/test_imagej_parity.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from skimage.draw import disk, ellipse

pytestmark = pytest.mark.imagej

if not os.environ.get("FUJI_TEST_IMAGEJ"):
    pytest.skip("définir FUJI_TEST_IMAGEJ=1 pour comparer avec ImageJ", allow_module_level=True)
pytest.importorskip("imagej")

from fuji.analysis import Image, Labels, run  # noqa: E402
from fuji.analysis.imagej import ImageJBridge  # noqa: E402

COLUMNS = [
    "Area",
    "Perim.",
    "Circ.",
    "Major",
    "Minor",
    "Angle",
    "Feret",
    "MinFeret",
    "AR",
    "Round",
    "Solidity",
    "X",
    "Y",
]


@pytest.fixture(scope="module")
def bridge():
    return ImageJBridge.get()


def test_analyze_particles_parity(bridge):
    m = np.zeros((240, 320), np.uint8)
    rr, cc = disk((60, 60), 25)
    m[rr, cc] = 255
    rr, cc = ellipse(150, 170, 18, 55, rotation=math.radians(35))
    m[rr, cc] = 255
    m[30:70, 200:290] = 255
    m[190:220, 30:45] = 255
    ours, _ = run("Analyse de particules", Labels((m > 0).astype(np.uint8), 0.5))
    macro = """
setThreshold(128, 255);
run("Set Scale...", "distance=2 known=1 unit=micron");
run("Set Measurements...", "area centroid perimeter fit shape feret's redirect=None decimal=6");
run("Analyze Particles...", "display clear");
"""
    res = bridge.run_macro(macro, Image(m, 0.5, "parite"))
    theirs = res.table.data
    assert len(theirs) == len(ours)
    a = ours.data.sort_values(["Y", "X"]).reset_index(drop=True)
    b = theirs.sort_values(["Y", "X"]).reset_index(drop=True)
    for col in COLUMNS:
        np.testing.assert_allclose(a[col], b[col], rtol=2e-3, atol=2e-3, err_msg=col)
