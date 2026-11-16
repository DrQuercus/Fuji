from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Chaque test a son propre dossier utilisateur (profils, champ plat...)."""
    monkeypatch.setenv("FUJI_HOME", str(tmp_path / "fuji-home"))


@pytest.fixture
def demo_scope():
    from fuji.core import Microscope

    scope = Microscope.from_file("demo")
    scope.connect()
    scope.camera._timing = False  # pas d'attente réelle de l'exposition
    yield scope
    scope.disconnect()


@pytest.fixture
def mono_demo_profile():
    """Petit microscope simulé monochrome, rapide pour les tests."""
    from fuji.core import MicroscopeProfile

    return MicroscopeProfile.from_yaml(
        """
name: Test mono
devices:
  - {name: z, driver: demo.focus}
  - {name: obj, driver: demo.state}
  - {name: cam, driver: demo.camera,
     options: {width: 160, height: 120, color: false, bit_depth: 12, simulate_timing: false}}
roles: {camera: cam, focus: z, objective_changer: obj}
objectives:
  - {label: "10x/0.25", magnification: 10, na: 0.25}
  - {label: "40x/0.65", magnification: 40, na: 0.65}
channels:
  - {name: BF, exposure_ms: 5}
default_channel: BF
"""
    )
