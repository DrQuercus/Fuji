"""Empaquetage de l'application (Fuji.app, .dmg) : cohérence de la configuration."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _name(requirement: str) -> str:
    return re.split(r"[\[<>=!~ ]", requirement, maxsplit=1)[0].lower()


def test_packaged_app_contains_every_optional_feature(pyproject):
    """L'application doit offrir tout ce qu'offre « pip install fuji-microscopy[all] »."""
    extras = pyproject["project"]["optional-dependencies"]
    wanted = {r for group in ("gui", "micromanager", "imagej", "moteurs", "ia") for r in extras[group]}
    bundled = set(pyproject["tool"]["briefcase"]["app"]["fuji"]["requires"])
    assert bundled == wanted
    assert {_name(r) for r in extras["all"]} == {"fuji-microscopy"}  # « all » renvoie aux groupes ci-dessus


def test_version_is_the_same_everywhere(pyproject):
    import fuji

    assert pyproject["project"]["version"] == fuji.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", fuji.__version__)  # lisible dans le Finder


def test_app_icons_exist_and_are_complete(pyproject):
    from PIL import Image

    base = ROOT / pyproject["tool"]["briefcase"]["app"]["fuji"]["icon"]
    icns = Image.open(base.with_suffix(".icns"))
    assert icns.format == "ICNS" and icns.size == (1024, 1024)
    assert {(16, 16, 2), (128, 128, 2), (512, 512, 2)} <= set(icns.info["sizes"])
    assert Image.open(base.with_suffix(".png")).size == (1024, 1024)
    assert base.with_suffix(".ico").exists()


def test_macos_settings(pyproject):
    app = pyproject["tool"]["briefcase"]["app"]["fuji"]
    mac = app["macOS"]
    assert app["sources"] == ["src/fuji"] and (ROOT / "src/fuji/__main__.py").exists()
    assert mac["universal_build"] is False
    assert tuple(int(p) for p in mac["min_os_version"].split(".")) >= (
        12,
        0,
    )  # scipy : macOS 12 sur Apple Silicon


def test_min_os_check_reads_otool_output():
    sys.path.insert(0, str(ROOT / "packaging" / "macos"))
    try:
        import check_min_os
    finally:
        sys.path.pop(0)
    modern = "Load command 9\n      cmd LC_BUILD_VERSION\n  cmdsize 32\n platform 1\n    minos 13.0\n"
    legacy = "Load command 8\n      cmd LC_VERSION_MIN_MACOSX\n  cmdsize 16\n  version 10.9\n      sdk 10.9\n"
    assert [m.group(2) for m in check_min_os.PATTERN.finditer(modern + legacy)] == ["13.0", "10.9"]
    assert check_min_os.version("10.15") < check_min_os.version("11.0") < check_min_os.version("13.1")


def test_finder_process_number_is_ignored(monkeypatch, capsys):
    """Lancée depuis le Finder, l'application peut recevoir « -psn_… » : ce n'est pas une commande."""
    import runpy

    monkeypatch.setattr(sys, "argv", ["Fuji", "-psn_0_12345", "drivers"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("fuji", run_name="__main__")
    assert stop.value.code == 0 and "gcode" in capsys.readouterr().out


def test_errors_are_kept_in_a_log_file(tmp_path, monkeypatch):
    """Lancée d'un double-clic, l'application n'a pas de terminal : les erreurs vont au journal."""
    pytest.importorskip("qtpy")
    import logging

    from fuji.ui import app

    monkeypatch.setenv("FUJI_HOME", str(tmp_path / "maison"))
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        path = app.start_log_file()
        assert path == tmp_path / "maison" / "fuji.log" and app.start_log_file() == path  # une seule fois
        assert len(root.handlers) == len(before) + 1
        logging.getLogger("fuji.essai").warning("caméra « %s » introuvable", "ToupTek")
        logging.getLogger("fuji.essai").info("détail sans importance")
        for handler in root.handlers:
            handler.flush()
        text = path.read_text(encoding="utf-8")
        assert "caméra « ToupTek » introuvable" in text and "sans importance" not in text
    finally:
        for handler in root.handlers[:]:
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


@pytest.mark.gui
def test_selftest_passes_on_the_simulated_microscope(tmp_path):
    """Le même essai que celui lancé sur l'application empaquetée avant de la distribuer."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("pas d'affichage (lancer avec xvfb-run)")
    pytest.importorskip("napari")
    import json
    import subprocess

    report, shot = tmp_path / "essai.json", tmp_path / "essai.png"
    done = subprocess.run(
        [sys.executable, "-m", "fuji", "selftest", "--report", str(report), "--screenshot", str(shot)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "FUJI_HOME": str(tmp_path / "home")},
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    failed = [(s["etape"], s["detail"]) for s in result["etapes"] if not s["ok"]]
    assert done.returncode == 0 and result["ok"] and not failed, (failed, done.stderr[-2000:])
    assert {
        "fenêtre",
        "live",
        "capture",
        "ouverture dans Analyse",
        "analyse",
        "enregistrement",
        "affichage",
        "compositions en direct",
        "moteurs simulés",
    } <= {s["etape"] for s in result["etapes"]}
    assert shot.stat().st_size > 20_000
