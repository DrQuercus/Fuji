"""Éditeur de configuration : logique (sans Qt) et pilotes décrits."""

from __future__ import annotations

import inspect

import pytest

from fuji.core.catalog import FILTER_SETS, OBJECTIVES, TEMPLATES
from fuji.core.profile import DeviceSpec, MicroscopeProfile, builtin_profiles, user_profile_dir
from fuji.drivers import DriverContext, create_devices, drivers_info
from fuji.ui.config import model as M


@pytest.mark.parametrize(
    "info", [i for i in drivers_info() if i.name != "micromanager"], ids=lambda i: i.name
)
def test_driver_defaults_build_a_device(info):
    """Chaque pilote accepte ses options par défaut (le formulaire ne peut pas casser)."""
    prof = MicroscopeProfile(name="t", objectives=[{"label": "10x", "magnification": 10, "na": 0.25}])
    opts = info.defaults()
    if info.name.endswith(".state"):
        opts["labels"] = ["A", "B"]
    spec = DeviceSpec(name="x", driver=info.name, options=opts)
    devices = create_devices(spec, DriverContext(prof))
    if info.kind is not None:
        assert devices and all(d.kind.value == info.kind for d in devices)
    else:  # pilote qui crée plusieurs périphériques : il annonce lesquels
        announced = info.provided("x", opts)
        assert announced and {(d.name, d.kind.value) for d in devices[1:]} == set(announced)


def test_option_names_match_constructors():
    from fuji.drivers.demo import DemoCamera
    from fuji.drivers.touptek import ToupTekCamera

    for cls, driver in ((DemoCamera, "demo.camera"), (ToupTekCamera, "touptek.camera")):
        params = set(inspect.signature(cls.__init__).parameters)
        info = next(i for i in drivers_info() if i.name == driver)
        assert {o.name for o in info.options} <= params


def test_rename_and_remove_keep_references_consistent():
    d = MicroscopeProfile.load(builtin_profiles()["demo"]).model_dump(mode="json")
    dapi = next(c for c in d["channels"] if c["name"] == "DAPI")
    M.rename_device(d, "cubes", "tourelle_cubes")
    assert dapi["states"] == {"tourelle_cubes": "DAPI"}
    M.rename_device(d, "lampe", "halogene")
    assert d["roles"]["transmitted_light"] == "halogene"
    with pytest.raises(ValueError):
        M.rename_device(d, "halogene", "camera")
    M.remove_device(d, "led_365")
    assert "led_365" not in dapi["lights"]
    profile = MicroscopeProfile.model_validate(d)
    assert profile.check() == []


def test_add_device_assigns_obvious_roles():
    d = M.empty_profile()
    M.add_device(d, "touptek.camera")
    M.add_device(d, "manual.light")
    M.add_device(d, "manual.focus")
    assert d["roles"]["camera"] == "camera"
    assert d["roles"]["transmitted_light"] == "lampe"
    assert d["roles"]["focus"] == "mise_au_point"
    assert M.unique_name(d, "camera") == "camera_2"


def test_setup_fluorescence_creates_turret_lamp_and_channels():
    d = MicroscopeProfile.model_validate(TEMPLATES["manuel-touptek"]["profile"]).model_dump(mode="json")
    d["filter_sets"] = [f for f in FILTER_SETS if f["name"].startswith("Leitz")]
    actions = M.setup_fluorescence(d)
    assert actions
    profile = MicroscopeProfile.model_validate(d)
    assert profile.check() == []
    cubes = next(x for x in profile.devices if x.name == "cubes")
    assert cubes.options["labels"][0] == "Vide" and len(cubes.options["labels"]) == 4
    fluo = [c for c in profile.channels if c.kind == "fluorescence"]
    assert len(fluo) == 3 and all(c.filter_set for c in fluo)
    assert profile.channel("Fond clair").states == {"cubes": "Vide"}
    assert M.setup_fluorescence(d) == []  # idempotent


def test_templates_and_catalog_are_valid():
    from fuji.ui.config.wizard import template_data

    for key in TEMPLATES:
        prof = MicroscopeProfile.model_validate(template_data(key))
        assert prof.name
    for o in OBJECTIVES:
        MicroscopeProfile(name="x", objectives=[o])


def test_save_path_never_overwrites_builtin(tmp_path):
    builtin = builtin_profiles()["demo"]
    assert M.save_path_for({}, builtin) == user_profile_dir() / "demo.yaml"
    new = M.save_path_for({"name": "Mon Diaplan"}, None)
    assert new.name == "mon-diaplan.yaml"
    new.write_text("x")
    assert M.save_path_for({"name": "Mon Diaplan"}, None).name == "mon-diaplan-2.yaml"


def test_profile_check_reports_problems():
    prof = MicroscopeProfile.model_validate(
        {
            "name": "x",
            "devices": [{"name": "t", "driver": "manual.state"}, {"name": "l", "driver": "manual.light"}],
            "roles": {"camera": "l"},
            "channels": [{"name": "A", "states": {"t": "Z"}, "filter_set": "absent"}],
        }
    )
    issues = "\n".join(prof.check())
    assert "positions non définies" in issues
    assert "pas du bon type" in issues
    assert "cube « absent »" in issues
    assert "Aucune caméra" in issues
