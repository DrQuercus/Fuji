from __future__ import annotations

import pytest
from pydantic import ValidationError

from fuji.core import Microscope, MicroscopeProfile
from fuji.core.profile import builtin_profiles, resolve_profile, user_profile_dir


@pytest.mark.parametrize("name", sorted(builtin_profiles()))
def test_builtin_profiles_are_valid(name):
    prof = MicroscopeProfile.load(builtin_profiles()[name])
    assert prof.name
    assert prof.devices


def test_yaml_on_key_is_not_a_boolean():
    prof = MicroscopeProfile.from_yaml(
        """
name: x
devices: [{name: lamp, driver: demo.light, options: {on: true}}]
channels: [{name: BF, lights: {lamp: {on: false, intensity: 30}}}]
"""
    )
    assert prof.devices[0].options == {"on": True}
    assert prof.channels[0].lights["lamp"].on is False


def test_duplicate_names_rejected():
    with pytest.raises(ValidationError):
        MicroscopeProfile.from_yaml(
            "name: x\ndevices: [{name: a, driver: demo.focus}, {name: a, driver: demo.focus}]"
        )


def test_roundtrip_yaml():
    prof = MicroscopeProfile.load(builtin_profiles()["demo"])
    again = MicroscopeProfile.from_yaml(prof.to_yaml())
    assert again == prof


def test_saving_builtin_profile_goes_to_user_dir():
    scope = Microscope.from_file("demo")
    path = scope.save_profile()
    assert path.parent == user_profile_dir()
    # le profil utilisateur devient prioritaire
    assert resolve_profile("demo") == path
