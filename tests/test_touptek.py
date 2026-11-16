from __future__ import annotations

import time

import numpy as np
import pytest

import fake_toupcam as tc
from fuji.drivers.touptek import ToupTekCamera, ToupTekSDKNotFound, install_sdk, load_toupcam


@pytest.fixture
def camera():
    cam = ToupTekCamera("cam", bit_depth=12, toupcam_module=tc)
    cam.initialize()
    yield cam
    cam.shutdown()


def test_initialization_configures_12bit_rgb(camera):
    hcam = tc.OPENED[-1]
    assert camera.is_color and camera.bit_depth == 12
    assert hcam.options[tc.TOUPCAM_OPTION_BITDEPTH] == 1
    assert hcam.options[tc.TOUPCAM_OPTION_RGB] == 1  # RGB48
    assert hcam.options[tc.TOUPCAM_OPTION_TRIGGER] == 1  # déclenchement logiciel
    assert hcam.autoexpo == 0
    assert camera.sensor_shape() == (10, 26)
    assert camera.sensor_pixel_size_um == pytest.approx(2.9)


def test_snap_uses_software_trigger_and_handles_row_padding(camera):
    hcam = tc.OPENED[-1]
    img = camera.snap()
    assert img.shape == (10, 26, 3) and img.dtype == np.uint16
    np.testing.assert_array_equal(img, hcam.expected_frame(48))


def test_padding_with_odd_width_8bit():
    cam = ToupTekCamera("cam", bit_depth=8, resolution_index=1, toupcam_module=tc)
    cam.initialize()
    try:
        hcam = tc.OPENED[-1]
        img = cam.snap()
        assert img.shape == (5, 13, 3) and img.dtype == np.uint8
        np.testing.assert_array_equal(img, hcam.expected_frame(24))
        # pixel effectif plus grand en résolution réduite
        assert cam.sensor_pixel_size_um == pytest.approx(5.8)
    finally:
        cam.shutdown()


def test_live_switches_to_video_mode(camera):
    hcam = tc.OPENED[-1]
    camera.start_live()
    assert hcam.options[tc.TOUPCAM_OPTION_TRIGGER] == 0
    n0, _ = camera.latest_frame()
    time.sleep(0.2)
    n1, _ = camera.latest_frame()
    assert n1 > n0 + 2
    camera.stop_live()
    assert hcam.options[tc.TOUPCAM_OPTION_TRIGGER] == 1


def test_exposure_gain_and_properties(camera):
    hcam = tc.OPENED[-1]
    camera.set_exposure(25.5)
    assert hcam.expo_us == 25_500 and camera.get_exposure() == pytest.approx(25.5)
    camera.set_gain(300)
    assert camera.get_gain() == 300
    names = {p.name for p in camera.property_infos()}
    assert {"Résolution", "Profondeur de bits", "Gamma", "Température de couleur"} <= names
    camera.set_property("Résolution", "13x5")
    assert camera.sensor_shape() == (5, 13)
    assert camera.snap().shape == (5, 13, 3)
    camera.auto_white_balance()
    assert hcam.awb_calls == 1
    assert camera.describe()["serial"] == "SN-TEST"


def test_missing_sdk_has_helpful_message(tmp_path):
    with pytest.raises(ToupTekSDKNotFound, match="fuji touptek install"):
        load_toupcam(tmp_path)


def test_install_sdk_from_folder(tmp_path):
    from fuji.drivers.touptek import _platform_library

    libname, arch = _platform_library()
    sdk = tmp_path / "toupcamsdk"
    (sdk / "python").mkdir(parents=True)
    (sdk / "python" / "toupcam.py").write_text("X = 1\n")
    libdir = sdk / ("mac" if libname.endswith("dylib") else "linux") / arch[0]
    libdir.mkdir(parents=True)
    (libdir / libname).write_bytes(b"\0")
    dest = install_sdk(sdk, tmp_path / "out")
    assert (dest / "toupcam.py").exists() and (dest / libname).exists()
    assert load_toupcam(dest).X == 1
