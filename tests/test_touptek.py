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
        # en résolution réduite, chaque pixel image couvre 2 photosites
        assert cam.sensor_pixel_size_um == pytest.approx(2.9)
        assert cam.binning == 2
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


# --------------------------------------------------------------------- régressions
def test_frames_do_not_share_the_sdk_buffer(camera):
    """En 12 bits, l'image renvoyée ne doit pas être réécrite par l'image suivante."""
    first = camera.snap()
    reference = first.copy()
    second = camera.snap()
    assert not np.array_equal(second, reference)  # le faux SDK change chaque image
    np.testing.assert_array_equal(first, reference)
    assert not np.shares_memory(first, np.frombuffer(camera._buf, np.uint8))


def test_stale_video_frame_is_not_taken_for_the_triggered_one(camera):
    hcam = tc.OPENED[-1]
    camera.set_exposure(40)
    hcam.stale_frames_on_trigger = 1
    before = hcam.frame_no
    img = camera.snap()
    assert hcam.frame_no == before + 2  # l'image en transit, puis la vraie
    np.testing.assert_array_equal(img, hcam.expected_frame(48))


def test_snap_survives_a_failed_trigger(camera):
    hcam = tc.OPENED[-1]
    hcam.fail_next_trigger = True
    img = camera.snap()
    np.testing.assert_array_equal(img, hcam.expected_frame(48))


def test_changing_resolution_during_live_keeps_live_running(camera):
    hcam = tc.OPENED[-1]
    camera.start_live()
    camera.set_property("Résolution", "13x5")
    assert camera.is_live and hcam.options[tc.TOUPCAM_OPTION_TRIGGER] == 0
    n0, _ = camera.latest_frame()
    time.sleep(0.15)
    n1, frame = camera.latest_frame()
    assert n1 > n0 and frame.shape == (5, 13, 3)
    camera.stop_live()
    assert hcam.flushed >= 1  # images vidéo en transit jetées


def test_out_of_range_resolution_is_clamped():
    cam = ToupTekCamera("cam", resolution_index=7, toupcam_module=tc)
    cam.initialize()
    try:
        assert cam.sensor_shape() == (5, 13)
        assert cam.get_property("Résolution") == "13x5"
    finally:
        cam.shutdown()


def test_failed_initialization_releases_the_camera():
    cam = ToupTekCamera("cam", toupcam_module=tc)
    n = len(tc.OPENED)
    original = tc.Toupcam.put_eSize
    tc.Toupcam.put_eSize = lambda self, i: (_ for _ in ()).throw(tc.HRESULTException(0x80004005))
    try:
        with pytest.raises(tc.HRESULTException):
            cam.initialize()
    finally:
        tc.Toupcam.put_eSize = original
    assert len(tc.OPENED) == n + 1 and tc.OPENED[-1].closed
    assert not cam.initialized


def test_disconnection_fails_fast_with_clear_message(camera):
    hcam = tc.OPENED[-1]
    camera.set_exposure(2000)
    hcam.emit_event(tc.TOUPCAM_EVENT_DISCONNECTED)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="débranchée"):
        camera.snap()
    assert time.monotonic() - t0 < 1.0


def test_detects_values_spread_over_16_bits(camera):
    hcam = tc.OPENED[-1]
    assert camera.bit_depth == 12
    camera.snap()
    assert camera.bit_depth == 12
    hcam.scale_to_16_bits = True
    camera.snap()
    assert camera.bit_depth == 16


def test_pixel_calibration_follows_resolution(camera):
    from fuji.core import Microscope, MicroscopeProfile

    prof = MicroscopeProfile.model_validate(
        {
            "name": "t",
            "devices": [{"name": "cam", "driver": "demo.camera"}],
            "objectives": [{"label": "10x", "magnification": 10, "na": 0.25}],
        }
    )
    scope = Microscope(prof)
    scope.camera = camera
    assert scope.pixel_size_um() == pytest.approx(0.29)
    scope.set_pixel_calibration(0.30)
    camera.set_property("Résolution", "13x5")
    assert scope.pixel_size_um() == pytest.approx(0.60)  # pixels deux fois plus grands
    scope.set_pixel_calibration(0.62)  # calibration refaite en résolution réduite
    camera.set_property("Résolution", "26x10")
    assert scope.pixel_size_um() == pytest.approx(0.31)


def test_install_sdk_never_picks_the_android_library(tmp_path, monkeypatch):
    import fuji.drivers.touptek as tt

    monkeypatch.setattr(tt.sys, "platform", "linux")
    monkeypatch.setattr(tt.platform, "machine", lambda: "x86_64")
    sdk = tmp_path / "sdk"
    (sdk / "python").mkdir(parents=True)
    (sdk / "python" / "toupcam.py").write_text("X = 2\n")
    for folder, content in (("android/x86_64", b"android"), ("linux/x64", b"linux"), ("linux/arm64", b"arm")):
        (sdk / folder).mkdir(parents=True)
        (sdk / folder / "libtoupcam.so").write_bytes(content)
    dest = tt.install_sdk(sdk, tmp_path / "out")
    assert (dest / "libtoupcam.so").read_bytes() == b"linux"
