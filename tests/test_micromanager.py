"""Le pont Micro-Manager, testé avec des périphériques Python (UniMMCore) : aucun
adaptateur C++ ni matériel nécessaire."""

from __future__ import annotations

import numpy as np
import pytest

unicore = pytest.importorskip("pymmcore_plus.experimental.unicore")

from fuji.core import Microscope, MicroscopeProfile  # noqa: E402
from fuji.core.devices import Camera, StateDevice, XYStage  # noqa: E402


class PyCamera(unicore.SimpleCameraDevice):
    def __init__(self) -> None:
        super().__init__()
        self._exp = 10.0

    def sensor_shape(self):  # type: ignore[no-untyped-def]
        return (24, 32)

    def dtype(self):  # type: ignore[no-untyped-def]
        return np.uint16

    def get_exposure(self) -> float:
        return self._exp

    def set_exposure(self, exposure: float) -> None:
        self._exp = exposure

    def snap(self, buffer):  # type: ignore[no-untyped-def]
        buffer[:] = 1234
        return {}


class PyXY(unicore.XYStageDevice):
    def __init__(self) -> None:
        super().__init__()
        self._pos = (0.0, 0.0)

    def set_position_um(self, x: float, y: float) -> None:
        self._pos = (x, y)

    def get_position_um(self):  # type: ignore[no-untyped-def]
        return self._pos

    def busy(self) -> bool:
        return False

    def home(self) -> None:
        self._pos = (0.0, 0.0)

    def stop(self) -> None:
        pass

    def set_origin_x(self) -> None:
        pass

    def set_origin_y(self) -> None:
        pass


class PyTurret(unicore.StateDevice):
    def __init__(self) -> None:
        super().__init__({0: "10X", 1: "40X"})
        self._s = 0

    def get_state(self) -> int:
        return self._s

    def set_state(self, position: int) -> None:
        self._s = int(position)


@pytest.fixture
def mm_core():
    core = unicore.UniMMCore()
    core.loadPyDevice("Cam", PyCamera())
    core.loadPyDevice("XY", PyXY())
    core.loadPyDevice("Objective", PyTurret())
    for label in ("Cam", "XY", "Objective"):
        core.initializeDevice(label)
    core.setCameraDevice("Cam")
    core.setXYStageDevice("XY")
    return core


def test_micromanager_devices_are_wrapped(mm_core):
    prof = MicroscopeProfile(
        name="MM",
        devices=[{"name": "mm", "driver": "micromanager", "options": {"core": mm_core}}],
        objectives=[
            {"label": "10X", "magnification": 10, "na": 0.3},
            {"label": "40X", "magnification": 40, "na": 0.75},
        ],
    )
    with Microscope(prof) as scope:
        assert isinstance(scope.camera, Camera) and scope.camera.name == "Cam"
        assert isinstance(scope.xy_stage, XYStage)
        assert isinstance(scope.objective_changer, StateDevice)
        img = scope.snap().data
        assert img.shape == (24, 32) and int(img.max()) == 1234
        scope.camera.set_exposure(42)
        assert scope.camera.get_exposure() == pytest.approx(42)
        scope.xy_stage.move_to(100, -50)
        assert scope.xy_stage.get_position() == pytest.approx((100, -50))
        scope.set_objective("40X")
        assert scope.objective.magnification == 40


def _mm_profile(core) -> MicroscopeProfile:  # type: ignore[no-untyped-def]
    return MicroscopeProfile(
        name="MM", devices=[{"name": "mm", "driver": "micromanager", "options": {"core": core}}]
    )


def test_external_core_is_not_unloaded(mm_core):
    """Un cœur fourni par l'appelant (autre logiciel) reste utilisable après la déconnexion."""
    with Microscope(_mm_profile(mm_core)) as scope:
        scope.snap()
    assert "Cam" in mm_core.getLoadedDevices()
    mm_core.snapImage()  # toujours fonctionnel


def test_snap_while_live_and_live_stops_cleanly(mm_core):
    with Microscope(_mm_profile(mm_core)) as scope:
        cam = scope.camera
        cam.start_live()
        try:
            cam.wait_for_frame(cam.latest_frame()[0], timeout=5)
            img = scope.snap().data  # « snapImage » serait refusé pendant la séquence
            assert img.shape == (24, 32) and int(img.max()) == 1234
        finally:
            cam.stop_live()
        assert not cam.is_live and not mm_core.isSequenceRunning()
        assert int(scope.snap().data.max()) == 1234  # la caméra n'est pas restée « occupée »


def test_live_keeps_up_with_a_camera_faster_than_the_loop(mm_core, monkeypatch):
    """Une caméra qui produit plus vite que le live ne lit : le retard ne doit pas s'accumuler
    (retirer les images une à une figeait le live au bout de quelques secondes)."""
    import time

    real_pop = mm_core.popNextImage

    def slow_pop(*args, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(0.0005)  # ordinateur lent : chaque image retirée coûte du temps
        return real_pop(*args, **kwargs)

    monkeypatch.setattr(mm_core, "popNextImage", slow_pop)
    with Microscope(_mm_profile(mm_core)) as scope:
        cam = scope.camera
        cam.start_live()
        try:
            start, last, longest, frames = time.monotonic(), cam.latest_frame()[0], 0.0, 0
            while time.monotonic() - start < 3.0:
                before = time.monotonic()
                cam.wait_for_frame(last, timeout=2)
                longest = max(longest, time.monotonic() - before)
                last, frames = cam.latest_frame()[0], frames + 1
            assert frames > 30 and longest < 1.0, (frames, longest)
            assert int(scope.snap().data.max()) == 1234  # la capture pendant le live reste possible
        finally:
            cam.stop_live()
        assert not mm_core.isSequenceRunning()


def test_live_thread_failure_does_not_leave_camera_busy(mm_core, monkeypatch):
    with Microscope(_mm_profile(mm_core)) as scope:
        cam = scope.camera

        def boom(*_a, **_k):  # type: ignore[no-untyped-def]
            raise RuntimeError("tampon perdu")

        monkeypatch.setattr(mm_core, "getRemainingImageCount", boom)
        cam.start_live()
        cam._live_thread.join(timeout=10)  # le thread abandonne après quelques erreurs
        assert not cam.is_live
        monkeypatch.undo()
        cam.stop_live()  # doit quand même arrêter la séquence Micro-Manager
        assert not mm_core.isSequenceRunning()
        assert int(scope.snap().data.max()) == 1234
