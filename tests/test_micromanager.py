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
