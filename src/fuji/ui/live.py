"""Affichage du live dans napari (interrogation périodique, sans copie inutile)."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from napari.components import ViewerModel
from qtpy.QtCore import QObject, QTimer, Signal

from fuji.core.microscope import Microscope

LIVE_LAYER = "Live"


def units_for(px: float | None) -> tuple[str, str] | None:
    """Unités physiques des axes (barre d'échelle en µm) si l'image est calibrée."""
    return ("µm", "µm") if px else None


def contrast_for(data: np.ndarray, bit_depth: int | None) -> tuple[float, float]:
    """Contraste automatique robuste (0,1 % – 99,9 %), sur un sous-échantillon."""
    sample = data[::8, ::8]
    lo, hi = np.percentile(sample, (0.1, 99.9))
    if hi <= lo:
        hi = lo + 1
    if bit_depth and data.dtype.kind == "u":
        hi = min(hi, 2**bit_depth - 1)
    return float(lo), float(hi)


class LiveController(QObject):
    fps_changed = Signal(float)
    live_changed = Signal(bool)

    def __init__(self, microscope: Microscope, viewer: ViewerModel, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self.viewer = viewer
        self.auto_contrast = True
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)
        self._last_count = -1
        self._times: deque[float] = deque(maxlen=30)
        self._frames_since_contrast = 0

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        cam = self.microscope.camera
        if cam is None:
            return
        cam.start_live()
        self._timer.start()
        self.live_changed.emit(True)

    def stop(self) -> None:
        self._timer.stop()
        cam = self.microscope.camera
        if cam is not None:
            cam.stop_live()
        self.live_changed.emit(False)

    def toggle(self) -> None:
        self.stop() if self.running else self.start()

    def _poll(self) -> None:
        cam = self.microscope.camera
        if cam is None:
            return
        count, frame = cam.latest_frame()
        if frame is None or count == self._last_count:
            return
        self._last_count = count
        self.show_frame(frame, LIVE_LAYER)
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) > 2:
            self.fps_changed.emit((len(self._times) - 1) / (self._times[-1] - self._times[0]))

    def show_frame(self, frame: np.ndarray, name: str) -> None:
        calibrated = self.microscope.pixel_size_um()
        px = calibrated or 1.0
        rgb = frame.ndim == 3 and frame.shape[-1] in (3, 4)
        cam = self.microscope.camera
        bits = cam.bit_depth if cam else None
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if layer.data.shape == frame.shape and getattr(layer, "rgb", False) == rgb:
                layer.data = frame
                if tuple(layer.scale[-2:]) != (px, px):  # changement d'objectif
                    layer.scale = (px, px)
                    layer.metadata["pixel_size_um"] = calibrated
                self._frames_since_contrast += 1
                if self.auto_contrast and self._frames_since_contrast >= 10:
                    layer.contrast_limits = contrast_for(frame, bits)
                    self._frames_since_contrast = 0
                return
            self.viewer.layers.remove(name)
        layer = self.viewer.add_image(
            frame,
            name=name,
            rgb=rgb,
            scale=(px, px),
            units=units_for(calibrated),
            contrast_limits=contrast_for(frame, bits),
        )
        layer.metadata["pixel_size_um"] = self.microscope.pixel_size_um()
        self.viewer.layers.move(self.viewer.layers.index(layer), 0)
        self.viewer.reset_view()
