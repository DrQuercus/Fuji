"""Barre rapide sous l'image : exposition, gain, objectif, canal, histogramme."""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import QDoubleSpinBox, QHBoxLayout, QSlider, QVBoxLayout, QWidget

from fuji.core.microscope import Microscope

from .components import Segmented, button, label, section_title, separator
from .histogram import HistogramWidget


class QuickBar(QWidget):
    message = Signal(str)

    def __init__(self, microscope: Microscope, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("QuickBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.m = microscope
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(14)
        cam = microscope.camera
        caps = microscope.capabilities()

        # exposition
        if cam is not None:
            box = QVBoxLayout()
            box.setSpacing(2)
            box.addWidget(section_title("Exposition"))
            row = QHBoxLayout()
            row.setSpacing(6)
            lo, hi = cam.exposure_limits()
            self._log_lo, self._log_hi = math.log10(max(lo, 0.05)), math.log10(min(hi, 10_000))
            self.exp_slider = QSlider(Qt.Orientation.Horizontal)
            self.exp_slider.setRange(0, 1000)
            self.exp_slider.setMinimumWidth(130)
            self.exp_slider.valueChanged.connect(self._slider_changed)
            self.exposure = QDoubleSpinBox()
            self.exposure.setRange(lo, hi)
            self.exposure.setDecimals(1)
            self.exposure.setSuffix(" ms")
            self.exposure.setKeyboardTracking(False)
            self.exposure.setFixedWidth(100)
            self.exposure.valueChanged.connect(self._exposure_changed)
            auto = button("Auto", "wand", tooltip="Exposition automatique (70 % de la dynamique)")
            auto.clicked.connect(self.auto_exposure)
            row.addWidget(self.exp_slider, 1)
            row.addWidget(self.exposure)
            row.addWidget(auto)
            box.addLayout(row)
            lay.addLayout(box, 3)
            if caps["gain"]:
                gbox = QVBoxLayout()
                gbox.setSpacing(2)
                gbox.addWidget(section_title("Gain"))
                glo, ghi = cam.gain_limits()
                self.gain = QDoubleSpinBox()
                self.gain.setRange(glo, ghi)
                self.gain.setDecimals(1 if ghi < 100 else 0)
                self.gain.setSuffix(f" {cam.gain_unit}")
                self.gain.setKeyboardTracking(False)
                self.gain.setFixedWidth(92)
                self.gain.valueChanged.connect(lambda v: cam.set_gain(v))
                gbox.addWidget(self.gain)
                lay.addLayout(gbox)
            lay.addWidget(separator(vertical=True))

        # objectif
        changer = microscope.objective_changer
        if changer is not None and microscope.profile.objectives:
            obox = QVBoxLayout()
            obox.setSpacing(2)
            title = "Objectif" + (" (indiquer)" if changer.manual else "")
            obox.addWidget(section_title(title))
            self.objectives = Segmented()
            short = {lbl: lbl.split("/")[0].split(" ")[0] for lbl in changer.labels}
            # deux objectifs de même grossissement (10x/0.25 et 10x/0.30 Ph) : nom complet,
            # sinon l'un des deux boutons serait inutilisable
            counts: dict[str, int] = {}
            for text in short.values():
                counts[text] = counts.get(text, 0) + 1
            short = {lbl: (text if counts[text] == 1 else lbl) for lbl, text in short.items()}
            self._obj_short = short
            self.objectives.set_options(list(short.values()), {v: k for k, v in short.items()})
            self.objectives.selected.connect(self._objective_clicked)
            obox.addWidget(self.objectives)
            lay.addLayout(obox, 2)

        # canal
        if microscope.channel_names:
            cbox = QVBoxLayout()
            cbox.setSpacing(2)
            cbox.addWidget(section_title("Canal"))
            self.channels = Segmented(microscope.channel_names)
            self.channels.selected.connect(self._channel_clicked)
            cbox.addWidget(self.channels)
            lay.addLayout(cbox, 2)

        lay.addWidget(separator(vertical=True))
        self.histogram = HistogramWidget()
        self.histogram.setMinimumWidth(220)
        self.histogram.setMaximumWidth(360)
        lay.addWidget(self.histogram, 2)
        self.pixel_label = label("", "hint", wrap=False)

        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def shutdown(self) -> None:
        """Arrête le rafraîchissement avant la déconnexion du microscope."""
        self._timer.stop()

    # ------------------------------------------------------------- exposition
    def _sync_slider(self, value: float) -> None:
        pos = (math.log10(max(value, 1e-3)) - self._log_lo) / (self._log_hi - self._log_lo) * 1000
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(int(np.clip(pos, 0, 1000)))
        self.exp_slider.blockSignals(False)

    def _slider_changed(self, pos: int) -> None:
        value = 10 ** (self._log_lo + pos / 1000 * (self._log_hi - self._log_lo))
        self.exposure.setValue(round(value, 1))

    def _exposure_changed(self, value: float) -> None:
        cam = self.m.camera
        if cam is not None:
            cam.set_exposure(value)
        self._sync_slider(value)

    def auto_exposure(self) -> None:
        cam = self.m.require_camera()
        try:
            cam.auto_expose()
        except Exception as exc:
            self.message.emit(f"Exposition automatique impossible : {exc}")
            return
        self.message.emit(f"Exposition réglée à {cam.get_exposure():.1f} ms")
        self.refresh()

    # ------------------------------------------------------------- optique
    def _objective_clicked(self, short: str) -> None:
        label_full = next((k for k, v in self._obj_short.items() if v == short), short)
        try:
            self.m.set_objective(label_full, declare=True)
        except Exception as exc:
            self.message.emit(str(exc))
        self.refresh()

    def _channel_clicked(self, name: str) -> None:
        try:
            self.m.apply_channel(name)
            self.message.emit(f"Canal « {name} »")
        except Exception as exc:
            self.message.emit(f"Canal non appliqué : {exc}")
        self.refresh()

    # ------------------------------------------------------------- état
    def show_frame(self, frame: np.ndarray) -> None:
        cam = self.m.camera
        self.histogram.set_image(frame, cam.bit_depth if cam else None)

    def refresh(self) -> None:
        cam = self.m.camera
        if cam is not None and hasattr(self, "exposure") and not self.exposure.hasFocus():
            v = cam.get_exposure()
            if abs(self.exposure.value() - v) > 0.05:
                self.exposure.blockSignals(True)
                self.exposure.setValue(v)
                self.exposure.blockSignals(False)
                self._sync_slider(v)
            elif self.exp_slider.value() == 0 and v > 10**self._log_lo:
                self._sync_slider(v)
        if hasattr(self, "gain") and cam is not None and not self.gain.hasFocus():
            self.gain.blockSignals(True)
            self.gain.setValue(cam.get_gain())
            self.gain.blockSignals(False)
        if hasattr(self, "objectives") and self.m.objective_changer is not None:
            self.objectives.set_value(self._obj_short.get(self.m.objective_changer.get_label()))
        if hasattr(self, "channels"):
            self.channels.set_value(self.m.channel)
