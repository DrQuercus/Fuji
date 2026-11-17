"""Barre rapide sous l'image : exposition, gain, objectif, canal, histogramme."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLayout, QSizePolicy, QSlider, QVBoxLayout, QWidget

from fuji.core.microscope import Microscope

from .components import button, choice, section_title
from .flow import FlowLayout
from .histogram import HistogramWidget


def _group(title: str, content: QWidget | QLayout) -> QWidget:
    """Un réglage de la barre : son titre au-dessus, la commande en dessous."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(section_title(title))
    if isinstance(content, QLayout):
        lay.addLayout(content)
    else:
        lay.addWidget(content)
    return box


class QuickBar(QWidget):
    """Réglages de prise de vue toujours à portée de main.

    Les groupes (exposition, gain, objectif, canal, histogramme) se rangent sur une ligne
    quand la fenêtre est large, et passent à la ligne sur un petit écran."""

    message = Signal(str)

    def __init__(self, microscope: Microscope, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("QuickBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.m = microscope
        lay = FlowLayout(self, spacing=16, line_spacing=8)
        #: réglages qui doublent le panneau « Contrôle » (masqués quand il est ouvert)
        self._settings_groups: list[QWidget] = []
        lay.setContentsMargins(14, 8, 14, 8)
        cam = microscope.camera
        caps = microscope.capabilities()

        # exposition
        if cam is not None:
            row = QHBoxLayout()
            row.setSpacing(6)
            lo, hi = cam.exposure_limits()
            self._log_lo, self._log_hi = math.log10(max(lo, 0.05)), math.log10(min(hi, 10_000))
            self.exp_slider = QSlider(Qt.Orientation.Horizontal)
            self.exp_slider.setRange(0, 1000)
            self.exp_slider.setMinimumWidth(90)
            self.exp_slider.valueChanged.connect(self._slider_changed)
            self.exposure = QDoubleSpinBox()
            self.exposure.setRange(lo, hi)
            self.exposure.setDecimals(1)
            self.exposure.setSuffix(" ms")
            self.exposure.setKeyboardTracking(False)
            self.exposure.setFixedWidth(92)
            self.exposure.valueChanged.connect(self._exposure_changed)
            auto = button("Auto", "wand", tooltip="Exposition automatique (70 % de la dynamique)")
            auto.clicked.connect(self.auto_exposure)
            row.addWidget(self.exp_slider, 1)
            row.addWidget(self.exposure)
            row.addWidget(auto)
            self._add_setting(lay, _group("Exposition", row), stretch=3)
            if caps["gain"]:
                glo, ghi = cam.gain_limits()
                self.gain = QDoubleSpinBox()
                self.gain.setRange(glo, ghi)
                self.gain.setDecimals(1 if ghi < 100 else 0)
                self.gain.setSuffix(f" {cam.gain_unit}")
                self.gain.setKeyboardTracking(False)
                self.gain.setFixedWidth(88)
                self.gain.valueChanged.connect(lambda v: cam.set_gain(v))
                self._add_setting(lay, _group("Gain", self.gain))

        # objectif
        changer = microscope.objective_changer
        if changer is not None and microscope.profile.objectives:
            title = "Objectif" + (" (indiquer)" if changer.manual else "")
            short = {lbl: lbl.split("/")[0].split(" ")[0] for lbl in changer.labels}
            # deux objectifs de même grossissement (10x/0.25 et 10x/0.30 Ph) : nom complet,
            # sinon l'un des deux boutons serait inutilisable
            counts: dict[str, int] = {}
            for text in short.values():
                counts[text] = counts.get(text, 0) + 1
            short = {lbl: (text if counts[text] == 1 else lbl) for lbl, text in short.items()}
            self._obj_short = short
            self.objectives = choice(list(short.values()), max_buttons=6)
            self.objectives.set_options(list(short.values()), {v: k for k, v in short.items()})
            self.objectives.selected.connect(self._objective_clicked)
            self._add_setting(lay, _group(title, self.objectives))

        # canal
        if microscope.channel_names:
            self.channels = choice(microscope.channel_names, max_buttons=4)
            self.channels.selected.connect(self._channel_clicked)
            self._add_setting(lay, _group("Canal", self.channels))

        if caps["xy_motorized"] or caps["focus_motorized"]:
            stop = button("Stop", "stop", tooltip="Arrêt d'urgence : interrompt tous les moteurs")
            stop.setObjectName("danger")
            stop.clicked.connect(self.stop_motion)
            lay.addWidget(_group("Moteurs", stop))
        self.histogram = HistogramWidget()
        self.histogram.setMinimumWidth(180)
        self.histogram.setMaximumWidth(360)
        lay.addWidget(self.histogram, stretch=2)

        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def _add_setting(self, lay: FlowLayout, group: QWidget, stretch: int = 0) -> None:
        lay.addWidget(group, stretch=stretch)
        self._settings_groups.append(group)

    def set_compact(self, compact: bool) -> None:
        """Seulement l'histogramme (et l'arrêt des moteurs) : quand le panneau « Contrôle »
        est ouvert, il donne déjà exposition, gain, objectif et canal, et l'image a plus de place."""
        for group in self._settings_groups:
            group.setVisible(not compact)
        self.updateGeometry()

    def shutdown(self) -> None:
        """Arrête le rafraîchissement avant la déconnexion du microscope."""
        self._timer.stop()

    def stop_motion(self) -> None:
        """Arrêt d'urgence, hors du thread de l'interface (elle doit rester disponible)."""
        import threading

        threading.Thread(target=self.m.stop_motion, name="fuji-stop", daemon=True).start()
        self.message.emit("Arrêt d'urgence envoyé aux moteurs")

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
        auto_expose_async(self.m, self.message.emit, self.refresh, self)

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


def auto_expose_async(microscope: Microscope, message: Any, done: Any, owner: QWidget) -> None:
    """Exposition automatique sans figer l'interface : la caméra prend plusieurs images
    (jusqu'à seize poses), dans un fil de travail. Un seul réglage à la fois."""
    from superqt.utils import create_worker

    cam = microscope.camera
    if cam is None or getattr(microscope, "_auto_exposing", False):
        return
    microscope._auto_exposing = True  # type: ignore[attr-defined]
    message("Exposition automatique…")
    worker = create_worker(cam.auto_expose, _start_thread=False, _ignore_errors=True)

    def finished() -> None:
        microscope._auto_exposing = False  # type: ignore[attr-defined]

    def returned(value: float) -> None:
        message(f"Exposition réglée à {value:.1f} ms")
        try:
            done()
        except RuntimeError:  # panneau détruit entre-temps
            pass

    worker.returned.connect(returned)
    worker.errored.connect(lambda exc: message(f"Exposition automatique impossible : {exc}"))
    worker.finished.connect(finished)
    owner._auto_worker = worker  # type: ignore[attr-defined]  # garde une référence
    worker.start()
