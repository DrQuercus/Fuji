"""Panneau « Microscope » : caméra, optique, éclairage, platine, mise au point.

Le panneau se construit d'après les capacités du microscope : pas de platine
motorisée → pas de commandes de platine ; objectifs manuels → on *indique*
l'objectif en place au lieu de le commander.
"""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fuji.core.devices import LightSource
from fuji.core.microscope import Microscope

from ..widgets import hint_label, row, section


class MicroscopePanel(QWidget):
    message = Signal(str)

    def __init__(self, microscope: Microscope, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.m = microscope
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        prof = microscope.profile
        title = QLabel(f"<b>{prof.name}</b>")
        title.setWordWrap(True)
        title.setToolTip(prof.description)
        lay.addWidget(title)
        caps = microscope.capabilities()
        manual = [d.label for d in microscope.devices.values() if d.manual]
        if manual:
            lay.addWidget(hint_label("Éléments manuels : " + ", ".join(manual)))
        if caps["camera"]:
            lay.addWidget(self._camera_section(caps))
        if prof.objectives or prof.channels:
            lay.addWidget(self._optics_section())
        lights = microscope.devices_of(LightSource)
        if lights:
            lay.addWidget(self._lights_section(lights))
        if caps["xy_motorized"]:
            lay.addWidget(self._stage_section())
        if caps["focus_motorized"]:
            lay.addWidget(self._focus_section())
        lay.addStretch(1)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def shutdown(self) -> None:
        """Arrête le rafraîchissement avant la déconnexion du microscope."""
        self._timer.stop()

    # ------------------------------------------------------------- caméra
    def _camera_section(self, caps: dict[str, bool]) -> QWidget:
        cam = self.m.require_camera()
        box, lay = section("Caméra")
        h, w = cam.sensor_shape()
        kind = "couleur" if cam.is_color else "monochrome"
        lay.addWidget(hint_label(f"{cam.label} — {w}×{h}, {kind}, {cam.bit_depth} bits"))
        grid = QGridLayout()
        lo, hi = cam.exposure_limits()
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(lo, hi)
        self.exposure.setDecimals(2)
        self.exposure.setSuffix(" ms")
        self.exposure.setKeyboardTracking(False)
        self.exposure.setValue(cam.get_exposure())
        self.exposure.valueChanged.connect(lambda v: cam.set_exposure(v))
        # curseur logarithmique : plage confortable de 0,1 ms à 10 s
        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(0, 1000)
        self._log_lo, self._log_hi = math.log10(max(lo, 0.05)), math.log10(min(hi, 10_000))
        self.exp_slider.valueChanged.connect(self._slider_to_exposure)
        auto = QPushButton("Auto")
        auto.setToolTip("Exposition automatique : vise 70 % de la dynamique sur l'image actuelle")
        auto.clicked.connect(self._auto_exposure)
        grid.addWidget(QLabel("Exposition"), 0, 0)
        grid.addWidget(self.exposure, 0, 1)
        grid.addWidget(auto, 0, 2)
        grid.addWidget(self.exp_slider, 1, 0, 1, 3)
        if caps["gain"]:
            glo, ghi = cam.gain_limits()
            self.gain = QDoubleSpinBox()
            self.gain.setRange(glo, ghi)
            self.gain.setDecimals(1 if ghi < 100 else 0)
            self.gain.setSuffix(f" {cam.gain_unit}")
            self.gain.setKeyboardTracking(False)
            self.gain.setValue(cam.get_gain())
            self.gain.valueChanged.connect(lambda v: cam.set_gain(v))
            grid.addWidget(QLabel("Gain"), 2, 0)
            grid.addWidget(self.gain, 2, 1)
        lay.addLayout(grid)
        if caps["auto_white_balance"]:
            wb = QPushButton("Balance des blancs (sur une zone vide)")
            wb.clicked.connect(self._white_balance)
            lay.addWidget(wb)
        self._sync_slider()
        return box

    def _sync_slider(self) -> None:
        v = max(self.exposure.value(), 1e-3)
        pos = (math.log10(v) - self._log_lo) / (self._log_hi - self._log_lo) * 1000
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(int(np.clip(pos, 0, 1000)))
        self.exp_slider.blockSignals(False)

    def _slider_to_exposure(self, pos: int) -> None:
        value = 10 ** (self._log_lo + pos / 1000 * (self._log_hi - self._log_lo))
        self.exposure.setValue(round(value, 2))

    def _auto_exposure(self) -> None:
        cam = self.m.require_camera()
        try:
            cam.auto_expose()
        except Exception as exc:
            self.message.emit(f"Exposition automatique impossible : {exc}")
            return
        self.message.emit(f"Exposition réglée à {cam.get_exposure():.1f} ms")
        self.refresh()

    def _white_balance(self) -> None:
        cam = self.m.require_camera()
        try:
            cam.auto_white_balance()
        except Exception as exc:
            self.message.emit(f"Balance des blancs impossible : {exc}")
            return
        self.message.emit("Balance des blancs effectuée")

    # ------------------------------------------------------------- optique
    def _optics_section(self) -> QWidget:
        m = self.m
        box, lay = section("Optique")
        grid = QGridLayout()
        r = 0
        if m.profile.objectives and m.objective_changer is not None:
            self.objective = QComboBox()
            self.objective.addItems(m.objective_changer.labels)
            self.objective.setCurrentText(m.objective_changer.get_label())
            self.objective.currentTextChanged.connect(self._on_objective)
            label = "Objectif" + (" (indiquer)" if m.objective_changer.manual else "")
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(self.objective, r, 1)
            r += 1
        self.pixel_label = QLabel()
        grid.addWidget(QLabel("Pixel"), r, 0)
        grid.addWidget(self.pixel_label, r, 1)
        r += 1
        if m.profile.channels:
            self.channel = QComboBox()
            self.channel.addItems(m.channel_names)
            if m.channel:
                self.channel.setCurrentText(m.channel)
            apply = QPushButton("Appliquer")
            apply.setToolTip("Règle cubes, lampes et exposition pour ce canal")
            apply.clicked.connect(lambda: self._apply_channel(self.channel.currentText()))
            grid.addWidget(QLabel("Canal"), r, 0)
            grid.addWidget(row(self.channel, apply, stretch_last=False), r, 1)
        lay.addLayout(grid)
        return box

    def _on_objective(self, label: str) -> None:
        try:
            self.m.set_objective(label, declare=True)
        except Exception as exc:  # ex. annulation d'une consigne
            self.message.emit(str(exc))
        self.refresh()

    def _apply_channel(self, name: str) -> None:
        try:
            self.m.apply_channel(name)
            self.message.emit(f"Canal « {name} » appliqué")
        except Exception as exc:
            self.message.emit(f"Canal non appliqué : {exc}")
        self.refresh()

    # ------------------------------------------------------------- éclairage
    def _lights_section(self, lights: list[LightSource]) -> QWidget:
        box, lay = section("Éclairage")
        self._light_widgets: list[tuple[LightSource, QCheckBox, QSlider | None]] = []
        for light in lights:
            cb = QCheckBox(light.label + (" (manuel)" if light.manual else ""))
            cb.setChecked(light.is_on())
            cb.toggled.connect(lambda on, d=light: self._set_light(d, on=on))
            slider = None
            widgets: list[QWidget] = [cb]
            if light.supports_intensity:
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(int(light.get_intensity()))
                slider.setFixedWidth(110)
                slider.sliderReleased.connect(
                    lambda d=light, s=slider: self._set_light(d, intensity=s.value())
                )
                widgets.append(slider)
            lay.addWidget(row(*widgets))
            self._light_widgets.append((light, cb, slider))
        return box

    def _set_light(self, dev: LightSource, *, on: bool | None = None, intensity: float | None = None) -> None:
        if dev.manual:  # l'utilisateur déclare ce qu'il a fait
            dev.declare(on=on, intensity=intensity)  # type: ignore[attr-defined]
            return
        if on is not None:
            dev.set_on(on)
        if intensity is not None:
            dev.set_intensity(intensity)

    # ------------------------------------------------------------- platine
    def _stage_section(self) -> QWidget:
        stage = self.m.xy_stage
        assert stage is not None
        box, lay = section("Platine XY")
        self.xy_label = QLabel()
        lay.addWidget(self.xy_label)
        self.xy_step = QDoubleSpinBox()
        self.xy_step.setRange(0.1, 10_000)
        self.xy_step.setValue(100)
        self.xy_step.setSuffix(" µm")
        grid = QGridLayout()
        for text, dx, dy, r, c in (
            ("↑", 0, -1, 0, 1),
            ("←", -1, 0, 1, 0),
            ("→", 1, 0, 1, 2),
            ("↓", 0, 1, 2, 1),
        ):
            b = QPushButton(text)
            b.setFixedWidth(36)
            b.clicked.connect(
                lambda _=False, dx=dx, dy=dy: stage.move_by(
                    dx * self.xy_step.value(), dy * self.xy_step.value()
                )
            )
            grid.addWidget(b, r, c)
        grid.addWidget(self.xy_step, 1, 1)
        lay.addLayout(grid)
        return box

    # ------------------------------------------------------------- focus
    def _focus_section(self) -> QWidget:
        focus = self.m.focus
        assert focus is not None
        box, lay = section("Mise au point")
        self.z_label = QLabel()
        self.z_step = QDoubleSpinBox()
        self.z_step.setRange(0.01, 1000)
        self.z_step.setValue(1.0)
        self.z_step.setSuffix(" µm")
        up = QPushButton("▲")
        down = QPushButton("▼")
        up.clicked.connect(lambda: focus.move_by(self.z_step.value()))
        down.clicked.connect(lambda: focus.move_by(-self.z_step.value()))
        lay.addWidget(row(self.z_label, down, up, self.z_step))
        return box

    # ------------------------------------------------------------- rafraîchissement
    def refresh(self) -> None:
        m = self.m
        px, src = m.pixel_size_um(), m.pixel_size_source()
        if hasattr(self, "pixel_label"):
            self.pixel_label.setText(f"{px:.4f} µm ({src})" if px else "non calibré")
        cam = m.camera
        if cam is not None and hasattr(self, "exposure") and not self.exposure.hasFocus():
            self.exposure.blockSignals(True)
            self.exposure.setValue(cam.get_exposure())
            self.exposure.blockSignals(False)
            self._sync_slider()
        if hasattr(self, "objective") and m.objective_changer is not None:
            current = m.objective_changer.get_label()
            if self.objective.currentText() != current:
                self.objective.blockSignals(True)
                self.objective.setCurrentText(current)
                self.objective.blockSignals(False)
        if (
            hasattr(self, "channel")
            and m.channel
            and self.channel.currentText() != m.channel
            and not self.channel.view().isVisible()
        ):
            self.channel.setCurrentText(m.channel)
        for light, cb, slider in getattr(self, "_light_widgets", []):
            cb.blockSignals(True)
            cb.setChecked(light.is_on())
            cb.blockSignals(False)
            if slider is not None and not slider.isSliderDown():
                slider.blockSignals(True)
                slider.setValue(int(light.get_intensity()))
                slider.blockSignals(False)
        if hasattr(self, "xy_label") and m.xy_stage is not None:
            x, y = m.xy_stage.get_position()
            self.xy_label.setText(f"X {x:.1f} µm   Y {y:.1f} µm")
        if hasattr(self, "z_label") and m.focus is not None:
            self.z_label.setText(f"Z {m.focus.get_position():.2f} µm")
