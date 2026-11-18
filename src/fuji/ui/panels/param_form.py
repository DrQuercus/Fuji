"""Formulaire des réglages d'un outil d'analyse.

Chaque réglage d'une opération (voir :class:`fuji.analysis.registry.ParamSpec`) reçoit la
commande qui lui convient :

- un nombre borné : **un curseur et un champ**, liés (curseur logarithmique quand la plage
  couvre plusieurs ordres de grandeur) ;
- un nombre sans borne : un champ ;
- un choix : une liste ; une option oui / non : une case ; un texte : une ligne ou un bloc.

Sous la commande, une ligne d'aide quand le réglage en a une ; à côté d'une longueur en
pixels, son **équivalent en µm** si l'image est calibrée. Les réglages « avancés » sont
rangés dans un bloc repliable, présent dans chaque fiche (plus besoin du mode expert pour
les trouver). « Valeurs par défaut » remet tout comme au départ.

Les bornes de certains réglages dépendent de l'image (``Param.span == "image"`` : un seuil
va du pixel le plus sombre au plus clair) : :meth:`ParamForm.set_context` les met à jour.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fuji.analysis.registry import ParamSpec

from ..components import button, label

_SLIDER_STEPS = 1000
#: au-delà de ce rapport max / min, le curseur est logarithmique
_LOG_RATIO = 200.0


def _decimals(lo: float, hi: float, step: float | None) -> int:
    """Nombre de décimales utile pour une plage donnée."""
    if step:
        return max(0, min(6, -math.floor(math.log10(step) + 1e-9)))
    span = abs(hi - lo)
    if span >= 1000:
        return 0
    if span >= 100:
        return 1
    if span >= 1:
        return 2
    return 4


def _handy_top(lo: float, hi: float, default: float, step: float | None, integer: bool) -> float:
    """Haut de la course du curseur : assez près de la valeur habituelle pour que le curseur
    soit précis (un rayon de 2 pixels réglable jusqu'à 1000 serait inutilisable à la souris).
    Le champ, lui, accepte toute la plage."""
    reach = default - lo
    if reach > 0 and (hi - lo) > 20 * reach:
        top = lo + 8 * reach
    elif reach <= 0 and (hi - lo) > 2000 * (step or (1.0 if integer else 0.1)):
        top = lo + 200 * (step or (1.0 if integer else 0.1))
    else:
        return hi
    return float(min(hi, math.ceil(top) if integer or top > 10 else top))


def nice_label(spec: ParamSpec) -> str:
    """Nom affiché d'un réglage : majuscule en tête, unité entre parenthèses."""
    text = spec.label.rstrip("_")
    text = text[:1].upper() + text[1:]
    return f"{text} ({spec.meta.unit})" if spec.meta.unit else text


class SliderSpin(QWidget):
    """Un curseur et un champ numérique liés."""

    changed = Signal()

    def __init__(
        self, integer: bool, lo: float, hi: float, step: float | None, parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.integer = integer
        self._lo, self._hi, self._step = float(lo), float(hi), step
        self._log = False
        self._syncing = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimumWidth(70)
        self.spin: QSpinBox | QDoubleSpinBox = QSpinBox() if integer else QDoubleSpinBox()
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(72)
        self.spin.setMaximumWidth(110)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.set_range(lo, hi)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    # ------------------------------------------------------------------ plage
    def set_slider_span(self, top: float) -> None:
        """Limite la course du curseur à ``top`` (le champ garde toute la plage)."""
        self._top = float(top)
        self.set_range(self._lo, self._full_hi)

    def set_range(self, lo: float, hi: float) -> None:
        if hi <= lo:
            hi = lo + 1
        self._full_hi = float(hi)
        spin_hi = float(hi)
        top = getattr(self, "_top", None)
        if top is not None and lo < top < hi:
            hi = top  # course du curseur ; le champ va jusqu'à spin_hi
        self._lo, self._hi = float(lo), float(hi)
        self._log = lo > 0 and hi / lo >= _LOG_RATIO
        self._syncing = True
        try:
            if self.integer:
                assert isinstance(self.spin, QSpinBox)
                self.spin.setRange(int(round(lo)), int(round(spin_hi)))
                self.spin.setSingleStep(max(1, int(self._step or 1)))
                span = int(round(hi)) - int(round(lo))
                linear = span <= 4 * _SLIDER_STEPS and not self._log
                self.slider.setRange(
                    int(round(lo)) if linear else 0, int(round(hi)) if linear else _SLIDER_STEPS
                )
                self._direct = linear
            else:
                assert isinstance(self.spin, QDoubleSpinBox)
                self.spin.setDecimals(_decimals(lo, hi, self._step))
                self.spin.setRange(lo, spin_hi)
                self.spin.setSingleStep(self._step or (hi - lo) / 100)
                self.slider.setRange(0, _SLIDER_STEPS)
                self._direct = False
        finally:
            self._syncing = False
        self._to_slider(self.value())

    def _slider_value(self) -> float:
        pos = self.slider.value()
        if self._direct:
            return float(pos)
        f = pos / _SLIDER_STEPS
        if self._log:
            return self._lo * (self._hi / self._lo) ** f
        return self._lo + f * (self._hi - self._lo)

    def _to_slider(self, value: float) -> None:
        value = min(max(value, self._lo), self._hi)
        if self._direct:
            pos = int(round(value))
        elif self._log:
            pos = int(round(_SLIDER_STEPS * math.log(value / self._lo) / math.log(self._hi / self._lo)))
        else:
            pos = int(round(_SLIDER_STEPS * (value - self._lo) / (self._hi - self._lo)))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)

    # ------------------------------------------------------------------ valeur
    def value(self) -> float | int:
        return self.spin.value()

    def set_value(self, value: float) -> None:
        self.spin.setValue(int(round(value)) if self.integer else float(value))

    def _from_slider(self, _pos: int) -> None:
        if self._syncing:
            return
        value = self._slider_value()
        if self._step and not self.integer:  # le curseur s'arrête sur les pas du réglage
            value = round(value / self._step) * self._step
        self._syncing = True
        try:
            self.set_value(value)
        finally:
            self._syncing = False
        self.changed.emit()

    def _from_spin(self, value: float) -> None:
        self._to_slider(float(value))
        if not self._syncing:
            self.changed.emit()


class ParamControl(QObject):
    """Un réglage d'outil : nom, étiquette, commande et valeur."""

    changed = Signal()

    def __init__(self, spec: ParamSpec, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.name = spec.name
        self.label = nice_label(spec)
        self._pixel_size: float | None = None
        self.equivalent: QLabel | None = None
        meta, kind = spec.meta, spec.type
        self._kind = "text"
        self.editor: Any
        if kind is bool:
            self._kind = "bool"
            self.editor = QCheckBox(self.label)
            self.editor.setChecked(bool(spec.default))
            self.editor.toggled.connect(self._emit)
        elif meta.choices:
            self._kind = "choice"
            self.editor = QComboBox()
            self.editor.addItems([str(c) for c in meta.choices])
            self.editor.setCurrentText(str(spec.default))
            self.editor.currentTextChanged.connect(self._emit)
        elif kind in (int, float):
            self._kind = "number"
            integer = kind is int
            lo = meta.min if meta.min is not None else (-1e9 if not integer else -(10**9))
            hi = meta.max if meta.max is not None else (1e9 if not integer else 10**9)
            bounded = meta.min is not None and meta.max is not None
            if bounded or meta.span == "image":
                self.editor = SliderSpin(integer, lo if bounded else 0.0, hi if bounded else 255.0, meta.step)
                if bounded:
                    self.editor.set_slider_span(_handy_top(lo, hi, float(spec.default), meta.step, integer))
                self.editor.set_value(spec.default)
                self.editor.changed.connect(self._emit)
            else:
                self.editor = QSpinBox() if integer else QDoubleSpinBox()
                self.editor.setKeyboardTracking(False)
                if integer:
                    self.editor.setRange(int(lo), int(hi))
                    self.editor.setSingleStep(max(1, int(meta.step or 1)))
                else:
                    self.editor.setDecimals(3 if abs(float(spec.default)) < 10 else 2)
                    self.editor.setRange(float(lo), float(hi))
                    if meta.step:
                        self.editor.setSingleStep(meta.step)
                self.editor.setValue(spec.default)
                self.editor.valueChanged.connect(self._emit)
            if meta.unit == "px":
                self.equivalent = label("", "faint", wrap=False)
        elif meta.multiline:
            self._kind = "block"
            self.editor = QPlainTextEdit(str(spec.default))
            self.editor.setMinimumHeight(90)
            self.editor.textChanged.connect(self._emit)
        else:
            self.editor = QLineEdit(str(spec.default))
            self.editor.editingFinished.connect(self._emit)
        if meta.description:
            self.editor.setToolTip(meta.description)

    def _emit(self, *_: Any) -> None:
        self._update_equivalent()
        self.changed.emit()

    # ------------------------------------------------------------------ valeur
    @property
    def value(self) -> Any:
        if self._kind == "bool":
            return self.editor.isChecked()
        if self._kind == "choice":
            return self.editor.currentText()
        if self._kind == "number":
            v = self.editor.value()
            return int(v) if self.spec.type is int else float(v)
        if self._kind == "block":
            return self.editor.toPlainText()
        return self.editor.text()

    @value.setter
    def value(self, value: Any) -> None:
        if self._kind == "bool":
            self.editor.setChecked(bool(value))
        elif self._kind == "choice":
            if self.editor.findText(str(value)) < 0:
                raise ValueError(f"{self.label} : « {value} » n'est pas un choix possible")
            self.editor.setCurrentText(str(value))
        elif self._kind == "number":
            if isinstance(self.editor, SliderSpin):
                self.editor.set_value(float(value))
            else:
                self.editor.setValue(int(value) if self.spec.type is int else float(value))
        elif self._kind == "block":
            self.editor.setPlainText(str(value))
        else:
            self.editor.setText(str(value))
        self._update_equivalent()

    def reset(self) -> None:
        self.value = self._default

    @property
    def _default(self) -> Any:
        return getattr(self, "_start", self.spec.default)

    @property
    def is_default(self) -> bool:
        return self.value == self._default

    # ----------------------------------------------------------------- contexte
    def set_context(
        self, pixel_size_um: float | None, data_range: tuple[float, float] | None, integer_data: bool
    ) -> None:
        """Adapte le réglage à l'image choisie : équivalent en µm, bornes d'un seuil."""
        self._pixel_size = pixel_size_um
        meta = self.spec.meta
        if meta.span == "image" and data_range is not None and isinstance(self.editor, SliderSpin):
            lo, hi = data_range
            untouched = self.is_default
            self.editor.blockSignals(True)
            if integer_data and not self.editor.integer:
                self.editor._step = 1.0
            # le curseur parcourt les valeurs de l'image ; le champ accepte aussi 0 et les
            # valeurs au-delà (remplir en blanc, seuil hors de l'image…)
            self.editor._top = hi
            self.editor.set_range(min(lo, 0.0), max(hi * 1.0001 + 1, 65535.0 if integer_data else hi + 1))
            start = {"min": lo, "max": hi, "milieu": (lo + hi) / 2}.get(meta.start)
            if start is not None:
                self._start = round(start) if integer_data else start
                if untouched:  # un réglage déjà modifié à la main n'est pas remplacé
                    self.editor.set_value(self._start)
            self.editor.blockSignals(False)
        self._update_equivalent()

    def _update_equivalent(self) -> None:
        if self.equivalent is None:
            return
        px = self._pixel_size
        if not px:
            self.equivalent.setText("")
            return
        um = float(self.value) * px
        text = f"{um:.3g} µm" if um < 1000 else f"{um / 1000:.3g} mm"
        self.equivalent.setText("≈ " + text.replace(".", ","))


class ParamForm(QWidget):
    """Les réglages d'un outil. S'utilise comme une liste de :class:`ParamControl`
    (``name``, ``label``, ``value``)."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controls: list[ParamControl] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.main = QFormLayout()
        self.main.setContentsMargins(0, 0, 0, 0)
        self.main.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        lay.addLayout(self.main)
        self.advanced_btn = button("Réglages avancés", "sliders", object_name="ghost")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        lay.addWidget(self.advanced_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_box = QWidget()
        self.advanced = QFormLayout(self.advanced_box)
        self.advanced.setContentsMargins(10, 0, 0, 0)
        self.advanced.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        lay.addWidget(self.advanced_box)
        self.reset_btn = button(
            "Valeurs par défaut",
            "refresh",
            object_name="ghost",
            tooltip="Remet tous les réglages de cet outil",
        )
        self.reset_btn.clicked.connect(self.reset)
        lay.addWidget(self.reset_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.build([], False)

    def __iter__(self) -> Iterator[ParamControl]:
        return iter(self.controls)

    def __len__(self) -> int:
        return len(self.controls)

    # ------------------------------------------------------------ construction
    def build(self, specs: list[ParamSpec], expert: bool = False) -> None:
        """Construit les commandes de ces réglages (``expert`` : bloc avancé déplié)."""
        for form in (self.main, self.advanced):
            while form.rowCount():
                form.removeRow(0)
        for control in self.controls:
            control.deleteLater()
        self.controls = []
        advanced = 0
        for spec in specs:
            control = ParamControl(spec, self)
            control.changed.connect(self._changed)
            self.controls.append(control)
            form = self.advanced if spec.meta.advanced else self.main
            advanced += bool(spec.meta.advanced)
            self._add_row(form, control)
        self.advanced_btn.setVisible(advanced > 0)
        self.advanced_btn.setText(f"Réglages avancés ({advanced})")
        self.advanced_btn.blockSignals(True)
        self.advanced_btn.setChecked(expert and advanced > 0)
        self.advanced_btn.blockSignals(False)
        self.advanced_box.setVisible(expert and advanced > 0)
        self.reset_btn.setVisible(bool(specs))
        self._update_reset()

    def _add_row(self, form: QFormLayout, control: ParamControl) -> None:
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(1)
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        line.addWidget(control.editor, 1)
        if control.equivalent is not None:
            line.addWidget(control.equivalent)
        col.addLayout(line)
        if control.spec.meta.description:
            col.addWidget(label(control.spec.meta.description, "faint"))
        if isinstance(control.editor, QCheckBox):
            form.addRow(box)  # la case porte déjà son nom
        else:
            form.addRow(control.label, box)

    def _toggle_advanced(self, on: bool) -> None:
        self.advanced_box.setVisible(on)

    def _changed(self) -> None:
        self._update_reset()
        self.changed.emit()

    def _update_reset(self) -> None:
        self.reset_btn.setEnabled(any(not c.is_default for c in self.controls))

    # ------------------------------------------------------------------ valeurs
    def values(self) -> dict[str, Any]:
        return {c.name: c.value for c in self.controls}

    def set_values(self, values: dict[str, Any]) -> None:
        """Applique des valeurs enregistrées ; celles qui ne conviennent plus sont ignorées."""
        for control in self.controls:
            if control.name in values:
                try:
                    control.value = values[control.name]
                except (TypeError, ValueError):
                    pass
        self._update_reset()

    def reset(self) -> None:
        for control in self.controls:
            control.blockSignals(True)
            control.reset()
            control.blockSignals(False)
        self._changed()

    def set_context(
        self,
        pixel_size_um: float | None = None,
        data_range: tuple[float, float] | None = None,
        integer_data: bool = True,
    ) -> None:
        for control in self.controls:
            control.set_context(pixel_size_um, data_range, integer_data)
        self._update_reset()


__all__ = ["ParamControl", "ParamForm", "SliderSpin", "nice_label"]
