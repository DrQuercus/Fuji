"""Formulaire généré à partir des options déclarées par un pilote."""

from __future__ import annotations

from typing import Any

import yaml
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fuji.core.profile import load_yaml
from fuji.drivers import DriverInfo, OptionSpec

from ..components import ListEditor, PathEdit, ToggleSwitch, hint

AUTO = "(automatique)"


class OptionForm(QWidget):
    """Construit un champ par option ; ``values()`` renvoie les options saisies."""

    changed = Signal()

    def __init__(
        self,
        info: DriverInfo,
        options: dict[str, Any],
        devices_by_kind: dict[str | None, list[str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.info = info
        self._original = dict(options)
        self._widgets: dict[str, tuple[OptionSpec, QWidget]] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        basic = QFormLayout()
        basic.setVerticalSpacing(8)
        lay.addLayout(basic)
        adv_specs = [o for o in info.options if o.advanced]
        adv_host = QWidget()
        adv = QFormLayout(adv_host)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setVerticalSpacing(8)
        for spec in info.options:
            w = self._make(spec, options.get(spec.name), devices_by_kind)
            w.setToolTip(spec.help)
            self._widgets[spec.name] = (spec, w)
            title = spec.title + (
                f" ({spec.unit})"
                if spec.unit and spec.type in ("int", "float") and spec.default is None
                else ""
            )
            (adv if spec.advanced else basic).addRow(title, w)
            if spec.help and spec.type in ("list", "mapping", "device", "path", "dir"):
                (adv if spec.advanced else basic).addRow("", hint(spec.help))
        known = {o.name for o in info.options}
        extra = {k: v for k, v in options.items() if k not in known}
        self.extra = QPlainTextEdit(yaml.safe_dump(extra, allow_unicode=True) if extra else "")
        self.extra.setPlaceholderText("autres options, en YAML (clé: valeur)")
        self.extra.setFixedHeight(70)
        self.extra.textChanged.connect(self.changed.emit)
        if not info.options:
            basic.addRow(hint("Ce pilote n'a pas d'option."))
        if adv_specs or extra:
            self.adv_toggle = QPushButton("Options avancées ▸")
            self.adv_toggle.setObjectName("ghost")
            self.adv_toggle.setCheckable(True)
            self.adv_toggle.toggled.connect(self._toggle_adv)
            lay.addWidget(self.adv_toggle)
            adv.addRow("Autres options", self.extra)
            lay.addWidget(adv_host)
            adv_host.setVisible(bool(extra))
            self.adv_toggle.setChecked(bool(extra))
        self._adv_host = adv_host

    def _toggle_adv(self, on: bool) -> None:
        self._adv_host.setVisible(on)
        self.adv_toggle.setText("Options avancées ▾" if on else "Options avancées ▸")

    # --------------------------------------------------------------- widgets
    def _make(self, spec: OptionSpec, value: Any, devices_by_kind: dict[str | None, list[str]]) -> QWidget:
        v = spec.default if value is None else value
        t = spec.type
        if t == "bool":
            w = ToggleSwitch()
            w.setChecked(bool(v))
            w.toggled.connect(self.changed.emit)
            return w
        if t == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in spec.choices])
            if v is not None and str(v) not in spec.choices:
                w.addItem(str(v))
            w.setCurrentText(str(v) if v is not None else "")
            w.currentTextChanged.connect(self.changed.emit)
            return w
        if t in ("int", "float") and spec.default is not None:
            w = QSpinBox() if t == "int" else QDoubleSpinBox()
            lo = spec.min if spec.min is not None else -1e9
            hi = spec.max if spec.max is not None else 1e9
            if t == "int":
                w.setRange(int(lo), int(hi))
            else:
                w.setRange(lo, hi)
                w.setDecimals(3 if hi <= 50 else 2)
            if spec.unit:
                w.setSuffix(f" {spec.unit}")
            w.setValue(v if t == "float" else int(v))
            w.valueChanged.connect(self.changed.emit)
            return w
        if t in ("int", "float"):  # facultatif : vide = non défini
            w = QLineEdit("" if value is None else f"{value:g}" if isinstance(value, float) else str(value))
            w.setPlaceholderText("non défini")
            w.textChanged.connect(self.changed.emit)
            return w
        if t in ("path", "dir"):
            w = PathEdit("dir" if t == "dir" else "file", spec.file_filter)
            w.setText("" if v is None else str(v))
            w.changed.connect(lambda *_: self.changed.emit())
            return w
        if t == "list":
            w = ListEditor([str(x) for x in (value or [])])
            w.changed.connect(lambda *_: self.changed.emit())
            return w
        if t == "device":
            w = QComboBox()
            w.addItem(AUTO)
            w.addItems(devices_by_kind.get(spec.device_kind, []))
            if value:
                if w.findText(str(value)) < 0:
                    w.addItem(str(value))
                w.setCurrentText(str(value))
            w.currentTextChanged.connect(self.changed.emit)
            return w
        if t == "mapping":
            w = QPlainTextEdit(yaml.safe_dump(value, allow_unicode=True, sort_keys=False) if value else "")
            w.setPlaceholderText("YAML")
            w.setFixedHeight(90)
            w.textChanged.connect(self.changed.emit)
            return w
        w = QLineEdit("" if v is None else str(v))
        if spec.help:
            w.setPlaceholderText(spec.help)
        w.textChanged.connect(self.changed.emit)
        return w

    # --------------------------------------------------------------- valeurs
    def values(self) -> dict[str, Any]:
        """Options saisies (les champs vides sont omis)."""
        out: dict[str, Any] = {}
        for name, (spec, w) in self._widgets.items():
            value: Any
            if isinstance(w, ToggleSwitch):
                value = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                value = w.value()
            elif isinstance(w, ListEditor):
                value = w.items() or None
            elif isinstance(w, PathEdit):
                value = w.text().strip() or None
            elif isinstance(w, QPlainTextEdit):
                text = w.toPlainText().strip()
                value = load_yaml(text) if text else None
            elif isinstance(w, QComboBox):
                value = w.currentText()
                if spec.type == "device" and value == AUTO:
                    value = None
                elif isinstance(spec.default, int) and value.lstrip("-").isdigit():
                    value = int(value)
            elif isinstance(w, QLineEdit):
                text = w.text().strip()
                if not text:
                    value = None
                elif spec.type == "int":
                    value = int(float(text.replace(",", ".")))
                elif spec.type == "float":
                    value = float(text.replace(",", "."))
                else:
                    value = text
            else:  # pragma: no cover
                value = None
            if value is None or value == "":
                continue
            if name not in self._original and value == spec.default:
                continue  # valeur par défaut non saisie : le fichier reste minimal
            out[name] = value
        extra = self.extra.toPlainText().strip()
        if extra:
            parsed = load_yaml(extra)
            if isinstance(parsed, dict):
                out.update(parsed)
        return out
