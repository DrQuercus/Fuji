"""Petits composants réutilisables de l'interface."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fuji.analysis.image import Table
from fuji.core.properties import PropertyInfo, PropertyType


def section(title: str, *, hint: str = "") -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 10, 8, 8)
    lay.setSpacing(6)
    if hint:
        lbl = QLabel(hint)
        lbl.setWordWrap(True)
        lbl.setObjectName("hint")
        lay.addWidget(lbl)
    return box, lay


def hint_label(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setObjectName("hint")
    return lbl


def row(*widgets: QWidget, stretch_last: bool = False) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, child in enumerate(widgets):
        lay.addWidget(child, 1 if (stretch_last and i == len(widgets) - 1) else 0)
    if not stretch_last:
        lay.addStretch(1)
    return w


def property_editor(info: PropertyInfo, value: Any, on_change) -> QWidget:  # type: ignore[no-untyped-def]
    """Éditeur adapté au type d'une propriété (construit pour n'importe quel matériel)."""
    if info.read_only:
        lbl = QLabel(f"{value} {info.unit}".strip())
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return lbl
    if info.choices:
        w = QComboBox()
        w.addItems(list(info.choices))
        w.setCurrentText(str(value))
        w.currentTextChanged.connect(on_change)
        return w
    if info.type is PropertyType.BOOL:
        w = QCheckBox()
        w.setChecked(bool(value))
        w.toggled.connect(on_change)
        return w
    if info.type is PropertyType.INT:
        w = QSpinBox()
        lo, hi = info.limits or (-(2**31), 2**31 - 1)
        w.setRange(int(lo), int(hi))
        w.setValue(int(value))
        w.setSuffix(f" {info.unit}" if info.unit else "")
        w.setKeyboardTracking(False)
        w.valueChanged.connect(on_change)
        return w
    if info.type is PropertyType.FLOAT:
        w = QDoubleSpinBox()
        lo, hi = info.limits or (-1e12, 1e12)
        w.setRange(float(lo), float(hi))
        span = hi - lo
        w.setDecimals(3 if span < 10 else 2 if span < 1000 else 1)
        w.setSingleStep(max(span / 100, 0.001) if info.limits else 1.0)
        w.setValue(float(value))
        w.setSuffix(f" {info.unit}" if info.unit else "")
        w.setKeyboardTracking(False)
        w.valueChanged.connect(on_change)
        return w
    w = QLineEdit(str(value))
    w.editingFinished.connect(lambda: on_change(w.text()))
    return w


def _fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        if v != v:  # NaN
            return ""
        a = abs(v)
        if a != 0 and (a >= 1e5 or a < 1e-3):
            return f"{v:.4g}"
        return f"{v:.4f}".rstrip("0").rstrip(".") if a < 100 else f"{v:.2f}"
    return str(v)


class ResultsWidget(QWidget):
    """Tableau de résultats (comme « Results » dans ImageJ), exportable."""

    def __init__(self, table: Table, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = table
        lay = QVBoxLayout(self)
        df = table.data
        summary = "  ·  ".join(f"{k} : {_fmt(v)}" for k, v in table.summary.items())
        title = QLabel(f"<b>{table.name}</b> — {len(df)} ligne(s)")
        lay.addWidget(title)
        if summary:
            lay.addWidget(hint_label(summary))
        grid = QTableWidget(len(df), len(df.columns))
        headers = [f"{c}\n({table.units[c]})" if table.units.get(c) else str(c) for c in df.columns]
        grid.setHorizontalHeaderLabels(headers)
        grid.setVerticalHeaderLabels([str(i) for i in df.index])
        for r, (_, rec) in enumerate(df.iterrows()):
            for c, v in enumerate(rec.tolist()):
                item = QTableWidgetItem(_fmt(v))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                grid.setItem(r, c, item)
        grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        grid.resizeColumnsToContents()
        lay.addWidget(grid, 1)
        export = QPushButton("Exporter en CSV…")
        export.clicked.connect(self._export)
        copy = QPushButton("Copier (pour Excel)")
        copy.clicked.connect(self._copy)
        lay.addWidget(row(export, copy))

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exporter les résultats", "resultats.csv", "CSV (*.csv)")
        if path:
            self.table.to_csv(path)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.table.data.to_csv(sep="\t"))
