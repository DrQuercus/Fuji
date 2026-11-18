"""Petits composants réutilisables de l'interface."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen
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
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fuji.analysis.image import Plot, Table
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


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Graduations « rondes » (1, 2, 5 × 10ⁿ) couvrant l'intervalle."""
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    power = 10.0 ** np.floor(np.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(first, hi + step * 1e-9, step)]


class PlotWidget(QWidget):
    """Graphique simple (courbes, barres ou points) pour les profils et les histogrammes."""

    COLORS = ("#4f8cff", "#f2a93b", "#2fbf71", "#f05252", "#b07cff", "#14b8a6")

    def __init__(self, table: Table, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from .theme import ThemeManager

        assert table.plot is not None
        self.spec: Plot = table.plot
        df = table.data
        self.x = df[self.spec.x].to_numpy(dtype=np.float64)
        self.series = [(name, df[name].to_numpy(dtype=np.float64)) for name in self.spec.y if name in df]
        self.log_y = False
        self.setMinimumSize(280, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip("Double-clic : échelle verticale linéaire / logarithmique")
        ThemeManager.get().changed.connect(self._on_theme)  # méthode : coupée avec le widget

    def _on_theme(self, *_: object) -> None:
        self.update()

    def mouseDoubleClickEvent(self, _event: object) -> None:
        self.log_y = not self.log_y
        self.update()

    def _transform(self, y: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(y, 0.0) + 1.0) if self.log_y else y

    def paintEvent(self, _event: object) -> None:
        from .theme import palette

        p = palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(p.surface))
        font = painter.font()
        font.setPixelSize(max(10, p.font_size - 2))
        painter.setFont(font)
        area = QRectF(self.rect()).adjusted(74, 12, -14, -38)
        finite = [np.isfinite(y) for _, y in self.series]
        if not self.series or not any(f.any() for f in finite) or area.width() < 20 or area.height() < 20:
            painter.end()
            return
        ys = [self._transform(y) for _, y in self.series]
        x_lo, x_hi = float(np.nanmin(self.x)), float(np.nanmax(self.x))
        y_lo = min(float(np.nanmin(y)) for y in ys)
        y_hi = max(float(np.nanmax(y)) for y in ys)
        if self.spec.kind == "barres":
            y_lo = min(y_lo, 0.0)
            half = (x_hi - x_lo) / max(len(self.x) - 1, 1) / 2.0
            x_lo, x_hi = x_lo - half, x_hi + half
        if x_hi <= x_lo:
            x_hi = x_lo + 1.0
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0

        def to_x(v: Any) -> Any:
            return area.left() + (v - x_lo) / (x_hi - x_lo) * area.width()

        def to_y(v: Any) -> Any:
            return area.bottom() - (v - y_lo) / (y_hi - y_lo) * area.height()

        # grille et graduations
        painter.setPen(QPen(QColor(p.border), 1))
        painter.drawRect(area)
        align_right = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        align_top = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        for tick in _nice_ticks(y_lo, y_hi):
            y = to_y(tick)
            painter.setPen(QPen(QColor(p.border), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor(p.muted))
            value = 10.0**tick - 1.0 if self.log_y else tick
            painter.drawText(QRectF(18, y - 9, area.left() - 24, 18), align_right, f"{float(value):.5g}")
        for tick in _nice_ticks(x_lo, x_hi, 6):
            x = to_x(tick)
            painter.setPen(QPen(QColor(p.border), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.setPen(QColor(p.muted))
            painter.drawText(QRectF(x - 40, area.bottom() + 3, 80, 16), align_top, f"{float(tick):.5g}")
        painter.setPen(QColor(p.text))
        painter.drawText(
            QRectF(area.left(), area.bottom() + 19, area.width(), 16),
            align_top,
            self.spec.xlabel or self.spec.x,
        )
        painter.save()
        painter.translate(10, area.center().y())
        painter.rotate(-90)
        ylabel = (self.spec.ylabel or "") + (" (log)" if self.log_y else "")
        painter.drawText(
            QRectF(-area.height() / 2, -8, area.height(), 16), int(Qt.AlignmentFlag.AlignCenter), ylabel
        )
        painter.restore()

        # données
        painter.setClipRect(area)
        for k, ((_name, _raw), y) in enumerate(zip(self.series, ys, strict=True)):
            color = QColor(self.COLORS[k % len(self.COLORS)])
            ok = np.isfinite(y) & np.isfinite(self.x)
            xs_px, ys_px = to_x(self.x[ok]), to_y(y[ok])
            if self.spec.kind == "barres":
                width = max(area.width() / max(len(self.x), 1), 1.0)
                base = to_y(max(y_lo, 0.0))
                fill = QColor(color)
                fill.setAlpha(170)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                for xp, yp in zip(xs_px, ys_px, strict=True):
                    painter.drawRect(QRectF(xp - width / 2, yp, width, base - yp))
            elif self.spec.kind == "points":
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                for xp, yp in zip(xs_px, ys_px, strict=True):
                    painter.drawEllipse(QPointF(xp, yp), 2.5, 2.5)
            elif len(xs_px):
                path = QPainterPath(QPointF(xs_px[0], ys_px[0]))
                for xp, yp in zip(xs_px[1:], ys_px[1:], strict=True):
                    path.lineTo(QPointF(xp, yp))
                painter.setPen(QPen(color, 1.6))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
        painter.setClipping(False)
        if len(self.series) > 1:  # légende
            for k, (name, _) in enumerate(self.series):
                painter.setPen(QColor(self.COLORS[k % len(self.COLORS)]))
                painter.drawText(QPointF(area.left() + 8 + 70 * k, area.top() + 14), str(name))
        painter.end()


#: Au-delà, seules les premières lignes sont affichées (l'export contient tout).
MAX_ROWS_SHOWN = 2000


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
        self.plot: PlotWidget | None = None
        if table.plot is not None and len(df):
            self.plot = PlotWidget(table)
            lay.addWidget(self.plot, 2)
        shown = min(len(df), MAX_ROWS_SHOWN)
        if shown < len(df):
            lay.addWidget(hint_label(f"{shown} premières lignes affichées ; l'export les contient toutes."))
        df = df.iloc[:shown]
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
        buttons = [export, copy]
        if self.plot is not None:
            image = QPushButton("Enregistrer le graphique…")
            image.clicked.connect(self._save_plot)
            buttons.append(image)
        lay.addWidget(row(*buttons))

    def _save_plot(self) -> None:
        if self.plot is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le graphique", "graphique.png", "PNG (*.png)"
        )
        if path:
            self.plot.grab().save(path)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exporter les résultats", "resultats.csv", "CSV (*.csv)")
        if path:
            self.table.to_csv(path)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.table.data.to_csv(sep="\t"))
