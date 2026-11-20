"""Petits composants réutilisables de l'interface."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fuji.analysis.image import Plot, Table
from fuji.core.properties import PropertyInfo, PropertyType
from fuji.text import number

from .components import Card

#: icône des sections courantes (une section sans icône connue n'en a pas)
SECTION_ICONS = {
    "Caméra": "camera",
    "Optique": "objective",
    "Éclairage": "bulb",
    "Platine XY": "move",
    "Mise au point": "focus",
    "Enregistrement": "save",
    "Série dans le temps": "timer",
    "Pile Z": "stack",
    "Canaux": "channels",
    "Positions": "target",
    "Mosaïque": "mosaic",
    "Autofocus": "focus",
}


def section(title: str, *, hint: str = "", icon: str | None = None) -> tuple[Card, QVBoxLayout]:
    """Une section de panneau : une carte avec son titre (et son icône), comme partout dans
    Fuji. Renvoie la carte et la disposition où ranger son contenu."""
    card = Card(title, icon_name=icon or SECTION_ICONS.get(title))
    card.body.setSpacing(6)
    if hint:
        lbl = QLabel(hint)
        lbl.setWordWrap(True)
        lbl.setObjectName("hint")
        card.body.addWidget(lbl)
    return card, card.body


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

    def _color(self, k: int) -> str:
        """Couleur de la k-ième courbe : celle du graphique s'il en donne, sinon l'habituelle."""
        if k < len(self.spec.colors):
            return self.spec.colors[k]
        return self.COLORS[k % len(self.COLORS)]

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
            painter.drawText(
                QRectF(18, y - 9, area.left() - 24, 18), align_right, number(float(value), digits=5)
            )
        for tick in _nice_ticks(x_lo, x_hi, 6):
            x = to_x(tick)
            painter.setPen(QPen(QColor(p.border), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.setPen(QColor(p.muted))
            painter.drawText(
                QRectF(x - 40, area.bottom() + 3, 80, 16), align_top, number(float(tick), digits=5)
            )
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
            color = QColor(self._color(k))
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
                painter.setPen(QColor(self._color(k)))
                painter.drawText(QPointF(area.left() + 8 + 70 * k, area.top() + 14), str(name))
        painter.end()
