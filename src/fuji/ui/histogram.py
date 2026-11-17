"""Histogramme en direct avec indicateur de saturation."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

from .theme import ThemeManager, palette

_RGB = ("#ff5a5a", "#3ddc84", "#4f8cff")


class HistogramWidget(QWidget):
    """Histogramme compact : courbes par canal, échelle logarithmique, saturation en %."""

    BINS = 128

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 54)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(58)
        self.log_scale = True
        self._hists: list[np.ndarray] = []
        self._saturated = 0.0
        self._stats = ""
        self.setToolTip("Histogramme de l'image (double-clic : échelle linéaire / logarithmique)")
        ThemeManager.get().changed.connect(self._on_theme)

    def _on_theme(self, *_: object) -> None:
        self.update()

    def mouseDoubleClickEvent(self, _event: object) -> None:
        self.log_scale = not self.log_scale
        self.update()

    def set_image(self, data: np.ndarray, bit_depth: int | None) -> None:
        # ~250 000 pixels suffisent, quelle que soit la taille de l'image
        step = max(1, int(np.sqrt(data.shape[0] * data.shape[1] / 250_000)))
        sample = data[::step, ::step]
        maxval = float(2**bit_depth - 1) if bit_depth else float(sample.max() or 1)
        planes = [sample[..., i] for i in range(3)] if sample.ndim == 3 else [sample]
        self._hists = [
            np.histogram(pl, bins=self.BINS, range=(0, maxval + 1))[0].astype(np.float64) for pl in planes
        ]
        self._saturated = float(np.mean(sample >= maxval)) * 100
        self._stats = f"{int(sample.min())} – {int(sample.max())} / {int(maxval)}"
        self.update()

    def paintEvent(self, _event: object) -> None:
        p = palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(p.border)))
        painter.setBrush(QColor(p.input))
        painter.drawRoundedRect(r, 8, 8)
        plot = r.adjusted(6, 6, -6, -18)
        if self._hists:
            colors = _RGB if len(self._hists) == 3 else (p.text,)
            for hist, color in zip(self._hists, colors, strict=False):
                h = np.log1p(hist) if self.log_scale else hist
                top = h.max() or 1.0
                path = QPainterPath(QPointF(plot.left(), plot.bottom()))
                for i, v in enumerate(h):
                    x = plot.left() + plot.width() * i / (len(h) - 1)
                    path.lineTo(QPointF(x, plot.bottom() - plot.height() * v / top))
                path.lineTo(QPointF(plot.right(), plot.bottom()))
                path.closeSubpath()
                c = QColor(color)
                fill = QColor(c)
                fill.setAlpha(60)
                painter.setPen(QPen(c, 1.2))
                painter.setBrush(fill)
                painter.drawPath(path)
        painter.setPen(QColor(p.muted))
        f = painter.font()
        f.setPixelSize(max(9, p.font_size - 3))
        painter.setFont(f)
        text_rect = QRectF(r.left() + 8, r.bottom() - 16, r.width() - 16, 14)
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._stats or "histogramme",
        )
        if self._hists:
            sat = self._saturated
            top = QRectF(r.left() + 8, r.top() + 4, r.width() - 14, 14)
            if sat > 0.1:
                painter.setPen(QColor(p.danger))
                f.setBold(True)
                painter.setFont(f)
                painter.drawText(
                    top, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop), f"⚠ saturé {sat:.1f} %"
                )
            else:
                painter.setPen(QColor(p.faint))
                painter.drawText(
                    top,
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
                    "log" if self.log_scale else "lin",
                )
        painter.end()
