"""Barre d'affichage sous l'image de l'espace Analyse (« Luminosité / contraste » d'ImageJ).

Toujours visible : le calque affiché, son histogramme (plan courant) avec les bornes du
contraste que l'on peut tirer à la souris, les valeurs min / max, « Auto », « Tout », le
gamma et la palette de couleurs.

Elle ne change **que l'affichage** : les mesures et les opérations utilisent toujours les
valeurs d'origine des pixels (comme dans ImageJ tant qu'on ne clique pas sur « Apply »).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image as ImageLayer
from qtpy.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import documents
from .components import button, label, section_title
from .flow import FlowLayout
from .theme import ThemeManager, palette

_RGB = ("#ff5a5a", "#3ddc84", "#4f8cff")
#: palettes proposées pour une image en niveaux de gris : (nom affiché, nom napari)
COLORMAPS = (
    ("Gris", "gray"),
    ("Vert", "green"),
    ("Magenta", "magenta"),
    ("Rouge", "red"),
    ("Bleu", "blue"),
    ("Cyan", "cyan"),
    ("Jaune", "yellow"),
    ("Chaleur (inferno)", "inferno"),
    ("Viridis", "viridis"),
    ("Gris inversé", "gray_r"),
)
#: part des pixels laissée saturée de chaque côté par « Auto » (comme ImageJ : 0,35 %)
AUTO_SATURATION = 0.35
_GAMMA_LOG = 100  # le curseur du gamma va de -100 à 100 → gamma de 0,2 à 5 (log)


def _sample(data: np.ndarray, rgb: bool) -> np.ndarray:
    """~250 000 pixels suffisent pour l'histogramme, quelle que soit la taille de l'image."""
    h, w = data.shape[:2]
    step = max(1, int(np.sqrt(h * w / 250_000)))
    return np.asarray(data[::step, ::step]) if rgb or data.ndim == 2 else np.asarray(data)


def auto_limits(data: np.ndarray, saturation: float = AUTO_SATURATION) -> tuple[float, float]:
    """Bornes qui laissent ``saturation`` % des pixels saturés de chaque côté."""
    values = np.asarray(data, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 1.0
    lo, hi = np.percentile(values, (saturation, 100 - saturation))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def full_limits(data: np.ndarray) -> tuple[float, float]:
    """Du pixel le plus sombre au plus clair."""
    values = np.asarray(data, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    return (lo, hi) if hi > lo else (lo, lo + 1)


class ContrastHistogram(QWidget):
    """Histogramme avec les deux bornes du contraste, déplaçables à la souris."""

    limits_dragged = Signal(float, float)

    BINS = 256

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 60)
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.log_scale = True
        self._hists: list[np.ndarray] = []
        self._range = (0.0, 1.0)
        self._limits = (0.0, 1.0)
        self._drag: int | None = None  # 0 : borne basse, 1 : borne haute
        self._stats = ""
        self.setToolTip(
            "Histogramme du plan affiché. Tirez les traits pour régler le contraste ; "
            "double-clic : échelle linéaire / logarithmique"
        )
        ThemeManager.get().changed.connect(lambda *_: self.update())

    # ------------------------------------------------------------------ données
    def set_data(self, data: np.ndarray | None, rgb: bool = False) -> None:
        if data is None or not np.asarray(data).size:
            self._hists, self._stats = [], ""
            self.update()
            return
        values = np.asarray(data)
        lo, hi = full_limits(values)
        self._range = (lo, hi)
        planes = [values[..., i] for i in range(values.shape[-1])] if rgb else [values]
        self._hists = []
        for pl in planes:
            pl = np.asarray(pl, dtype=np.float64)
            pl = pl[np.isfinite(pl)]
            self._hists.append(np.histogram(pl, bins=self.BINS, range=(lo, hi))[0].astype(np.float64))
        self._stats = f"{_fmt(lo)} – {_fmt(hi)}"
        self.update()

    def set_limits(self, lo: float, hi: float) -> None:
        self._limits = (float(lo), float(hi))
        self.update()

    def _span(self) -> tuple[float, float]:
        """Plage affichée : celle des pixels, élargie aux bornes si elles en sortent."""
        lo = min(self._range[0], self._limits[0])
        hi = max(self._range[1], self._limits[1])
        return (lo, hi if hi > lo else lo + 1)

    def _plot_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(7, 7, -7, -19)

    def _x_of(self, value: float) -> float:
        lo, hi = self._span()
        r = self._plot_rect()
        return r.left() + r.width() * (value - lo) / (hi - lo)

    def _value_of(self, x: float) -> float:
        lo, hi = self._span()
        r = self._plot_rect()
        return lo + (hi - lo) * (x - r.left()) / max(r.width(), 1.0)

    # ------------------------------------------------------------------- souris
    def mouseDoubleClickEvent(self, _event: Any) -> None:
        self.log_scale = not self.log_scale
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        if not self._hists or event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        d = [abs(x - self._x_of(v)) for v in self._limits]
        self._drag = int(np.argmin(d))
        self._move_to(x)

    def mouseMoveEvent(self, event: Any) -> None:
        x = event.position().x()
        if self._drag is not None:
            self._move_to(x)
            return
        near = self._hists and min(abs(x - self._x_of(v)) for v in self._limits) < 6
        self.setCursor(Qt.CursorShape.SizeHorCursor if near else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, _event: Any) -> None:
        self._drag = None

    def _move_to(self, x: float) -> None:
        value = self._value_of(x)
        lo, hi = self._limits
        gap = (self._span()[1] - self._span()[0]) / 1000
        if self._drag == 0:
            lo = min(value, hi - gap)
        else:
            hi = max(value, lo + gap)
        self._limits = (lo, hi)
        self.update()
        self.limits_dragged.emit(lo, hi)

    # ------------------------------------------------------------------- dessin
    def paintEvent(self, _event: Any) -> None:
        p = palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(p.border)))
        painter.setBrush(QColor(p.input))
        painter.drawRoundedRect(r, 8, 8)
        plot = self._plot_rect()
        f = painter.font()
        f.setPixelSize(max(9, p.font_size - 3))
        painter.setFont(f)
        if self._hists:
            # rampe de gris entre les bornes : ce qui sera noir, gris, blanc à l'écran
            x0, x1 = self._x_of(self._limits[0]), self._x_of(self._limits[1])
            shade = QColor(p.muted)
            shade.setAlpha(40)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(shade)
            painter.drawRect(QRectF(plot.left(), plot.top(), max(0.0, x0 - plot.left()), plot.height()))
            painter.drawRect(QRectF(x1, plot.top(), max(0.0, plot.right() - x1), plot.height()))
            colors = _RGB if len(self._hists) == 3 else (p.text,)
            for hist, color in zip(self._hists, colors, strict=False):
                h = np.log1p(hist) if self.log_scale else hist
                top = h.max() or 1.0
                path = QPainterPath(QPointF(self._x_of(self._range[0]), plot.bottom()))
                n = len(h)
                for i, v in enumerate(h):
                    value = self._range[0] + (self._range[1] - self._range[0]) * (i + 0.5) / n
                    path.lineTo(QPointF(self._x_of(value), plot.bottom() - plot.height() * v / top))
                path.lineTo(QPointF(self._x_of(self._range[1]), plot.bottom()))
                path.closeSubpath()
                c = QColor(color)
                fill = QColor(c)
                fill.setAlpha(60)
                painter.setPen(QPen(c, 1.1))
                painter.setBrush(fill)
                painter.drawPath(path)
            # les deux bornes
            pen = QPen(QColor(p.accent), 2)
            painter.setPen(pen)
            for x in (x0, x1):
                painter.drawLine(QPointF(x, plot.top() - 3), QPointF(x, plot.bottom() + 2))
            painter.setPen(QColor(p.accent))
            painter.drawLine(QPointF(x0, plot.bottom()), QPointF(x1, plot.top()))
        painter.setPen(QColor(p.muted))
        text_rect = QRectF(r.left() + 8, r.bottom() - 16, r.width() - 16, 14)
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"pixels : {self._stats}" if self._stats else "aucune image",
        )
        if self._hists:
            painter.setPen(QColor(p.faint))
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                "log" if self.log_scale else "lin",
            )
        painter.end()


def _fmt(v: float) -> str:
    if abs(v) >= 1000 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.3g}"


def _group(title: str, *widgets: QWidget, row: bool = True) -> QWidget:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    lay.addWidget(section_title(title))
    inner = QHBoxLayout() if row else QVBoxLayout()
    inner.setSpacing(6)
    for w in widgets:
        inner.addWidget(w)
    lay.addLayout(inner)
    return box


class DisplayBar(QWidget):
    """Réglages d'affichage de l'image active, toujours sous l'image (espace Analyse)."""

    message = Signal(str)
    #: « Fermer » : fermer l'image affichée
    close_requested = Signal()
    #: une autre des images ouvertes est choisie
    document_chosen = Signal(str)

    def __init__(self, viewer: ViewerModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("QuickBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.viewer = viewer
        self.layer: ImageLayer | None = None
        self._plane: np.ndarray | None = None
        self._syncing = False
        lay = FlowLayout(self, spacing=16, line_spacing=8)
        lay.setContentsMargins(14, 8, 14, 8)

        # l'image réglée
        self.name_label = label("Aucune image", "ToolTitle", wrap=False)
        self.info_label = label("", "hint", wrap=False)
        info = QWidget()
        il = QVBoxLayout(info)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(1)
        il.addWidget(section_title("Image affichée"))
        # plusieurs images ouvertes : on choisit celle à afficher
        self.doc_combo = QComboBox()
        self.doc_combo.setToolTip("Images ouvertes : choisissez celle à afficher")
        self.doc_combo.setMinimumWidth(120)
        self.doc_combo.activated.connect(
            lambda i: self.document_chosen.emit(self.doc_combo.itemData(i) or "")
        )
        self.doc_combo.hide()
        self.close_btn = button("Fermer", "close", tooltip="Fermer cette image et ses résultats (Ctrl+W)")
        self.close_btn.setObjectName("ghost")
        self.close_btn.clicked.connect(self.close_requested.emit)
        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        name_row.addWidget(self.name_label, 1)
        name_row.addWidget(self.doc_combo, 1)
        name_row.addWidget(self.close_btn)
        il.addLayout(name_row)
        il.addWidget(self.info_label)
        info.setMaximumWidth(270)
        lay.addWidget(info)

        # histogramme + bornes
        self.histogram = ContrastHistogram()
        self.histogram.setMinimumWidth(200)
        self.histogram.setMaximumWidth(420)
        self.histogram.limits_dragged.connect(self._set_limits)
        lay.addWidget(self.histogram, stretch=2)

        # min / max et boutons
        self.min_box = QDoubleSpinBox()
        self.max_box = QDoubleSpinBox()
        for box, tip in (
            (self.min_box, "Valeur affichée en noir"),
            (self.max_box, "Valeur affichée en blanc"),
        ):
            box.setRange(-1e12, 1e12)
            box.setKeyboardTracking(False)
            box.setToolTip(tip)
            box.setMinimumWidth(70)
            box.setMaximumWidth(92)
            box.valueChanged.connect(self._boxes_changed)
        self.auto_btn = button(
            "Auto",
            "wand",
            tooltip="Contraste automatique : 0,35 % des pixels saturés de chaque côté (comme ImageJ)",
        )
        self.auto_btn.clicked.connect(self.auto_contrast)
        self.full_btn = button("Tout", "refresh", tooltip="Du pixel le plus sombre au plus clair")
        self.full_btn.clicked.connect(self.full_contrast)
        contrast = QWidget()
        cl = QVBoxLayout(contrast)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(3)
        title = section_title("Contraste — affichage seulement")
        contrast.setToolTip(
            "Valeur affichée en noir → valeur affichée en blanc. Ne change que l'affichage : "
            "les mesures et les opérations utilisent toujours les vraies valeurs des pixels."
        )
        cl.addWidget(title)
        r1 = QHBoxLayout()
        r1.setSpacing(6)
        r1.addWidget(self.min_box)
        r1.addWidget(QLabel("→"))
        r1.addWidget(self.max_box)
        r1.addWidget(self.auto_btn)
        r1.addWidget(self.full_btn)
        cl.addLayout(r1)
        lay.addWidget(contrast)

        # gamma et palette
        self.gamma = QSlider(Qt.Orientation.Horizontal)
        self.gamma.setRange(-_GAMMA_LOG, _GAMMA_LOG)
        self.gamma.setValue(0)
        self.gamma.setMinimumWidth(80)
        self.gamma.setMaximumWidth(140)
        self.gamma.setToolTip("Gamma : < 1 éclaircit les zones sombres, > 1 les assombrit")
        self.gamma.valueChanged.connect(self._gamma_changed)
        self.gamma_label = QPushButton("1,00")  # un clic : retour à 1
        self.gamma_label.setObjectName("ghost")
        self.gamma_label.setToolTip("Revenir à 1 (pas de correction)")
        self.gamma_label.setMinimumWidth(44)
        self.gamma_label.clicked.connect(lambda: self.gamma.setValue(0))
        self.colormap = QComboBox()
        for text, name in COLORMAPS:
            self.colormap.addItem(text, name)
        self.colormap.setToolTip("Couleur d'affichage d'une image en niveaux de gris")
        self.colormap.currentIndexChanged.connect(self._colormap_changed)
        self.gamma_group = _group("Gamma", self.gamma, self.gamma_label)
        self.colormap_group = _group("Couleur", self.colormap)
        lay.addWidget(self.gamma_group)
        lay.addWidget(self.colormap_group)
        self.controls = [self.histogram, contrast, self.gamma_group, self.colormap_group]

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._refresh_plane)
        viewer.layers.selection.events.active.connect(self._retarget)
        viewer.layers.events.inserted.connect(self._retarget)
        viewer.layers.events.removed.connect(self._retarget)
        viewer.dims.events.current_step.connect(lambda *_: self._timer.start())
        self._retarget()
        if self.layer is None:
            self._refresh_all()

    # ------------------------------------------------------------------ cible
    def _target(self) -> ImageLayer | None:
        """L'image réglée : le calque actif si c'est une image, sinon l'image du dessus."""
        active = self.viewer.layers.selection.active
        if isinstance(active, ImageLayer) and not active.name.startswith(documents.PREVIEW_PREFIX):
            return active
        for layer in reversed(self.viewer.layers):
            if (
                isinstance(layer, ImageLayer)
                and layer.visible
                and not layer.name.startswith(documents.PREVIEW_PREFIX)
            ):
                return layer
        return None

    def _retarget(self, *_: Any) -> None:
        layer = self._target()
        self._sync_docs()
        if layer is self.layer:
            return
        if self.layer is not None:
            for name, cb in self._callbacks():
                try:
                    getattr(self.layer.events, name).disconnect(cb)
                except Exception:  # calque déjà détruit
                    pass
        self.layer = layer
        if layer is not None:
            for name, cb in self._callbacks():
                getattr(layer.events, name).connect(cb)
        self._refresh_all()

    def _callbacks(self) -> list[tuple[str, Any]]:
        return [
            ("contrast_limits", self._sync_controls),
            ("gamma", self._sync_controls),
            ("colormap", self._sync_controls),
            ("data", self._data_changed),
            ("name", self._sync_info),
        ]

    def _data_changed(self, *_: Any) -> None:
        self._timer.start()

    # -------------------------------------------------------------- affichage
    def _refresh_all(self) -> None:
        for w in self.controls:  # sans image, seulement « Aucune image » : la vue garde la place
            w.setVisible(self.layer is not None)
        self._sync_info()
        self._refresh_plane()
        self._sync_controls()

    def _sync_docs(self) -> None:
        """Liste des images ouvertes (montrée quand il y en a plusieurs)."""
        docs = documents.documents(self.viewer)
        current = documents.doc_of(self.layer) if self.layer is not None else None
        self.doc_combo.blockSignals(True)
        self.doc_combo.clear()
        for doc in docs:
            self.doc_combo.addItem(_elide(doc, 34), doc)
        if current in docs:
            self.doc_combo.setCurrentIndex(docs.index(current))
        self.doc_combo.blockSignals(False)
        several = len(docs) > 1
        self.doc_combo.setVisible(several)
        self.name_label.setVisible(not several)
        self.close_btn.setVisible(bool(self.viewer.layers))

    def _sync_info(self, *_: Any) -> None:
        layer = self.layer
        self._sync_docs()
        if layer is None:
            self.name_label.setText("Aucune image")
            self.info_label.setText("Ouvrez une image (liste à gauche)")
            return
        self.name_label.setText(_elide(layer.name, 30))
        self.name_label.setToolTip(layer.name)
        data = layer.data
        shape = tuple(int(n) for n in data.shape)
        rgb = bool(layer.rgb)
        h, w = shape[-3:-1] if rgb else shape[-2:]
        parts = [f"{w}×{h}"]
        extra = shape[: len(shape) - (3 if rgb else 2)]
        if extra:
            parts.append("pile " + "×".join(str(n) for n in extra))
        parts.append(_type_text(np.dtype(data.dtype), rgb))
        px = layer.metadata.get("pixel_size_um")
        if px is None and float(layer.scale[-1]) != 1.0:
            px = float(layer.scale[-1])
        parts.append(f"{px:.3g} µm/px".replace(".", ",") if px else "sans échelle")
        self.info_label.setText(" · ".join(parts))

    def _refresh_plane(self) -> None:
        from .panels.analysis_panel import displayed_plane

        layer = self.layer
        if layer is None:
            self._plane = None
            self.histogram.set_data(None)
            return
        try:
            plane, _ = displayed_plane(layer, self.viewer)
        except Exception:  # calque en cours de changement : on réessaiera
            return
        self._plane = _sample(plane, bool(layer.rgb))
        self.histogram.set_data(self._plane, rgb=bool(layer.rgb) and self._plane.ndim == 3)

    def _sync_controls(self, *_: Any) -> None:
        layer = self.layer
        if layer is None:
            return
        self._syncing = True
        try:
            lo, hi = (float(v) for v in layer.contrast_limits)
            integer = np.issubdtype(np.dtype(layer.data.dtype), np.integer)
            for box in (self.min_box, self.max_box):
                box.setDecimals(0 if integer else 3)
                box.setSingleStep(1 if integer else max((hi - lo) / 100, 1e-6))
            self.min_box.setValue(lo)
            self.max_box.setValue(hi)
            self.histogram.set_limits(lo, hi)
            g = float(layer.gamma)
            self.gamma.setValue(int(round(_GAMMA_LOG * math.log(max(g, 0.2), 5))))
            self.gamma_label.setText(f"{g:.2f}".replace(".", ","))
            rgb = bool(layer.rgb)
            self.colormap_group.setVisible(not rgb)
            if not rgb:
                name = layer.colormap.name
                i = self.colormap.findData(name)
                if i < 0:  # palette posée ailleurs (canaux d'un fichier…) : ajoutée à la liste
                    self.colormap.addItem(name, name)
                    i = self.colormap.count() - 1
                self.colormap.setCurrentIndex(i)
        finally:
            self._syncing = False

    # ---------------------------------------------------------------- réglages
    def _set_limits(self, lo: float, hi: float) -> None:
        if self.layer is None or hi <= lo:
            return
        lo_r, hi_r = self.layer.contrast_limits_range
        if lo < lo_r or hi > hi_r:  # sinon napari ramène les bornes dans la plage
            self.layer.contrast_limits_range = (min(lo, lo_r), max(hi, hi_r))
        self.layer.contrast_limits = (lo, hi)

    def _boxes_changed(self, *_: Any) -> None:
        if self._syncing:
            return
        lo, hi = self.min_box.value(), self.max_box.value()
        if hi <= lo:
            self._sync_controls()  # valeurs impossibles : on revient à l'état du calque
            return
        self._set_limits(lo, hi)

    def _whole(self, lo: float, hi: float) -> tuple[float, float]:
        """Bornes entières pour une image en entiers (comme ImageJ)."""
        if self.layer is not None and np.issubdtype(np.dtype(self.layer.data.dtype), np.integer):
            lo, hi = float(math.floor(lo)), float(math.ceil(hi))
            hi = max(hi, lo + 1)
        return lo, hi

    def auto_contrast(self) -> None:
        if self._plane is None:
            return
        self._set_limits(*self._whole(*auto_limits(self._plane)))
        self.message.emit("Contraste automatique (affichage seulement)")

    def full_contrast(self) -> None:
        if self._plane is None:
            return
        self._set_limits(*self._whole(*full_limits(self._plane)))

    def _gamma_changed(self, value: int) -> None:
        g = 5 ** (value / _GAMMA_LOG)
        self.gamma_label.setText(f"{g:.2f}".replace(".", ","))
        if not self._syncing and self.layer is not None:
            self.layer.gamma = g

    def _colormap_changed(self, index: int) -> None:
        if self._syncing or self.layer is None or index < 0 or self.layer.rgb:
            return
        self.layer.colormap = self.colormap.itemData(index)


def _type_text(dtype: np.dtype, rgb: bool) -> str:
    if rgb:
        return "couleur " + ("8 bits" if dtype == np.uint8 else f"{dtype.itemsize * 8} bits")
    if dtype == np.bool_:
        return "binaire"
    if np.issubdtype(dtype, np.floating):
        return f"{dtype.itemsize * 8} bits (réels)"
    return f"{dtype.itemsize * 8} bits"


def _elide(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


__all__ = ["COLORMAPS", "ContrastHistogram", "DisplayBar", "auto_limits", "full_limits"]
