"""Barre d'affichage sous l'image de l'espace Analyse (« Luminosité / contraste » d'ImageJ).

Toujours visible : l'image affichée, le zoom, son histogramme (plan courant) avec les bornes
du contraste que l'on peut tirer à la souris, les valeurs min / max, « Auto » et « Tout ».
Dépliables : luminosité, contraste, gamma, palette de couleurs (et son négatif), transparence
des masques. Dans les options : saturation d'« Auto », lissage, barre d'échelle, « appliquer
aux pixels ».

Elle ne change **que l'affichage** : les mesures et les opérations utilisent toujours les
valeurs d'origine des pixels (comme dans ImageJ tant qu'on ne clique pas sur « Apply »).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image as ImageLayer
from napari.layers import Labels as LabelsLayer
from qtpy.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QAction, QActionGroup, QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import documents
from .components import button, icon_button, label, section_title
from .flow import FlowLayout
from .prefs import settings
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
)
#: part des pixels laissée saturée de chaque côté par « Auto » (comme ImageJ : 0,35 %)
AUTO_SATURATION = 0.35
_GAMMA_LOG = 100  # le curseur du gamma va de -100 à 100 → gamma de 0,2 à 5 (log)
#: choix proposés pour la saturation d'« Auto » (en % de chaque côté)
SATURATIONS = (0.0, 0.1, 0.35, 1.0, 2.0, 5.0)
#: agrandissements proposés (100 % : un pixel de l'image = un pixel de l'écran)
ZOOMS = (25, 50, 100, 200, 400, 800)
ZOOM_STEP = 1.25
_CONTRAST_BASE = 100.0  # curseur « contraste » à 100 : fenêtre 100 fois plus étroite que l'image
_INVERTED = " (inversé)"  # fin du nom d'une palette à l'envers
_CONTOUR = 2  # épaisseur du contour des masques (pixels)
_LEVELS_WIDTH = 330  # largeur souhaitée du bloc histogramme + bornes
_ROOMY = 600  # largeur à partir de laquelle les curseurs sont dépliés d'office

#: barre d'échelle : réglages par défaut, puis les choix proposés (texte, valeur)
SCALE_BAR_DEFAULTS: dict[str, Any] = {
    "visible": True,
    "position": "bottom_right",
    "color": "auto",
    "font_size": 10.0,
    "length": 0.0,  # 0 : longueur choisie automatiquement
    "box": True,
    "ticks": True,
}
SCALE_BAR_POSITIONS = (
    ("En bas à droite", "bottom_right"),
    ("En bas à gauche", "bottom_left"),
    ("En haut à droite", "top_right"),
    ("En haut à gauche", "top_left"),
)
SCALE_BAR_COLORS = (
    ("Automatique (selon le fond)", "auto"),
    ("Blanc", "white"),
    ("Noir", "black"),
    ("Jaune", "yellow"),
    ("Magenta", "magenta"),
)
SCALE_BAR_SIZES = (("Petit", 8.0), ("Normal", 10.0), ("Grand", 14.0), ("Très grand", 20.0))
SCALE_BAR_LENGTHS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


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
        ThemeManager.get().changed.connect(self._on_theme)  # méthode : coupée avec le widget

    def _on_theme(self, *_: Any) -> None:
        self.update()

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
    lay.addStretch(1)  # dans une ligne plus haute que lui, le groupe reste en haut
    return box


class _Block(QWidget):
    """Bloc dont la largeur souhaitée est fixée : la barre se replie d'après elle, et non
    d'après la largeur « idéale » de son contenu (des champs de saisie très larges)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self._width = width

    def sizeHint(self) -> QSize:
        return QSize(self._width, super().sizeHint().height())


def _slider(lo: int, hi: int, tooltip: str) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setFixedWidth(70)  # quatre groupes de curseurs tiennent sur une ligne d'écran de portable
    s.setToolTip(tooltip)
    return s


def _value_button(text: str, tooltip: str) -> QPushButton:
    """La valeur d'un curseur ; un clic la remet au neutre."""
    b = QPushButton(text)
    b.setObjectName("ghost")
    b.setToolTip(tooltip)
    b.setMinimumWidth(38)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


# ----------------------------------------------------------- luminosité / contraste
def window_to_sliders(lo: float, hi: float, data_range: tuple[float, float]) -> tuple[int, int]:
    """Bornes du contraste → positions des curseurs « luminosité » et « contraste ».

    Comme dans ImageJ, ce sont deux autres façons de déplacer les mêmes bornes : la
    luminosité glisse la fenêtre (son centre), le contraste la resserre ou l'élargit.
    À zéro et zéro, la fenêtre couvre exactement les valeurs de l'image."""
    f_lo, f_hi = data_range
    full = max(f_hi - f_lo, 1e-12)
    centre, width = (lo + hi) / 2, max(hi - lo, 1e-12)
    brightness = round(200 * ((f_lo + f_hi) / 2 - centre) / full)
    contrast = round(-100 * math.log(width / full, _CONTRAST_BASE))
    return max(-100, min(100, brightness)), max(-100, min(100, contrast))


def sliders_to_window(
    brightness: float, contrast: float, data_range: tuple[float, float]
) -> tuple[float, float]:
    """L'inverse de :func:`window_to_sliders`."""
    f_lo, f_hi = data_range
    full = max(f_hi - f_lo, 1e-12)
    centre = (f_lo + f_hi) / 2 - brightness / 200 * full
    width = full * _CONTRAST_BASE ** (-contrast / 100)
    return centre - width / 2, centre + width / 2


# ------------------------------------------------------------------ barre d'échelle
def scale_bar_of(viewer: ViewerModel) -> Any:
    """La barre d'échelle d'une vue (son emplacement a changé avec napari 0.9)."""
    overlays = getattr(getattr(viewer, "canvas", None), "overlays", None)
    bar = getattr(overlays, "scale_bar", None)
    return bar if bar is not None else viewer.scale_bar


def scale_bar_prefs() -> dict[str, Any]:
    """Les réglages retenus de la barre d'échelle."""
    s = settings()
    out: dict[str, Any] = {}
    for key, default in SCALE_BAR_DEFAULTS.items():
        kind = type(default)
        try:
            out[key] = s.value(f"display/scale_bar/{key}", default, type=kind)
        except (TypeError, ValueError):
            out[key] = default
    if out["position"] not in dict(SCALE_BAR_POSITIONS).values():
        out["position"] = SCALE_BAR_DEFAULTS["position"]
    if out["color"] not in dict(SCALE_BAR_COLORS).values():
        out["color"] = SCALE_BAR_DEFAULTS["color"]
    return out


def apply_scale_bar(bar: Any, prefs: dict[str, Any] | None = None) -> None:
    """Applique les réglages à la barre d'échelle d'une vue."""
    p = prefs if prefs is not None else scale_bar_prefs()
    try:
        bar.visible = bool(p["visible"])
        bar.position = p["position"]
        bar.colored = p["color"] != "auto"
        if p["color"] != "auto":
            bar.color = p["color"]
        bar.font_size = float(p["font_size"])
        bar.length = float(p["length"]) if float(p["length"]) > 0 else None
        bar.box = bool(p["box"])
        bar.ticks = bool(p["ticks"])
    except Exception:  # réglage inconnu de cette version de napari : la barre reste telle quelle
        pass


def _reversed_colormap(name: str) -> Any:
    """La même palette, à l'envers (du clair au sombre)."""
    from napari.utils.colormaps import Colormap, ensure_colormap

    cm = ensure_colormap(name)
    controls = 1.0 - np.asarray(cm.controls, dtype=np.float64)[::-1]
    return Colormap(
        colors=np.asarray(cm.colors)[::-1],
        controls=controls,
        name=f"{name}{_INVERTED}",
        interpolation=cm.interpolation,
    )


class DisplayBar(QWidget):
    """Réglages d'affichage de l'image active, toujours sous l'image (espace Analyse).

    Toujours là : l'image, le zoom, l'histogramme et les bornes du contraste. Le bouton
    « Réglages » déplie les curseurs (luminosité, contraste, gamma, couleur, masque) ; la
    roue dentée ouvre les options (saturation d'« Auto », lissage, barre d'échelle…)."""

    message = Signal(str)
    #: « Fermer » : fermer l'image affichée
    close_requested = Signal()
    #: une autre des images ouvertes est choisie
    document_chosen = Signal(str)
    #: « Appliquer aux pixels » : nom du calque, noir, blanc, gamma
    apply_requested = Signal(str, float, float, float)

    def __init__(
        self,
        viewer: ViewerModel,
        parent: QWidget | None = None,
        *,
        scale_bars: Sequence[Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("QuickBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.viewer = viewer
        self.layer: ImageLayer | None = None
        self._plane: np.ndarray | None = None
        self._data_range = (0.0, 1.0)
        self._syncing = False
        self._driving = ""  # le curseur que l'on est en train de tirer n'est pas resynchronisé
        self._scale_bars = list(scale_bars) if scale_bars is not None else [scale_bar_of(viewer)]
        prefs = settings()
        self.auto_saturation = _pref(prefs, "display/auto_saturation", AUTO_SATURATION)
        self.smooth = _pref(prefs, "display/smooth", False)
        self._more_chosen = prefs.contains("display/more")
        lay = FlowLayout(self, spacing=16, line_spacing=8)
        lay.setContentsMargins(14, 8, 14, 8)

        # ---- l'image réglée, le zoom ------------------------------------------------
        self.name_label = label("Aucune image", "ToolTitle", wrap=False)
        self.info_label = label("", "hint", wrap=False)
        info = QWidget()
        il = QVBoxLayout(info)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(2)
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

        self.zoom_out_btn = icon_button("minus", "Dézoomer")
        self.zoom_in_btn = icon_button("plus", "Zoomer")
        for b in (self.zoom_out_btn, self.zoom_in_btn):
            b.setFixedWidth(30)
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_by(1 / ZOOM_STEP))
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_by(ZOOM_STEP))
        self.zoom_btn = QPushButton("100 %")
        self.zoom_btn.setToolTip(
            "Agrandissement de l'image. À 100 %, un pixel de l'image occupe un pixel de l'écran : "
            "c'est là que l'on juge la netteté"
        )
        self.zoom_btn.setMinimumWidth(74)
        self.zoom_menu = QMenu(self.zoom_btn)
        self.zoom_menu.addAction("Ajuster à la fenêtre", self.zoom_fit)
        self.zoom_menu.addSeparator()
        for percent in ZOOMS:
            self.zoom_menu.addAction(f"{percent} %", lambda p=percent: self.set_zoom(p))
        self.zoom_btn.setMenu(self.zoom_menu)
        self.more_btn = icon_button(
            "sliders",
            "Réglages : luminosité, contraste, gamma, couleur, masque (replier / déplier)",
            object_name="toggle",
            checkable=True,
        )
        self.more_btn.setChecked(_pref(prefs, "display/more", True))
        self.more_btn.toggled.connect(self._more_toggled)
        self.options_btn = icon_button(
            "gear", "Options d'affichage : contraste automatique, lissage, barre d'échelle…"
        )
        self.options_btn.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0px; }")
        for b in (self.more_btn, self.options_btn):
            b.setFixedWidth(34)
        self.view_row = QWidget()
        vl = QHBoxLayout(self.view_row)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(3)
        for w in (self.zoom_out_btn, self.zoom_btn, self.zoom_in_btn):
            vl.addWidget(w)
        vl.addStretch(1)
        vl.addWidget(self.more_btn)
        vl.addWidget(self.options_btn)
        il.addWidget(self.view_row)
        info.setMinimumWidth(236)
        info.setMaximumWidth(270)
        lay.addWidget(info)

        # ---- histogramme + bornes ---------------------------------------------------
        self.histogram = ContrastHistogram()
        self.histogram.limits_dragged.connect(self._set_limits)
        self.min_box = QDoubleSpinBox()
        self.max_box = QDoubleSpinBox()
        for box, tip in (
            (self.min_box, "Valeur affichée en noir"),
            (self.max_box, "Valeur affichée en blanc"),
        ):
            box.setRange(-1e12, 1e12)
            box.setKeyboardTracking(False)
            box.setToolTip(tip)
            box.setMinimumWidth(66)
            box.setMaximumWidth(92)
            box.valueChanged.connect(self._boxes_changed)
        self.auto_btn = button("Auto")
        self.auto_btn.clicked.connect(self.auto_contrast)
        self.full_btn = button("Tout", tooltip="Du pixel le plus sombre au plus clair")
        self.full_btn.clicked.connect(self.full_contrast)
        self.levels = _Block(_LEVELS_WIDTH)
        self.levels.setToolTip(
            "Valeur affichée en noir → valeur affichée en blanc. Ne change que l'affichage : "
            "les mesures et les opérations utilisent toujours les vraies valeurs des pixels."
        )
        cl = QVBoxLayout(self.levels)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(self.histogram)
        r1 = QHBoxLayout()
        r1.setSpacing(5)
        r1.addWidget(self.min_box)
        r1.addWidget(QLabel("→"))
        r1.addWidget(self.max_box)
        r1.addWidget(self.auto_btn)
        r1.addWidget(self.full_btn)
        r1.addStretch(1)
        cl.addLayout(r1)
        self.levels.setMaximumWidth(460)
        lay.addWidget(self.levels, stretch=2)

        # ---- réglages dépliables ----------------------------------------------------
        self.brightness = _slider(
            -100, 100, "Luminosité : déplace les bornes du contraste, sans changer leur écart"
        )
        self.brightness_label = _value_button("0", "Revenir à 0 (fenêtre centrée sur les valeurs de l'image)")
        self.brightness.valueChanged.connect(self._brightness_changed)
        self.brightness.sliderReleased.connect(self._released)
        self.brightness_label.clicked.connect(lambda: self._reset_slider(self.brightness))
        self.contrast = _slider(-100, 100, "Contraste : resserre (à droite) ou écarte (à gauche) les bornes")
        self.contrast_label = _value_button("0", "Revenir à 0 (du pixel le plus sombre au plus clair)")
        self.contrast.valueChanged.connect(self._contrast_changed)
        self.contrast.sliderReleased.connect(self._released)
        self.contrast_label.clicked.connect(lambda: self._reset_slider(self.contrast))
        self.gamma = _slider(
            -_GAMMA_LOG, _GAMMA_LOG, "Gamma : < 1 éclaircit les zones sombres, > 1 les assombrit"
        )
        self.gamma.setValue(0)
        self.gamma.valueChanged.connect(self._gamma_changed)
        self.gamma_label = _value_button("1,00", "Revenir à 1 (pas de correction)")
        self.gamma_label.clicked.connect(lambda: self.gamma.setValue(0))
        self.colormap = QComboBox()
        for text, name in COLORMAPS:
            self.colormap.addItem(text, name)
        self.colormap.setToolTip("Couleur d'affichage d'une image en niveaux de gris")
        self.colormap.currentIndexChanged.connect(self._colormap_changed)
        self.invert = QCheckBox("Inverser")
        self.invert.setToolTip(
            "Affiche le négatif (clair ↔ sombre) : une fluorescence en noir sur fond blanc, par "
            "exemple. Affichage seulement ; pour changer les pixels : l'outil « Inverser »"
        )
        self.invert.toggled.connect(self._colormap_changed)
        self.mask_opacity = _slider(0, 100, "Transparence des masques posés sur l'image")
        self.mask_opacity.valueChanged.connect(self._mask_changed)
        self.mask_contour = QCheckBox("Contours")
        self.mask_contour.setToolTip("Ne dessiner que le contour des objets du masque")
        self.mask_contour.toggled.connect(self._mask_changed)
        self.brightness_group = _group("Luminosité", self.brightness, self.brightness_label)
        self.contrast_group = _group("Contraste", self.contrast, self.contrast_label)
        self.gamma_group = _group("Gamma", self.gamma, self.gamma_label)
        self.colormap_group = _group("Couleur", self.colormap, self.invert)
        self.mask_group = _group("Masque", self.mask_opacity, self.mask_contour)
        self.drawer = [
            self.brightness_group,
            self.contrast_group,
            self.gamma_group,
            self.colormap_group,
            self.mask_group,
        ]
        for w in self.drawer:
            lay.addWidget(w)
        #: tout ce qui n'a de sens qu'avec une image
        self.controls = [self.histogram, self.levels, self.view_row, *self.drawer]
        self._build_options()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._refresh_plane)
        self._layers_timer = QTimer(self)  # calques montrés / cachés : une seule mise à jour
        self._layers_timer.setSingleShot(True)
        self._layers_timer.setInterval(0)
        self._layers_timer.timeout.connect(self._retarget)
        viewer.layers.selection.events.active.connect(self._retarget)
        viewer.layers.events.inserted.connect(self._inserted)
        viewer.layers.events.removed.connect(self._retarget)
        viewer.dims.events.current_step.connect(lambda *_: self._timer.start())
        viewer.camera.events.zoom.connect(self._sync_zoom)
        for bar in self._scale_bars:
            apply_scale_bar(bar)
        self._sync_auto_tip()
        self._retarget()
        if self.layer is None:
            self._refresh_all()

    # ---------------------------------------------------------------- options
    def _build_options(self) -> None:
        """Le menu de la roue dentée : ce que l'on règle une fois pour toutes."""
        menu = self.options_menu = QMenu(self.options_btn)
        sat = menu.addMenu("Contraste « Auto » : pixels saturés")
        group = QActionGroup(sat)
        self.saturation_actions: dict[float, QAction] = {}
        for value in SATURATIONS:
            text = "aucun (du plus sombre au plus clair)" if value == 0 else f"{value:g} %".replace(".", ",")
            if value == AUTO_SATURATION:
                text += "  (comme ImageJ)"
            act = sat.addAction(text)
            act.setCheckable(True)
            act.setChecked(abs(value - self.auto_saturation) < 1e-9)
            act.triggered.connect(lambda _=False, v=value: self.set_auto_saturation(v))
            group.addAction(act)
            self.saturation_actions[value] = act
        self.smooth_action = menu.addAction("Lisser l'image agrandie")
        self.smooth_action.setCheckable(True)
        self.smooth_action.setChecked(self.smooth)
        self.smooth_action.setToolTip("Adoucit les pixels quand on zoome fort (affichage seulement)")
        self.smooth_action.toggled.connect(self.set_smooth)

        bar = menu.addMenu("Barre d'échelle")
        p = scale_bar_prefs()
        self.scale_actions: dict[str, QAction] = {}
        self.scale_choices: dict[tuple[str, Any], QAction] = {}

        def check(text: str, key: str) -> None:
            act = bar.addAction(text)
            act.setCheckable(True)
            act.setChecked(bool(p[key]))
            act.toggled.connect(lambda on, k=key: self.set_scale_bar(**{k: bool(on)}))
            self.scale_actions[key] = act

        def choices(title: str, key: str, options: Sequence[tuple[str, Any]]) -> QMenu:
            sub = bar.addMenu(title)
            grp = QActionGroup(sub)
            grp.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
            for text, value in options:
                act = sub.addAction(text)
                act.setCheckable(True)
                act.setChecked(p[key] == value)
                act.triggered.connect(lambda _=False, k=key, v=value: self.set_scale_bar(**{k: v}))
                grp.addAction(act)
                self.scale_choices[(key, value)] = act
            return sub

        check("Afficher", "visible")
        choices("Position", "position", SCALE_BAR_POSITIONS)
        choices("Couleur", "color", SCALE_BAR_COLORS)
        choices("Taille du texte", "font_size", SCALE_BAR_SIZES)
        lengths = [("Automatique", 0.0)] + [(f"{v:g}", float(v)) for v in SCALE_BAR_LENGTHS]
        sub = choices("Longueur (µm, ou pixels sans échelle)", "length", lengths)
        sub.addSeparator()
        self.scale_other = sub.addAction("Autre…", self._ask_scale_length)
        check("Fond sous le texte", "box")
        check("Traits aux extrémités", "ticks")

        menu.addSeparator()
        self.reset_action = menu.addAction("Revenir à l'affichage d'origine", self.reset_display)
        self.apply_action = menu.addAction("Appliquer le contraste aux pixels…", self.apply_to_pixels)
        self.apply_action.setToolTip(
            "Crée une nouvelle image dont les pixels valent ce que l'on voit (« Apply » d'ImageJ)"
        )
        menu.setToolTipsVisible(True)
        self.options_btn.setMenu(menu)

    def set_auto_saturation(self, percent: float) -> None:
        """Part des pixels que « Auto » laisse saturés de chaque côté (0 : aucun)."""
        self.auto_saturation = float(percent)
        settings().setValue("display/auto_saturation", self.auto_saturation)
        for value, act in self.saturation_actions.items():
            act.setChecked(abs(value - self.auto_saturation) < 1e-9)
        self._sync_auto_tip()
        if self.layer is not None:
            self.auto_contrast()

    def _sync_auto_tip(self) -> None:
        s = f"{self.auto_saturation:g}".replace(".", ",")
        self.auto_btn.setToolTip(
            f"Contraste automatique : {s} % des pixels saturés de chaque côté "
            "(réglable dans les options, roue dentée)"
        )

    def set_smooth(self, on: bool) -> None:
        """Pixels lissés (ou nets) quand on agrandit l'image : affichage seulement."""
        self.smooth = bool(on)
        settings().setValue("display/smooth", self.smooth)
        if self.smooth_action.isChecked() != self.smooth:
            self.smooth_action.setChecked(self.smooth)
        for layer in self.viewer.layers:
            self._apply_smooth(layer)

    def _apply_smooth(self, layer: Any) -> None:
        if isinstance(layer, ImageLayer):
            try:
                layer.interpolation2d = "linear" if self.smooth else "nearest"
            except Exception:  # calque sans ce réglage
                pass

    def set_scale_bar(self, **changes: Any) -> None:
        """Change des réglages de la barre d'échelle (retenus, appliqués aux deux espaces)."""
        s = settings()
        for key, value in changes.items():
            if key not in SCALE_BAR_DEFAULTS:
                raise KeyError(key)
            s.setValue(f"display/scale_bar/{key}", value)
        p = scale_bar_prefs()
        for bar in self._scale_bars:
            apply_scale_bar(bar, p)
        for key, act in self.scale_actions.items():
            if act.isChecked() != bool(p[key]):
                act.blockSignals(True)
                act.setChecked(bool(p[key]))
                act.blockSignals(False)
        known = False
        for (key, value), act in self.scale_choices.items():
            on = p[key] == value
            act.setChecked(on)
            known = known or (key == "length" and on)
        length = f"{p['length']:g}".replace(".", ",")
        self.scale_other.setText("Autre…" if known else f"Autre… ({length})")

    def _ask_scale_length(self) -> None:
        current = float(scale_bar_prefs()["length"]) or 100.0
        value, ok = QInputDialog.getDouble(
            self,
            "Barre d'échelle",
            "Longueur de la barre (en µm ; en pixels si l'image n'a pas d'échelle) :",
            current,
            0.001,
            1e6,
            3,
        )
        if ok:
            self.set_scale_bar(length=float(value))

    def reset_display(self) -> None:
        """Du pixel le plus sombre au plus clair, gamma à 1, palette à l'endroit."""
        if self.layer is None:
            return
        self.full_contrast()
        self.layer.gamma = 1.0
        if self.invert.isChecked():
            self.invert.setChecked(False)
        self.message.emit("Affichage d'origine")

    def apply_to_pixels(self) -> None:
        """Demande une nouvelle image dont les pixels valent ce que l'on voit."""
        if self.layer is None:
            return
        lo, hi = (float(v) for v in self.layer.contrast_limits)
        self.apply_requested.emit(self.layer.name, lo, hi, float(self.layer.gamma))

    # -------------------------------------------------------------------- zoom
    def _unit(self) -> float:
        """Taille d'un pixel de l'image dans les unités de la vue."""
        if self.layer is None:
            return 1.0
        return abs(float(self.layer.scale[-1])) or 1.0

    def zoom_percent(self) -> float:
        """Pixels de l'écran par pixel de l'image, en pour cent."""
        return float(self.viewer.camera.zoom) * self._unit() * self.devicePixelRatioF() * 100

    def set_zoom(self, percent: float) -> None:
        percent = min(max(float(percent), 1.0), 6400.0)
        self.viewer.camera.zoom = percent / 100 / (self._unit() * self.devicePixelRatioF())

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self.zoom_percent() * factor)

    def zoom_fit(self) -> None:
        """Toute l'image dans la fenêtre."""
        if self.layer is not None and hasattr(self.viewer, "fit_to_view"):
            try:
                self.viewer.fit_to_view(layers=[self.layer])
                return
            except Exception:  # anciennes versions : on recentre tout
                pass
        self.viewer.reset_view()

    def _sync_zoom(self, *_: Any) -> None:
        percent = self.zoom_percent()
        text = f"{percent:.0f} %" if percent >= 10 else f"{percent:.1f} %".replace(".", ",")
        self.zoom_btn.setText(text)

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

    def _masks(self) -> list[LabelsLayer]:
        return [ly for ly in self.viewer.layers if isinstance(ly, LabelsLayer)]

    def _mask(self) -> LabelsLayer | None:
        """Le masque visible du dessus (celui que l'on règle)."""
        return next((ly for ly in reversed(self._masks()) if ly.visible), None)

    def _inserted(self, event: Any = None) -> None:
        layer = getattr(event, "value", None)
        if layer is not None:
            self._apply_smooth(layer)
            layer.events.visible.connect(lambda *_: self._layers_timer.start())
            if isinstance(layer, LabelsLayer):
                self._style_mask(layer)
        self._retarget()

    def _retarget(self, *_: Any) -> None:
        layer = self._target()
        self._sync_docs()
        if layer is self.layer:
            self._update_visibility()
            self._sync_mask()
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
            ("scale", self._sync_zoom),
        ]

    def _data_changed(self, *_: Any) -> None:
        self._timer.start()

    # -------------------------------------------------------------- affichage
    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not self._more_chosen and self.width() > 0:
            # premier affichage : curseurs dépliés si la fenêtre a la place, repliés sinon
            self._more_chosen = True
            self.more_btn.blockSignals(True)
            self.more_btn.setChecked(self.width() >= _ROOMY)
            self.more_btn.blockSignals(False)
            self._update_visibility()

    def _more_toggled(self, on: bool) -> None:
        self._more_chosen = True
        settings().setValue("display/more", bool(on))
        self._update_visibility()

    def _update_visibility(self) -> None:
        has = self.layer is not None
        more = has and self.more_btn.isChecked()
        for w in (self.histogram, self.levels, self.view_row):
            w.setVisible(has)  # sans image, seulement « Aucune image » : la vue garde la place
        for w in (self.brightness_group, self.contrast_group, self.gamma_group):
            w.setVisible(more)
        self.colormap_group.setVisible(more and not bool(self.layer.rgb))
        self.mask_group.setVisible(more and self._mask() is not None)

    def _refresh_all(self) -> None:
        self._update_visibility()
        self._sync_info()
        self._refresh_plane()
        self._sync_controls()
        self._sync_zoom()
        self._sync_mask()

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
        self.name_label.setText(_elide(layer.name, 26))
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
        self._data_range = full_limits(self._plane)
        self.histogram.set_data(self._plane, rgb=bool(layer.rgb) and self._plane.ndim == 3)
        self._sync_faders()

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
            if not layer.rgb:
                name = layer.colormap.name
                inverted = name.endswith(_INVERTED)
                if inverted:
                    name = name[: -len(_INVERTED)]
                i = self.colormap.findData(name)
                if i < 0:  # palette posée ailleurs (canaux d'un fichier…) : ajoutée à la liste
                    self.colormap.addItem(name, name)
                    i = self.colormap.count() - 1
                self.colormap.setCurrentIndex(i)
                self.invert.setChecked(inverted)
        finally:
            self._syncing = False
        self._sync_faders()

    def _sync_faders(self) -> None:
        """Curseurs « luminosité » et « contraste » : une autre lecture des mêmes bornes."""
        if self.layer is None:
            return
        lo, hi = (float(v) for v in self.layer.contrast_limits)
        b, c = window_to_sliders(lo, hi, self._data_range)
        self._syncing = True
        try:
            if self._driving != "brightness":
                self.brightness.setValue(b)
            if self._driving != "contrast":
                self.contrast.setValue(c)
        finally:
            self._syncing = False
        self.brightness_label.setText(f"{self.brightness.value():+d}".replace("+0", "0"))
        self.contrast_label.setText(f"{self.contrast.value():+d}".replace("+0", "0"))

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
        self._set_limits(*self._whole(*auto_limits(self._plane, self.auto_saturation)))
        self.message.emit("Contraste automatique (affichage seulement)")

    def full_contrast(self) -> None:
        if self._plane is None:
            return
        self._set_limits(*self._whole(*full_limits(self._plane)))

    def _brightness_changed(self, value: int) -> None:
        if self._syncing or self.layer is None:
            return
        lo, hi = (float(v) for v in self.layer.contrast_limits)
        centre = sum(sliders_to_window(value, 0, self._data_range)) / 2
        self._drive("brightness", centre - (hi - lo) / 2, centre + (hi - lo) / 2)

    def _contrast_changed(self, value: int) -> None:
        if self._syncing or self.layer is None:
            return
        lo, hi = (float(v) for v in self.layer.contrast_limits)
        w_lo, w_hi = sliders_to_window(0, value, self._data_range)
        centre, half = (lo + hi) / 2, (w_hi - w_lo) / 2
        self._drive("contrast", centre - half, centre + half)

    def _drive(self, name: str, lo: float, hi: float) -> None:
        self._driving = name
        try:
            self._set_limits(lo, hi)
        finally:
            self._driving = ""
        self._sync_faders()

    def _released(self) -> None:
        self._sync_faders()

    def _reset_slider(self, slider: QSlider) -> None:
        if slider.value() == 0:  # déjà au neutre d'après le curseur : on recale quand même les bornes
            slider.valueChanged.emit(0)
        else:
            slider.setValue(0)

    def _gamma_changed(self, value: int) -> None:
        g = 5 ** (value / _GAMMA_LOG)
        self.gamma_label.setText(f"{g:.2f}".replace(".", ","))
        if not self._syncing and self.layer is not None:
            self.layer.gamma = g

    def _colormap_changed(self, *_: Any) -> None:
        if self._syncing or self.layer is None or self.layer.rgb or self.colormap.currentIndex() < 0:
            return
        name = self.colormap.currentData()
        self.layer.colormap = _reversed_colormap(name) if self.invert.isChecked() else name

    # ------------------------------------------------------------------ masque
    def _style_mask(self, layer: LabelsLayer) -> None:
        """Un nouveau masque prend la transparence et le style choisis pour les précédents."""
        s = settings()
        if s.contains("display/mask_opacity"):
            layer.opacity = _pref(s, "display/mask_opacity", 70) / 100
        if s.contains("display/mask_contour"):
            layer.contour = _CONTOUR if _pref(s, "display/mask_contour", False) else 0

    def _sync_mask(self) -> None:
        mask = self._mask()
        if mask is None:
            return
        self._syncing = True
        try:
            self.mask_opacity.setValue(int(round(float(mask.opacity) * 100)))
            self.mask_contour.setChecked(int(mask.contour) > 0)
        finally:
            self._syncing = False

    def _mask_changed(self, *_: Any) -> None:
        if self._syncing:
            return
        s = settings()
        s.setValue("display/mask_opacity", self.mask_opacity.value())
        s.setValue("display/mask_contour", self.mask_contour.isChecked())
        for layer in self._masks():
            layer.opacity = self.mask_opacity.value() / 100
            layer.contour = _CONTOUR if self.mask_contour.isChecked() else 0


def _pref(store: Any, key: str, default: Any) -> Any:
    """Une préférence, du type de sa valeur par défaut."""
    try:
        return store.value(key, default, type=type(default))
    except (TypeError, ValueError):
        return default


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


__all__ = [
    "COLORMAPS",
    "ContrastHistogram",
    "DisplayBar",
    "apply_scale_bar",
    "auto_limits",
    "full_limits",
    "scale_bar_of",
    "scale_bar_prefs",
    "sliders_to_window",
    "window_to_sliders",
]
