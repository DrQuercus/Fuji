"""Dessin sur l'image : style des ROI et des points, numéros et mesures affichés.

Les formes dessinées (calque « ROI ») et les points posés (calque « Points ») prennent le
style choisi dans les réglages du dessin : couleur, épaisseur du trait, remplissage, taille
des points. L'épaisseur et la taille sont données en pixels **de l'écran** : un trait reste
visible quand on regarde en entier une image 4K, et ne grossit pas quand on zoome.

À côté de chaque forme s'affichent son numéro (celui de sa ligne dans le tableau de
« Mesurer les ROI ») et sa mesure, mise à jour pendant qu'on la dessine : longueur d'une
ligne ou d'un trajet, surface d'une région, angle d'un tracé à trois points.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from napari.components import ViewerModel
from napari.layers import Image as ImageLayer
from napari.layers import Points as PointsLayer
from napari.layers import Shapes as ShapesLayer
from qtpy.QtCore import QObject, QTimer, Signal
from qtpy.QtGui import QAction, QActionGroup, QColor
from qtpy.QtWidgets import QMenu, QWidget

from fuji.analysis import Rois

from .documents import camera_of
from .prefs import settings

#: choix proposés dans le menu des réglages : (texte, valeur)
COLORS = (
    ("Automatique (selon l'image)", "auto"),
    ("Jaune", "#ffd400"),
    ("Rouge", "#ff4d4d"),
    ("Vert", "#3ddc84"),
    ("Cyan", "#22d3ee"),
    ("Magenta", "#ff4fd8"),
    ("Blanc", "#ffffff"),
    ("Noir", "#000000"),
)
WIDTHS = (("Très fin", 1.0), ("Fin", 1.5), ("Moyen", 2.5), ("Épais", 4.0), ("Très épais", 6.0))
FILLS = (("Aucun", 0.0), ("Léger", 0.08), ("Moyen", 0.25), ("Fort", 0.5))
POINT_SIZES = (("Très petits", 6.0), ("Petits", 9.0), ("Moyens", 12.0), ("Gros", 18.0), ("Très gros", 26.0))
TEXT_SIZES = (("Petit", 8.0), ("Normal", 10.0), ("Grand", 13.0), ("Très grand", 18.0))
_TEXT = "texte"  # colonne des étiquettes affichées
#: couleur « automatique » : jaune sur une image sombre (fluorescence), vert foncé sur une
#: image claire (fond clair), où le jaune se lit mal
ON_DARK, ON_LIGHT = "#ffd400", "#00843d"
#: bornes des réglages chiffrés (des préférences abîmées ne donnent ni trait nul ni trait géant)
_LIMITS = {"width": (0.5, 20.0), "fill": (0.0, 1.0), "point_size": (2.0, 100.0), "text_size": (5.0, 40.0)}
#: outils où la forme naît d'un glisser : un simple clic n'en crée pas
_DRAG_MODES = ("add_rectangle", "add_ellipse", "add_line")
_MIN_DRAG = 4.0  # en pixels de l'écran


@dataclass
class DrawingStyle:
    """Les réglages du dessin (retenus d'une séance à l'autre)."""

    color: str = "auto"
    width: float = 1.5  # épaisseur du trait, en pixels de l'écran
    fill: float = 0.08  # opacité du remplissage des régions (0 : aucun)
    point_size: float = 12.0  # diamètre des points, en pixels de l'écran
    numbers: bool = True
    measures: bool = True
    text_size: float = 10.0

    @classmethod
    def load(cls) -> DrawingStyle:
        s = settings()
        values = {}
        for f in fields(cls):
            try:
                values[f.name] = s.value(f"drawing/{f.name}", f.default, type=type(f.default))
            except (TypeError, ValueError):
                values[f.name] = f.default
        style = cls(**values)
        if style.color != "auto" and not QColor(style.color).isValid():
            style.color = cls.color
        for name, (lo, hi) in _LIMITS.items():
            value = float(getattr(style, name))
            setattr(style, name, min(hi, max(lo, value)) if np.isfinite(value) else getattr(cls, name))
        return style

    def save(self) -> None:
        s = settings()
        for f in fields(self):
            s.setValue(f"drawing/{f.name}", getattr(self, f.name))


def number_text(value: float) -> str:
    """Un nombre lisible à côté d'une forme : quatre chiffres utiles, virgule décimale."""
    text = f"{value:.0f}" if abs(value) >= 10000 else f"{value:.4g}"
    return text.replace(".", ",")


def shape_text(rois: Rois, index: int, *, numbers: bool = True, measures: bool = True) -> str:
    """L'étiquette d'une forme : « 3 : 12,5 µm » (ligne), « 1 : 340 µm² » (région),
    « 2 : 47,5° » (tracé à trois points : l'angle au sommet du milieu)."""
    parts = []
    if numbers:
        parts.append(str(index + 1))
    if measures:
        if rois.kind(index) == "path" and len(rois.vertices(index)) == 3:
            parts.append(f"{number_text(rois.angle(index))}°")
        elif rois.is_line(index):
            parts.append(f"{number_text(rois.length(index) * rois.px)} {rois.unit}")
        else:
            parts.append(f"{number_text(rois.area(index) * rois.px**2)} {rois.unit}²")
    return " : ".join(parts)


def _pixel_size(layer: Any) -> float | None:
    px = layer.metadata.get("pixel_size_um")
    if px is None and float(layer.scale[-1]) != 1.0:
        px = float(layer.scale[-1])
    return float(px) if px else None


def layer_rois(layer: ShapesLayer, reference: Any | None = None) -> Rois:
    """Les formes d'un calque, en pixels de l'image ``reference`` (celle qui est affichée) et
    avec son échelle : le calque de dessin est commun à toutes les images ouvertes, qui n'ont
    pas toutes la même. Sans image : en pixels du calque lui-même."""
    shapes = []
    for kind, verts in zip(layer.shape_type, layer.data, strict=True):
        v = np.asarray(verts, dtype=np.float64)[:, -2:]
        if reference is not None:
            world = v * np.asarray(layer.scale[-2:]) + np.asarray(layer.translate[-2:])
            v = (world - np.asarray(reference.translate[-2:])) / np.asarray(reference.scale[-2:])
        shapes.append((str(kind), v))
    return Rois(shapes, _pixel_size(reference if reference is not None else layer), layer.name)


def _drop_stale_selection(layer: Any) -> None:
    """Oublie la sélection de formes ou de points qui n'existent plus (calque vidé pendant
    qu'ils étaient sélectionnés) : napari leur appliquerait le style « en cours », et lèverait
    une erreur au zoom suivant."""
    count = len(layer.data)
    if any(i >= count for i in layer.selected_data):
        layer.selected_data = {i for i in layer.selected_data if i < count}


class DrawingStyler(QObject):
    """Tient les calques de dessin d'une vue au style choisi, avec leurs étiquettes."""

    #: le nombre de formes et de points a changé
    counted = Signal(int, int)

    def __init__(
        self, viewer: ViewerModel, shapes_name: str, points_name: str, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.style = DrawingStyle.load()
        self._names = (shapes_name, points_name)
        self._texts: dict[int, list[str]] = {}
        self._applied_color = ON_DARK
        self._timer = QTimer(self)  # plusieurs changements d'affilée : une seule mise à jour
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self.refresh)
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(80)
        self._zoom_timer.timeout.connect(self._rescale)
        # formes ou points collés (Ctrl+V) : napari ne prévient pas, on compare les comptes
        self._watch = QTimer(self)
        self._watch.setInterval(400)
        self._watch.timeout.connect(self._check_counts)
        self._watch.start()
        viewer.layers.events.inserted.connect(self._inserted)
        viewer.layers.events.removed.connect(self._schedule)
        camera_of(viewer).events.zoom.connect(self._zoomed)
        for layer in viewer.layers:
            self._follow_image(layer)
            self._adopt(layer)

    def shutdown(self) -> None:
        """Fenêtre fermée : plus rien à surveiller."""
        for timer in (self._timer, self._zoom_timer, self._watch):
            timer.stop()

    # ------------------------------------------------------------------ calques
    def shapes(self) -> ShapesLayer | None:
        layer = self.viewer.layers[self._names[0]] if self._names[0] in self.viewer.layers else None
        return layer if isinstance(layer, ShapesLayer) else None

    def points(self) -> PointsLayer | None:
        layer = self.viewer.layers[self._names[1]] if self._names[1] in self.viewer.layers else None
        return layer if isinstance(layer, PointsLayer) else None

    def _schedule(self, *_: Any) -> None:
        self._timer.start()

    def _zoomed(self, *_: Any) -> None:
        self._zoom_timer.start()

    def image(self) -> ImageLayer | None:
        """L'image affichée (celle du dessus) : c'est sur elle que l'on dessine et mesure."""
        return next(
            (ly for ly in reversed(self.viewer.layers) if isinstance(ly, ImageLayer) and ly.visible), None
        )

    def _inserted(self, event: Any) -> None:
        layer = getattr(event, "value", None)
        self._follow_image(layer)
        self._adopt(layer)

    def _follow_image(self, layer: Any) -> None:
        """Autre image affichée, ou échelle changée : la couleur automatique et l'unité des
        mesures peuvent changer."""
        if isinstance(layer, ImageLayer):
            layer.events.visible.connect(self._schedule)
            layer.events.scale.connect(self._schedule)
            self._timer.start()

    def _check_counts(self) -> None:
        for layer in (self.shapes(), self.points()):
            if layer is not None and len(layer.data) != len(self._texts.get(id(layer), ())):
                self._timer.start()
                return

    def _adopt(self, layer: Any) -> None:
        if layer is None or layer.name not in self._names:
            return
        if not isinstance(layer, (ShapesLayer, PointsLayer)):
            return
        layer.events.data.connect(self._schedule)
        layer.events.scale.connect(self._zoomed)
        layer.events.name.connect(self._keep_name)
        if self._follow_drag not in layer.mouse_drag_callbacks:
            layer.mouse_drag_callbacks.append(self._follow_drag)
        self._texts.pop(id(layer), None)
        self.apply(layer)
        self._relabel(layer)
        self._count()

    def _keep_name(self, event: Any) -> None:
        """Un calque de dessin garde son nom : c'est par lui que les outils et les mesures le
        retrouvent (renommé, ses étiquettes se figeaient et un second calque « ROI » naissait)."""
        layer = getattr(event, "source", None)
        if not isinstance(layer, (ShapesLayer, PointsLayer)) or layer not in self.viewer.layers:
            return
        wanted = self._names[0] if isinstance(layer, ShapesLayer) else self._names[1]
        taken = wanted in self.viewer.layers and self.viewer.layers[wanted] is not layer
        if layer.name != wanted and not taken:
            layer.name = wanted

    def _follow_drag(self, layer: Any, event: Any) -> Any:
        """Pendant que l'on dessine ou déplace une forme, sa mesure suit la souris."""
        drawn = isinstance(layer, ShapesLayer) and str(layer.mode) in _DRAG_MODES
        before = len(layer.data)
        yield
        while event.type == "mouse_move":
            if not self._timer.isActive():  # relancé à chaque mouvement, il n'expirait jamais
                self._timer.start()
            yield
        if drawn:
            QTimer.singleShot(0, lambda: self._drop_click(layer, before))
        self._timer.start()

    def _drop_click(self, layer: Any, before: int) -> None:
        """Un simple clic avec le rectangle, l'ellipse ou la ligne laissait une forme de deux
        pixels : invisible, mais comptée et mesurée. Elle est retirée."""
        try:
            if len(layer.data) != before + 1:
                return
            verts = np.asarray(layer.data[-1], dtype=np.float64)[:, -2:]
            extent = (verts.max(axis=0) - verts.min(axis=0)) * np.abs(np.asarray(layer.scale[-2:]))
            if float(extent.max()) * float(camera_of(self.viewer).zoom) >= _MIN_DRAG:
                return
            layer.selected_data = {before}
            layer.remove_selected()
        except Exception:  # calque fermé entre-temps
            return
        self._timer.start()

    # -------------------------------------------------------------------- style
    def set_style(self, **changes: Any) -> None:
        """Change des réglages du dessin : appliqués aux formes déjà dessinées, et retenus."""
        for key, value in changes.items():
            if key not in {f.name for f in fields(self.style)}:
                raise KeyError(key)
            setattr(self.style, key, value)
        self.style.save()
        for layer in (self.shapes(), self.points()):
            if layer is not None:
                self.apply(layer)
                self._relabel(layer, force=True)

    def _on_screen(self, layer: Any, pixels: float) -> float:
        """Taille, dans les unités du calque, de ``pixels`` pixels de l'écran."""
        zoom = float(camera_of(self.viewer).zoom) * abs(float(layer.scale[-1]))
        return pixels / zoom if zoom > 0 else pixels

    def color(self) -> str:
        """La couleur du dessin ; « automatique » : d'après la clarté de l'image affichée."""
        if self.style.color != "auto":
            return self.style.color
        image = self.image()
        if image is None:
            return ON_DARK
        try:
            data = image.data
            lead = data.ndim - (3 if image.rgb else 2)
            plane = data[(0,) * lead] if lead > 0 else data
            step = max(1, max(plane.shape[:2]) // 64)
            sample = np.asarray(plane[::step, ::step], dtype=np.float64)
            lo, hi = (float(v) for v in image.contrast_limits)
            level = (float(sample.mean()) - lo) / (hi - lo) if hi > lo else 0.0
        except Exception:  # image illisible pour l'instant : la couleur d'origine
            return ON_DARK
        return ON_LIGHT if level > 0.55 else ON_DARK

    def apply(self, layer: Any) -> None:
        """Met un calque de dessin au style choisi."""
        _drop_stale_selection(layer)
        style = self.style
        color = self._applied_color = self.color()
        rgba = QColor(color).getRgbF()
        if isinstance(layer, ShapesLayer):
            face = [rgba[0], rgba[1], rgba[2], style.fill]
            width = self._on_screen(layer, style.width)
            layer.current_edge_color = color
            layer.current_face_color = face
            layer.current_edge_width = width
            if len(layer.data):
                layer.edge_color = color
                layer.face_color = face
                layer.edge_width = [width] * len(layer.data)
            anchor = "upper_left"
        else:
            size = self._on_screen(layer, style.point_size)
            dark = QColor(color).lightness() < 90
            layer.current_face_color = color
            layer.current_border_color = "#ffffff" if dark else "#000000"
            layer.current_size = size
            if len(layer.data):
                layer.face_color = color
                layer.border_color = "#ffffff" if dark else "#000000"
                layer.size = size
            anchor = "upper_right"
        layer.text = {
            "string": "{" + _TEXT + "}",
            "size": style.text_size,
            "color": color,
            "anchor": anchor,
            "visible": style.numbers or (style.measures and isinstance(layer, ShapesLayer)),
        }

    def _rescale(self) -> None:
        """Après un zoom : traits et points gardent leur taille à l'écran."""
        shapes, points = self.shapes(), self.points()
        for layer in (shapes, points):
            if layer is not None:
                _drop_stale_selection(layer)
        if shapes is not None:
            width = self._on_screen(shapes, self.style.width)
            shapes.current_edge_width = width
            if len(shapes.data):
                shapes.edge_width = [width] * len(shapes.data)
        if points is not None:
            size = self._on_screen(points, self.style.point_size)
            points.current_size = size
            if len(points.data):
                points.size = size

    # --------------------------------------------------------------- étiquettes
    def texts(self, layer: Any) -> list[str]:
        """Les étiquettes d'un calque de dessin, dans l'ordre de ses formes ou de ses points."""
        style = self.style
        if isinstance(layer, ShapesLayer):
            rois = layer_rois(layer, self.image())
            return [
                shape_text(rois, i, numbers=style.numbers, measures=style.measures) for i in range(len(rois))
            ]
        return [str(i + 1) if style.numbers else "" for i in range(len(layer.data))]

    def _relabel(self, layer: Any, force: bool = False) -> None:
        try:
            texts = self.texts(layer)
        except Exception:  # forme en cours de construction (sommets incomplets) : au prochain passage
            return
        if not force and self._texts.get(id(layer)) == texts:
            return
        self._texts[id(layer)] = texts
        layer.features = pd.DataFrame({_TEXT: pd.Series(texts, dtype=object)})
        # une forme en cours de tracé n'a pas encore d'étiquette : rien, plutôt que « None » ou
        # celle de la forme précédente (la valeur par défaut de napari)
        layer.feature_defaults = {_TEXT: ""}
        layer.refresh_text()

    def _count(self) -> None:
        shapes, points = self.shapes(), self.points()
        self.counted.emit(
            len(shapes.data) if shapes is not None else 0, len(points.data) if points is not None else 0
        )

    def refresh(self) -> None:
        """Remet à jour les étiquettes et le compte des formes et des points."""
        recolor = self.style.color == "auto" and self.color() != self._applied_color
        for layer in (self.shapes(), self.points()):
            if layer is not None:
                if recolor:
                    self.apply(layer)
                self._relabel(layer, force=recolor)
        self._count()

    # --------------------------------------------------------------------- menu
    def build_menu(self, parent: QWidget) -> QMenu:
        """Le menu des réglages du dessin (bouton à roue dentée de la rangée « Dessiner »)."""
        menu = QMenu(parent)
        self.choices: dict[tuple[str, Any], QAction] = {}
        self.checks: dict[str, QAction] = {}

        def choices(title: str, key: str, options: Any) -> None:
            sub = menu.addMenu(title)
            group = QActionGroup(sub)
            for text, value in options:
                act = sub.addAction(text)
                act.setCheckable(True)
                act.setChecked(getattr(self.style, key) == value)
                act.triggered.connect(lambda _=False, k=key, v=value: self.set_style(**{k: v}))
                group.addAction(act)
                self.choices[(key, value)] = act

        def check(text: str, key: str, tip: str) -> None:
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(bool(getattr(self.style, key)))
            act.setToolTip(tip)
            act.toggled.connect(lambda on, k=key: self.set_style(**{k: bool(on)}))
            self.checks[key] = act

        choices("Couleur", "color", COLORS)
        choices("Épaisseur du trait", "width", WIDTHS)
        choices("Remplissage des régions", "fill", FILLS)
        choices("Taille des points", "point_size", POINT_SIZES)
        menu.addSeparator()
        check("Numéros", "numbers", "Le numéro de chaque forme et de chaque point (celui du tableau)")
        check(
            "Mesures sur l'image",
            "measures",
            "Longueur des lignes, surface des régions, angle des tracés à trois points",
        )
        choices("Taille du texte", "text_size", TEXT_SIZES)
        menu.setToolTipsVisible(True)
        return menu


__all__ = ["DrawingStyle", "DrawingStyler", "layer_rois", "number_text", "shape_text"]
