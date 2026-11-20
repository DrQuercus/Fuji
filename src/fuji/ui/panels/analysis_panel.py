"""Panneau « Outils » : toutes les opérations, rangées par tâche, avec leur mode d'emploi.

Trois pages :

- les **menus** (« Améliorer l'image », « Isoler les objets », « Compter et mesurer »…),
  avec une ligne d'explication par outil, la recherche et les démarches guidées ;
- la **fiche** d'un outil : ce qu'il fait, quand l'utiliser, son équivalent ImageJ, puis
  ses réglages (construits à partir de la signature Python de l'opération), l'aperçu en
  direct et « Exécuter » ;
- une **démarche guidée** : les étapes à suivre pour un besoin courant (compter des
  cellules, mesurer une surface…), chacune ouvrant le bon outil.

Chaque opération du registre (y compris celles ajoutées par des extensions) apparaît
ici. Les calques napari servent d'entrées et de sorties :

=================  =========================================================
calque             rôle
=================  =========================================================
image              image à traiter ou à mesurer
étiquettes         masque binaire ou objets numérotés
formes             régions d'intérêt (ROI) : rectangle, ellipse, polygone, ligne
points             points posés à la main ou trouvés (maxima)
=================  =========================================================

Comme dans ImageJ : « Aperçu » montre le résultat pendant le réglage, et une opération
peut s'appliquer au plan affiché ou à toute la pile.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image as ImageLayer
from napari.layers import Labels as LabelsLayer
from napari.layers import Points as PointsLayer
from napari.layers import Shapes as ShapesLayer
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji.analysis import Image, Labels, Points, Rois, Table, list_operations
from fuji.analysis.guide import GUESS, RECIPES, Recipe, help_of
from fuji.analysis.imagej import open_in_fiji
from fuji.analysis.ops.adjust import WHITE_KEY, white_level
from fuji.analysis.registry import Operation, combine_planes

from .. import documents
from ..components import button, icon_button, label, separator
from ..drawing import DrawingStyler
from ..flow import FlowLayout
from ..prefs import settings
from .param_form import ParamForm, labelled
from .tool_menu import RecipePage, ToolMenu

_NONE = "(aucune)"
ROI_LAYER = documents.ROI_LAYER
POINTS_LAYER = documents.POINTS_LAYER
PREVIEW_PREFIX = documents.PREVIEW_PREFIX
_LAYER_TYPES = {Image: ImageLayer, Labels: LabelsLayer, Rois: ShapesLayer, Points: PointsLayer}
_INPUT_LABELS = {
    "image": "Image",
    "image1": "Image 1",
    "image2": "Image 2",
    "mask": "Masque",
    "mask1": "Masque 1",
    "mask2": "Masque 2",
    "intensity": "Intensités",
    "flat": "Champ plat",
    "dark": "Noir",
    "roi": "ROI",
    "background": "Zone de fond",
    "points": "Points",
    "red": "Rouge",
    "green": "Vert",
    "blue": "Bleu",
}
#: ce qu'il faut faire quand une entrée manque, selon son type
_MISSING_INPUT = {
    Image: "Il manque une image pour « {label} » : ouvrez-en une (double-clic dans la liste « Images »).",
    Labels: "Il manque un masque pour « {label} » : créez-le d'abord (menu « Isoler les objets »).",
    Rois: "Il manque une ROI : dessinez-la sur l'image (outils « Dessiner », en haut du panneau).",
    Points: "Il manque des points : posez-les sur l'image (outil « Points », en haut du panneau).",
}
_TOOL_WIDTH = 32  # largeur d'un bouton de la rangée « Dessiner »
#: outils de dessin : (icône, info-bulle, type de calque, mode napari)
_TOOLS = (
    ("roi_rect", "Rectangle", "shapes", "add_rectangle"),
    ("roi_ellipse", "Ellipse", "shapes", "add_ellipse"),
    ("roi_polygon", "Polygone (double-clic pour fermer)", "shapes", "add_polygon"),
    ("roi_free", "Main levée : contour tracé à la souris, bouton enfoncé", "shapes", "add_polygon_lasso"),
    ("roi_line", "Ligne (profil, distance)", "shapes", "add_line"),
    (
        "roi_path",
        "Trajet : ligne brisée, un clic par point (double-clic pour finir)",
        "shapes",
        "add_polyline",
    ),
    (
        "roi_angle",
        "Angle : trois clics, le sommet de l'angle en deuxième (double-clic au troisième)",
        "shapes",
        "add_polyline",
    ),
    ("roi_point", "Points (comptage)", "points", "add"),
    ("pointer", "Sélectionner et déplacer les formes (Suppr : effacer la forme choisie)", "shapes", "select"),
    ("roi_edit", "Modifier une forme : tirer ses sommets", "shapes", "direct"),
    ("move", "Déplacer l'image (outil normal)", "", "pan_zoom"),
)


# ------------------------------------------------------------ calques → analyse
def _pixel_size(layer: Any) -> float | None:
    px = layer.metadata.get("pixel_size_um")
    if px is None and float(layer.scale[-1]) != 1.0:
        px = float(layer.scale[-1])
    return px


#: au-delà, les valeurs extrêmes d'une pile sont lues sur le plan affiché seulement
_RANGE_LIMIT = 200_000_000


def value_range(layer: Any, viewer: ViewerModel | None = None) -> tuple[float, float] | None:
    """Valeurs extrêmes de l'image, exactes et sans les NaN : celles de toute la pile quand
    elle tient en mémoire (un seuil proposé ne doit écarter aucun pixel d'aucun plan), sinon
    celles du plan affiché. ``None`` si l'image n'a aucune valeur finie."""
    data = layer.data
    if int(np.prod(data.shape)) > _RANGE_LIMIT and viewer is not None:
        data, _ = displayed_plane(layer, viewer)
    values = np.asarray(data)
    if _is_rgb(layer) and values.shape[-1] == 4:
        values = values[..., :3]  # la transparence n'est pas une valeur de l'image
    if values.dtype.kind == "f":
        if not np.isfinite(values).any():
            return None
        return float(np.nanmin(np.where(np.isfinite(values), values, np.nan))), float(
            np.nanmax(np.where(np.isfinite(values), values, np.nan))
        )
    if not values.size:
        return None
    return float(values.min()), float(values.max())


def layer_white(layer: Any) -> float | None:
    """Le « blanc » d'un calque : celui de la caméra quand on la connaît (4095 en 12 bits),
    sinon celui de toute la pile, pour que chaque plan soit traité à la même échelle. ``None`` :
    une image seule, qui le dira elle-même."""
    data = layer.data
    bits = layer.metadata.get("bit_depth")
    try:
        if bits and np.dtype(data.dtype).kind == "u" and 0 < int(bits) <= np.dtype(data.dtype).itemsize * 8:
            return float(2 ** int(bits) - 1)
    except (TypeError, ValueError):
        pass
    if int(data.ndim) - (3 if _is_rgb(layer) else 2) <= 0:
        return None
    cached = layer.metadata.get("_fuji_white")
    if cached and cached[0] == id(data):
        return float(cached[1])
    if int(np.prod(data.shape)) > _RANGE_LIMIT:
        return None
    values = np.asarray(data)
    value = white_level(values[..., :3] if _is_rgb(layer) else values)
    layer.metadata["_fuji_white"] = (id(data), value)
    return value


def _is_rgb(layer: Any) -> bool:
    return bool(getattr(layer, "rgb", False))


def plane_index(layer: Any, viewer: ViewerModel) -> tuple[int, ...]:
    """Indices du plan affiché, pour un calque qui a plus de deux dimensions."""
    extra = int(layer.ndim) - 2  # layer.ndim ne compte pas l'axe des couleurs
    if extra <= 0:
        return ()
    coords = layer.world_to_data(viewer.dims.point)
    shape = layer.data.shape
    return tuple(
        int(np.clip(round(float(c)), 0, n - 1)) for c, n in zip(coords[:extra], shape[:extra], strict=True)
    )


def displayed_plane(layer: Any, viewer: ViewerModel) -> tuple[np.ndarray, str]:
    """Plan 2D actuellement affiché d'un calque (et un suffixe de nom qui le désigne).

    Les opérations d'analyse travaillent sur une image 2D : pour une pile (temps, Z...),
    on prend le plan visible, comme ImageJ traite la coupe courante.
    """
    index = plane_index(layer, viewer)
    if not index:
        return np.asarray(layer.data), ""
    return np.asarray(layer.data[index]), " [" + ", ".join(str(i + 1) for i in index) + "]"


def layer_to_image(layer: Any, viewer: ViewerModel | None = None) -> Image:
    """Image d'analyse à partir d'un calque. Avec ``viewer`` : seulement le plan affiché."""
    md, rgb = dict(layer.metadata), _is_rgb(layer)
    white = layer_white(layer)
    if white:
        md[WHITE_KEY] = white
    if viewer is None:
        return Image(np.asarray(layer.data), _pixel_size(layer), layer.name, md, rgb)
    data, suffix = displayed_plane(layer, viewer)
    return Image(data, _pixel_size(layer), layer.name + suffix, md, rgb)


def layer_to_labels(layer: Any, viewer: ViewerModel | None = None) -> Labels:
    if viewer is None:
        return Labels(np.asarray(layer.data), _pixel_size(layer), layer.name)
    data, suffix = displayed_plane(layer, viewer)
    return Labels(data, _pixel_size(layer), layer.name + suffix)


def _to_reference_pixels(layer: Any, coords: np.ndarray, reference: Any | None) -> np.ndarray:
    """Coordonnées (ligne, colonne) d'un calque de formes ou de points, exprimées en pixels
    du calque de référence (l'image mesurée) : les deux calques peuvent avoir des échelles
    différentes."""
    world = coords[:, -2:] * np.asarray(layer.scale[-2:]) + np.asarray(layer.translate[-2:])
    if reference is None:
        return coords[:, -2:].astype(np.float64)
    return (world - np.asarray(reference.translate[-2:])) / np.asarray(reference.scale[-2:])


def _on_displayed_plane(layer: Any, coords: np.ndarray, viewer: ViewerModel | None) -> bool:
    """Vrai si une forme (ou un point) d'un calque à plus de deux dimensions est sur le
    plan affiché : ce sont les seules que l'on voit."""
    extra = coords.shape[1] - 2
    if extra <= 0 or viewer is None:
        return True
    current = np.round(np.asarray(layer.world_to_data(viewer.dims.point))[:extra])
    return bool(np.all(np.round(coords[0, :extra]) == current))


def layer_to_rois(layer: Any, reference: Any | None = None, viewer: ViewerModel | None = None) -> Rois:
    shapes: list[tuple[str, np.ndarray]] = []
    for kind, verts in zip(layer.shape_type, layer.data, strict=True):
        v = np.asarray(verts, dtype=np.float64)
        if not _on_displayed_plane(layer, v, viewer):
            continue
        shapes.append((str(kind), _to_reference_pixels(layer, v, reference)))
    px = _pixel_size(reference) if reference is not None else _pixel_size(layer)
    return Rois(shapes, px, layer.name)


def layer_to_points(layer: Any, reference: Any | None = None, viewer: ViewerModel | None = None) -> Points:
    data = np.asarray(layer.data, dtype=np.float64).reshape(-1, max(2, int(layer.ndim)))
    keep = [i for i in range(len(data)) if _on_displayed_plane(layer, data[i : i + 1], viewer)]
    px = _pixel_size(reference) if reference is not None else _pixel_size(layer)
    return Points(_to_reference_pixels(layer, data[keep], reference), px, layer.name)


def _same_kind(layer: Any, item: Any) -> bool:
    """Ce résultat peut-il remplacer sur place les pixels de ce calque d'aperçu ?"""
    if isinstance(item, Labels):
        ok = isinstance(layer, LabelsLayer)
    elif isinstance(item, Image):
        ok = isinstance(layer, ImageLayer) and bool(layer.rgb) == bool(item.is_rgb)
    else:
        return False
    data = np.asarray(item.data)
    return ok and tuple(layer.data.shape) == tuple(data.shape) and layer.data.dtype == data.dtype


def _display_like(layer: Any, like: Any | None) -> None:
    """Donne à une image de résultat les réglages d'affichage de l'image traitée, quand elles
    sont de même nature (même type de pixels, toutes deux en couleur ou en gris)."""
    if not isinstance(like, ImageLayer) or not isinstance(layer, ImageLayer):
        return
    if bool(like.rgb) != bool(layer.rgb) or like.data.dtype != layer.data.dtype:
        return
    try:
        lo, hi = (float(v) for v in like.contrast_limits)
        r_lo, r_hi = layer.contrast_limits_range
        layer.contrast_limits_range = (min(lo, r_lo), max(hi, r_hi))
        layer.contrast_limits = (lo, hi)
        layer.gamma = float(like.gamma)
        if not layer.rgb:
            layer.colormap = like.colormap
    except Exception:  # réglage refusé par napari : l'affichage automatique reste
        pass


def run_on_planes(op: Operation, plane_inputs: list[list[Any]], params: dict[str, Any]) -> list[Any]:
    """Applique une opération à chaque plan (exécuté dans un thread de travail)."""
    return [op.func(*inputs, **params) for inputs in plane_inputs]


class AnalysisPanel(QWidget):
    table_ready = Signal(object)  # Table
    message = Signal(str)

    def __init__(self, viewer: ViewerModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.expert = False
        self._op: Operation | None = None
        self._inputs: dict[str, QComboBox] = {}
        self._params = ParamForm()
        self._params.changed.connect(self._schedule_preview)
        viewer.dims.events.current_step.connect(self._schedule_preview)  # autre plan d'une pile
        self._busy = False
        self._preview_pending = False
        self._run_pending = False
        self._pending_op: Operation | None = None
        self._recipe: Recipe | None = None
        self._step = 0
        self._done: set[int] = set()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # outils de dessin des ROI (ils agissent sur un calque de formes ou de points) ; la
        # rangée passe à la ligne si le panneau est étroit
        tools_box = QWidget()
        tools = FlowLayout(tools_box, spacing=2, line_spacing=2)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.addWidget(label("Dessiner :", "hint", wrap=False))
        self.tool_buttons: dict[str, QPushButton] = {}
        self._tool = "move"  # l'outil actif (clé de son bouton)
        for icon_name, tip, kind, mode in _TOOLS:
            b = icon_button(icon_name, tip, checkable=True)
            b.setFixedWidth(_TOOL_WIDTH)
            b.clicked.connect(lambda _=False, k=kind, m=mode, n=icon_name: self.set_tool(k, m, n))
            tools.addWidget(b)
            self.tool_buttons[icon_name] = b
        self.tool_buttons["move"].setChecked(True)
        clear = icon_button("trash", "Effacer toutes les ROI et tous les points")
        clear.clicked.connect(self.clear_rois)
        clear.setFixedWidth(_TOOL_WIDTH)
        tools.addWidget(clear)
        # réglages du dessin : couleur, épaisseur, numéros et mesures affichés sur l'image
        self.drawing = DrawingStyler(viewer, ROI_LAYER, POINTS_LAYER, self)
        self.drawing_btn = icon_button(
            "gear", "Réglages du dessin : couleur, épaisseur, remplissage, numéros, mesures sur l'image"
        )
        self.drawing_menu = self.drawing.build_menu(self.drawing_btn)
        self.drawing_btn.setMenu(self.drawing_menu)
        self.drawing_btn.setFixedWidth(_TOOL_WIDTH)
        tools.addWidget(self.drawing_btn)
        self.roi_count = label("", "hint", wrap=False)
        self.roi_count.setToolTip("Ce qui est dessiné sur l'image (les points servent à compter)")
        self.roi_count.hide()
        self.drawing.counted.connect(self._show_count)
        tools.addWidget(self.roi_count)
        policy = tools_box.sizePolicy()
        policy.setHeightForWidth(True)
        tools_box.setSizePolicy(policy)
        tools_box.setToolTip("Dessiner sur l'image : régions (ROI), lignes, points")
        lay.addWidget(tools_box)

        # page 1 : les menus ------------------------------------------------------
        self.menu_page = QWidget()
        mlay = QVBoxLayout(self.menu_page)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.setSpacing(8)
        self.menu = ToolMenu()
        self.menu.operation_chosen.connect(self.select_operation)
        self.menu.recipe_chosen.connect(self.open_recipe)
        self.search = self.menu.search
        mlay.addWidget(self.menu)
        fiji = button("Ouvrir le calque sélectionné dans Fiji")
        fiji.clicked.connect(self._open_in_fiji)
        mlay.addWidget(fiji)

        # page 2 : la fiche d'un outil -------------------------------------------------
        self.op_page = QWidget()
        olay = QVBoxLayout(self.op_page)
        olay.setContentsMargins(0, 0, 0, 0)
        olay.setSpacing(6)
        self.back_btn = button("Tous les outils", "left", object_name="ghost")
        self.back_btn.clicked.connect(self._back)
        olay.addWidget(self.back_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.banner = QFrame()
        self.banner.setObjectName("Banner")
        self.banner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        blay = QVBoxLayout(self.banner)
        blay.setContentsMargins(10, 8, 10, 8)
        blay.setSpacing(6)
        self.banner_text = label("")
        blay.addWidget(self.banner_text)
        self.next_btn = button("Étape suivante", "right")
        self.next_btn.clicked.connect(self.next_step)
        blay.addWidget(self.next_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.banner.hide()
        olay.addWidget(self.banner)
        self.op_title = label("", "step")
        olay.addWidget(self.op_title)
        self.desc = label("")
        olay.addWidget(self.desc)
        self.when = label("", "hint")
        olay.addWidget(self.when)
        self.imagej_label = label("", "faint")
        olay.addWidget(self.imagej_label)
        self.details_btn = button("Plus de détails", "info", object_name="ghost")
        self.details_btn.setCheckable(True)
        self.details_btn.toggled.connect(self._toggle_details)
        olay.addWidget(self.details_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.details = label("", "hint")
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.hide()
        olay.addWidget(self.details)
        olay.addWidget(separator())
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 0, 0, 0)
        olay.addWidget(self.form_host)
        self.preview = QCheckBox("Aperçu en direct")
        self.preview.setToolTip(
            "Montre le résultat sur le plan affiché pendant le réglage (comme « Preview » d'ImageJ)"
        )
        self.preview.toggled.connect(self._preview_toggled)
        self.whole_stack = QCheckBox("Toute la pile")
        self.whole_stack.setToolTip(
            "Applique l'opération à chaque plan de la pile, et pas seulement au plan affiché"
        )
        # pendant l'aperçu : revoir l'image d'origine (bouton à maintenir), régler la
        # transparence de ce qui est superposé
        self.compare_btn = button(
            "Voir l'original",
            "search",
            object_name="ghost",
            tooltip="Maintenez le bouton enfoncé : l'aperçu se cache, pour comparer avant / après",
        )
        self.compare_btn.pressed.connect(lambda: self._set_preview_visible(False))
        self.compare_btn.released.connect(lambda: self._set_preview_visible(True))
        self.preview_opacity = QSlider(Qt.Orientation.Horizontal)
        self.preview_opacity.setRange(10, 100)
        try:
            opacity = int(settings().value("analysis/preview_opacity", 100, type=int))
        except (TypeError, ValueError):
            opacity = 100
        self.preview_opacity.setValue(opacity)
        self.preview_opacity.setMaximumWidth(110)
        self.preview_opacity.setToolTip("Opacité de l'aperçu (un masque à 50 % laisse voir l'image dessous)")
        self.preview_opacity.valueChanged.connect(self._preview_opacity_changed)
        self.opacity_box = QWidget()
        ob = QHBoxLayout(self.opacity_box)
        ob.setContentsMargins(0, 0, 0, 0)
        ob.setSpacing(6)
        ob.addWidget(label("Opacité de l'aperçu", "hint", wrap=False))
        ob.addWidget(self.preview_opacity)
        options = FlowLayout(spacing=10, line_spacing=4)
        options.setContentsMargins(0, 0, 0, 0)
        for w in (self.preview, self.compare_btn, self.opacity_box, self.whole_stack):
            options.addWidget(w)
        options_box = QWidget()
        options_box.setLayout(options)
        policy = options_box.sizePolicy()
        policy.setHeightForWidth(True)
        options_box.setSizePolicy(policy)
        olay.addWidget(options_box)
        self.run_btn = button("Exécuter")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        olay.addWidget(self.run_btn)

        # page 3 : une démarche guidée ------------------------------------------------
        self.recipe_page = RecipePage()
        self.recipe_page.back.connect(self.show_menu)
        self.recipe_page.open_step.connect(self._open_step)
        # trois pages, une seule visible (les pages cachées ne prennent pas de place)
        self.pages = [self.menu_page, self.op_page, self.recipe_page]
        for page in self.pages:
            lay.addWidget(page)
        lay.addStretch(1)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._run_preview)
        self._show_page(self.menu_page)
        self._update_options()
        viewer.layers.events.inserted.connect(self._layer_inserted)
        viewer.layers.events.removed.connect(self._refresh_inputs)

    def set_expert(self, expert: bool) -> None:
        self.expert = expert
        if self._op is not None:
            self._build_form(self._op)

    # ------------------------------------------------------------------ outils
    def _reference_layer(self) -> Any | None:
        """Image sur laquelle on dessine : le calque actif s'il convient, sinon la première image."""
        active = self.viewer.layers.selection.active
        if isinstance(active, (ImageLayer, LabelsLayer)) and not active.name.startswith(PREVIEW_PREFIX):
            return active
        images = [ly for ly in reversed(self.viewer.layers) if isinstance(ly, ImageLayer)]
        return images[0] if images else None

    def _drawing_layer(self, kind: str) -> Any:
        """Calque de formes (« ROI ») ou de points, créé au besoin à l'échelle de l'image."""
        name, cls = (ROI_LAYER, ShapesLayer) if kind == "shapes" else (POINTS_LAYER, PointsLayer)
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], cls):
            return self.viewer.layers[name]
        ref = self._reference_layer()
        common: dict[str, Any] = {"name": name, "ndim": 2}
        if ref is not None:
            common["scale"] = tuple(ref.scale[-2:])
            common["translate"] = tuple(ref.translate[-2:])
            if ref.metadata.get("pixel_size_um"):
                common["units"] = ("µm", "µm")
        if kind == "shapes":
            layer = self.viewer.add_shapes(
                edge_color="#ffd400", face_color=[1.0, 0.83, 0.0, 0.08], edge_width=1.5, **common
            )
        else:
            layer = self.viewer.add_points(face_color="#ffd400", border_color="#000000", size=12, **common)
        if ref is not None:
            layer.metadata["pixel_size_um"] = ref.metadata.get("pixel_size_um")
        layer.events.mode.connect(self._mode_changed)
        return layer

    def set_tool(self, kind: str, mode: str, name: str | None = None) -> None:
        """Passe en mode dessin (rectangle, ellipse, ligne, points...) sur le calque voulu.
        ``name`` : le bouton de l'outil, quand deux outils partagent un mode (trajet, angle)."""
        if not kind:  # retour à l'outil normal : déplacer / zoomer
            for layer in self.viewer.layers:
                if isinstance(layer, (ShapesLayer, PointsLayer)):
                    layer.mode = "pan_zoom"
            self._show_tool("move")
            return
        layer = self._drawing_layer(kind)
        if self.viewer.dims.ndisplay != 2:
            self.viewer.dims.ndisplay = 2  # on ne dessine pas dans la vue en volume
        layer.visible = True  # un calque caché par son œil ne laisserait rien dessiner
        self.viewer.layers.selection.active = layer
        layer.mode = mode
        self._show_tool(name or next((n for n, _t, k, m in _TOOLS if (k, m) == (kind, mode)), "move"))
        self.message.emit("Dessinez sur l'image ; l'outil « déplacer » redonne la main à la souris")

    def _show_tool(self, name: str) -> None:
        """Coche le bouton de l'outil actif (un seul à la fois)."""
        self._tool = name
        for key, b in self.tool_buttons.items():
            b.setChecked(key == name)

    def _mode_changed(self, event: Any) -> None:
        """Le mode d'un calque de dessin a changé sans passer par nos boutons (touches E, P, L,
        S, Z… de napari, changement de section) : la rangée d'outils le reflète."""
        layer = getattr(event, "source", None)
        if layer is None or layer.name not in (ROI_LAYER, POINTS_LAYER):
            return
        mode = str(layer.mode)
        kind = "shapes" if isinstance(layer, ShapesLayer) else "points"
        if mode == "pan_zoom":
            others = [
                ly
                for ly in self.viewer.layers
                if ly is not layer and ly.name in (ROI_LAYER, POINTS_LAYER) and str(ly.mode) != "pan_zoom"
            ]
            if not others:
                self._show_tool("move")
            return
        current = next((t for t in _TOOLS if t[0] == self._tool), None)
        if current is not None and (current[2], current[3]) == (kind, mode):
            return  # déjà le bon bouton (trajet ou angle : même mode)
        self._show_tool(next((n for n, _t, k, m in _TOOLS if (k, m) == (kind, mode)), self._tool))

    def leave(self) -> None:
        """En quittant la section : la souris redevient « déplacer / zoomer » et l'aperçu en
        direct s'arrête (sinon l'outil resterait actif ailleurs)."""
        self.set_tool("", "")
        self._stop_preview()

    def _show_count(self, shapes: int, points: int) -> None:
        parts = []
        if shapes:
            parts.append(f"{shapes} forme{'s' if shapes > 1 else ''}")
        if points:
            parts.append(f"{points} point{'s' if points > 1 else ''}")
        self.roi_count.setText(" · ".join(parts))
        self.roi_count.setVisible(bool(parts))

    def clear_rois(self) -> None:
        for name in (ROI_LAYER, POINTS_LAYER):
            if name in self.viewer.layers:
                layer = self.viewer.layers[name]
                layer.selected_data = set()  # sinon napari garde une sélection de points disparus
                layer.data = [] if isinstance(layer, ShapesLayer) else np.zeros((0, layer.ndim))

    # ---------------------------------------------------------------- pages
    def _show_page(self, page: QWidget) -> None:
        """Affiche une page et revient en haut du panneau."""
        for w in self.pages:
            w.setVisible(w is page)
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            parent.verticalScrollBar().setValue(0)

    def current_page(self) -> str:
        """« menu », « operation » ou « recipe »."""
        if not self.menu_page.isHidden():
            return "menu"
        return "operation" if not self.op_page.isHidden() else "recipe"

    def show_menu(self) -> None:
        """Retour aux menus (l'aperçu en direct s'arrête)."""
        self._recipe = None
        self._stop_preview()
        if self._op is not None:
            self.menu.open_menu_of(self._op.name)
        self._show_page(self.menu_page)

    def _back(self) -> None:
        if self._recipe is not None:
            self._stop_preview()
            self._show_recipe_page()
        else:
            self.show_menu()

    def _stop_preview(self) -> None:
        self._preview_timer.stop()
        if self.preview.isChecked():
            self.preview.setChecked(False)
        self._clear_preview()

    def select_operation(self, name: str, *, in_recipe: bool = False) -> None:
        """Ouvre la fiche d'une opération (menus, démarches, raccourcis, tests)."""
        op = next((o for o in list_operations() if o.name == name), None)
        if op is None:
            raise KeyError(name)
        if not in_recipe:
            self._recipe = None
        self._clear_preview()
        self._op = op
        info = help_of(op.name, op.description)
        self.op_title.setText(op.name)
        self.desc.setText(info.summary + ".")
        self.when.setText(f"Quand l'utiliser : {info.when}" if info.when else "")
        self.when.setVisible(bool(info.when))
        self.imagej_label.setText(f"Équivalent ImageJ : {op.imagej}" if op.imagej else "")
        self.imagej_label.setVisible(bool(op.imagej))
        self.details.setText(op.description.strip())
        self.details_btn.setVisible(bool(op.description.strip()))
        self.details_btn.setChecked(False)
        self._build_form(op)
        self._update_banner()
        self._show_page(self.op_page)
        self._schedule_preview()

    def _toggle_details(self, on: bool) -> None:
        self.details.setVisible(on)
        self.details_btn.setText("Moins de détails" if on else "Plus de détails")
        self._show_page(self.op_page)

    # ---------------------------------------------------------------- démarches
    def open_recipe(self, key: str) -> None:
        """Ouvre une démarche guidée à sa première étape."""
        recipe = next(r for r in RECIPES if r.key == key)
        self._stop_preview()
        self._recipe, self._step, self._done = recipe, 0, set()
        self._show_recipe_page()

    def _show_recipe_page(self) -> None:
        assert self._recipe is not None
        self.recipe_page.show_recipe(self._recipe, self._step, self._done)
        self._show_page(self.recipe_page)

    def _open_step(self, index: int) -> None:
        assert self._recipe is not None
        step = self._recipe.steps[index]
        self._step = index
        if step.operation:
            self.select_operation(step.operation, in_recipe=True)
            for name, value in step.params:
                widget = next((w for w in self._params if w.name == name), None)
                if widget is None:
                    continue
                if value == GUESS and name == "objects":
                    value = self._guess_objects()
                    if value is None:
                        continue
                    self.message.emit(
                        f"Objets « {value} » : le fond de l'image est plutôt "
                        + ("clair" if value == "sombres" else "sombre")
                    )
                widget.value = value
        elif step.tool:
            self.set_tool(*step.tool)
            self._done.add(index)
            self._step = min(index + 1, len(self._recipe.steps) - 1)
            self._show_recipe_page()

    def next_step(self) -> None:
        """Passe à l'étape suivante de la démarche en cours."""
        if self._recipe is None:
            return
        self._stop_preview()
        self._done.add(self._step)
        following = self._step + 1
        if following >= len(self._recipe.steps):
            self.message.emit(f"Démarche « {self._recipe.title} » terminée")
            self._show_recipe_page()
            return
        self._step = following
        if self._recipe.steps[following].operation:
            self._open_step(following)
        else:
            self._show_recipe_page()

    def _guess_objects(self) -> str | None:
        """« sombres » si le fond de l'image est clair (fond clair, coloration), « clairs »
        sinon (fluorescence) : la plupart des pixels appartiennent au fond."""
        layer = self._primary_layer()
        if layer is None:
            return None
        data, _ = displayed_plane(layer, self.viewer)
        # sous-échantillonner d'abord : 100 000 pixels suffisent (une image 4K entière en
        # nombres réels coûtait 0,2 s et 200 Mo)
        step = max(1, int(np.sqrt(data.shape[0] * data.shape[1] / 100_000)))
        sample = np.asarray(data[::step, ::step], dtype=np.float32)
        if _is_rgb(layer):
            sample = sample[..., :3].mean(axis=-1)
        lo, mid, hi = np.percentile(sample, (1, 50, 99))
        if hi <= lo:
            return None
        return "sombres" if mid > (lo + hi) / 2 else "clairs"

    def _update_banner(self, done: bool = False) -> None:
        recipe = self._recipe
        if recipe is None or self._op is None:
            self.banner.hide()
            self.back_btn.setText("Tous les outils")
            return
        self.back_btn.setText("Étapes de la démarche")
        n, i = len(recipe.steps), self._step
        text = f"<b>{recipe.title}</b><br>Étape {i + 1} sur {n} : {recipe.steps[i].text}"
        if done:
            text += "<br>✓ Fait. Vérifiez le résultat, puis passez à la suite."
        self.banner_text.setText(text)
        last = i + 1 >= n
        self.next_btn.setText("Terminer la démarche" if last else "Étape suivante")
        self.banner.show()

    def visible_operations(self) -> list[str]:
        """Outils affichés dans les menus (après la recherche)."""
        return self.menu.visible_operations()

    # ------------------------------------------------------------- formulaire
    def _build_form(self, op: Operation) -> None:
        self._params.setParent(None)  # le formulaire des réglages est réutilisé d'un outil à l'autre
        while self.form.rowCount():
            self.form.removeRow(0)
        self._inputs = {}
        for spec in op.inputs:
            combo = QComboBox()
            combo.currentTextChanged.connect(self._input_changed)
            self._inputs[spec.name] = combo
            self.form.addRow(labelled(_INPUT_LABELS.get(spec.name, spec.name), combo))
        self.form.addRow(self._params)
        self._params.build(list(op.params), expert=self.expert)
        self._refresh_inputs()  # choisit l'image : les seuils prennent ses valeurs
        # comme dans ImageJ, un outil garde les réglages de sa dernière utilisation — repris
        # une fois l'image connue, sinon un seuil au-dessus de 255 était ramené à 255
        self._params.blockSignals(True)
        self._params.restore(self._remembered(op))
        self._params.blockSignals(False)

    def _refresh_inputs(self, *_: Any) -> None:
        if self._op is None:
            return
        for spec in self._op.inputs:
            combo = self._inputs.get(spec.name)
            if combo is None:
                continue
            wanted = _LAYER_TYPES[spec.kind]
            names = [
                ly.name
                for ly in reversed(self.viewer.layers)
                if isinstance(ly, wanted) and not ly.name.startswith(PREVIEW_PREFIX)
            ]
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if spec.optional:
                combo.addItem(_NONE)
            combo.addItems(names)
            if current in names or (current == _NONE and spec.optional):
                combo.setCurrentText(current)
            elif names and not spec.optional:
                sel = self.viewer.layers.selection.active
                if sel is not None and sel.name in names:
                    combo.setCurrentText(sel.name)
            combo.blockSignals(False)
        self._update_options()

    def _layer_inserted(self, event: Any) -> None:
        layer = getattr(event, "value", None)
        if layer is not None:  # un calque renommé garde sa place dans les listes d'entrées
            layer.events.name.connect(self._refresh_inputs)
        self._refresh_inputs()

    def _input_changed(self, *_: Any) -> None:
        self._update_options()
        self._schedule_preview()

    def _primary_layer(self) -> Any | None:
        """Premier calque image ou masque choisi : il fixe la grille de pixels et la pile."""
        if self._op is None:
            return None
        for spec in self._op.inputs:
            if spec.kind not in (Image, Labels):
                continue
            name = self._inputs[spec.name].currentText()
            if name and name != _NONE and name in self.viewer.layers:
                return self.viewer.layers[name]
        return None

    def _update_options(self) -> None:
        op, primary = self._op, self._primary_layer()
        is_stack = primary is not None and int(primary.ndim) > 2
        produces_layers = op is not None and op.category != "Fiji / ImageJ"
        self.whole_stack.setVisible(bool(op is not None and not op.stack and is_stack))
        self.preview.setVisible(bool(produces_layers and op is not None and not op.stack))
        if self.whole_stack.isHidden():  # isHidden : état propre à la case, même panneau replié
            self.whole_stack.setChecked(False)
        for w in (self.compare_btn, self.opacity_box):
            w.setVisible(not self.preview.isHidden() and self.preview.isChecked())
        self._update_context(primary)

    def _update_context(self, primary: Any | None) -> None:
        """Adapte les réglages à l'image choisie : équivalent en µm des longueurs en pixels,
        bornes des seuils (du pixel le plus sombre au plus clair de l'image, pile comprise)."""
        if primary is None:
            self._params.set_context(None, None)
            return
        data_range = None
        integer = np.issubdtype(np.dtype(primary.data.dtype), np.integer)
        if any(c.spec.meta.span == "image" for c in self._params):
            try:
                data_range = value_range(primary, self.viewer)
            except Exception:  # calque en cours de changement
                data_range = None
        self._params.blockSignals(True)
        self._params.set_context(_pixel_size(primary), data_range, bool(integer))
        self._params.blockSignals(False)

    def _remembered(self, op: Operation) -> dict[str, Any]:
        try:
            return dict(json.loads(settings().value(f"analysis/params/{op.name}", "{}", type=str)))
        except (TypeError, ValueError):
            return {}

    def _remember(self, op: Operation) -> None:
        settings().setValue(f"analysis/params/{op.name}", json.dumps(self._params.values()))

    # ---------------------------------------------------------------- exécution
    def _convert(self, spec: Any, layer: Any, primary: Any | None, plane: bool) -> Any:
        viewer = self.viewer if plane else None
        if spec.kind is Image:
            return layer_to_image(layer, viewer)
        if spec.kind is Labels:
            return layer_to_labels(layer, viewer)
        if spec.kind is Rois:
            return layer_to_rois(layer, primary, self.viewer)
        return layer_to_points(layer, primary, self.viewer)

    def _layers(self) -> list[tuple[Any, Any | None]]:
        assert self._op is not None
        chosen: list[tuple[Any, Any | None]] = []
        for spec in self._op.inputs:
            name = self._inputs[spec.name].currentText()
            if not name or name == _NONE:
                if not spec.optional:
                    label = _INPUT_LABELS.get(spec.name, spec.name)
                    raise ValueError(
                        _MISSING_INPUT.get(spec.kind, "Il manque « {label} ».").format(label=label)
                    )
                chosen.append((spec, None))
                continue
            chosen.append((spec, self.viewer.layers[name]))
        return chosen

    def _collect(self, whole_stack: bool | None = None) -> tuple[list[Any], dict[str, Any]]:
        """Entrées et réglages de l'opération : le plan affiché, ou la pile entière pour les
        opérations qui travaillent sur une pile."""
        assert self._op is not None
        plane = not self._op.stack if whole_stack is None else not whole_stack
        primary = self._primary_layer()
        inputs = [
            None if layer is None else self._convert(spec, layer, primary, plane)
            for spec, layer in self._layers()
        ]
        return inputs, self._param_values()

    def _param_values(self) -> dict[str, Any]:
        return self._params.values()

    def _collect_planes(self) -> tuple[list[list[Any]], list[tuple[int, ...]], tuple[int, ...]]:
        """Une liste d'entrées par plan de la pile principale. Les autres piles de même
        forme sont parcourues en parallèle ; le reste est repris tel quel à chaque plan."""
        primary = self._primary_layer()
        assert primary is not None
        lead = tuple(primary.data.shape[: int(primary.ndim) - 2])
        indices = list(np.ndindex(*lead))
        chosen = self._layers()
        fixed = [
            None
            if layer is None
            else self._convert(spec, layer, primary, True)
            if spec.kind in (Rois, Points) or tuple(layer.data.shape[: int(layer.ndim) - 2]) != lead
            else None
            for spec, layer in chosen
        ]
        plane_inputs = []
        for index in indices:
            row = []
            for (spec, layer), constant in zip(chosen, fixed, strict=True):
                if layer is None or constant is not None:
                    row.append(constant)
                    continue
                data = np.asarray(layer.data[index])
                if spec.kind is Image:
                    row.append(
                        Image(data, _pixel_size(layer), layer.name, dict(layer.metadata), _is_rgb(layer))
                    )
                else:
                    row.append(Labels(data, _pixel_size(layer), layer.name))
            plane_inputs.append(row)
        return plane_inputs, indices, lead

    def _run(self) -> None:
        if self._op is None:
            return
        if self._busy:  # un aperçu se calcule : « Exécuter » partira dès qu'il aura fini
            self._run_pending = True
            self._pending_op = self._op
            self.run_btn.setEnabled(False)
            return
        self._run_pending = False
        op = self._op
        self._remember(op)
        self._preview_timer.stop()
        self._clear_preview()
        try:
            # l'image traitée est notée maintenant : les résultats lui reviennent même si l'on
            # change d'outil ou d'image pendant le calcul
            target = self._target()
            if self.whole_stack.isChecked() and not op.stack:
                plane_inputs, indices, lead = self._collect_planes()
                params = self._param_values()
                worker = create_worker(
                    run_on_planes, op, plane_inputs, params, _start_thread=False, _ignore_errors=True
                )
                worker.returned.connect(
                    lambda results: self._show_outputs(
                        op, combine_planes(results, indices, lead), target=target
                    )
                )
                self.message.emit(f"{op.name} sur {len(indices)} plans…")
            else:
                inputs, params = self._collect()
                worker = create_worker(op.func, *inputs, **params, _start_thread=False, _ignore_errors=True)
                worker.returned.connect(lambda out: self._show_outputs(op, out, target=target))
                self.message.emit(f"{op.name} en cours…")
        except Exception as exc:
            QMessageBox.warning(self, "Analyse", str(exc))
            return
        self._busy = True
        self.run_btn.setEnabled(False)
        worker.errored.connect(lambda exc: self._error(op, exc))
        worker.finished.connect(self._finished)
        worker.start()

    def _target(self) -> tuple[tuple[float, float], str | None, Any]:
        """(décalage, image ouverte, calque traité) de l'entrée principale : où poser les
        résultats, et d'après quelle image les afficher."""
        primary = self._primary_layer()
        if primary is None:
            return (0.0, 0.0), None, None
        offset = (float(primary.translate[-2]), float(primary.translate[-1]))
        return offset, documents.doc_of(primary), primary

    def _finished(self) -> None:
        self._busy = False
        self.run_btn.setEnabled(True)
        self.preview.setText("Aperçu en direct")
        if self._run_pending and self._pending_op is self._op:
            self._run()
        elif self._preview_pending:
            self._schedule_preview()
        self._run_pending = False  # on a changé d'outil entre-temps : la demande est oubliée

    def _error(self, op: Operation, exc: Exception) -> None:
        QMessageBox.warning(
            self,
            op.name,
            f"« {op.name} » n'a pas abouti : {exc}\n\nVérifiez l'image choisie et les réglages.",
        )
        self.message.emit(f"{op.name} : erreur")

    # ------------------------------------------------------------------- aperçu
    def _preview_toggled(self, on: bool) -> None:
        for w in (self.compare_btn, self.opacity_box):
            w.setVisible(on)
        if on:
            self._schedule_preview()
        else:
            self._preview_timer.stop()
            self._clear_preview()

    def _schedule_preview(self, *_: Any) -> None:
        if self.preview.isChecked() and not self.preview.isHidden():
            self._preview_timer.start()

    def _run_preview(self) -> None:
        if self._op is None or not self.preview.isChecked():
            return
        if self._busy:  # un calcul est en cours : on recommencera quand il sera fini
            self._preview_pending = True
            return
        self._preview_pending = False
        op = self._op
        try:
            inputs, params = self._collect(whole_stack=False)
        except Exception as exc:
            self.message.emit(f"Aperçu : {exc}")
            return
        self._busy = True
        self.preview.setText("Aperçu en direct — calcul en cours…")
        target = self._target()
        worker = create_worker(op.func, *inputs, **params, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(lambda out: self._show_outputs(op, out, preview=True, target=target))
        worker.errored.connect(lambda exc: self.message.emit(f"Aperçu : {exc}"))
        worker.finished.connect(self._finished)
        worker.start()

    def _preview_layers(self) -> list[Any]:
        return [ly for ly in self.viewer.layers if ly.name.startswith(PREVIEW_PREFIX)]

    def _clear_preview(self) -> None:
        for layer in self._preview_layers():
            self.viewer.layers.remove(layer)

    def _set_preview_visible(self, visible: bool) -> None:
        """« Voir l'original » : cache l'aperçu tant que le bouton est enfoncé."""
        for layer in self._preview_layers():
            layer.visible = visible

    def _preview_opacity_changed(self, value: int) -> None:
        settings().setValue("analysis/preview_opacity", value)
        for layer in self._preview_layers():
            layer.opacity = value / 100

    # ------------------------------------------------------------------ sorties
    def _show_outputs(
        self,
        op: Operation,
        out: Any,
        preview: bool = False,
        target: tuple[tuple[float, float], str | None, Any] | None = None,
    ) -> None:
        if preview and (self._op is not op or not self.preview.isChecked()):
            return  # l'utilisateur est passé à autre chose pendant le calcul
        # les résultats appartiennent à l'image traitée : ils se ferment avec elle
        offset, doc, source = target if target is not None else self._target()
        if source is not None and source not in self.viewer.layers:
            # l'image a été fermée pendant le calcul : son résultat ne la ferait pas revenir
            self._clear_preview() if preview else self.message.emit(
                f"{op.name} : l'image a été fermée pendant le calcul, le résultat est abandonné"
            )
            return
        items = [i for i in (out if isinstance(out, tuple) else (out,)) if i is not None]
        if preview:
            self._show_preview(items, offset, doc, source)
            return
        shown = source is None or bool(source.visible)  # une autre image est affichée entre-temps
        for item in items:
            if isinstance(item, Table):
                self.table_ready.emit(item)
                continue
            layer = self._add_layer(item, offset, "", doc, like=source)
            if layer is not None and not shown:
                layer.visible = False  # il attend avec son image, sans se poser sur une autre
        self.message.emit(f"{op.name} : terminé")
        if self.preview.isChecked():  # le résultat est là : l'aperçu n'a plus rien à montrer
            self.preview.setChecked(False)
        if self._recipe is not None and op is self._op:
            self._done.add(self._step)
            self._update_banner(done=True)

    def _show_preview(
        self, items: list[Any], offset: tuple[float, float], doc: str | None, source: Any
    ) -> None:
        """Pose l'aperçu. Un aperçu déjà là est mis à jour sur place (ses pixels sont remplacés) :
        pas de clignotement, et le calque choisi dans « Calques » reste choisi."""
        selected = self.viewer.layers.selection.active
        old = {layer.name: layer for layer in self._preview_layers()}
        kept: set[str] = set()
        for item in items:
            if isinstance(item, Table):
                continue
            name = PREVIEW_PREFIX + item.name
            layer = old.get(name)
            if layer is not None and _same_kind(layer, item):
                layer.data = item.data
            else:
                if layer is not None:
                    self.viewer.layers.remove(layer)
                layer = self._add_layer(item, offset, PREVIEW_PREFIX, doc, like=source)
            if layer is not None:
                layer.opacity = self.preview_opacity.value() / 100
                layer.visible = not self.compare_btn.isDown()
                kept.add(name)
        for name, layer in old.items():
            if name not in kept and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        if selected is not None and selected in self.viewer.layers:
            self.viewer.layers.selection.active = selected

    def _add_layer(
        self,
        item: Any,
        offset: tuple[float, float],
        prefix: str,
        doc: str | None = None,
        like: Any | None = None,
    ) -> Any | None:
        """Ajoute un résultat à la vue. ``like`` : l'image traitée ; une image de même nature
        reprend ses réglages d'affichage, pour que l'on voie ce que l'outil a changé (un
        contraste recalculé sur chaque résultat masquait un simple changement de luminosité)."""
        px = item.pixel_size_um or 1.0
        name = prefix + item.name
        if isinstance(item, Rois):
            if not len(item):
                return None
            layer = self.viewer.add_shapes(
                [v for _, v in item.shapes],
                shape_type=[k for k, _ in item.shapes],
                name=name,
                scale=(px, px),
                translate=offset,
                edge_color="#ffd400",
                face_color="transparent",
                edge_width=1.5,
            )
        elif isinstance(item, Points):
            data = np.asarray(item.data, dtype=np.float64)
            lead = data.shape[1] - 2 if data.ndim == 2 else 0
            layer = self.viewer.add_points(
                data,
                name=name,
                scale=(1.0,) * lead + (px, px),
                translate=(0.0,) * lead + tuple(offset),
                face_color="transparent",
                border_color="#ffd400",
                size=10,
            )
        else:
            data = item.data
            rgb = isinstance(item, Image) and item.is_rgb
            lead = data.ndim - (3 if rgb else 2)
            kwargs: dict[str, Any] = {
                "name": name,
                "scale": (1.0,) * lead + (px, px),
                "translate": (0.0,) * lead + tuple(offset),
            }
            if item.pixel_size_um and lead == 0:
                kwargs["units"] = ("µm", "µm")
            if isinstance(item, Labels):
                layer = self.viewer.add_labels(data, **kwargs)
            else:
                layer = self.viewer.add_image(data, rgb=rgb, **kwargs)
                _display_like(layer, like)
        layer.metadata["pixel_size_um"] = item.pixel_size_um
        documents.tag(layer, doc)
        return layer

    # ------------------------------------------------------- pour le reste de Fuji
    @property
    def busy(self) -> bool:
        """Vrai pendant un calcul (opération ou aperçu)."""
        return self._busy

    def add_result(self, item: Any, prefix: str = "", doc: str | None = None) -> Any | None:
        """Affiche un résultat venu d'ailleurs (assistant IA, script) : image, masque, ROI
        ou points. Renvoie le calque ajouté."""
        return self._add_layer(item, (0.0, 0.0), prefix, doc)

    def run_operation(
        self, name: str, inputs: dict[str, str] | None = None, params: dict[str, Any] | None = None
    ) -> None:
        """Ouvre une opération, choisit ses entrées (nom de l'entrée → nom du calque, quand le
        calque existe), pose les réglages donnés et l'exécute : les raccourcis « Mesurer »,
        « Histogramme », « Appliquer le contraste aux pixels »…"""
        if self.preview.isChecked():
            self.preview.setChecked(False)
        self.select_operation(name)
        for key, layer_name in (inputs or {}).items():
            combo = self._inputs.get(key)
            if combo is not None and combo.findText(layer_name) >= 0:
                combo.setCurrentText(layer_name)
        if params:
            self._params.set_values(params)
        self._run()

    def _open_in_fiji(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is None or not isinstance(layer, (ImageLayer, LabelsLayer)):
            QMessageBox.information(self, "Fiji", "Sélectionnez d'abord une image ou un masque.")
            return
        try:
            data = (
                layer_to_image(layer)
                if not isinstance(layer, LabelsLayer)
                else Image(
                    np.asarray(layer.data).astype(np.uint16), layer_to_labels(layer).pixel_size_um, layer.name
                )
            )
            path = open_in_fiji(data)
            self.message.emit(f"Ouvert dans Fiji : {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Fiji", f"L'image n'a pas pu être ouverte dans Fiji : {exc}")


__all__ = [
    "AnalysisPanel",
    "displayed_plane",
    "layer_to_image",
    "layer_to_labels",
    "layer_to_points",
    "layer_to_rois",
    "plane_index",
]
