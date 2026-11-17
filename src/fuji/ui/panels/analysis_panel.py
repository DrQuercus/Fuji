"""Panneau « Analyse » : toutes les opérations, avec formulaires générés automatiquement.

Chaque opération du registre (y compris celles ajoutées par des extensions) apparaît
ici ; ses réglages sont construits à partir de sa signature Python. Les calques napari
servent d'entrées et de sorties :

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

from typing import Any

import numpy as np
from magicgui.widgets import Container, create_widget
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
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji.analysis import Image, Labels, Points, Rois, Table, list_operations
from fuji.analysis.imagej import open_in_fiji
from fuji.analysis.registry import Operation, combine_planes

from ..components import icon_button
from ..flow import FlowLayout
from ..widgets import hint_label

_NONE = "(aucune)"
ROI_LAYER = "ROI"
POINTS_LAYER = "Points"
PREVIEW_PREFIX = "Aperçu — "
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
#: outils de dessin : (icône, info-bulle, type de calque, mode napari)
_TOOLS = (
    ("roi_rect", "Rectangle", "shapes", "add_rectangle"),
    ("roi_ellipse", "Ellipse", "shapes", "add_ellipse"),
    ("roi_polygon", "Polygone (double-clic pour fermer)", "shapes", "add_polygon"),
    ("roi_line", "Ligne (profil, distance)", "shapes", "add_line"),
    ("roi_point", "Points (comptage)", "points", "add"),
    ("pointer", "Sélectionner et déplacer les formes", "shapes", "select"),
    ("move", "Déplacer l'image (outil normal)", "", "pan_zoom"),
)


# ------------------------------------------------------------ calques → analyse
def _pixel_size(layer: Any) -> float | None:
    px = layer.metadata.get("pixel_size_um")
    if px is None and float(layer.scale[-1]) != 1.0:
        px = float(layer.scale[-1])
    return px


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
        self._params: Container | None = None
        self._busy = False
        self._preview_pending = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        # outils de dessin des ROI (ils agissent sur un calque de formes ou de points) ; la
        # rangée passe à la ligne si le panneau est étroit
        tools_box = QWidget()
        tools = FlowLayout(tools_box, spacing=2, line_spacing=2)
        tools.setContentsMargins(0, 0, 0, 0)
        for icon_name, tip, kind, mode in _TOOLS:
            b = icon_button(icon_name, tip)
            b.clicked.connect(lambda _=False, k=kind, m=mode: self.set_tool(k, m))
            tools.addWidget(b)
        clear = icon_button("trash", "Effacer toutes les ROI et tous les points")
        clear.clicked.connect(self.clear_rois)
        tools.addWidget(clear)
        policy = tools_box.sizePolicy()
        policy.setHeightForWidth(True)
        tools_box.setSizePolicy(policy)
        lay.addWidget(tools_box)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher (ex. seuil, flou, profil, Threshold, Z Project)")
        self.search.textChanged.connect(self._fill_list)
        lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.setMinimumHeight(170)
        self.list.setMaximumHeight(260)
        self.list.currentItemChanged.connect(self._select)
        lay.addWidget(self.list)
        self.desc = hint_label()
        lay.addWidget(self.desc)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        lay.addWidget(self.form_host)
        self.preview = QCheckBox("Aperçu en direct")
        self.preview.setToolTip(
            "Montre le résultat sur le plan affiché pendant le réglage (comme « Preview » d'ImageJ)"
        )
        self.preview.toggled.connect(self._preview_toggled)
        self.whole_stack = QCheckBox("Toute la pile")
        self.whole_stack.setToolTip(
            "Applique l'opération à chaque plan de la pile, et pas seulement au plan affiché"
        )
        options = QHBoxLayout()
        options.addWidget(self.preview)
        options.addWidget(self.whole_stack)
        options.addStretch(1)
        lay.addLayout(options)
        self.run_btn = QPushButton("Exécuter")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        lay.addWidget(self.run_btn)
        fiji = QPushButton("Ouvrir le calque sélectionné dans Fiji")
        fiji.clicked.connect(self._open_in_fiji)
        lay.addWidget(fiji)
        lay.addStretch(1)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._run_preview)
        self._fill_list()
        self._update_options()
        viewer.layers.events.inserted.connect(self._refresh_inputs)
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
        return layer

    def set_tool(self, kind: str, mode: str) -> None:
        """Passe en mode dessin (rectangle, ellipse, ligne, points...) sur le calque voulu."""
        if not kind:  # retour à l'outil normal : déplacer / zoomer
            for layer in self.viewer.layers:
                if isinstance(layer, (ShapesLayer, PointsLayer)):
                    layer.mode = "pan_zoom"
            return
        layer = self._drawing_layer(kind)
        self.viewer.layers.selection.active = layer
        layer.mode = mode
        self.message.emit("Dessinez sur l'image ; l'outil « déplacer » redonne la main à la souris")

    def leave(self) -> None:
        """En quittant la section : la souris redevient « déplacer / zoomer » et l'aperçu en
        direct s'arrête (sinon l'outil resterait actif ailleurs)."""
        self.set_tool("", "")
        if self.preview.isChecked():
            self.preview.setChecked(False)

    def clear_rois(self) -> None:
        for name in (ROI_LAYER, POINTS_LAYER):
            if name in self.viewer.layers:
                layer = self.viewer.layers[name]
                layer.data = [] if isinstance(layer, ShapesLayer) else np.zeros((0, layer.ndim))

    # ---------------------------------------------------------------- liste
    def _fill_list(self) -> None:
        text = self.search.text().strip().lower()
        self.list.blockSignals(True)
        self.list.clear()
        current_cat = None
        for op in list_operations():
            hay = f"{op.name} {op.category} {op.description} {op.imagej or ''}".lower()
            if text and text not in hay:
                continue
            if op.category != current_cat:
                current_cat = op.category
                header = QListWidgetItem(op.category.upper())
                header.setFlags(Qt.ItemFlag.NoItemFlags)  # titre de catégorie
                self.list.addItem(header)
            item = QListWidgetItem("   " + op.name)
            item.setData(256, op.name)
            if op.imagej:
                item.setToolTip(f"ImageJ : {op.imagej}")
            self.list.addItem(item)
        self.list.blockSignals(False)

    def select_operation(self, name: str) -> None:
        """Sélectionne une opération par son nom (menus, tests)."""
        self.search.clear()
        for i in range(self.list.count()):
            if self.list.item(i).data(256) == name:
                self.list.setCurrentRow(i)
                return
        raise KeyError(name)

    def _select(self, item: QListWidgetItem | None, _prev: Any = None) -> None:
        if item is None or item.data(256) is None:
            return
        op = next(o for o in list_operations() if o.name == item.data(256))
        self._clear_preview()
        self._op = op
        text = op.description.split("\n\n")[0]
        if op.imagej:
            text += f"\n\nÉquivalent ImageJ : {op.imagej}"
        self.desc.setText(text)
        self._build_form(op)
        self._schedule_preview()

    # ------------------------------------------------------------- formulaire
    def _build_form(self, op: Operation) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._inputs = {}
        for spec in op.inputs:
            combo = QComboBox()
            combo.currentTextChanged.connect(self._input_changed)
            self._inputs[spec.name] = combo
            self.form.addRow(_INPUT_LABELS.get(spec.name, spec.name), combo)
        widgets = []
        for p in op.params:
            if p.meta.advanced and not self.expert:
                continue
            opts: dict[str, Any] = {}
            if p.meta.min is not None:
                opts["min"] = p.meta.min
            if p.meta.max is not None:
                opts["max"] = p.meta.max
            elif p.type in (int, float):
                opts["max"] = 1e9
            if p.meta.step is not None:
                opts["step"] = p.meta.step
            if p.meta.choices:
                opts["choices"] = list(p.meta.choices)
            label = p.label.rstrip("_") + (f" ({p.meta.unit})" if p.meta.unit else "")
            kwargs: dict[str, Any] = {"value": p.default, "label": label, "name": p.name, "options": opts}
            if p.meta.multiline:
                kwargs["widget_type"] = "TextEdit"
            else:
                kwargs["annotation"] = p.type
            w = create_widget(**kwargs)
            if p.meta.description:
                w.tooltip = p.meta.description
            widgets.append(w)
        self._params = Container(widgets=widgets)
        self._params.changed.connect(self._schedule_preview)
        if widgets:
            self.form.addRow(self._params.native)
        self._refresh_inputs()

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
                    kind = "une ROI (outils en haut du panneau)" if spec.kind is Rois else "un calque"
                    label = _INPUT_LABELS.get(spec.name, spec.name)
                    raise ValueError(f"Choisissez {kind} pour « {label} »")
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
        return {w.name: w.value for w in self._params} if self._params is not None else {}

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
        if self._op is None or self._busy:
            return
        op = self._op
        self._preview_timer.stop()
        self._clear_preview()
        try:
            if self.whole_stack.isChecked() and not op.stack:
                plane_inputs, indices, lead = self._collect_planes()
                params = self._param_values()
                worker = create_worker(
                    run_on_planes, op, plane_inputs, params, _start_thread=False, _ignore_errors=True
                )
                worker.returned.connect(
                    lambda results: self._show_outputs(op, combine_planes(results, indices, lead))
                )
                self.message.emit(f"{op.name} sur {len(indices)} plans…")
            else:
                inputs, params = self._collect()
                worker = create_worker(op.func, *inputs, **params, _start_thread=False, _ignore_errors=True)
                worker.returned.connect(lambda out: self._show_outputs(op, out))
                self.message.emit(f"{op.name} en cours…")
        except Exception as exc:
            QMessageBox.warning(self, "Analyse", str(exc))
            return
        self._busy = True
        self.run_btn.setEnabled(False)
        worker.errored.connect(lambda exc: self._error(op, exc))
        worker.finished.connect(self._finished)
        worker.start()

    def _finished(self) -> None:
        self._busy = False
        self.run_btn.setEnabled(True)
        if self._preview_pending:
            self._schedule_preview()

    def _error(self, op: Operation, exc: Exception) -> None:
        QMessageBox.warning(self, op.name, f"Erreur : {exc}")
        self.message.emit(f"{op.name} : erreur")

    # ------------------------------------------------------------------- aperçu
    def _preview_toggled(self, on: bool) -> None:
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
        worker = create_worker(op.func, *inputs, **params, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(lambda out: self._show_outputs(op, out, preview=True))
        worker.errored.connect(lambda exc: self.message.emit(f"Aperçu : {exc}"))
        worker.finished.connect(self._finished)
        worker.start()

    def _clear_preview(self) -> None:
        for layer in [ly for ly in self.viewer.layers if ly.name.startswith(PREVIEW_PREFIX)]:
            self.viewer.layers.remove(layer)

    # ------------------------------------------------------------------ sorties
    def _show_outputs(self, op: Operation, out: Any, preview: bool = False) -> None:
        if preview:
            if self._op is not op or not self.preview.isChecked():
                return  # l'utilisateur est passé à autre chose pendant le calcul
            selected = self.viewer.layers.selection.active
            self._clear_preview()
        primary = self._primary_layer()
        offset = tuple(primary.translate[-2:]) if primary is not None else (0.0, 0.0)
        items = out if isinstance(out, tuple) else (out,)
        for item in items:
            if item is None:
                continue
            if isinstance(item, Table):
                if not preview:
                    self.table_ready.emit(item)
                continue
            self._add_layer(item, offset, PREVIEW_PREFIX if preview else "")
        if preview:
            if selected is not None and selected.name in self.viewer.layers:
                self.viewer.layers.selection.active = selected
            return
        self.message.emit(f"{op.name} terminé")

    def _add_layer(self, item: Any, offset: tuple[float, float], prefix: str) -> None:
        px = item.pixel_size_um or 1.0
        name = prefix + item.name
        if isinstance(item, Rois):
            if not len(item):
                return
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
        layer.metadata["pixel_size_um"] = item.pixel_size_um

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
            QMessageBox.warning(self, "Fiji", str(exc))


__all__ = [
    "AnalysisPanel",
    "displayed_plane",
    "layer_to_image",
    "layer_to_labels",
    "layer_to_points",
    "layer_to_rois",
    "plane_index",
]
