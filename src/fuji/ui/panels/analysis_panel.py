"""Panneau « Analyse » : toutes les opérations, avec formulaires générés automatiquement.

Chaque opération du registre (y compris celles ajoutées par des extensions)
apparaît ici ; ses réglages sont construits à partir de sa signature Python.
Les calques napari servent d'entrées et de sorties.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from magicgui.widgets import Container, create_widget
from napari.components import ViewerModel
from napari.layers import Image as ImageLayer
from napari.layers import Labels as LabelsLayer
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji.analysis import Image, Labels, Table, list_operations
from fuji.analysis.imagej import open_in_fiji
from fuji.analysis.registry import Operation

from ..widgets import hint_label

_NONE = "(aucune)"


def layer_to_image(layer: Any) -> Image:
    px = layer.metadata.get("pixel_size_um")
    if px is None and float(layer.scale[-1]) != 1.0:
        px = float(layer.scale[-1])
    return Image(np.asarray(layer.data), px, layer.name, dict(layer.metadata))


def layer_to_labels(layer: Any) -> Labels:
    px = layer.metadata.get("pixel_size_um")
    if px is None and float(layer.scale[-1]) != 1.0:
        px = float(layer.scale[-1])
    return Labels(np.asarray(layer.data), px, layer.name)


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
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher une opération (ex. seuil, particules, fond)")
        self.search.textChanged.connect(self._fill_list)
        lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.setMaximumHeight(190)
        self.list.currentItemChanged.connect(self._select)
        lay.addWidget(self.list)
        self.desc = hint_label()
        lay.addWidget(self.desc)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        lay.addWidget(self.form_host)
        self.run_btn = QPushButton("Exécuter")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        lay.addWidget(self.run_btn)
        fiji = QPushButton("Ouvrir le calque sélectionné dans Fiji")
        fiji.clicked.connect(self._open_in_fiji)
        lay.addWidget(fiji)
        lay.addStretch(1)
        self._fill_list()
        viewer.layers.events.inserted.connect(self._refresh_inputs)
        viewer.layers.events.removed.connect(self._refresh_inputs)

    def set_expert(self, expert: bool) -> None:
        self.expert = expert
        if self._op is not None:
            self._build_form(self._op)

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
            self.list.addItem(item)
        self.list.blockSignals(False)

    def _select(self, item: QListWidgetItem | None, _prev: Any = None) -> None:
        if item is None or item.data(256) is None:
            return
        op = next(o for o in list_operations() if o.name == item.data(256))
        self._op = op
        text = op.description.split("\n\n")[0]
        if op.imagej:
            text += f"\n\nÉquivalent ImageJ : {op.imagej}"
        self.desc.setText(text)
        self._build_form(op)

    # ------------------------------------------------------------- formulaire
    def _build_form(self, op: Operation) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._inputs = {}
        for spec in op.inputs:
            combo = QComboBox()
            self._inputs[spec.name] = combo
            label = {
                "image": "Image",
                "mask": "Masque",
                "intensity": "Intensités",
                "flat": "Champ plat",
                "dark": "Noir",
            }.get(spec.name, spec.name)
            self.form.addRow(label, combo)
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
            kwargs: dict[str, Any] = {"value": p.default, "label": p.label, "name": p.name, "options": opts}
            if p.meta.multiline:
                kwargs["widget_type"] = "TextEdit"
            else:
                kwargs["annotation"] = p.type
            w = create_widget(**kwargs)
            if p.meta.description:
                w.tooltip = p.meta.description
            widgets.append(w)
        self._params = Container(widgets=widgets)
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
            wanted = ImageLayer if spec.kind is Image else LabelsLayer
            names = [ly.name for ly in reversed(self.viewer.layers) if isinstance(ly, wanted)]
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if spec.optional:
                combo.addItem(_NONE)
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            elif names:
                sel = self.viewer.layers.selection.active
                if sel is not None and sel.name in names:
                    combo.setCurrentText(sel.name)
            combo.blockSignals(False)

    # ---------------------------------------------------------------- exécution
    def _collect(self) -> tuple[list[Any], dict[str, Any]]:
        assert self._op is not None
        inputs: list[Any] = []
        for spec in self._op.inputs:
            name = self._inputs[spec.name].currentText()
            if not name or name == _NONE:
                if not spec.optional:
                    raise ValueError(f"Choisissez un calque pour « {spec.name} »")
                inputs.append(None)
                continue
            layer = self.viewer.layers[name]
            inputs.append(layer_to_image(layer) if spec.kind is Image else layer_to_labels(layer))
        params = {w.name: w.value for w in self._params} if self._params is not None else {}
        return inputs, params

    def _run(self) -> None:
        if self._op is None:
            return
        try:
            inputs, params = self._collect()
        except Exception as exc:
            QMessageBox.warning(self, "Analyse", str(exc))
            return
        op = self._op
        self.run_btn.setEnabled(False)
        self.message.emit(f"{op.name} en cours…")
        worker = create_worker(op.func, *inputs, **params, _start_thread=False)
        worker.returned.connect(lambda out: self._show_outputs(op, out))
        worker.errored.connect(lambda exc: self._error(op, exc))
        worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        worker.start()

    def _error(self, op: Operation, exc: Exception) -> None:
        QMessageBox.warning(self, op.name, f"Erreur : {exc}")
        self.message.emit(f"{op.name} : erreur")

    def _show_outputs(self, op: Operation, out: Any) -> None:
        items = out if isinstance(out, tuple) else (out,)
        for item in items:
            if item is None:
                continue
            if isinstance(item, Table):
                self.table_ready.emit(item)
                continue
            px = item.pixel_size_um or 1.0
            units = ("µm", "µm") if item.pixel_size_um else None
            if isinstance(item, Labels):
                layer = self.viewer.add_labels(item.data, name=item.name, scale=(px, px), units=units)
            else:
                layer = self.viewer.add_image(
                    item.data, name=item.name, scale=(px, px), rgb=item.is_rgb, units=units
                )
            layer.metadata["pixel_size_um"] = item.pixel_size_um
        self.message.emit(f"{op.name} terminé")

    def _open_in_fiji(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is None:
            QMessageBox.information(self, "Fiji", "Sélectionnez d'abord un calque.")
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


__all__ = ["AnalysisPanel", "layer_to_image", "layer_to_labels"]
