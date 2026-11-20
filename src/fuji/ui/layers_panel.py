"""Panneau « Calques » : ce qui est posé sur l'image, dit simplement.

Une ligne par calque (un œil pour le montrer ou le cacher, son genre, son nom), les actions
courantes (monter, descendre, renommer, enregistrer, supprimer) et les réglages utiles du
calque choisi : opacité, superposition, contours d'un masque. Les réglages complets de napari
— en anglais — ne sont montrés qu'en mode expert.
"""

from __future__ import annotations

from typing import Any

from napari.components import ViewerModel
from napari.layers import Image as ImageLayer
from napari.layers import Labels as LabelsLayer
from napari.layers import Points as PointsLayer
from napari.layers import Shapes as ShapesLayer
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import documents
from .components import Card, EmptyState, button, flow, hint, icon_button, label, set_icon, shortcut

#: genre d'un calque : (nom, icône)
KINDS: tuple[tuple[type, str, str], ...] = (
    (LabelsLayer, "Masque", "roi_free"),
    (ShapesLayer, "Formes dessinées", "roi_rect"),
    (PointsLayer, "Points", "roi_point"),
    (ImageLayer, "Image", "gallery"),
)
#: façons de superposer une image à celles du dessous : (texte, mode napari)
BLENDINGS = (
    ("Normale (cache ce qui est dessous)", "translucent_no_depth"),
    ("Additive (canaux de fluorescence)", "additive"),
    ("Minimum (colorations en fond clair)", "minimum"),
)
_CONTOUR = 2  # épaisseur du contour d'un masque (pixels)


def kind_of(layer: Any) -> tuple[str, str]:
    """Le genre d'un calque et son icône."""
    for cls, name, icon_name in KINDS:
        if isinstance(layer, cls):
            if cls is ImageLayer and layer.name.startswith(documents.PREVIEW_PREFIX):
                return "Aperçu", "sparkle"
            return name, icon_name
    return "Calque", "layers"


class LayerRow(QFrame):
    """Une ligne de la liste : œil, genre, nom."""

    clicked = Signal(object)
    renamed = Signal(object)

    def __init__(self, layer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layer = layer
        self.setObjectName("LayerRow")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 3, 8, 3)
        lay.setSpacing(6)
        self.eye = icon_button("eye", "Montrer ou cacher ce calque", checkable=True)
        self.eye.setFixedWidth(30)
        self.eye.clicked.connect(self._toggle)
        lay.addWidget(self.eye)
        kind, icon_name = kind_of(layer)
        self.icon = QLabel()
        set_icon(self.icon, icon_name, "muted", 16)
        self.icon.setToolTip(kind)
        lay.addWidget(self.icon)
        self.name = label("", wrap=False)
        self.name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.name, 1)
        self.refresh()

    def refresh(self) -> None:
        layer = self.layer
        self.name.setText(layer.name)
        self.name.setToolTip(f"{kind_of(layer)[0]} : {layer.name}")
        self.name.setEnabled(bool(layer.visible))
        self.eye.setChecked(bool(layer.visible))
        set_icon(self.eye, "eye" if layer.visible else "eye_off", "text" if layer.visible else "faint", 16)

    def set_current(self, on: bool) -> None:
        if self.property("current") != on:
            self.setProperty("current", on)
            self.style().unpolish(self)
            self.style().polish(self)

    def _toggle(self) -> None:
        self.layer.visible = not self.layer.visible
        self.refresh()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit(self.layer)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, _event: Any) -> None:
        self.renamed.emit(self.layer)


class LayersPanel(QWidget):
    """La liste des calques de l'image affichée et les réglages du calque choisi."""

    message = Signal(str)
    #: « Enregistrer… » le calque choisi
    save_requested = Signal()
    #: supprimer une image (et ses résultats) : c'est la fenêtre qui la ferme, avec sa question
    close_requested = Signal(str)

    def __init__(
        self, viewer: ViewerModel, napari_widgets: list[QWidget], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._rows: list[LayerRow] = []
        self._watched: Any = None
        self._doc: str | None = None  # l'image dont on montre les calques
        self._syncing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.empty = EmptyState(
            "layers", "Aucun calque", "Ouvrez une image : elle apparaîtra ici avec ses résultats."
        )
        lay.addWidget(self.empty)
        self.list_box = QWidget()
        self.list_lay = QVBoxLayout(self.list_box)
        self.list_lay.setContentsMargins(0, 0, 0, 0)
        self.list_lay.setSpacing(1)
        lay.addWidget(self.list_box)

        self.up_btn = icon_button("up", "Monter : passer devant le calque du dessus")
        self.down_btn = icon_button("down", "Descendre : passer derrière le calque du dessous")
        self.rename_btn = button("Renommer", "edit")
        self.save_btn = button(
            "Enregistrer…", "save", tooltip=f"Enregistrer ce calque dans un fichier ({shortcut('Ctrl+S')})"
        )
        self.delete_btn = button("Supprimer", "trash", object_name="danger")
        for b in (self.up_btn, self.down_btn):
            b.setFixedWidth(34)
        self.up_btn.clicked.connect(lambda: self.move_selected(+1))
        self.down_btn.clicked.connect(lambda: self.move_selected(-1))
        self.rename_btn.clicked.connect(lambda: self.rename(self.selected()))
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.actions_row = flow(self.up_btn, self.down_btn, self.rename_btn, self.save_btn, self.delete_btn)
        lay.addWidget(self.actions_row)

        # réglages du calque choisi
        self.card = Card("Réglages du calque", icon_name="sliders")
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setToolTip("Transparence du calque : à 0 % il ne se voit plus")
        self.opacity.valueChanged.connect(self._opacity_changed)
        self.opacity_label = label("100 %", wrap=False)
        self.opacity_label.setMinimumWidth(44)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Opacité"))
        row.addWidget(self.opacity, 1)
        row.addWidget(self.opacity_label)
        self.card.add(layout=row)
        self.blending = QComboBox()
        for text, mode in BLENDINGS:
            self.blending.addItem(text, mode)
        self.blending.setToolTip("Comment cette image se mêle à celles qui sont dessous")
        self.blending.currentIndexChanged.connect(self._blending_changed)
        self.blending_row = QWidget()
        bl = QVBoxLayout(self.blending_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)
        bl.addWidget(QLabel("Superposition"))
        bl.addWidget(self.blending)
        self.card.add(self.blending_row)
        self.contour = QCheckBox("Contours seulement")
        self.contour.setToolTip("Ne dessiner que le contour des objets du masque")
        self.contour.toggled.connect(self._contour_changed)
        self.card.add(self.contour)
        self.tip = hint("")
        self.card.add(self.tip)
        lay.addWidget(self.card)

        # mode expert : les réglages complets de napari
        self.expert_box = Card("Réglages complets (napari, en anglais)", icon_name="code")
        for w in napari_widgets:
            self.expert_box.add(w)
        self.expert_box.hide()
        lay.addWidget(self.expert_box)
        lay.addStretch(1)

        self._timer = QTimer(self)  # plusieurs changements d'affilée : une seule mise à jour
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.rebuild)
        events = viewer.layers.events
        events.inserted.connect(self._inserted)
        events.removed.connect(self._schedule)
        events.reordered.connect(self._schedule)
        viewer.layers.selection.events.active.connect(self._selection_changed)
        for layer in viewer.layers:
            self._watch(layer)
        self.rebuild()

    # ------------------------------------------------------------------ liste
    def set_expert(self, expert: bool) -> None:
        self.expert_box.setVisible(bool(expert))

    def _schedule(self, *_: Any) -> None:
        self._timer.start()

    def _inserted(self, event: Any) -> None:
        layer = getattr(event, "value", None)
        if layer is not None:
            self._watch(layer)
        self._timer.start()

    def _watch(self, layer: Any) -> None:
        layer.events.visible.connect(self._schedule)
        layer.events.name.connect(self._schedule)

    def layers(self) -> list[Any]:
        """Les calques montrés, celui du dessus en premier : ceux de l'image affichée et les
        calques communs (ROI, points). Les autres images ouvertes ont leur propre liste."""
        shown = [
            ly
            for ly in self.viewer.layers
            if ly.visible and documents.is_source(ly) and documents.doc_of(ly) is not None
        ]
        if shown:
            self._doc = documents.doc_of(shown[-1])
        elif self._doc not in documents.documents(self.viewer):
            self._doc = None  # image cachée par son œil : on garde sa liste ; fermée : on l'oublie
        current = self._doc
        keep = [ly for ly in self.viewer.layers if current is None or documents.doc_of(ly) in (None, current)]
        return list(reversed(keep))

    def selected(self) -> Any | None:
        active = self.viewer.layers.selection.active
        return active if active is not None and active in self.layers() else None

    def rebuild(self) -> None:
        """Refait la liste d'après les calques de la vue."""
        layers = self.layers()
        if [r.layer for r in self._rows] != layers:
            for row in self._rows:
                self.list_lay.removeWidget(row)
                row.deleteLater()
            self._rows = []
            for layer in layers:
                row = LayerRow(layer)
                row.clicked.connect(self.select)
                row.renamed.connect(self.rename)
                self.list_lay.addWidget(row)
                self._rows.append(row)
        else:
            for row in self._rows:
                row.refresh()
        has = bool(layers)
        self.empty.setVisible(not has)
        self.list_box.setVisible(has)
        self.actions_row.setVisible(has)
        self._selection_changed()

    def select(self, layer: Any) -> None:
        if layer in self.viewer.layers:
            self.viewer.layers.selection.active = layer

    # ------------------------------------------------------------- calque choisi
    def _selection_changed(self, *_: Any) -> None:
        layer = self.selected()
        for row in self._rows:
            row.set_current(row.layer is layer)
        if self._watched is not None and self._watched is not layer:
            for name in ("opacity", "blending", "contour"):
                emitter = getattr(self._watched.events, name, None)
                if emitter is not None:
                    try:
                        emitter.disconnect(self._sync)
                    except Exception:  # calque déjà détruit
                        pass
        if layer is not None and layer is not self._watched:
            for name in ("opacity", "blending", "contour"):
                emitter = getattr(layer.events, name, None)
                if emitter is not None:
                    emitter.connect(self._sync)
        self._watched = layer
        self._sync()

    def _sync(self, *_: Any) -> None:
        layer = self.selected()
        rows = self.layers()
        self.card.setVisible(layer is not None)
        for b in (self.up_btn, self.down_btn, self.rename_btn, self.save_btn, self.delete_btn):
            b.setEnabled(layer is not None)
        if layer is None:
            return
        index = rows.index(layer)
        self.up_btn.setEnabled(index > 0)
        self.down_btn.setEnabled(index < len(rows) - 1)
        source = documents.is_source(layer) and documents.doc_of(layer) is not None
        self.rename_btn.setEnabled(not source and not _is_drawing(layer))
        self.rename_btn.setToolTip(
            "Une image se renomme dans la liste « Images » (son fichier suit)"
            if source
            else "Le calque du dessin garde son nom : c'est par lui que les mesures le retrouvent"
            if _is_drawing(layer)
            else "Changer le nom du calque"
        )
        self.save_btn.setEnabled(isinstance(layer, (ImageLayer, LabelsLayer)))
        self._syncing = True
        try:
            self.opacity.setValue(int(round(float(layer.opacity) * 100)))
            self.opacity_label.setText(f"{self.opacity.value()} %")
            is_image = isinstance(layer, ImageLayer)
            self.blending_row.setVisible(is_image)
            if is_image:
                i = self.blending.findData(str(layer.blending))
                if i < 0:  # mode posé ailleurs : ajouté tel quel
                    self.blending.addItem(str(layer.blending), str(layer.blending))
                    i = self.blending.count() - 1
                self.blending.setCurrentIndex(i)
            is_mask = isinstance(layer, LabelsLayer)
            self.contour.setVisible(is_mask)
            if is_mask:
                self.contour.setChecked(int(layer.contour) > 0)
        finally:
            self._syncing = False
        if isinstance(layer, LabelsLayer):
            tip = "La transparence de tous les masques se règle aussi dans la barre sous l'image."
        elif isinstance(layer, (ShapesLayer, PointsLayer)):
            tip = "Couleur, épaisseur, numéros et mesures : roue dentée de la rangée « Dessiner » (Outils)."
        else:
            tip = "Contraste, luminosité et couleur : dans la barre sous l'image."
        self.tip.setText(tip)

    def _opacity_changed(self, value: int) -> None:
        self.opacity_label.setText(f"{value} %")
        layer = self.selected()
        if not self._syncing and layer is not None:
            layer.opacity = value / 100

    def _blending_changed(self, *_: Any) -> None:
        layer = self.selected()
        if not self._syncing and isinstance(layer, ImageLayer) and self.blending.currentIndex() >= 0:
            layer.blending = self.blending.currentData()

    def _contour_changed(self, on: bool) -> None:
        layer = self.selected()
        if not self._syncing and isinstance(layer, LabelsLayer):
            layer.contour = _CONTOUR if on else 0

    # ------------------------------------------------------------------ actions
    def move_selected(self, step: int) -> None:
        """Monte (+1) ou descend (−1) le calque choisi dans l'empilement."""
        layer = self.selected()
        if layer is None:
            return
        rows = self.layers()
        target = rows.index(layer) - step  # la liste montre le dessus en premier
        if not 0 <= target < len(rows):
            return
        layers = self.viewer.layers
        src, dst = layers.index(layer), layers.index(rows[target])
        layers.move(src, dst + 1 if dst > src else dst)
        self.select(layer)

    def rename(self, layer: Any | None) -> None:
        if layer is None or (documents.is_source(layer) and documents.doc_of(layer) is not None):
            return
        if _is_drawing(layer):  # renommé, ses étiquettes se figeaient et « Mesurer » l'ignorait
            self.message.emit(f"Le calque « {layer.name} » garde son nom (les mesures le retrouvent par lui)")
            return
        name, ok = QInputDialog.getText(self, "Renommer le calque", "Nouveau nom :", text=layer.name)
        name = name.strip()
        if ok and name and name != layer.name:
            layer.name = name
            self.message.emit(f"Calque renommé : {layer.name}")

    def delete_selected(self) -> None:
        """Supprime le calque choisi ; une image est fermée par la fenêtre (avec ses résultats)."""
        layer = self.selected()
        if layer is None:
            return
        doc = documents.doc_of(layer)
        if documents.is_source(layer) and doc is not None:
            self.close_requested.emit(doc)
        elif layer in self.viewer.layers:
            self.viewer.layers.remove(layer)


def _is_drawing(layer: Any) -> bool:
    """Un des deux calques de dessin (« ROI », « Points ») ?"""
    return (isinstance(layer, ShapesLayer) and layer.name == documents.ROI_LAYER) or (
        isinstance(layer, PointsLayer) and layer.name == documents.POINTS_LAYER
    )


__all__ = ["BLENDINGS", "LayerRow", "LayersPanel", "kind_of"]
