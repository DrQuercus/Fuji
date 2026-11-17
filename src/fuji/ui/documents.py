"""Images ouvertes dans l'espace Analyse.

Une **image ouverte** (un « document ») regroupe le calque de l'image elle-même et les
calques qui en découlent : masques, résultats des opérations, aperçus. On l'ouvre, on
passe d'une image à l'autre et on la ferme d'un bloc, comme une fenêtre d'ImageJ.

Les calques de dessin (« ROI », « Points ») sont communs à toutes les images, comme le
ROI Manager d'ImageJ ; ils sont fermés avec la dernière image.

Le document d'un calque est noté dans ses métadonnées (clé ``fuji_document``). Une image
ajoutée sans cette note (par un script, par l'IA) forme son propre document.
"""

from __future__ import annotations

from typing import Any

from napari.components import ViewerModel
from napari.layers import Image as ImageLayer

DOC_KEY = "fuji_document"
SOURCE_KEY = "fuji_source"
PREVIEW_PREFIX = "Aperçu — "


def tag(layer: Any, doc: str | None, *, source: bool = False) -> None:
    """Rattache un calque à une image ouverte (``source`` : c'est l'image elle-même)."""
    if doc is None:
        return
    layer.metadata[DOC_KEY] = doc
    if source:
        layer.metadata[SOURCE_KEY] = True


def doc_of(layer: Any) -> str | None:
    doc = layer.metadata.get(DOC_KEY)
    if doc is not None:
        return str(doc)
    if isinstance(layer, ImageLayer) and not layer.name.startswith(PREVIEW_PREFIX):
        return layer.name  # image ajoutée sans note : son propre document
    return None


def is_source(layer: Any) -> bool:
    if layer.metadata.get(SOURCE_KEY):
        return True
    return DOC_KEY not in layer.metadata and doc_of(layer) is not None


def documents(viewer: ViewerModel) -> list[str]:
    """Les images ouvertes, dans l'ordre d'ouverture."""
    out: list[str] = []
    for layer in viewer.layers:
        doc = doc_of(layer)
        if doc is not None and doc not in out:
            out.append(doc)
    return out


def layers_of(viewer: ViewerModel, doc: str) -> list[Any]:
    return [ly for ly in viewer.layers if doc_of(ly) == doc]


def source_layer(viewer: ViewerModel, doc: str) -> Any | None:
    layers = layers_of(viewer, doc)
    return next((ly for ly in layers if is_source(ly)), layers[0] if layers else None)


def shared_layers(viewer: ViewerModel) -> list[Any]:
    """Calques communs à toutes les images (ROI, points, calques ajoutés à la main)."""
    return [ly for ly in viewer.layers if doc_of(ly) is None]


def _has_content(layer: Any) -> bool:
    """Faux pour un calque de dessin vide (aucune forme, aucun point) ou un aperçu."""
    if layer.name.startswith(PREVIEW_PREFIX):
        return False
    try:
        return len(layer.data) > 0
    except TypeError:
        return True


def results_lost(viewer: ViewerModel, docs: list[str]) -> list[Any]:
    """Calques de résultats (masques, ROI dessinées…) qui disparaîtraient en fermant ``docs``."""
    lost = [
        ly
        for doc in docs
        for ly in layers_of(viewer, doc)
        if not is_source(ly) and not ly.name.startswith(PREVIEW_PREFIX)
    ]
    if set(docs) >= set(documents(viewer)):  # dernière image : les calques communs partent aussi
        lost += [ly for ly in shared_layers(viewer) if _has_content(ly)]
    return lost


def close(viewer: ViewerModel, docs: list[str]) -> None:
    """Ferme ces images (et ce qui en découle) ; avec la dernière, les calques communs."""
    remaining = [d for d in documents(viewer) if d not in docs]
    doomed = [ly for ly in viewer.layers if doc_of(ly) in docs]
    if not remaining:
        doomed = list(viewer.layers)
    for layer in doomed:
        if layer in viewer.layers:
            viewer.layers.remove(layer)
    if remaining:
        show_only(viewer, remaining[-1])


def show_only(viewer: ViewerModel, doc: str) -> None:
    """Affiche cette image (et ses résultats) ; les autres images restent ouvertes, cachées."""
    for layer in viewer.layers:
        d = doc_of(layer)
        if d is not None:
            layer.visible = d == doc
    source = source_layer(viewer, doc)
    if source is not None:
        viewer.layers.selection.active = source


def rename(viewer: ViewerModel, old: str, new: str) -> None:
    """L'image a été renommée dans la liste : son calque et ses résultats suivent."""
    for layer in layers_of(viewer, old):
        source = is_source(layer)
        if layer.name == old:
            layer.name = new
        tag(layer, new, source=source)


__all__ = [
    "DOC_KEY",
    "close",
    "doc_of",
    "documents",
    "is_source",
    "layers_of",
    "rename",
    "results_lost",
    "shared_layers",
    "show_only",
    "source_layer",
    "tag",
]
