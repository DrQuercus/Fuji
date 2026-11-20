"""Images ouvertes dans l'espace Analyse.

Une **image ouverte** (un « document ») regroupe le calque de l'image elle-même et les
calques qui en découlent : masques, résultats des opérations, aperçus. On l'ouvre, on
passe d'une image à l'autre et on la ferme d'un bloc, comme une fenêtre d'ImageJ.

Les calques de dessin (« ROI », « Points ») sont communs à toutes les images, comme le
ROI Manager d'ImageJ ; ils sont fermés avec la dernière image.

Le document d'un calque est noté dans ses métadonnées (clé ``fuji_document``). Une image
ajoutée sans cette note (par un script, par l'IA) forme son propre document.

Une image peut avoir **plusieurs canaux** (DAPI, GFP, fond clair…) : chacun est un calque
« source » de la même image, avec sa couleur ; ils se superposent à l'écran.
"""

from __future__ import annotations

from typing import Any

from napari.components import ViewerModel
from napari.layers import Image as ImageLayer

DOC_KEY = "fuji_document"
SOURCE_KEY = "fuji_source"
#: nom du canal d'un calque source (image à plusieurs canaux)
CHANNEL_KEY = "fuji_channel"
#: visibilité d'un calque au moment où son image a été cachée (pour la lui rendre)
_SHOWN_KEY = "fuji_shown"
#: nombre d'images ouvertes au-delà duquel les plus anciennes sans résultat sont refermées
MAX_OPEN = 12
PREVIEW_PREFIX = "Aperçu — "
#: les deux calques de dessin, retrouvés par leur nom (qui ne se change donc pas)
ROI_LAYER = "ROI"
POINTS_LAYER = "Points"


def camera_of(viewer: ViewerModel) -> Any:
    """La caméra d'une vue (zoom, centre). Son emplacement a changé avec napari 0.9 :
    ``viewer.scene.camera`` ; l'ancien ``viewer.camera`` avertit à chaque lecture."""
    scene = getattr(viewer, "scene", None)
    camera = getattr(scene, "camera", None)
    return camera if camera is not None else viewer.camera


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


def unique_name(viewer: ViewerModel, base: str) -> str:
    """Un nom d'image ouverte libre : ``base``, sinon « base (2) », « base (3) »…"""
    taken = set(documents(viewer)) | {ly.name for ly in viewer.layers}
    if base not in taken:
        return base
    i = 2
    while f"{base} ({i})" in taken:
        i += 1
    return f"{base} ({i})"


def path_of(viewer: ViewerModel, doc: str) -> str | None:
    """Le fichier d'une image ouverte (None pour une image calculée)."""
    for layer in layers_of(viewer, doc):
        if is_source(layer) and layer.metadata.get("source"):
            return str(layer.metadata["source"])
    return None


def find_path(viewer: ViewerModel, path: Any) -> str | None:
    """L'image ouverte qui vient de ce fichier, ou None."""
    from pathlib import Path

    try:
        wanted = Path(str(path)).resolve()
    except (OSError, RuntimeError):
        return None
    for doc in documents(viewer):
        source = path_of(viewer, doc)
        if source is not None:
            try:
                if Path(source).resolve() == wanted:
                    return doc
            except (OSError, RuntimeError):
                continue
    return None


def channels_of(viewer: ViewerModel, doc: str) -> list[Any]:
    """Les canaux d'une image : ses calques sources (un seul pour une image simple)."""
    return [ly for ly in layers_of(viewer, doc) if is_source(ly) and isinstance(ly, ImageLayer)]


def channel_name(layer: Any) -> str:
    return str(layer.metadata.get(CHANNEL_KEY) or layer.metadata.get("channel") or layer.name)


def shown(viewer: ViewerModel) -> str | None:
    """L'image affichée (celle dont un calque source est visible), ou None."""
    for doc in reversed(documents(viewer)):
        if any(ly.visible for ly in channels_of(viewer, doc)):
            return doc
    return None


def hide_layer(layer: Any) -> None:
    """Cache un calque d'une image qui n'est pas affichée ; il reparaîtra avec elle."""
    layer.metadata[_SHOWN_KEY] = True
    layer.visible = False


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
    """Ferme ces images (et ce qui en découle) ; avec la dernière, les calques communs.
    Si l'image affichée est fermée, la dernière ouverte des autres prend sa place."""
    remaining = [d for d in documents(viewer) if d not in docs]
    displayed = shown(viewer)
    doomed = [ly for ly in viewer.layers if doc_of(ly) in docs]
    if not remaining:
        doomed = list(viewer.layers)
    for layer in doomed:
        if layer in viewer.layers:
            viewer.layers.remove(layer)
    if remaining and (displayed in docs or displayed is None or shown(viewer) is None):
        show_only(viewer, remaining[-1])


def show_only(viewer: ViewerModel, doc: str) -> None:
    """Affiche cette image (et ses résultats) ; les autres images restent ouvertes, cachées.
    Chaque calque retrouve en revenant l'état qu'il avait : un canal ou un masque que l'on
    avait caché le reste."""
    for layer in viewer.layers:
        d = doc_of(layer)
        if d is None:
            continue
        if d == doc:
            if _SHOWN_KEY in layer.metadata:
                layer.visible = bool(layer.metadata.pop(_SHOWN_KEY))
        elif _SHOWN_KEY not in layer.metadata:
            layer.metadata[_SHOWN_KEY] = bool(layer.visible)
            layer.visible = False
    active = viewer.layers.selection.active
    if active is not None and doc_of(active) == doc and active.visible:
        return  # on garde le canal ou le résultat que l'on réglait
    source = next((ly for ly in channels_of(viewer, doc) if ly.visible), None) or source_layer(viewer, doc)
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
    "CHANNEL_KEY",
    "DOC_KEY",
    "MAX_OPEN",
    "channel_name",
    "channels_of",
    "close",
    "doc_of",
    "documents",
    "find_path",
    "hide_layer",
    "is_source",
    "layers_of",
    "path_of",
    "rename",
    "results_lost",
    "shared_layers",
    "show_only",
    "shown",
    "source_layer",
    "tag",
    "unique_name",
]
