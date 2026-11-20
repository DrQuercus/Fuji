"""Ouvrir des images dans l'espace Analyse : une image à un ou plusieurs canaux, ou
plusieurs images superposées en canaux (DAPI + GFP + fond clair…).

Chaque canal est un calque de l'image ouverte, en niveaux de gris avec sa couleur (une
capture couleur de fluorescence est ramenée à la couleur de la caméra qui porte son
signal), additionné aux autres à l'écran ; un fond clair reste en couleur, en dessous.
On règle ensuite chaque canal dans la barre sous l'image (couleur, contraste, visible ou
non), et l'on en fait une image couleur pour une figure.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image as ImageLayer

from fuji.analysis.channels import (
    ChannelImage,
    Shown,
    channel_plane,
    composite_rgb,
    order_channels,
)
from fuji.core.frame import is_rgb

from . import documents
from .live import contrast_for

#: au-delà, les plans d'une acquisition sont lus à la demande (pas tout en mémoire)
MEMMAP_BYTES = 1 << 30
#: (type, couleur, longueur d'onde d'émission) d'un canal d'après son nom (configuration)
Lookup = Callable[[str], tuple[str | None, str | None, float | None]]


@dataclass
class Loaded:
    """Une image lue sur le disque."""

    data: np.ndarray
    axes: str = ""
    pixel_size_um: float | None = None
    channel_names: list[str] = field(default_factory=list)

    @property
    def rgb(self) -> bool:
        return self.axes.endswith("S") or (not self.axes and is_rgb(self.data))


def read_file(path: str | Path) -> Loaded:
    """Lit un fichier image : TIFF (dont OME-TIFF, avec ses canaux), PNG, JPEG, BMP."""
    from fuji.analysis.image import read_pixel_size_um

    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        from skimage import io

        return Loaded(io.imread(path))
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        px = read_pixel_size_um(tif)
        names = _channel_names(tif)
        data = None
        if series.size * series.dtype.itemsize > MEMMAP_BYTES:
            try:  # grande acquisition : les plans sont lus à la demande
                data = tifffile.memmap(path, mode="r")
            except Exception:  # fichier compressé ou en mosaïque
                data = None
        if data is None:
            data = series.asarray()
    return Loaded(data, axes, px, names)


def _channel_names(tif: Any) -> list[str]:
    try:
        import tifffile

        if not tif.ome_metadata:
            return []
        pixels = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]
        pixels = (pixels[0] if isinstance(pixels, list) else pixels)["Pixels"]
        channels = pixels.get("Channel") or []
        channels = channels if isinstance(channels, list) else [channels]
        return [str(c.get("Name") or "") for c in channels]
    except Exception:  # métadonnées d'un autre logiciel : pas de noms
        return []


def sidecar_metadata(path: str | Path) -> dict[str, Any]:
    """Les renseignements enregistrés à côté d'une image prise par Fuji (canal, objectif…)."""
    from fuji.io.captures import CaptureInfo

    info = CaptureInfo(path=Path(path))
    try:
        return dict(json.loads(info.info_path.read_text(encoding="utf-8")).get("metadata") or {})
    except (OSError, ValueError):
        return {}


def split_channels(
    loaded: Loaded, lookup: Lookup | None = None, metadata: dict[str, Any] | None = None
) -> list[ChannelImage] | None:
    """Les canaux d'une image à plusieurs canaux (axe C), prêts à superposer ; None si
    l'image n'a qu'un canal."""
    axes, data = loaded.axes, loaded.data
    if "C" not in axes or data.shape[axes.index("C")] < 2:
        return None
    c_axis = axes.index("C")
    count = data.shape[c_axis]
    names = list(loaded.channel_names) + [""] * count
    out: list[ChannelImage] = []
    for i in range(count):
        plane = data[(slice(None),) * c_axis + (i,)]  # une vue, sans copie
        name = names[i] or f"C{i + 1}"
        kind, color, emission = lookup(name) if lookup is not None else (None, None, None)
        md = {**(metadata or {}), "channel": name}
        if emission:
            md["emission_nm"] = emission
        out.append(
            channel_plane(plane, name, kind=kind, color=color, taken=[c.color for c in out], metadata=md)
        )
    return out


def auto_limits(data: np.ndarray, rgb: bool = False) -> tuple[float, float]:
    """Contraste d'un canal à son ouverture : 0,35 % de pixels saturés de chaque côté, comme
    le « Auto » d'ImageJ, sur le premier plan d'une pile."""
    from .display_bar import auto_limits as limits

    extra = data.ndim - (3 if rgb else 2)
    plane = np.asarray(data[(0,) * extra] if extra > 0 else data)
    step = max(1, int(np.sqrt(plane.shape[0] * plane.shape[1] / 250_000)))
    return limits(plane[::step, ::step])


def add_channels(
    viewer: ViewerModel,
    doc: str,
    channels: Sequence[ChannelImage],
    *,
    pixel_size_um: float | None = None,
    metadata: dict[str, Any] | None = None,
    source: str | None = None,
) -> list[ImageLayer]:
    """Ajoute une image à plusieurs canaux : un calque par canal, dans sa couleur, additionnés
    à l'écran (un fond clair en dessous, tel quel)."""
    layers: list[ImageLayer] = []
    for i, ch in enumerate(order_channels(channels)):
        extra = ch.data.ndim - (3 if ch.rgb else 2)
        scale = [1.0] * extra + [pixel_size_um or 1.0] * 2
        units = ("pixel",) * extra + ("µm", "µm") if pixel_size_um else None
        kwargs: dict[str, Any] = {"rgb": ch.rgb}
        if ch.rgb:
            kwargs["contrast_limits"] = contrast_for(
                np.asarray(ch.data[(0,) * extra] if extra else ch.data), None
            )
        else:
            kwargs["colormap"] = ch.color
            kwargs["contrast_limits"] = auto_limits(ch.data)
        bottom = i == 0 and (ch.rgb or ch.kind == "transmission")
        # les renseignements sont posés dès la création : la barre sous l'image et les
        # onglets reconnaissent aussitôt un canal de cette image
        md: dict[str, Any] = {**(metadata or {}), **ch.metadata}
        md[documents.CHANNEL_KEY] = ch.name
        md["pixel_size_um"] = pixel_size_um
        if ch.kind:
            md["contrast"] = ch.kind
        if source:
            md["source"] = source
        md[documents.DOC_KEY] = doc
        md[documents.SOURCE_KEY] = True
        layer = viewer.add_image(
            ch.data,
            name=documents.unique_name(viewer, f"{doc} · {ch.name}"),
            scale=scale,
            units=units,
            blending="translucent_no_depth" if bottom else "additive",
            metadata=md,
            **kwargs,
        )
        layers.append(layer)
    return layers


def _tint(layer: ImageLayer) -> tuple[float, float, float]:
    """La couleur pure d'un canal (celle de ses pixels les plus clairs)."""
    try:
        colors = np.asarray(layer.colormap.colors, dtype=np.float32)
        return tuple(float(v) for v in colors[-1][:3])  # type: ignore[return-value]
    except Exception:
        return (1.0, 1.0, 1.0)


def shown_channels(viewer: ViewerModel, doc: str) -> list[Shown]:
    """Les canaux d'une image tels qu'ils sont montrés (plan affiché d'une pile)."""
    from .panels.analysis_panel import displayed_plane

    out = []
    for layer in documents.channels_of(viewer, doc):
        plane, _ = displayed_plane(layer, viewer)
        out.append(
            Shown(
                plane,
                color="",
                limits=tuple(float(v) for v in layer.contrast_limits),  # type: ignore[arg-type]
                gamma=float(layer.gamma),
                rgb=bool(layer.rgb),
                visible=bool(layer.visible),
                additive=str(layer.blending) == "additive",
                opacity=float(layer.opacity),
                tint=None if layer.rgb else _tint(layer),
            )
        )
    return out


def flatten(viewer: ViewerModel, doc: str) -> np.ndarray:
    """L'image couleur (8 bits) d'une image à plusieurs canaux, telle qu'on la voit."""
    return composite_rgb(shown_channels(viewer, doc))


__all__ = [
    "MEMMAP_BYTES",
    "Loaded",
    "add_channels",
    "auto_limits",
    "flatten",
    "read_file",
    "shown_channels",
    "sidecar_metadata",
    "split_channels",
]
