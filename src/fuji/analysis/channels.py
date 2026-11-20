"""Canaux d'une image : les reconnaître, les colorer, les superposer.

Une image à plusieurs canaux (DAPI, GFP, fond clair…) se montre comme dans le « Composite »
d'ImageJ : chaque canal en niveaux de gris, coloré par sa table de couleurs, les canaux
additionnés à l'écran. Ici, les fonctions sans interface :

- :func:`channel_color` — la couleur d'un canal, d'après ce que l'on sait de lui (couleur
  choisie dans la configuration, longueur d'onde, nom du fluorochrome) ;
- :func:`channel_plane` — ce qui sera montré d'une image : une capture couleur de
  fluorescence devient un plan d'intensité (celui des trois couleurs de la caméra qui porte
  le signal) ; un fond clair ou un fond noir en couleur reste en couleur ;
- :func:`composite_rgb` — l'image couleur que l'on voit, à enregistrer pour une figure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: tables de couleurs proposées pour un canal (noms napari), dans l'ordre des menus
COLORS = ("blue", "cyan", "green", "yellow", "red", "magenta", "gray")
#: leur nom en français
COLOR_NAMES = {
    "blue": "Bleu",
    "cyan": "Cyan",
    "green": "Vert",
    "yellow": "Jaune",
    "red": "Rouge",
    "magenta": "Magenta",
    "gray": "Gris",
}
#: couleur RVB de chaque table (pour l'image couleur finale)
RGB = {
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "gray": (1.0, 1.0, 1.0),
}
#: couleurs données aux canaux de fluorescence inconnus, dans cet ordre (contrastées entre elles)
_SPARE = ("green", "magenta", "cyan", "yellow", "red", "blue")
#: fluorochromes courants → couleur (comparaison sans majuscules, sur une partie du nom)
_DYES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dapi", "hoechst", "bleu", "blue", "uv", "bfp", "405", "350"), "blue"),
    (("cfp", "cyan", "440"), "cyan"),
    (
        ("gfp", "fitc", "488", "alexa 488", "af488", "cy2", "vert", "green", "yfp", "calcein", "sytox"),
        "green",
    ),
    (
        ("tritc", "cy3", "rfp", "mcherry", "tomato", "texas", "568", "594", "555", "561", "rhodamine", "pi"),
        "red",
    ),
    (("cy5", "647", "far red", "rouge lointain", "633", "640", "alexa 647"), "magenta"),
    (("rouge", "red", "orange"), "red"),
    (("jaune", "yellow"), "yellow"),
)
#: types de canal qui restent en couleur dans une superposition (on les voit tels quels)
KEEP_COLOR = ("transmission", "fond noir", "réflexion")


def color_from_name(name: str | None) -> str | None:
    """La couleur d'un fluorochrome d'après son nom (« GFP » → vert), ou None."""
    if not name:
        return None
    low = name.lower()
    for words, color in _DYES:
        for word in words:
            # « pi » ou « uv » seuls ne doivent pas reconnaître « pile » ou « uvée »
            if len(word) <= 3:
                if any(part == word for part in low.replace("-", " ").replace("_", " ").split()):
                    return color
            elif word in low:
                return color
    return None


def channel_color(
    name: str | None = None,
    kind: str | None = None,
    stored: str | None = None,
    emission_nm: float | None = None,
    taken: Iterable[str] = (),
) -> str:
    """La couleur d'affichage d'un canal.

    Dans l'ordre : la couleur choisie dans la configuration (sauf le gris d'un canal de
    fluorescence, valeur par défaut), la longueur d'onde d'émission, le nom du fluorochrome ;
    un fond clair ou un fond noir est gris ; sinon une couleur encore libre."""
    if stored in RGB and stored != "gray":
        return str(stored)
    if kind in KEEP_COLOR:
        return "gray"
    if emission_nm:
        from fuji.core.catalog import color_for_wavelength

        return color_for_wavelength(emission_nm)
    guess = color_from_name(name)
    if guess:
        return guess
    words = (name or "").lower().replace("_", " ").replace("-", " ").split()
    if any(w in words for w in ("fond", "clair", "noir", "bf", "df", "brightfield", "darkfield")):
        return "gray"
    used = set(taken)
    return next((c for c in _SPARE if c not in used), "gray")


def looks_transmitted(data: np.ndarray) -> bool:
    """Vrai pour une image au fond clair (fond clair, coloration) : son fond, la valeur la plus
    courante, est dans la moitié haute des valeurs."""
    sample = np.asarray(data[::8, ::8], dtype=np.float32)
    gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
    lo, hi = np.percentile(gray, (0.5, 99.5))
    return bool(hi > lo and np.median(gray) > lo + 0.5 * (hi - lo))


def best_component(rgb: np.ndarray, color: str | None = None) -> np.ndarray:
    """Le plan d'intensité d'une capture couleur de fluorescence.

    Avec une couleur connue, la ou les couleurs de la caméra correspondantes (bleu pour
    DAPI, vert pour GFP, rouge et bleu pour un magenta…). Sinon, celle des trois qui porte le
    plus de signal au-dessus du fond."""
    data = np.asarray(rgb)[..., :3]
    weights = RGB.get(color or "")
    if weights is not None and color != "gray":
        picked = [data[..., i] for i in range(3) if weights[i] > 0]
        if len(picked) == 1:
            return picked[0]  # une vue : une grande acquisition reste lue à la demande
        return np.maximum.reduce(picked)
    sample = np.asarray(data[::8, ::8], dtype=np.float32)
    signal = [float(np.percentile(sample[..., i], 99.9) - np.median(sample[..., i])) for i in range(3)]
    return data[..., int(np.argmax(signal))]


@dataclass
class ChannelImage:
    """Un canal à superposer : ses pixels (gris, ou couleur s'il reste en couleur), son nom,
    sa couleur et son type."""

    data: np.ndarray
    name: str
    color: str = "gray"
    kind: str | None = None
    rgb: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plane_shape(self) -> tuple[int, int]:
        shape = self.data.shape
        return (int(shape[-3]), int(shape[-2])) if self.rgb else (int(shape[-2]), int(shape[-1]))


def channel_plane(
    data: np.ndarray,
    name: str,
    *,
    kind: str | None = None,
    color: str | None = None,
    taken: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
) -> ChannelImage:
    """Prépare une image pour la superposer aux autres (voir le module)."""
    md = dict(metadata or {})
    data = np.asarray(data)
    rgb = data.ndim >= 3 and data.shape[-1] in (3, 4)
    if kind is None and rgb and looks_transmitted(data):
        kind = "transmission"
    color = channel_color(name, kind, color or md.get("channel_color"), md.get("emission_nm"), taken)
    if rgb and kind in KEEP_COLOR:
        return ChannelImage(np.ascontiguousarray(data[..., :3]), name, "gray", kind, True, md)
    if rgb:
        data = best_component(data, color)
    return ChannelImage(data, name, color, kind, False, md)


def order_channels(channels: Sequence[ChannelImage]) -> list[ChannelImage]:
    """Le fond clair en dessous (c'est le fond de la superposition), les autres par-dessus,
    dans l'ordre donné."""
    return sorted(channels, key=lambda c: 0 if c.kind == "transmission" else 1)


def scaled(plane: np.ndarray, limits: tuple[float, float], gamma: float = 1.0) -> np.ndarray:
    """Valeurs ramenées entre 0 et 1 entre les deux bornes du contraste (et le gamma)."""
    lo, hi = (float(v) for v in limits)
    out = (np.asarray(plane, dtype=np.float32) - np.float32(lo)) / np.float32(max(hi - lo, 1e-12))
    np.clip(out, 0.0, 1.0, out=out)
    if gamma != 1.0:
        np.power(out, np.float32(gamma), out=out)
    return out


@dataclass
class Shown:
    """Un canal tel qu'il est montré : ses pixels, sa couleur, son contraste, s'il est visible
    et s'il est additionné aux autres (sinon il couvre ce qui est dessous)."""

    data: np.ndarray
    color: str = "gray"
    limits: tuple[float, float] = (0.0, 1.0)
    gamma: float = 1.0
    rgb: bool = False
    visible: bool = True
    additive: bool = True
    opacity: float = 1.0
    #: couleur pure du canal (R, V, B entre 0 et 1), si elle n'est pas une des couleurs nommées
    tint: tuple[float, float, float] | None = None


def composite_rgb(channels: Sequence[Shown]) -> np.ndarray:
    """L'image couleur que l'on voit (8 bits) : chaque canal visible, dans sa couleur et avec
    son contraste, additionné à ce qui est dessous (ou posé dessus s'il n'est pas additif)."""
    shown = [c for c in channels if c.visible]
    if not shown:
        raise ValueError("Aucun canal visible à mettre dans l'image")
    first = shown[0]
    h, w = first.data.shape[-3:-1] if first.rgb else first.data.shape[-2:]
    out = np.zeros((h, w, 3), np.float32)
    for c in shown:
        unit = scaled(c.data, c.limits, c.gamma)
        if unit.shape[:2] != (h, w):
            raise ValueError("Les canaux n'ont pas tous la même taille")
        tint = c.tint if c.tint is not None else RGB.get(c.color, (1.0, 1.0, 1.0))
        layer = unit[..., :3] if c.rgb else unit[..., None] * np.asarray(tint, np.float32)
        layer = layer * np.float32(c.opacity)
        if c.additive:
            out += layer
        else:
            out = out * np.float32(1.0 - c.opacity) + layer
    np.clip(out, 0.0, 1.0, out=out)
    return np.round(out * 255).astype(np.uint8)


__all__ = [
    "COLORS",
    "COLOR_NAMES",
    "KEEP_COLOR",
    "RGB",
    "ChannelImage",
    "Shown",
    "best_component",
    "channel_color",
    "channel_plane",
    "color_from_name",
    "composite_rgb",
    "looks_transmitted",
    "order_channels",
    "scaled",
]
