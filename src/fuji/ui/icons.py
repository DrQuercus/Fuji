"""Icônes de Fuji (dessinées pour le projet, SVG en trait, colorées selon le thème)."""

from __future__ import annotations

import math
from functools import lru_cache

from qtpy.QtCore import QByteArray, QRectF, Qt
from qtpy.QtGui import QIcon, QPainter, QPixmap

_STROKE = 'fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'


def _rays(cx: float, cy: float, r1: float, r2: float, n: int, offset: float = 0) -> str:
    parts = []
    for i in range(n):
        a = 2 * math.pi * i / n + offset
        parts.append(
            f"M{cx + r1 * math.cos(a):.2f} {cy + r1 * math.sin(a):.2f}"
            f"L{cx + r2 * math.cos(a):.2f} {cy + r2 * math.sin(a):.2f}"
        )
    return f'<path d="{"".join(parts)}"/>'


_PATHS: dict[str, str] = {
    "microscope": '<path d="M7.6 3.6l3-1.5 4.5 8.4-3 1.6z"/><path d="M12.6 12.4l1 2"/>'
    '<path d="M8 16h8"/><path d="M15.4 6.6c3.6 1.6 4.6 6.6 2 10.2"/><path d="M12 16v4.4"/>'
    '<path d="M5 20.5h14"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.6"/>'
    '<path d="M12 2.5v3.5M12 18v3.5M2.5 12H6M18 12h3.5"/>',
    "timer": '<circle cx="12" cy="13.5" r="7.5"/><path d="M12 10v3.5l2.6 2.4"/>'
    '<path d="M9.5 2.5h5M12 2.5V6"/>',
    "chart": '<path d="M4 20h16"/><path d="M7 16.5v-4"/><path d="M12 16.5V7"/><path d="M17 16.5v-7"/>',
    "layers": '<path d="M12 3.5l8.5 4.6L12 12.7 3.5 8.1z"/><path d="M3.5 12.3l8.5 4.6 8.5-4.6"/>'
    '<path d="M3.5 16.3l8.5 4.6 8.5-4.6"/>',
    "sliders": '<path d="M4 6.5h9M17 6.5h3M4 12h3M11 12h9M4 17.5h11M19 17.5h1"/>'
    '<circle cx="15" cy="6.5" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17.5" r="2"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="6.6"/>' + _rays(12, 12, 6.6, 9.4, 8),
    "play": '<path d="M8 5.5v13l10.5-6.5z" fill="{c}"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="{c}"/>',
    "pause": '<path d="M8.5 6v12M15.5 6v12" stroke-width="3"/>',
    "record": '<circle cx="12" cy="12" r="6" fill="{c}"/>',
    "camera": '<path d="M4 8.5h3.2L8.8 6h6.4l1.6 2.5H20v10.5H4z"/><circle cx="12" cy="13.5" r="3.4"/>',
    "save": '<path d="M12 4v10.5"/><path d="M7.5 10L12 14.5 16.5 10"/><path d="M4.5 16.5v3h15v-3"/>',
    "folder": '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.7l2 2h7.3a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
    "open": '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.7l2 2h7.3a2 2 0 0 1 2 2v1"/>'
    '<path d="M3.5 18.5l2.6-7h15.4l-2.6 7z"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "trash": '<path d="M4.5 7h15"/><path d="M9.5 7V4.5h5V7"/><path d="M6.5 7l1 12.5h9l1-12.5"/>'
    '<path d="M10.5 11v5M13.5 11v5"/>',
    "copy": '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/>'
    '<path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5"/>',
    "up": '<path d="M6 15l6-6 6 6"/>',
    "down": '<path d="M6 9l6 6 6-6"/>',
    "right": '<path d="M9 6l6 6-6 6"/>',
    "left": '<path d="M15 6l-6 6 6 6"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "close": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    "warning": '<path d="M12 4l9 15.5H3z"/><path d="M12 10v4.5"/><path d="M12 17.3v.2"/>',
    "info": '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5"/><path d="M12 7.6v.2"/>',
    "sun": '<circle cx="12" cy="12" r="3.8"/>' + _rays(12, 12, 6.3, 8.8, 8),
    "moon": '<path d="M19.5 14.6A7.8 7.8 0 1 1 9.4 4.5a6.2 6.2 0 0 0 10.1 10.1z"/>',
    "focus": '<path d="M4 8.5V5.5a1.5 1.5 0 0 1 1.5-1.5h3M15.5 4h3A1.5 1.5 0 0 1 20 5.5v3'
    'M20 15.5v3a1.5 1.5 0 0 1-1.5 1.5h-3M8.5 20h-3A1.5 1.5 0 0 1 4 18.5v-3"/>'
    '<circle cx="12" cy="12" r="3"/>',
    "bulb": '<path d="M9.5 18h5M10.5 21h3"/>'
    '<path d="M12 3a6 6 0 0 0-3.6 10.8c.7.5 1.1 1.3 1.1 2.1V16h5v-.1c0-.8.4-1.6 1.1-2.1A6 6 0 0 0 12 3z"/>',
    "ruler": '<rect x="2.5" y="8" width="19" height="8" rx="1.5"/>'
    '<path d="M6 8v3M9.5 8v4.5M13 8v3M16.5 8v4.5M20 8v2"/>',
    "wand": '<path d="M4.5 19.5l9-9"/><path d="M16.5 3.5l1 2.6 2.6 1-2.6 1-1 2.6-1-2.6-2.6-1 2.6-1z"/>',
    "sparkle": '<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9z"/>'
    '<path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
    "move": '<path d="M12 3v18M3 12h18"/>'
    '<path d="M9.2 5.8L12 3l2.8 2.8M9.2 18.2L12 21l2.8-2.8M5.8 9.2L3 12l2.8 2.8M18.2 9.2L21 12l-2.8 2.8"/>',
    "updown": '<path d="M12 4v16"/><path d="M8.5 7.5L12 4l3.5 3.5M8.5 16.5L12 20l3.5-3.5"/>',
    "turret": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="7.6" r="1.7"/>'
    '<circle cx="16.4" cy="12" r="1.7"/><circle cx="12" cy="16.4" r="1.7"/><circle cx="7.6" cy="12" r="1.7"/>',
    "shutter": '<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5L14.5 12M20.5 12L12 14.5M12 20.5L9.5 12M3.5 12L12 9.5"/>',
    "chip": '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/>'
    '<path d="M9.5 3v3.5M14.5 3v3.5M9.5 17.5V21M14.5 17.5V21M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21"/>',
    "objective": '<path d="M8 3h8v5l-1.5 1.6v6.4L12 18.5 9.5 16V9.6L8 8z"/><path d="M8 5.6h8"/>'
    '<path d="M12 18.5V21"/>',
    "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5"/><path d="M12 12v9"/>',
    "channels": '<circle cx="9" cy="9.5" r="5"/><circle cx="15" cy="9.5" r="5"/><circle cx="12" cy="14.8" r="5"/>',
    "code": '<path d="M9 7l-5 5 5 5M15 7l5 5-5 5"/>',
    "link": '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/>'
    '<path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
    "panel": '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M14.5 4.5v15"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>',
    "refresh": '<path d="M19.5 8.5A8 8 0 0 0 5 9"/><path d="M4.5 15.5A8 8 0 0 0 19 15"/>'
    '<path d="M19.5 4v4.5H15M4.5 20v-4.5H9"/>',
    "export": '<path d="M12 15V4"/><path d="M7.5 8.5L12 4l4.5 4.5"/><path d="M4.5 13.5v6h15v-6"/>',
    "roi_rect": '<rect x="4.5" y="6.5" width="15" height="11" rx="1"/>',
    "roi_ellipse": '<ellipse cx="12" cy="12" rx="8" ry="5.5"/>',
    "roi_polygon": '<path d="M5 9.5l6.5-5.5 7.5 4-2 10.5H7z"/>',
    "roi_line": '<path d="M5.5 18.5l13-13"/><circle cx="5.5" cy="18.5" r="1.7" fill="{c}"/>'
    '<circle cx="18.5" cy="5.5" r="1.7" fill="{c}"/>',
    "roi_point": '<circle cx="12" cy="12" r="2.3" fill="{c}"/><path d="M12 4v4M12 16v4M4 12h4M16 12h4"/>',
    "pointer": '<path d="M6 4l11.5 7.6-5.2 1.2 3 5.6-2.2 1.2-3-5.6L6 17z"/>',
    "plug": '<path d="M9 3v5M15 3v5"/><path d="M6.5 8h11v3a5.5 5.5 0 0 1-11 0z"/><path d="M12 16.5V21"/>',
}

_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<path d="M1.5 20.5L9.4 7l3.4 5.6 2.4-3.6 7.3 11.5z" fill="{c}"/>'
    '<path d="M9.4 7L7.2 10.8l1.4-.7 1 1.1 1.4-1.1z" fill="#ffffff" opacity="0.95"/>'
    "</svg>"
)


def svg(name: str, color: str) -> str:
    if name == "logo":
        return _LOGO.format(c=color)
    body = _PATHS[name].format(c=color)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE.format(c=color)}>{body}</svg>'
    )


@lru_cache(maxsize=512)
def pixmap(name: str, color: str, size: int = 20, ratio: float = 2.0) -> QPixmap:
    from qtpy.QtSvg import QSvgRenderer

    px = QPixmap(int(size * ratio), int(size * ratio))
    px.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg(name, color).encode()))
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size * ratio, size * ratio))
    painter.end()
    px.setDevicePixelRatio(ratio)
    return px


def icon(name: str, color: str | None = None, size: int = 20, *, checked_color: str | None = None) -> QIcon:
    """Icône colorée ; ``checked_color`` : couleur quand le bouton est enfoncé/sélectionné."""
    from .theme import palette

    p = palette()
    ic = QIcon()
    ic.addPixmap(pixmap(name, color or p.muted, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(pixmap(name, checked_color or color or p.accent, size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(pixmap(name, p.faint, size), QIcon.Mode.Disabled, QIcon.State.Off)
    return ic


NAMES = tuple(sorted([*_PATHS, "logo"]))
