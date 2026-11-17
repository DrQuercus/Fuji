"""Fabrique les icônes de l'application à partir du logo de Fuji.

    python packaging/make_icons.py

Écrit ``packaging/icons/fuji.icns`` (macOS), ``fuji.ico`` (Windows) et ``fuji.png``. Les
fichiers produits sont versionnés : inutile de relancer ce script, sauf si le logo change.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from qtpy.QtCore import QBuffer, QByteArray, QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath
from qtpy.QtSvg import QSvgRenderer

SIZE = 1024
#: gabarit des icônes macOS : un carré arrondi de 824 px centré dans 1024 px
MARGIN, RADIUS = 100, 185

MOUNTAIN = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    "<defs>"
    '<linearGradient id="m" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#7db4ff"/><stop offset="1" stop-color="#2f6fe4"/></linearGradient>'
    "</defs>"
    '<path d="M1.5 20.5L9.4 7l3.4 5.6 2.4-3.6 7.3 11.5z" fill="url(#m)"/>'
    '<path d="M9.4 7L7.2 10.8l1.4-.7 1 1.1 1.4-1.1z" fill="#ffffff"/>'
    "</svg>"
)


def render() -> Image.Image:
    _app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # Qt en a besoin pour dessiner
    image = QImage(SIZE, SIZE, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    square = QRectF(MARGIN, MARGIN, SIZE - 2 * MARGIN, SIZE - 2 * MARGIN)
    path = QPainterPath()
    path.addRoundedRect(square, RADIUS, RADIUS)
    background = QLinearGradient(QPointF(0, MARGIN), QPointF(0, SIZE - MARGIN))
    background.setColorAt(0.0, QColor("#1b2740"))
    background.setColorAt(1.0, QColor("#0b111e"))
    painter.fillPath(path, background)
    painter.setClipPath(path)
    # la montagne, un peu plus bas que le centre (son poids visuel est en bas)
    side = 0.80 * square.width()
    box = QRectF(SIZE / 2 - side / 2, SIZE / 2 - side / 2 - 0.04 * side, side, side)
    QSvgRenderer(QByteArray(MOUNTAIN.encode())).render(painter, box)
    painter.end()
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    data = bytes(buffer.data())
    buffer.close()
    return Image.open(io.BytesIO(data)).convert("RGBA")


def main() -> None:
    out = Path(__file__).parent / "icons"
    out.mkdir(exist_ok=True)
    icon = render()
    icon.save(out / "fuji.png")
    icon.save(out / "fuji.icns", sizes=[(s, s) for s in (16, 32, 64, 128, 256, 512, 1024)])
    icon.save(out / "fuji.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    for path in sorted(out.iterdir()):
        print(f"{path.name:12s} {path.stat().st_size // 1024} Ko")


if __name__ == "__main__":
    main()
