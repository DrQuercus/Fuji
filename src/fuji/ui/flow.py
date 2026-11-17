"""Disposition « en lignes qui se replient » : les éléments se rangent de gauche à droite
et passent à la ligne quand la place manque (comme des mots dans un paragraphe).

Utilisée par la barre rapide sous l'image : sur un écran large tout tient sur une ligne,
sur un portable elle passe sur deux, au lieu d'imposer une fenêtre trop large.
"""

from __future__ import annotations

from qtpy.QtCore import QPoint, QRect, QSize, Qt
from qtpy.QtWidgets import QLayout, QLayoutItem, QWidget, QWidgetItem


class FlowLayout(QLayout):  # les noms de méthodes sont imposés par Qt
    """Les éléments dont le widget porte la propriété ``flow_stretch`` (> 0) se partagent la
    place restante sur leur ligne, comme un ``stretch`` dans une ligne ordinaire."""

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 12, line_spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h, self._v = spacing, line_spacing

    # ------------------------------------------------------------ interface QLayout
    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def addWidget(self, widget: QWidget, stretch: int = 0) -> None:
        if stretch:
            widget.setProperty("flow_stretch", stretch)
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        """Tout sur une ligne, si la place le permet."""
        items = self._visible()
        width = sum(i.sizeHint().width() for i in items) + self._h * max(0, len(items) - 1)
        height = max((i.sizeHint().height() for i in items), default=0)
        m = self.contentsMargins()
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    def minimumSize(self) -> QSize:
        """L'élément le plus large : en dessous, plus rien ne peut se replier."""
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    # ------------------------------------------------------------ calcul
    def _visible(self) -> list[QLayoutItem]:
        return [i for i in self._items if i.widget() is None or not i.widget().isHidden()]

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        lines: list[list[QLayoutItem]] = [[]]
        used = 0
        for item in self._visible():
            w = item.sizeHint().width()
            if lines[-1] and used + self._h + w > area.width():
                lines.append([])
                used = 0
            used += (self._h if lines[-1] else 0) + w
            lines[-1].append(item)
        y = area.y()
        for line in lines:
            if not line:
                continue
            height = max(i.sizeHint().height() for i in line)
            if apply:
                natural = sum(i.sizeHint().width() for i in line) + self._h * (len(line) - 1)
                spare = max(0, area.width() - natural)
                weights = [_stretch(i) for i in line]
                total = sum(weights)
                x = area.x()
                for item, weight in zip(line, weights, strict=True):
                    extra = spare * weight // total if total else 0
                    w = min(item.sizeHint().width() + extra, item.maximumSize().width())
                    item.setGeometry(QRect(QPoint(x, y), QSize(w, height)))
                    x += w + self._h
            y += height + self._v
        return y - self._v - rect.y() + m.bottom() if lines[0] else m.top() + m.bottom()


def _stretch(item: QLayoutItem) -> int:
    widget = item.widget()
    return int(widget.property("flow_stretch") or 0) if widget is not None else 0
