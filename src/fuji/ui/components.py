"""Composants d'interface réutilisables (cartes, interrupteurs, boutons segmentés...)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from qtpy.QtCore import Property, QPropertyAnimation, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .theme import ThemeManager, palette


# ------------------------------------------------------------------ textes
def label(text: str = "", object_name: str | None = None, *, wrap: bool = True) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(wrap)
    if object_name:
        lbl.setObjectName(object_name)
    return lbl


def hint(text: str = "") -> QLabel:
    return label(text, "hint")


def badge(text: str, tone: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName({"accent": "badgeAccent", "warning": "badgeWarning"}.get(tone, "badge"))
    lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lbl


def separator(vertical: bool = False) -> QFrame:
    f = QFrame()
    f.setObjectName("VSeparator" if vertical else "Separator")
    return f


def section_title(text: str) -> QLabel:
    return label(text.upper(), "SectionTitle", wrap=False)


# ------------------------------------------------------------------ cartes
class Card(QFrame):
    """Carte arrondie avec titre facultatif et icône."""

    def __init__(
        self,
        title: str | None = None,
        subtitle: str | None = None,
        icon_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 12)
        self.body.setSpacing(8)
        if title:
            head = QHBoxLayout()
            head.setSpacing(8)
            if icon_name:
                ic = QLabel()
                set_icon(ic, icon_name, "accent", 18)
                head.addWidget(ic)
            t = QLabel(title)
            t.setObjectName("CardTitle")
            head.addWidget(t)
            head.addStretch(1)
            self.header = head
            self.body.addLayout(head)
            if subtitle:
                self.body.addWidget(hint(subtitle))

    def add(self, widget: QWidget | None = None, *, layout: Any = None, stretch: int = 0) -> None:
        if widget is not None:
            self.body.addWidget(widget, stretch)
        if layout is not None:
            self.body.addLayout(layout)


class PanelHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(2)
        self.title = label(title, "PanelTitle", wrap=False)
        self.subtitle = label(subtitle, "PanelSubtitle")
        lay.addWidget(self.title)
        lay.addWidget(self.subtitle)
        self.subtitle.setVisible(bool(subtitle))

    def set_text(self, title: str, subtitle: str = "") -> None:
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self.subtitle.setVisible(bool(subtitle))


def hbox(*widgets: QWidget | None, spacing: int = 8, stretch: bool = True, margins: int = 0) -> QWidget:
    w = QWidget()
    w.setObjectName("Row")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    for child in widgets:
        if child is None:
            lay.addStretch(1)
        else:
            lay.addWidget(child)
    if stretch and None not in widgets:
        lay.addStretch(1)
    return w


# ------------------------------------------------------------------ icônes et thème
def _role_color(role: str) -> str:
    p = palette()
    return {
        "text": p.text,
        "muted": p.muted,
        "accent": p.accent,
        "accent_text": p.accent_text,
        "faint": p.faint,
    }.get(role, p.text)


def set_icon(widget: Any, name: str, role: str = "text", size: int = 18) -> None:
    """Pose une icône qui sera redessinée automatiquement au changement de thème."""
    widget.setProperty("fuji_icon", name)
    widget.setProperty("fuji_icon_role", role)
    widget.setProperty("fuji_icon_size", size)
    color = _role_color(role)
    if isinstance(widget, QLabel):
        widget.setPixmap(icons.pixmap(name, color, size))
    else:
        widget.setIcon(icons.icon(name, color, size, checked_color=color))
        widget.setIconSize(QSize(size, size))


def retheme(root: QWidget) -> None:
    """Redessine toutes les icônes posées avec :func:`set_icon` sous ``root``."""
    for w in [root, *root.findChildren(QWidget)]:
        name = w.property("fuji_icon")
        if name:
            set_icon(w, name, w.property("fuji_icon_role") or "text", int(w.property("fuji_icon_size") or 18))


# ------------------------------------------------------------------ boutons
def icon_button(
    icon_name: str,
    tooltip: str = "",
    *,
    text: str = "",
    object_name: str = "ghost",
    size: int = 18,
    checkable: bool = False,
) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(object_name)
    set_icon(b, icon_name, "text" if text or object_name != "ghost" else "muted", size)
    b.setToolTip(tooltip)
    b.setCheckable(checkable)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def button(
    text: str,
    icon_name: str | None = None,
    *,
    primary: bool = False,
    object_name: str | None = None,
    tooltip: str = "",
) -> QPushButton:
    b = QPushButton(text)
    if primary:
        b.setObjectName("primary")
    elif object_name:
        b.setObjectName(object_name)
    if icon_name:
        set_icon(b, icon_name, "accent_text" if primary else "text", 16)
    b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class Segmented(QWidget):
    """Boutons « pilule » exclusifs (objectifs, canaux...)."""

    selected = Signal(str)

    def __init__(self, options: Sequence[str] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonClicked.connect(lambda b: self.selected.emit(b.property("value")))
        self.set_options(options)

    def set_options(self, options: Iterable[str], tooltips: dict[str, str] | None = None) -> None:
        for b in self._group.buttons():
            self._group.removeButton(b)
            b.deleteLater()
        while self._lay.count():
            self._lay.takeAt(0)
        for opt in options:
            b = QPushButton(opt)
            b.setObjectName("chip")
            b.setCheckable(True)
            b.setProperty("value", opt)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if tooltips and opt in tooltips:
                b.setToolTip(tooltips[opt])
            self._group.addButton(b)
            self._lay.addWidget(b)
        self._lay.addStretch(1)

    def set_value(self, value: str | None) -> None:
        for b in self._group.buttons():
            b.blockSignals(True)
            b.setChecked(b.property("value") == value)
            b.blockSignals(False)

    def value(self) -> str | None:
        b = self._group.checkedButton()
        return b.property("value") if b else None

    def options(self) -> list[str]:
        return [b.property("value") for b in self._group.buttons()]


class ChoiceCombo(QComboBox):
    """Même usage que :class:`Segmented`, sous forme de liste déroulante : pour les longues
    listes (beaucoup de canaux), qui prendraient trop de place en boutons."""

    selected = Signal(str)

    def __init__(self, options: Sequence[str] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.activated.connect(lambda i: self.selected.emit(self.itemText(i)))
        self.set_options(options)

    def set_options(self, options: Iterable[str], tooltips: dict[str, str] | None = None) -> None:
        self.blockSignals(True)
        self.clear()
        for opt in options:
            self.addItem(opt)
            if tooltips and opt in tooltips:
                self.setItemData(self.count() - 1, tooltips[opt], Qt.ItemDataRole.ToolTipRole)
        self.blockSignals(False)

    def set_value(self, value: str | None) -> None:
        self.blockSignals(True)
        self.setCurrentIndex(self.findText(value) if value is not None else -1)
        self.blockSignals(False)

    def value(self) -> str | None:
        return self.currentText() or None

    def options(self) -> list[str]:
        return [self.itemText(i) for i in range(self.count())]


def choice(options: Sequence[str], max_buttons: int = 5) -> Segmented | ChoiceCombo:
    """Boutons pour un petit nombre d'options, liste déroulante au-delà."""
    return Segmented(options) if len(options) <= max_buttons else ChoiceCombo(options)


class ToggleSwitch(QAbstractButton):
    """Interrupteur à glissière (style macOS)."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self.toggled.connect(self._animate)
        ThemeManager.get().changed.connect(self._on_theme)

    def _on_theme(self, *_: object) -> None:
        self.update()

    def _get_knob(self) -> float:
        return self._pos

    def _set_knob(self, v: float) -> None:
        self._pos = v
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked: bool) -> None:
        super().setChecked(checked)
        self._pos = 1.0 if checked else 0.0
        self.update()

    def sizeHint(self) -> QSize:
        w = 36 + (self.fontMetrics().horizontalAdvance(self.text()) + 10 if self.text() else 0)
        return QSize(w, 22)

    def paintEvent(self, _event: Any) -> None:
        p = palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(0, 3, 34, 18)
        off, on = QColor(p.border_strong), QColor(p.accent)
        t = self._pos
        color = QColor(
            round(off.red() + (on.red() - off.red()) * t),
            round(off.green() + (on.green() - off.green()) * t),
            round(off.blue() + (on.blue() - off.blue()) * t),
        )
        if not self.isEnabled():
            color = QColor(p.surface3)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(track, 9, 9)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QRectF(2 + t * 16, 5, 14, 14))
        if self.text():
            painter.setPen(QColor(p.text if self.isEnabled() else p.faint))
            painter.drawText(
                QRectF(44, 0, self.width() - 44, 24),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                self.text(),
            )
        painter.end()


# ------------------------------------------------------------------ champs composés
class PathEdit(QWidget):
    """Champ de chemin + bouton « Parcourir »."""

    changed = Signal(str)

    def __init__(self, mode: str = "file", file_filter: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode, self.file_filter = mode, file_filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("dossier…" if mode == "dir" else "fichier…")
        self.edit.editingFinished.connect(lambda: self.changed.emit(self.edit.text()))
        browse = icon_button("folder", "Parcourir…", object_name="")
        browse.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(browse)

    def _browse(self) -> None:
        start = self.edit.text() or ""
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Choisir un dossier", start)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Choisir un fichier", start, self.file_filter or "Tous les fichiers (*)"
            )
        if path:
            self.edit.setText(path)
            self.changed.emit(path)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, text: str) -> None:
        self.edit.setText(text)


class ListEditor(QWidget):
    """Liste de textes modifiable (positions d'une tourelle, fluorophores...)."""

    changed = Signal(list)

    def __init__(
        self, items: Sequence[str] = (), placeholder: str = "nouvelle position", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        self.list.itemChanged.connect(lambda *_: self.changed.emit(self.items()))
        lay.addWidget(self.list)
        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.input.returnPressed.connect(self._add)
        add = icon_button("plus", "Ajouter", object_name="")
        add.clicked.connect(self._add)
        rm = icon_button("trash", "Supprimer la ligne sélectionnée", object_name="")
        rm.clicked.connect(self._remove)
        up = icon_button("up", "Monter", object_name="")
        up.clicked.connect(lambda: self._move(-1))
        down = icon_button("down", "Descendre", object_name="")
        down.clicked.connect(lambda: self._move(1))
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.input, 1)
        for b in (add, rm, up, down):
            row.addWidget(b)
        lay.addLayout(row)
        self.set_items(items)

    def set_items(self, items: Sequence[str]) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for it in items:
            item = QListWidgetItem(str(it))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem(item)
        self.list.blockSignals(False)

    def items(self) -> list[str]:
        return [
            self.list.item(i).text().strip()
            for i in range(self.list.count())
            if self.list.item(i).text().strip()
        ]

    def _add(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(item)
        self.input.clear()
        self.changed.emit(self.items())

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            self.list.takeItem(row)
            self.changed.emit(self.items())

    def _move(self, delta: int) -> None:
        row = self.list.currentRow()
        new = row + delta
        if row < 0 or not 0 <= new < self.list.count():
            return
        item = self.list.takeItem(row)
        self.list.insertItem(new, item)
        self.list.setCurrentRow(new)
        self.changed.emit(self.items())


#: Couleurs d'affichage proposées pour les canaux (tables de couleurs napari).
CHANNEL_COLORS = {
    "gray": "#d0d4da",
    "blue": "#3b6cff",
    "cyan": "#22d3ee",
    "green": "#22c55e",
    "yellow": "#facc15",
    "magenta": "#e040fb",
    "red": "#ef4444",
}


class ColorChooser(QWidget):
    """Pastilles de couleur exclusives."""

    changed = Signal(str)

    def __init__(self, value: str = "gray", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._group = QButtonGroup(self)
        for name, hexcolor in CHANNEL_COLORS.items():
            b = QPushButton()
            b.setObjectName("swatch")
            b.setCheckable(True)
            b.setToolTip(name)
            b.setProperty("value", name)
            b.setStyleSheet(
                f"QPushButton#swatch {{ background: {hexcolor}; border: 2px solid transparent; }}"
                f"QPushButton#swatch:checked {{ border: 2px solid {palette().text}; }}"
            )
            self._group.addButton(b)
            lay.addWidget(b)
        lay.addStretch(1)
        self._group.buttonClicked.connect(lambda b: self.changed.emit(b.property("value")))
        self.set_value(value)

    def set_value(self, value: str) -> None:
        for b in self._group.buttons():
            b.setChecked(b.property("value") == value)

    def value(self) -> str:
        b = self._group.checkedButton()
        return b.property("value") if b else "gray"


class EmptyState(QWidget):
    def __init__(self, icon_name: str, title: str, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic = QLabel()
        set_icon(ic, icon_name, "faint", 42)
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ic)
        t = label(title, "CardTitle")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        if text:
            h = hint(text)
            h.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(h)


def nav_button(text: str, icon_name: str) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setIcon(icons.icon(icon_name, palette().muted, 22, checked_color=palette().accent))
    b.setIconSize(QSize(22, 22))
    b.setCheckable(True)
    b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    b.setFixedSize(68, 58)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setProperty("icon_name", icon_name)
    return b
