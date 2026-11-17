"""Cadre de la fenêtre : barre supérieure et navigation latérale."""

from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .components import ToggleSwitch, icon_button, label, nav_button, separator
from .theme import ThemeManager, palette

#: Les deux espaces de travail : se servir du microscope, ou analyser les images prises.
#: Chacun a sa propre vue d'image et ses propres sections, pour ne pas mélanger les deux.
MODES = (("microscope", "Microscope", "microscope"), ("analyse", "Analyse", "chart"))

#: Sections de navigation : clé, libellé, icône, en bas de la barre ?, espaces où elle apparaît
SECTIONS = (
    ("control", "Contrôle", "microscope", False, ("microscope",)),
    ("assistants", "Assistants", "target", False, ("microscope",)),
    ("acquisition", "Acquisition", "timer", False, ("microscope",)),
    ("images", "Images", "gallery", False, ("microscope", "analyse")),
    ("analysis", "Outils", "ruler", False, ("analyse",)),
    ("layers", "Calques", "layers", False, ("analyse",)),
    ("config", "Microscope", "sliders", True, ("microscope",)),
    ("settings", "Réglages", "gear", True, ("microscope", "analyse")),
)


def sections_of(mode: str) -> list[str]:
    return [key for key, *_rest, modes in SECTIONS if mode in modes]


def mode_of(key: str, current: str) -> str:
    """Espace où afficher une section : l'espace actuel s'il la contient, sinon le sien."""
    modes = next(m for k, *_rest, m in SECTIONS if k == key)
    return current if current in modes else modes[0]


class Sidebar(QWidget):
    """Barre de navigation à icônes. Recliquer la section active replie le panneau."""

    selected = Signal(str)
    collapse_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(80)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 10, 6, 10)
        lay.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        bottom_started = False
        for key, text, icon_name, bottom, modes in SECTIONS:
            if bottom and not bottom_started:
                lay.addStretch(1)
                bottom_started = True
            b = nav_button(text, icon_name)
            b.setProperty("key", key)
            b.setProperty("modes", list(modes))
            b.clicked.connect(lambda _=False, k=key: self._clicked(k))
            self._group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
        self._current: str | None = None
        ThemeManager.get().changed.connect(self._retheme)

    def set_mode(self, mode: str) -> None:
        """N'affiche que les sections de l'espace ``mode``."""
        for b in self.buttons.values():
            b.setVisible(mode in b.property("modes"))

    def set_badge(self, key: str, text: str) -> None:
        """Petit compteur sous le nom d'une section (ex. nombre d'images prises)."""
        b = self.buttons[key]
        base = next(title for k, title, *_ in SECTIONS if k == key)
        b.setText(f"{base} {text}".strip())

    def _clicked(self, key: str) -> None:
        if key == self._current:
            self.collapse_requested.emit()
            return
        self.select(key)

    def select(self, key: str) -> None:
        self._current = key
        self.buttons[key].setChecked(True)
        self.selected.emit(key)

    def current(self) -> str | None:
        return self._current

    def _retheme(self, *_: object) -> None:
        p = palette()
        for b in self.buttons.values():
            b.setIcon(icons.icon(b.property("icon_name"), p.muted, 22, checked_color=p.accent))


class ModeSwitch(QFrame):
    """« Microscope | Analyse » : bascule entre les deux espaces de travail."""

    changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ModeSwitch")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for key, text, icon_name in MODES:
            b = QPushButton(text)
            b.setObjectName("mode")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setProperty("icon_name", icon_name)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            self._group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
        tips = {
            "microscope": "Régler le microscope, voir le live, prendre des images (Ctrl+1)",
            "analyse": "Ouvrir les images prises, mesurer, analyser (Ctrl+2)",
        }
        for key, tip in tips.items():
            self.buttons[key].setToolTip(tip)
        self._compact = False
        ThemeManager.get().changed.connect(self._retheme)
        self._retheme()

    def set_mode(self, mode: str) -> None:
        self.buttons[mode].setChecked(True)
        self._retheme()

    def set_compact(self, compact: bool) -> None:
        """Sans icônes : moins large, pour les petits écrans."""
        if compact != self._compact:
            self._compact = compact
            self._retheme()

    def _retheme(self, *_: object) -> None:
        p = palette()
        for b in self.buttons.values():
            if self._compact:
                b.setIcon(QIcon())
                continue
            color = p.accent_text if b.isChecked() else p.muted
            b.setIcon(icons.icon(b.property("icon_name"), color, 16))
            b.setIconSize(QSize(16, 16))


class TopBar(QWidget):
    """Logo, choix du microscope, live, capture, état de l'acquisition, thème."""

    mode_changed = Signal(str)
    profile_selected = Signal(str)
    live_toggled = Signal(bool)
    capture_clicked = Signal()
    expert_toggled = Signal(bool)
    manage_profiles = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(56)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(10)

        self.logo = QLabel()
        lay.addWidget(self.logo)
        self.name = label("Fuji", "AppName", wrap=False)
        lay.addWidget(self.name)
        lay.addSpacing(4)
        self.mode = ModeSwitch()
        self.mode.changed.connect(self.mode_changed.emit)
        lay.addWidget(self.mode)
        lay.addSpacing(4)

        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(130)
        self.profile_combo.setMaximumWidth(340)
        self.profile_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.profile_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.profile_combo.setMinimumContentsLength(12)
        self.profile_combo.setToolTip("Microscope utilisé")
        self.profile_combo.activated.connect(
            lambda i: self.profile_selected.emit(self.profile_combo.itemData(i))
        )
        lay.addWidget(self.profile_combo)
        manage = icon_button("edit", "Configurer les microscopes")
        manage.setFixedSize(34, 34)
        manage.clicked.connect(self.manage_profiles.emit)
        lay.addWidget(manage)
        self.manage = manage
        self.status = label("", "badge", wrap=False)
        lay.addWidget(self.status)
        lay.addStretch(1)

        self.acq_label = label("", "hint", wrap=False)
        self.acq_progress = QProgressBar()
        self.acq_progress.setFixedWidth(140)
        for w in (self.acq_label, self.acq_progress):
            w.setVisible(False)
            lay.addWidget(w)

        self.live = QPushButton("Live")
        self.live.setObjectName("live")
        self.live.setCheckable(True)
        self.live.setToolTip("Démarrer / arrêter le live (Ctrl+L)")
        self.live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.live.toggled.connect(self.live_toggled.emit)
        self.capture = QPushButton("Capturer")
        self.capture.setObjectName("capture")
        self.capture.setToolTip("Capturer une image avec ses métadonnées (Ctrl+K)")
        self.capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.capture.clicked.connect(self.capture_clicked.emit)
        lay.addWidget(self.live)
        lay.addWidget(self.capture)
        lay.addSpacing(6)
        lay.addWidget(separator(vertical=True))

        self.expert = ToggleSwitch("Expert")
        self.expert.setToolTip("Affiche tous les réglages avancés")
        self.expert.toggled.connect(self.expert_toggled.emit)
        lay.addWidget(self.expert)
        self._separator = lay.itemAt(lay.count() - 2).widget()
        self.theme_btn = icon_button("moon", "Thème clair / sombre")
        self.theme_btn.setFixedSize(34, 34)
        self.theme_btn.clicked.connect(ThemeManager.get().toggle_mode)
        lay.addWidget(self.theme_btn)
        ThemeManager.get().changed.connect(self._retheme)
        self._retheme()
        # sur un petit écran, ces éléments disparaissent plutôt que d'élargir la fenêtre
        # (le mode expert reste dans Réglages, le type de microscope dans Contrôle)
        self._optional = (self.name, self.status, self.expert, self._separator)
        for w in self._optional:
            w.setMinimumWidth(1)
        self._wanted: dict[QWidget, bool] = {w: True for w in self._optional}
        self._compact = False
        self._full_width = 0
        self._mode = "microscope"

    def set_mode(self, mode: str) -> None:
        """Le choix du microscope, le live et la capture n'ont de sens que dans l'espace
        Microscope."""
        self._mode = mode
        self.mode.set_mode(mode)
        scope = mode == "microscope"
        for w in (self.profile_combo, self.manage, self.live, self.capture):
            w.setVisible(scope)
        self._wanted[self.status] = scope
        self._fit()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if not self._compact:
            for w in self._optional:
                w.setVisible(self._wanted[w])
            self._full_width = self.layout().sizeHint().width()
        compact = self.width() < self._full_width
        self._compact = compact
        for w in self._optional:
            w.setVisible(self._wanted[w] and not compact)
        self.mode.set_compact(compact)

    def _retheme(self, *_: object) -> None:
        p = palette()
        self.logo.setPixmap(icons.pixmap("logo", p.accent, 26))
        self.theme_btn.setIcon(icons.icon("sun" if p.mode == "sombre" else "moon", p.muted, 18))
        self.live.setIcon(icons.icon("play", p.text, 14, checked_color="#04130a"))
        self.live.setIconSize(QSize(14, 14))
        self.capture.setIcon(icons.icon("camera", p.accent_text, 16, checked_color=p.accent_text))
        self.capture.setIconSize(QSize(16, 16))

    def set_profiles(self, profiles: dict[str, str], current: str | None) -> None:
        """``profiles`` : clé → titre affiché."""
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for key, title in profiles.items():
            self.profile_combo.addItem(icons.icon("microscope", palette().muted, 16), title, key)
        if current is not None:
            idx = self.profile_combo.findData(current)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def set_live(self, on: bool) -> None:
        self.live.blockSignals(True)
        self.live.setChecked(on)
        self.live.setText("Live ●" if on else "Live")
        self.live.blockSignals(False)

    def set_acquisition(self, done: int, total: int, running: bool) -> None:
        self.acq_label.setVisible(running)
        self.acq_progress.setVisible(running)
        if running:
            self.acq_label.setText("Acquisition")
            self.acq_progress.setRange(0, max(total, 1))
            self.acq_progress.setValue(done)
            self.acq_progress.setFormat(f"{done}/{total}")
