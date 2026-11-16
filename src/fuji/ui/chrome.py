"""Cadre de la fenêtre : barre supérieure et navigation latérale."""

from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .components import ToggleSwitch, icon_button, label, nav_button, separator
from .theme import ThemeManager, palette

#: Sections de navigation : clé, libellé, icône, en bas de la barre ?
SECTIONS = (
    ("control", "Contrôle", "microscope", False),
    ("assistants", "Assistants", "target", False),
    ("acquisition", "Acquisition", "timer", False),
    ("analysis", "Analyse", "chart", False),
    ("layers", "Calques", "layers", False),
    ("config", "Microscope", "sliders", True),
    ("settings", "Réglages", "gear", True),
)


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
        for key, text, icon_name, bottom in SECTIONS:
            if bottom and not bottom_started:
                lay.addStretch(1)
                bottom_started = True
            b = nav_button(text, icon_name)
            b.setProperty("key", key)
            b.clicked.connect(lambda _=False, k=key: self._clicked(k))
            self._group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
        self._current: str | None = None
        ThemeManager.get().changed.connect(self._retheme)

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


class TopBar(QWidget):
    """Logo, choix du microscope, live, capture, état de l'acquisition, thème."""

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
        name = label("Fuji", "AppName", wrap=False)
        lay.addWidget(name)
        lay.addSpacing(8)
        lay.addWidget(separator(vertical=True))
        lay.addSpacing(4)

        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(260)
        self.profile_combo.setToolTip("Microscope utilisé")
        self.profile_combo.activated.connect(
            lambda i: self.profile_selected.emit(self.profile_combo.itemData(i))
        )
        lay.addWidget(self.profile_combo)
        manage = icon_button("edit", "Configurer les microscopes")
        manage.clicked.connect(self.manage_profiles.emit)
        lay.addWidget(manage)
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
        self.theme_btn = icon_button("moon", "Thème clair / sombre")
        self.theme_btn.clicked.connect(ThemeManager.get().toggle_mode)
        lay.addWidget(self.theme_btn)
        ThemeManager.get().changed.connect(self._retheme)
        self._retheme()

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
