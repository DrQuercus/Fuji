"""« Nouveau microscope » : choisir un point de départ, puis tout ajuster dans l'éditeur."""

from __future__ import annotations

import copy
from typing import Any

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import QDialog, QGridLayout, QPushButton, QVBoxLayout, QWidget

from fuji.core.catalog import TEMPLATES
from fuji.core.profile import MicroscopeProfile, builtin_profiles

from .. import icons
from ..components import PanelHeader, button
from ..theme import palette

_TEMPLATE_ICONS = {
    "manuel-touptek": "microscope",
    "manuel-fluo": "cube",
    "micromanager": "chip",
    "demo": "sparkle",
    "vide": "plus",
    "copie": "copy",
}


def template_data(key: str, current: dict[str, Any] | None = None) -> dict[str, Any]:
    if key == "copie" and current is not None:
        data = copy.deepcopy(current)
        data["name"] = f"{data.get('name', 'Microscope')} (copie)"
        return data
    tpl = TEMPLATES[key]
    if tpl["profile"] is None:
        data = MicroscopeProfile.load(builtin_profiles()["demo"]).model_dump(mode="json")
        data["name"] = "Démo (copie)"
        return data
    return MicroscopeProfile.model_validate(copy.deepcopy(tpl["profile"])).model_dump(mode="json")


class NewMicroscopeDialog(QDialog):
    def __init__(self, current: dict[str, Any] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nouveau microscope")
        self.resize(820, 520)
        self.current = current
        self.choice: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.addWidget(
            PanelHeader("Nouveau microscope", "Choisissez un point de départ : tout restera modifiable.")
        )
        grid = QGridLayout()
        grid.setSpacing(12)
        entries = [(k, t["title"], t["description"]) for k, t in TEMPLATES.items()]
        if current is not None:
            entries.append(
                ("copie", "Copier le microscope actuel", f"Repartir de « {current.get('name', '')} ».")
            )
        for i, (key, title, desc) in enumerate(entries):
            b = QPushButton(f"{title}\n{desc}")
            b.setObjectName("templateCard")
            b.setIcon(icons.icon(_TEMPLATE_ICONS.get(key, "microscope"), palette().accent, 28))
            b.setIconSize(QSize(28, 28))
            b.setMinimumHeight(86)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                "QPushButton#templateCard { text-align: left; padding: 12px 14px; border-radius: 12px; }"
            )
            b.clicked.connect(lambda _=False, k=key: self._choose(k))
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)
        lay.addStretch(1)
        cancel = button("Annuler")
        cancel.clicked.connect(self.reject)
        lay.addWidget(cancel, 0, Qt.AlignmentFlag.AlignRight)

    def _choose(self, key: str) -> None:
        self.choice = key
        self.accept()

    def data(self) -> dict[str, Any] | None:
        return template_data(self.choice, self.current) if self.choice else None
