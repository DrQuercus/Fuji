"""Fenêtre d'édition complète d'un microscope (profil)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji.core.profile import MicroscopeProfile

from .. import icons
from ..components import PanelHeader, button, label
from ..theme import palette
from . import model as M
from .pages import PAGES, Page


class ConfigEditor(QDialog):
    """Édite tout un profil : statif, périphériques, rôles, objectifs, cubes, canaux."""

    saved = Signal(object)  # Path

    def __init__(self, data: dict[str, Any], path: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configuration du microscope")
        self.resize(1180, 780)
        self.setMinimumSize(900, 600)
        self.data = M.normalize(copy.deepcopy(data))
        self.path = path
        self._current = 0

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        nav_host = QWidget()
        nav_host.setObjectName("Sidebar")
        nav_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        nav_host.setFixedWidth(230)
        nav_lay = QVBoxLayout(nav_host)
        nav_lay.setContentsMargins(12, 16, 12, 12)
        self.title_label = label("", "PanelTitle")
        nav_lay.addWidget(self.title_label)
        self.path_label = label("", "faint")
        nav_lay.addWidget(self.path_label)
        nav_lay.addSpacing(10)
        self.nav = QListWidget()
        self.nav.setIconSize(QSize(18, 18))
        self.nav.setStyleSheet(
            "QListWidget { background: transparent; border: none; }QListWidget::item { padding: 8px 8px; }"
        )
        nav_lay.addWidget(self.nav, 1)
        root.addWidget(nav_host)

        main = QVBoxLayout()
        main.setContentsMargins(24, 18, 24, 14)
        main.setSpacing(10)
        self.header = PanelHeader("", "")
        main.addWidget(self.header)
        self.stack = QStackedWidget()
        main.addWidget(self.stack, 1)
        self.pages: list[Page] = []
        for cls in PAGES:
            page = cls(self)
            page.changed.connect(self._update_issues)
            self.pages.append(page)
            if cls.__name__ in ("DevicesPage", "ChannelsPage", "ObjectivesPage", "FiltersPage", "YamlPage"):
                self.stack.addWidget(page)
            else:
                area = QScrollArea()
                area.setWidgetResizable(True)
                area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                area.setWidget(page)
                self.stack.addWidget(area)
            item = QListWidgetItem(
                icons.icon(cls.icon, palette().muted, 18, checked_color=palette().accent), cls.title
            )
            self.nav.addItem(item)

        bottom = QHBoxLayout()
        self.issues = label("", "warning")
        self.issues.setCursor(Qt.CursorShape.PointingHandCursor)
        self.issues.mousePressEvent = lambda _e: self.show_issues()  # type: ignore[assignment]
        bottom.addWidget(self.issues, 1)
        check = button("Vérifier", "check")
        check.clicked.connect(self.show_issues)
        test = button("Tester la connexion", "plug")
        test.clicked.connect(self.test_connection)
        cancel = button("Annuler")
        cancel.clicked.connect(self.reject)
        self.save_btn = button("Enregistrer", "save", primary=True)
        self.save_btn.clicked.connect(self.save)
        for b in (check, test, cancel, self.save_btn):
            bottom.addWidget(b)
        main.addLayout(bottom)
        root.addLayout(main, 1)

        self.nav.currentRowChanged.connect(self._go)
        self.pages[0].load(self.data)
        self.nav.setCurrentRow(0)
        self._go(0)

    # ------------------------------------------------------------ navigation
    def _go(self, index: int) -> None:
        if index < 0:
            return
        if index != self._current:
            try:
                self.pages[self._current].store(self.data)
            except Exception as exc:
                QMessageBox.warning(self, "Saisie invalide", str(exc))
                self.nav.blockSignals(True)
                self.nav.setCurrentRow(self._current)
                self.nav.blockSignals(False)
                return
        self._current = index
        page = self.pages[index]
        page.load(self.data)
        self.stack.setCurrentIndex(index)
        self.header.title.setText(page.title)
        self.header.subtitle.setText(page.subtitle)
        self._update_issues()

    def go_to(self, title: str) -> None:
        for i, page in enumerate(self.pages):
            if page.title == title:
                self.nav.setCurrentRow(i)

    def collect(self) -> MicroscopeProfile:
        """Enregistre la page affichée et valide le profil complet."""
        self.pages[self._current].store(self.data)
        return MicroscopeProfile.model_validate(self.data)

    # ------------------------------------------------------------ vérifications
    def _update_issues(self) -> None:
        self.title_label.setText(self.data.get("name") or "Microscope")
        self.path_label.setText(str(self.path.name) if self.path else "nouveau profil")
        try:
            issues = MicroscopeProfile.model_validate(self.data).check()
        except Exception as exc:
            self.issues.setText(f"⚠ Profil invalide : {str(exc).splitlines()[0]}")
            return
        self.issues.setText(f"⚠ {len(issues)} point(s) à vérifier — cliquer pour le détail" if issues else "")

    def show_issues(self) -> None:
        try:
            profile = self.collect()
        except Exception as exc:
            QMessageBox.warning(self, "Profil invalide", str(exc))
            return
        issues = profile.check()
        if issues:
            QMessageBox.warning(self, "À vérifier", "\n".join(f"• {i}" for i in issues))
        else:
            QMessageBox.information(self, "Vérification", "Aucun problème détecté.")
        self._update_issues()

    def test_connection(self) -> None:
        try:
            profile = self.collect()
        except Exception as exc:
            QMessageBox.warning(self, "Profil invalide", str(exc))
            return

        def job() -> list[str]:
            from fuji.core.microscope import Microscope

            lines = []
            scope = Microscope(profile, manual_handler=lambda _a: True)
            scope.connect()
            try:
                for dev in scope.devices.values():
                    lines.append(f"✔ {dev.label} — {dev.kind.value}" + (" (manuel)" if dev.manual else ""))
                caps = scope.capabilities()
                if scope.camera is not None:
                    frame = scope.snap()
                    lines.append(
                        f"✔ Capture : {frame.data.shape} {frame.data.dtype}, pixel "
                        f"{frame.metadata.pixel_size_um or '?'} µm"
                    )
                lines.append("")
                lines.append("Capacités : " + ", ".join(k for k, v in caps.items() if v))
            finally:
                scope.disconnect()
            return lines

        self.setCursor(Qt.CursorShape.WaitCursor)
        worker = create_worker(job, _start_thread=False)
        worker.returned.connect(
            lambda lines: QMessageBox.information(self, "Connexion réussie", "\n".join(lines))
        )
        worker.errored.connect(lambda exc: QMessageBox.warning(self, "Connexion impossible", f"{exc}"))
        worker.finished.connect(lambda: self.unsetCursor())
        worker.start()

    # ------------------------------------------------------------ enregistrement
    def save(self) -> None:
        try:
            profile = self.collect()
        except Exception as exc:
            QMessageBox.warning(self, "Profil invalide", str(exc))
            return
        target = M.save_path_for(self.data, self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        profile.save(target)
        self.path = target
        self.saved.emit(target)
        self.accept()
