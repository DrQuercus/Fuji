"""Panneau « Microscope » : le microscope actuel et la gestion des profils."""

from __future__ import annotations

import shutil
from pathlib import Path

from qtpy.QtCore import QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from fuji.core.microscope import Microscope
from fuji.core.profile import (
    MicroscopeProfile,
    available_profiles,
    is_builtin_profile,
    user_profile_dir,
)

from .. import icons
from ..components import Card, badge, button, hint, label
from ..theme import palette


class ConfigPanel(QWidget):
    edit_requested = Signal(object)  # chemin du profil à modifier (None = actuel)
    new_requested = Signal()
    switch_requested = Signal(str)  # chemin du profil

    def __init__(self, microscope: Microscope, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.m = microscope
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        prof = microscope.profile
        card = Card(prof.name, icon_name="microscope")
        if microscope.profile_path is not None:
            card.header.addWidget(
                badge("fourni" if is_builtin_profile(microscope.profile_path) else "personnel", "accent")
            )
        if prof.description:
            card.add(hint(prof.description))
        ins = prof.instrument
        parts = [x for x in (ins.manufacturer, ins.model) if x]
        parts += [f"statif {ins.stand}", f"tube {ins.tube_length}"]
        card.add(label(" · ".join(parts), "hint"))
        card.add(
            label(
                f"{len(prof.devices)} périphérique(s) · {len(prof.objectives)} objectif(s) · "
                f"{len(prof.channels)} canal(aux) · {len(prof.filter_sets)} cube(s)",
                "hint",
            )
        )
        manual = [d.label for d in microscope.devices.values() if d.manual]
        if manual:
            card.add(hint("Manuel : " + ", ".join(manual)))
        issues = prof.check()
        if issues:
            warn = label("⚠ " + "\n⚠ ".join(issues[:4]), "warning")
            card.add(warn)
        edit = button("Modifier la configuration…", "edit", primary=True)
        edit.clicked.connect(lambda: self.edit_requested.emit(None))
        card.add(edit)
        lay.addWidget(card)

        card = Card("Microscopes disponibles", icon_name="layers")
        self.list = QListWidget()
        self.list.setMinimumHeight(170)
        self.list.itemDoubleClicked.connect(lambda *_: self._use())
        card.add(self.list)
        use = button("Utiliser", "check")
        use.clicked.connect(self._use)
        modify = button("Modifier", "edit")
        modify.clicked.connect(self._edit_selected)
        delete = button("Supprimer", "trash", object_name="danger")
        delete.clicked.connect(self._delete)
        new = button("Nouveau microscope…", "plus", primary=True)
        new.clicked.connect(self.new_requested.emit)
        imp = button("Importer…", "open")
        imp.clicked.connect(self._import)
        exp = button("Exporter…", "export")
        exp.clicked.connect(self._export)
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, b in enumerate((use, modify, imp, exp, delete)):
            grid.addWidget(b, i // 2, i % 2)
        card.add(layout=grid)
        card.add(new)
        lay.addWidget(card)

        card = Card("Dossier des configurations", icon_name="folder")
        card.add(label(str(user_profile_dir()), "faint"))
        opener = button("Ouvrir le dossier", "folder")
        opener.clicked.connect(self._open_folder)
        card.add(opener)
        lay.addWidget(card)
        lay.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        current = self.m.profile_path.resolve() if self.m.profile_path else None
        for key, path in available_profiles().items():
            try:
                title = MicroscopeProfile.load(path).name
            except Exception:
                title = f"{key} (invalide)"
            kind = "fourni" if is_builtin_profile(path) else "personnel"
            item = QListWidgetItem(
                icons.icon("microscope", palette().muted, 18), f"{title}\n{kind} · {path.name}"
            )
            item.setData(256, str(path))
            if current is not None and path.resolve() == current:
                item.setIcon(icons.icon("check", palette().accent, 18))
            self.list.addItem(item)

    def _selected(self) -> Path | None:
        item = self.list.currentItem()
        return Path(item.data(256)) if item else None

    def _use(self) -> None:
        path = self._selected()
        if path is not None:
            self.switch_requested.emit(str(path))

    def _edit_selected(self) -> None:
        path = self._selected()
        if path is not None:
            self.edit_requested.emit(path)

    def _delete(self) -> None:
        path = self._selected()
        if path is None:
            return
        if is_builtin_profile(path):
            QMessageBox.information(
                self,
                "Supprimer",
                "Les profils fournis avec Fuji ne peuvent pas être supprimés (leur copie personnelle, si).",
            )
            return
        answer = QMessageBox.question(self, "Supprimer", f"Supprimer la configuration {path.name} ?")
        if answer == QMessageBox.StandardButton.Yes:
            path.unlink(missing_ok=True)
            self.refresh()

    def _import(self) -> None:
        src, _ = QFileDialog.getOpenFileName(
            self, "Importer une configuration", str(Path.home()), "Configuration Fuji (*.yaml *.yml)"
        )
        if not src:
            return
        try:
            MicroscopeProfile.load(src)
        except Exception as exc:
            QMessageBox.warning(
                self, "Importer", f"Ce fichier n'est pas une configuration de microscope valable :\n{exc}"
            )
            return
        user_profile_dir().mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, user_profile_dir() / Path(src).name)
        self.refresh()

    def _export(self) -> None:
        path = self._selected() or self.m.profile_path
        if path is None:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Exporter le profil", str(Path.home() / path.name), "Profil Fuji (*.yaml)"
        )
        if dest:
            shutil.copy2(path, dest)

    def _open_folder(self) -> None:
        folder = user_profile_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
