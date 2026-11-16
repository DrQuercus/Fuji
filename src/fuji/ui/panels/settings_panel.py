"""Panneau « Réglages » : apparence, interface, Fiji, ToupTek, dossiers."""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtCore import QSettings, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji import __version__

from ..components import Card, PathEdit, Segmented, ToggleSwitch, button, hbox, hint, label
from ..theme import ACCENTS, FONT_SIZES, ThemeManager


class SettingsPanel(QWidget):
    expert_changed = Signal(bool)
    auto_contrast_changed = Signal(bool)
    message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        tm = ThemeManager.get()
        mode, accent, size = tm.settings()
        s = QSettings("Fuji", "Fuji")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # apparence
        card = Card("Apparence", icon_name="sun")
        form = QFormLayout()
        self.mode = Segmented(["Sombre", "Clair"])
        self.mode.set_value("Sombre" if mode == "sombre" else "Clair")
        self.mode.selected.connect(lambda v: tm.set(mode=v.lower()))
        form.addRow("Thème", self.mode)
        sl = QWidget()
        sl_lay = QHBoxLayout(sl)
        sl_lay.setContentsMargins(0, 0, 0, 0)
        sl_lay.setSpacing(6)
        self._accent_buttons: list[QPushButton] = []
        for name, color in ACCENTS.items():
            b = QPushButton()
            b.setObjectName("swatch")
            b.setCheckable(True)
            b.setToolTip(name)
            b.setChecked(name == accent)
            b.setStyleSheet(
                f"QPushButton#swatch {{ background: {color}; border: 2px solid "
                "transparent; } QPushButton#swatch:checked { border: 2px solid white; }"
            )
            b.clicked.connect(lambda _=False, n=name: self._accent(n))
            sl_lay.addWidget(b)
            self._accent_buttons.append(b)
        sl_lay.addStretch(1)
        form.addRow("Couleur d'accent", sl)
        self.size = Segmented(list(FONT_SIZES))
        self.size.set_value(size)
        self.size.selected.connect(lambda v: tm.set(size=v))
        form.addRow("Taille du texte", self.size)
        card.add(layout=form)
        lay.addWidget(card)

        # interface
        card = Card("Interface", icon_name="sliders")
        self.expert = ToggleSwitch("Mode expert (tous les réglages avancés)")
        self.expert.setChecked(s.value("expert", False, type=bool))
        self.expert.toggled.connect(self.expert_changed.emit)
        self.auto_contrast = ToggleSwitch("Contraste automatique du live")
        self.auto_contrast.setChecked(s.value("live/auto_contrast", True, type=bool))
        self.auto_contrast.toggled.connect(self._auto_contrast)
        card.add(self.expert)
        card.add(self.auto_contrast)
        lay.addWidget(card)

        # Fiji
        card = Card(
            "Fiji / ImageJ",
            icon_name="chart",
            subtitle="Ton installation Fiji (avec tes plugins) est utilisée pour les "
            "macros et « Ouvrir dans Fiji ».",
        )
        from fuji.analysis.imagej import find_fiji

        self.fiji = PathEdit("dir")
        found = (
            s.value("fiji/path", "")
            or os.environ.get("FUJI_FIJI_PATH", "")
            or (str(find_fiji()) if find_fiji() else "")
        )
        self.fiji.setText(found)
        self.fiji.changed.connect(self._set_fiji)
        if found:
            os.environ["FUJI_FIJI_PATH"] = found
        test = button("Tester Fiji", "plug")
        test.clicked.connect(self._test_fiji)
        card.add(self.fiji)
        card.add(test)
        lay.addWidget(card)

        # ToupTek
        card = Card("Caméras ToupTek", icon_name="camera")
        self.tp_status = label("", "hint")
        card.add(self.tp_status)
        install = button("Installer le SDK…", "save")
        install.clicked.connect(self._install_touptek)
        detect = button("Détecter les caméras", "search")
        detect.clicked.connect(self._refresh_touptek)
        card.add(hbox(install, detect, spacing=6))
        lay.addWidget(card)

        # dossiers
        from fuji.core.profile import user_profile_dir

        home = user_profile_dir().parent
        card = Card("Dossiers", icon_name="folder")
        card.add(label(f"Données de Fuji : {home}", "faint"))
        opener = button("Ouvrir", "folder")
        opener.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(home))))
        card.add(opener)
        lay.addWidget(card)

        card = Card(f"Fuji {__version__}", icon_name="info")
        card.add(
            hint(
                "Logiciel libre (licence BSD-3-Clause). Construit sur napari, "
                "pymmcore-plus (Micro-Manager) et PyImageJ (Fiji)."
            )
        )
        lay.addWidget(card)
        lay.addStretch(1)
        self._refresh_touptek(quiet=True)

    def _accent(self, name: str) -> None:
        for b in self._accent_buttons:
            b.setChecked(b.toolTip() == name)
        ThemeManager.get().set(accent=name)

    def _auto_contrast(self, on: bool) -> None:
        QSettings("Fuji", "Fuji").setValue("live/auto_contrast", on)
        self.auto_contrast_changed.emit(on)

    def _set_fiji(self, path: str) -> None:
        QSettings("Fuji", "Fuji").setValue("fiji/path", path)
        if path:
            os.environ["FUJI_FIJI_PATH"] = path

    def _test_fiji(self) -> None:
        from fuji.analysis.imagej import ImageJBridge

        self.message.emit("Démarrage de Fiji (Java)…")
        worker = create_worker(ImageJBridge.get().version, _start_thread=False)
        worker.returned.connect(lambda v: QMessageBox.information(self, "Fiji", f"ImageJ {v} prêt."))
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Fiji", str(e)))
        worker.start()

    def _refresh_touptek(self, quiet: bool = False) -> None:
        from fuji.drivers.touptek import ToupTekSDKNotFound, default_sdk_dir, list_cameras

        try:
            cams = list_cameras()
            names = ", ".join(c["nom"] for c in cams) or "aucune caméra branchée"
            self.tp_status.setText(f"SDK installé ({default_sdk_dir()}) — {names}")
        except ToupTekSDKNotFound:
            self.tp_status.setText(
                "SDK non installé : téléchargez « ToupTek SDK » sur le site du "
                "fabricant, puis « Installer le SDK… »."
            )
        except Exception as exc:
            self.tp_status.setText(f"Erreur du SDK : {exc}")
            if not quiet:
                QMessageBox.warning(self, "ToupTek", str(exc))

    def _install_touptek(self) -> None:
        from fuji.drivers.touptek import install_sdk

        src, _ = QFileDialog.getOpenFileName(
            self, "SDK ToupTek (toupcamsdk.zip)", str(Path.home()), "SDK (*.zip);;Tous les fichiers (*)"
        )
        if not src:
            return
        try:
            dest = install_sdk(src)
        except Exception as exc:
            QMessageBox.warning(self, "ToupTek", f"Installation impossible :\n{exc}")
            return
        self.message.emit(f"SDK ToupTek installé dans {dest}")
        self._refresh_touptek()
