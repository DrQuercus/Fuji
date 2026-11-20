"""Panneau « Préférences » : apparence, interface, Fiji, ToupTek, dossiers."""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtCore import QSettings, Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji import __version__

from ..components import Card, PathEdit, Segmented, ToggleSwitch, button, flow, hint, label
from ..flow import FlowLayout
from ..prefs import settings
from ..theme import ACCENTS, FONT_SIZES, ThemeManager


class SettingsPanel(QWidget):
    expert_changed = Signal(bool)
    auto_contrast_changed = Signal(bool)
    #: demande de démarrage ou d'arrêt du serveur pour IA : (actif, port, autoriser le matériel)
    ai_server_requested = Signal(bool, int, bool)
    message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        tm = ThemeManager.get()
        mode, accent, size = tm.settings()
        s = settings()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # apparence
        card = Card("Apparence", icon_name="sun")
        form = QFormLayout()
        # panneau étroit : le champ passe sous son libellé
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.mode = Segmented(["Sombre", "Clair"])
        self.mode.set_value("Sombre" if mode == "sombre" else "Clair")
        self.mode.selected.connect(lambda v: tm.set(mode=v.lower()))
        form.addRow("Thème", self.mode)
        sl = QWidget()
        sl_lay = FlowLayout(sl, spacing=6, line_spacing=6)  # les pastilles passent à la ligne
        sl_lay.setContentsMargins(0, 0, 0, 0)
        self._accent_buttons: list[QPushButton] = []
        for name, color in ACCENTS.items():
            b = QPushButton()
            b.setObjectName("swatch")
            b.setCheckable(True)
            b.setToolTip(name)
            b.setChecked(name == accent)
            b.setText("✓" if name == accent else "")  # la couleur choisie porte une coche
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton#swatch {{ background: {color}; border: 2px solid {color}; "
                "color: #ffffff; font-weight: 800; }"
            )
            b.clicked.connect(lambda _=False, n=name: self._accent(n))
            sl_lay.addWidget(b)
            self._accent_buttons.append(b)
        form.addRow("Couleur d'accent", sl)
        self.size = Segmented(list(FONT_SIZES))
        self.size.set_value(size)
        self.size.selected.connect(lambda v: tm.set(size=v))
        form.addRow("Taille du texte", self.size)
        card.add(layout=form)
        lay.addWidget(card)

        # interface
        card = Card("Interface", icon_name="sliders")
        self.expert = ToggleSwitch("Mode expert (réglages avancés)")
        self.expert.setChecked(s.value("expert", False, type=bool))
        self.expert.toggled.connect(self.expert_changed.emit)
        self.auto_contrast = ToggleSwitch("Contraste automatique du live")
        self.auto_contrast.setChecked(s.value("live/auto_contrast", True, type=bool))
        self.auto_contrast.toggled.connect(self._auto_contrast)
        card.add(self.expert)
        card.add(self.auto_contrast)
        lay.addWidget(card)
        lay.addWidget(self._ai_card(s))

        # Fiji
        card = Card(
            "Fiji / ImageJ",
            icon_name="chart",
            subtitle="Votre installation de Fiji (avec vos plugins) sert aux macros et à "
            "« Ouvrir dans Fiji ».",
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
        card.add(flow(test))  # les boutons secondaires gardent leur largeur, calés à gauche
        lay.addWidget(card)

        # ToupTek
        card = Card("Caméras ToupTek", icon_name="camera")
        self.tp_status = label("", "hint")
        card.add(self.tp_status)
        install = button("Installer le SDK…", "save")
        install.clicked.connect(self._install_touptek)
        detect = button("Détecter les caméras", "search")
        detect.clicked.connect(self._refresh_touptek)
        card.add(flow(install, detect))
        lay.addWidget(card)

        # dossiers
        from fuji.core.profile import user_profile_dir

        home = user_profile_dir().parent
        card = Card("Dossiers", icon_name="folder")
        card.add(label(f"Données de Fuji : {home}", "faint"))
        opener = button("Ouvrir", "folder")
        opener.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(home))))
        card.add(flow(opener))
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

    # --------------------------------------------------------------- assistant IA
    def _ai_card(self, s: QSettings) -> QWidget:
        from qtpy.QtWidgets import QSpinBox

        card = Card("Assistant IA", icon_name="sparkle")
        card.add(
            hint(
                "Laisse un assistant IA (Claude Desktop, Claude Code ou tout client MCP) voir les images, "
                "lancer des analyses et, si vous l'autorisez, régler le microscope. Vous voyez dans Fuji "
                "tout ce qu'il fait, et les manipulations manuelles vous sont demandées à l'écran."
            )
        )
        self.ai_enabled = ToggleSwitch("Serveur pour IA (MCP) actif")
        self.ai_hardware = ToggleSwitch("L'IA peut régler le matériel")
        self.ai_hardware.setToolTip("Désactivé : l'IA peut seulement regarder les images et les analyser")
        self.ai_hardware.setChecked(s.value("ia/hardware", True, type=bool))
        self.ai_port = QSpinBox()
        self.ai_port.setRange(1024, 65535)
        self.ai_port.setValue(int(s.value("ia/port", 8765)))
        self.ai_status = label("Arrêté", "hint")
        self.ai_status.setWordWrap(True)  # « Actif — http://… » ne doit pas élargir tout le panneau
        copy = button(
            "Copier la configuration", "copy", tooltip="À coller dans la configuration de votre assistant"
        )
        copy.clicked.connect(self._copy_ai_config)
        form = QFormLayout()
        form.addRow(self.ai_enabled)
        form.addRow(self.ai_hardware)
        form.addRow("Port", self.ai_port)
        form.addRow("État", self.ai_status)
        card.add(layout=form)
        card.add(flow(copy))
        self.ai_enabled.toggled.connect(self._ai_toggled)
        self.ai_hardware.toggled.connect(self._ai_hardware_toggled)
        return card

    def _ai_toggled(self, on: bool) -> None:
        self.ai_port.setEnabled(not on)
        settings().setValue("ia/port", self.ai_port.value())
        self.ai_server_requested.emit(on, self.ai_port.value(), self.ai_hardware.isChecked())

    def _ai_hardware_toggled(self, on: bool) -> None:
        settings().setValue("ia/hardware", on)
        if self.ai_enabled.isChecked():  # applique tout de suite le nouveau réglage
            self.ai_server_requested.emit(True, self.ai_port.value(), on)

    def set_ai_state(self, running: bool, text: str) -> None:
        """Appelé par la fenêtre principale : état réel du serveur."""
        self.ai_enabled.blockSignals(True)
        self.ai_enabled.setChecked(running)
        self.ai_enabled.blockSignals(False)
        self.ai_port.setEnabled(not running)
        self.ai_status.setText(text)

    def ai_config(self) -> str:
        """Configuration à donner au client MCP pour rejoindre ce Fuji."""
        import json

        url = f"http://127.0.0.1:{self.ai_port.value()}/mcp"
        return json.dumps({"mcpServers": {"fuji": {"type": "http", "url": url}}}, indent=2)

    def _copy_ai_config(self) -> None:
        from qtpy.QtGui import QGuiApplication

        QGuiApplication.clipboard().setText(self.ai_config())
        self.message.emit("Configuration copiée (voir docs/IA.md pour la coller dans votre assistant)")

    def _accent(self, name: str) -> None:
        for b in self._accent_buttons:
            b.setChecked(b.toolTip() == name)
            b.setText("✓" if b.toolTip() == name else "")
        ThemeManager.get().set(accent=name)

    def _auto_contrast(self, on: bool) -> None:
        settings().setValue("live/auto_contrast", on)
        self.auto_contrast_changed.emit(on)

    def _set_fiji(self, path: str) -> None:
        settings().setValue("fiji/path", path)
        if path:
            os.environ["FUJI_FIJI_PATH"] = path

    def _test_fiji(self) -> None:
        from fuji.analysis.imagej import ImageJBridge

        self.message.emit("Démarrage de Fiji (Java)…")
        worker = create_worker(ImageJBridge.get().version, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(lambda v: QMessageBox.information(self, "Fiji", f"ImageJ {v} prêt."))
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Fiji", f"Fiji n'a pas pu démarrer : {e}"))
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
                QMessageBox.warning(self, "ToupTek", f"Le SDK ToupTek ne répond pas : {exc}")

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
