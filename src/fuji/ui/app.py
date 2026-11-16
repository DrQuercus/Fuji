"""Lancement de l'interface graphique."""

from __future__ import annotations

import logging
import sys

from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QApplication, QMessageBox

from fuji.core.microscope import Microscope

log = logging.getLogger(__name__)

EXTRA_STYLE = """
QLabel#hint { color: #9aa4ae; font-size: 11px; }
QLabel#warning { color: #f2a93b; }
QLabel#step { font-size: 15px; font-weight: 600; padding-top: 6px; }
QPushButton#primary { background-color: #2f6fd6; color: white; border-radius: 4px;
                      padding: 6px 10px; font-weight: 600; }
QPushButton#primary:disabled { background-color: #555; color: #aaa; }
QGroupBox { font-weight: 600; margin-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; }
"""


def _style(app: QApplication) -> None:
    try:
        from napari.qt import get_stylesheet

        app.setStyleSheet(get_stylesheet("dark") + EXTRA_STYLE)
    except Exception:  # pragma: no cover - thème indisponible
        app.setStyleSheet(EXTRA_STYLE)


def _connect(name: str) -> Microscope:
    scope = Microscope.from_file(name)
    scope.connect()
    return scope


def create_window(profile: str | None = None):  # type: ignore[no-untyped-def]
    """Crée la fenêtre (sans boucle d'événements) : utile pour les tests et scripts."""
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Fuji")
    app.setOrganizationName("Fuji")
    _style(app)  # type: ignore[arg-type]
    settings = QSettings("Fuji", "Fuji")
    name = profile or settings.value("last_profile", "demo")
    try:
        scope = _connect(name)
    except Exception as exc:
        log.exception("Connexion impossible au profil %s", name)
        if name == "demo":
            raise
        answer = QMessageBox.question(
            None,
            "Connexion impossible",
            f"Impossible de se connecter au microscope « {name} » :\n\n{exc}\n\n"
            "Ouvrir le microscope de démonstration à la place ?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            raise
        name = "demo"
        scope = _connect(name)
    settings.setValue("last_profile", name)
    window = MainWindow(scope)
    return app, window


def run_app(profile: str | None = None) -> int:
    app, window = create_window(profile)
    window.show()
    return app.exec()
