"""Lancement de l'interface graphique."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from qtpy.QtWidgets import QApplication, QMessageBox

from fuji.core.microscope import Microscope
from fuji.core.profile import user_profile_dir

from .prefs import settings

log = logging.getLogger(__name__)


def _style(app: QApplication) -> None:
    from .theme import apply

    apply(app)


def install_error_handler() -> None:
    """Une erreur imprévue dans l'interface est journalisée et affichée, sans fermer Fuji.

    (Par défaut, PyQt arrête le programme sur toute exception non interceptée.)
    """
    import traceback

    if getattr(sys.excepthook, "_fuji", False):
        return

    def hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("Erreur non interceptée :\n%s", text)
        app = QApplication.instance()
        if app is None:
            return
        try:
            box = QMessageBox(app.activeWindow())
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Erreur")
            box.setText("Une erreur inattendue s'est produite.")
            box.setInformativeText(
                f"{exc}\n\nFuji continue de fonctionner. Si elle revient, envoyez le fichier "
                "~/.fuji/fuji.log (les détails sont aussi ci-dessous)."
            )
            box.setDetailedText(text)
            box.show()
        except Exception:  # pragma: no cover - affichage impossible
            pass

    hook._fuji = True  # type: ignore[attr-defined]
    sys.excepthook = hook


def install_french(app: QApplication) -> None:
    """Boutons et dialogues standard de Qt en français (« Oui », « Non », « Annuler »…)."""
    from qtpy.QtCore import QLibraryInfo, QLocale, QTranslator

    if getattr(app, "_fuji_translator", None) is not None:
        return
    translator = QTranslator(app)
    try:
        folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    except AttributeError:  # Qt 5
        folder = QLibraryInfo.location(QLibraryInfo.TranslationsPath)  # type: ignore[attr-defined]
    if translator.load(QLocale(QLocale.Language.French), "qtbase", "_", folder):
        app.installTranslator(translator)
        app._fuji_translator = translator  # type: ignore[attr-defined]
    else:
        log.info("Traduction française de Qt introuvable dans %s", folder)


def isolate_settings(folder: str) -> None:
    """Range les réglages de l'interface dans ``folder`` plutôt que dans les préférences de
    l'utilisateur (tests, essai automatique). À appeler avant de créer la fenêtre."""
    from .prefs import isolate

    isolate(folder)


def _connect(name: str) -> Microscope:
    scope = Microscope.from_file(name)
    scope.connect()
    return scope


def create_window(profile: str | None = None):  # type: ignore[no-untyped-def]
    """Crée la fenêtre (sans boucle d'événements) : utile pour les tests et scripts."""
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    install_french(app)
    app.setApplicationName("Fuji")
    app.setOrganizationName("Fuji")
    _style(app)  # type: ignore[arg-type]
    prefs = settings()
    name = profile or prefs.value("last_profile", "demo")
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
    prefs.setValue("last_profile", name)
    window = MainWindow(scope)
    return app, window


def log_file() -> Path:
    """Journal de Fuji : ``~/.fuji/fuji.log`` (ou dans ``FUJI_HOME``)."""
    return user_profile_dir().parent / "fuji.log"


def start_log_file() -> Path | None:
    """Écrit les avertissements et les erreurs dans un fichier.

    Une application lancée d'un double-clic n'a pas de terminal : sans ce journal, la cause
    d'une panne (caméra absente, pilote refusé...) serait perdue. Deux fichiers de 500 Ko au
    plus sont gardés."""
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    if any(getattr(handler, "_fuji_journal", False) for handler in root.handlers):
        return log_file()
    try:
        path = log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=500_000, backupCount=1, encoding="utf-8")
    except OSError:  # dossier protégé : Fuji fonctionne quand même, sans journal
        return None
    handler._fuji_journal = True  # type: ignore[attr-defined]
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s : %(message)s"))
    root.addHandler(handler)
    return path


def run_app(profile: str | None = None) -> int:
    start_log_file()
    install_error_handler()
    app, window = create_window(profile)
    window.show()
    return app.exec()
