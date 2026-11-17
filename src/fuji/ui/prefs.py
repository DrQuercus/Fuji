"""Préférences de l'interface (thème, dernier microscope, dossiers...).

Toujours passer par :func:`settings` plutôt que ``QSettings("Fuji", "Fuji")`` : les tests et
l'essai automatique rangent leurs préférences dans un dossier temporaire (:func:`isolate`),
pour ne jamais toucher à celles de l'utilisateur. Sur macOS, les préférences « natives »
(fichier .plist) ne se déplacent pas : en mode isolé on utilise donc un fichier .ini.
"""

from __future__ import annotations

from qtpy.QtCore import QSettings

_isolated = False


def isolate(folder: str) -> None:
    """Préférences rangées dans ``folder`` jusqu'à la fin du programme."""
    global _isolated
    _isolated = True
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(folder))


def settings() -> QSettings:
    if _isolated:
        return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "Fuji", "Fuji")
    return QSettings("Fuji", "Fuji")
