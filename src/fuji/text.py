"""Petits outils d'écriture en français : pluriels et nombres à virgule.

Une seule convention dans toute l'interface : « 1 image », « 3 images », « 12,5 µm »
(jamais « 3 image(s) » ni « 12.5 µm »).
"""

from __future__ import annotations

__all__ = ["count", "number", "plural"]

#: pluriels qui ne se forment pas avec un « s »
_IRREGULAR = {"canal": "canaux", "manipulation manuelle": "manipulations manuelles"}


def plural(n: float, word: str, many: str | None = None) -> str:
    """Le mot accordé avec ``n`` : ``plural(1, "image")`` → « image », ``plural(3, "image")`` →
    « images ». En français, zéro et un s'écrivent au singulier."""
    if abs(n) < 2:
        return word
    if many is not None:
        return many
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    return " ".join(w if w.endswith(("s", "x", "z")) else w + "s" for w in word.split(" "))


def count(n: int, word: str, many: str | None = None) -> str:
    """Le nombre suivi du mot accordé : « 1 image », « 3 images », « 2 canaux »."""
    return f"{n} {plural(n, word, many)}"


def signed(value: float, decimals: int = 1) -> str:
    """Un écart, avec son signe : « +1,5 », « −0,3 » s'écrit « -0,3 »."""
    return f"{float(value):+.{decimals}f}".replace(".", ",")


def file_manager() -> str:
    """Le nom de l'explorateur de fichiers de l'ordinateur, avec son article."""
    import sys

    return {"darwin": "le Finder", "win32": "l'Explorateur"}.get(sys.platform, "le gestionnaire de fichiers")


def number(value: float, decimals: int | None = None, digits: int = 4) -> str:
    """Un nombre écrit à la française, avec une virgule décimale.

    ``decimals`` : nombre fixe de chiffres après la virgule (« 12,50 »). Sans lui, ``digits``
    chiffres utiles et pas de zéros inutiles (« 12,5 », « 0,2907 », « 1234 »)."""
    value = float(value)
    if decimals is not None:
        text = f"{value:.{decimals}f}"
    elif value != value:  # pas un nombre
        return "—"
    elif abs(value) >= 10**digits:
        text = f"{value:.0f}"
    else:
        text = f"{value:.{digits}g}"
        if "e" in text:  # très petit : notation scientifique gardée telle quelle
            return text.replace(".", ",")
    return text.replace(".", ",")
