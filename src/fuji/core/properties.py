"""Système de propriétés générique des périphériques.

Chaque périphérique expose ses réglages sous forme de *propriétés* décrites
(type, bornes, choix possibles, unité...). L'interface peut ainsi construire
automatiquement un panneau de réglages pour **n'importe quel** matériel, y compris
un périphérique Micro-Manager qu'elle n'a jamais vu. C'est la base de la
compatibilité « un peu tous les microscopes ».
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PropertyType(str, Enum):
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"


@dataclass(frozen=True)
class PropertyInfo:
    """Description d'un réglage de périphérique."""

    name: str
    type: PropertyType
    read_only: bool = False
    limits: tuple[float, float] | None = None
    choices: tuple[str, ...] | None = None
    unit: str = ""
    description: str = ""
    #: Réglage réservé au mode expert (masqué en mode simple).
    advanced: bool = False
    #: Regroupement dans l'interface (ex. « Image », « Couleur », « Simulation »).
    category: str = "Général"

    def coerce(self, value: Any) -> Any:
        """Convertit et valide une valeur pour cette propriété."""
        if self.read_only:
            raise PermissionError(f"La propriété {self.name!r} est en lecture seule.")
        if self.type is PropertyType.BOOL:
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on", "oui", "vrai"}
            return bool(value)
        if self.type in (PropertyType.FLOAT, PropertyType.INT):
            num = float(value)
            if self.limits is not None:
                lo, hi = self.limits
                if not lo <= num <= hi:
                    raise ValueError(f"{self.name} = {num} hors limites [{lo}, {hi}] {self.unit}".strip())
            return int(round(num)) if self.type is PropertyType.INT else num
        value = str(value)
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{self.name} = {value!r} ; choix possibles : {list(self.choices)}")
        return value


@dataclass
class _Entry:
    info: PropertyInfo
    getter: Callable[[], Any]
    setter: Callable[[Any], None] | None


@dataclass
class PropertyRegistry:
    """Stocke les propriétés d'un périphérique avec leurs accesseurs."""

    _entries: dict[str, _Entry] = field(default_factory=dict)

    def add(
        self,
        info: PropertyInfo,
        getter: Callable[[], Any],
        setter: Callable[[Any], None] | None = None,
    ) -> None:
        if setter is None and not info.read_only:
            info = PropertyInfo(**{**info.__dict__, "read_only": True})
        self._entries[info.name] = _Entry(info, getter, setter)

    def remove(self, name: str) -> None:
        self._entries.pop(name, None)

    def infos(self) -> list[PropertyInfo]:
        return [e.info for e in self._entries.values()]

    def info(self, name: str) -> PropertyInfo:
        return self._entry(name).info

    def get(self, name: str) -> Any:
        return self._entry(name).getter()

    def set(self, name: str, value: Any) -> Any:
        entry = self._entry(name)
        coerced = entry.info.coerce(value)
        assert entry.setter is not None
        entry.setter(coerced)
        return coerced

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def names(self) -> Sequence[str]:
        return list(self._entries)

    def _entry(self, name: str) -> _Entry:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"Propriété inconnue {name!r}. Disponibles : {list(self._entries)}") from None
