"""Registre des pilotes de périphériques.

Un pilote est une fonction ``fabrique(spec, contexte) -> Device | list[Device]``
enregistrée sous un nom (``demo.camera``, ``touptek.camera``, ``micromanager``...).
Les profils YAML référencent ces noms.

Pilotes fournis :

=====================  ======================================================
``demo.*``             microscope simulé complet (pour tester sans matériel)
``manual.*``           éléments non motorisés (tourelle, mise au point, lampe)
``touptek.camera``     caméras ToupTek via le SDK officiel (toupcam)
``micromanager``       toute configuration Micro-Manager (~250 périphériques)
=====================  ======================================================

Un paquet tiers peut ajouter ses pilotes via le point d'entrée ``fuji.drivers``
(dans son ``pyproject.toml``) sans modifier Fuji.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from fuji.core.devices import Device
    from fuji.core.profile import DeviceSpec, MicroscopeProfile

log = logging.getLogger(__name__)


@dataclass
class DriverContext:
    """Ce qu'un pilote peut consulter au moment de sa création."""

    profile: MicroscopeProfile
    #: Périphériques déjà créés (dans l'ordre du profil).
    devices: dict[str, Device] = field(default_factory=dict)

    def device(self, name: str) -> Device:
        try:
            return self.devices[name]
        except KeyError:
            raise KeyError(
                f"Périphérique {name!r} introuvable (il doit être déclaré plus haut dans le profil)"
            ) from None


DriverFactory = Callable[["DeviceSpec", DriverContext], Union["Device", list["Device"]]]

_REGISTRY: dict[str, DriverFactory] = {}
_BUILTIN_MODULES = (
    "fuji.drivers.demo",
    "fuji.drivers.manual",
    "fuji.drivers.touptek",
    "fuji.drivers.micromanager",
)
_loaded = False


def register_driver(name: str) -> Callable[[DriverFactory], DriverFactory]:
    """Décorateur : ``@register_driver("monlabo.platine")``."""

    def deco(factory: DriverFactory) -> DriverFactory:
        _REGISTRY[name] = factory
        return factory

    return deco


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for mod in _BUILTIN_MODULES:
        importlib.import_module(mod)
    for ep in entry_points(group="fuji.drivers"):
        try:
            obj = ep.load()
            if callable(obj) and ep.name not in _REGISTRY:
                _REGISTRY[ep.name] = obj
        except Exception:  # pragma: no cover - dépend des paquets installés
            log.exception("Impossible de charger le pilote externe %s", ep.name)


def available_drivers() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def create_devices(spec: DeviceSpec, ctx: DriverContext) -> list[Device]:
    _load_all()
    try:
        factory = _REGISTRY[spec.driver]
    except KeyError:
        raise KeyError(
            f"Pilote inconnu {spec.driver!r} pour {spec.name!r}. "
            f"Pilotes disponibles : {', '.join(sorted(_REGISTRY))}"
        ) from None
    result = factory(spec, ctx)
    return list(result) if isinstance(result, (list, tuple)) else [result]
