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
``gcode``              motorisation maison : GRBL, FluidNC, Marlin (port série)
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
from typing import TYPE_CHECKING, Any, Literal, Union

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

OptionType = Literal["str", "int", "float", "bool", "choice", "path", "dir", "list", "device", "mapping"]


@dataclass(frozen=True)
class OptionSpec:
    """Description d'une option de pilote (pour générer le formulaire de configuration)."""

    name: str
    type: OptionType = "str"
    label: str = ""
    default: Any = None
    help: str = ""
    choices: tuple[str, ...] = ()
    #: Pour ``type="device"`` : genre de périphérique attendu (``camera``, ``focus``...).
    device_kind: str | None = None
    min: float | None = None
    max: float | None = None
    unit: str = ""
    #: Filtre de fichiers pour ``type="path"`` (ex. ``"Configuration (*.cfg)"``).
    file_filter: str = ""
    advanced: bool = False

    @property
    def title(self) -> str:
        return self.label or self.name.replace("_", " ")


@dataclass(frozen=True)
class DriverInfo:
    """Un pilote et sa description pour l'interface."""

    name: str
    factory: DriverFactory
    title: str
    category: str = "Autres"
    #: Genre de périphérique créé (``camera``, ``state``...) ; ``None`` = plusieurs.
    kind: str | None = None
    description: str = ""
    options: tuple[OptionSpec, ...] = ()
    #: Pour un pilote qui crée plusieurs périphériques à partir d'une seule entrée de profil :
    #: fonction (nom de l'entrée, options) → [(nom du périphérique créé, genre), ...]. Permet à
    #: l'éditeur et aux vérifications de connaître ces périphériques sans se connecter.
    provides: Callable[[str, dict[str, Any]], list[tuple[str, str]]] | None = None

    def provided(self, name: str, options: dict[str, Any]) -> list[tuple[str, str]]:
        return list(self.provides(name, options)) if self.provides is not None else []

    def option(self, name: str) -> OptionSpec | None:
        return next((o for o in self.options if o.name == name), None)

    def defaults(self) -> dict[str, Any]:
        """Options par défaut d'un nouveau périphérique (seulement celles qui ont une valeur)."""
        return {o.name: o.default for o in self.options if o.default is not None}


#: Types de sources de lumière proposés.
LIGHT_TYPES = ("halogène", "LED", "mercure (HBO)", "xénon", "halogénure métallique", "laser", "autre")

#: Ordre d'affichage des catégories dans l'interface.
CATEGORIES = (
    "Caméras",
    "Platines et mise au point",
    "Tourelles, filtres et cubes",
    "Éclairage",
    "Systèmes complets",
    "Simulation",
    "Autres",
)

_REGISTRY: dict[str, DriverInfo] = {}
_BUILTIN_MODULES = (
    "fuji.drivers.demo",
    "fuji.drivers.manual",
    "fuji.drivers.touptek",
    "fuji.drivers.micromanager",
    "fuji.drivers.gcode",
)
_loaded = False


def register_driver(
    name: str,
    *,
    title: str | None = None,
    category: str = "Autres",
    kind: str | None = None,
    description: str = "",
    options: tuple[OptionSpec, ...] | list[OptionSpec] = (),
    provides: Callable[[str, dict[str, Any]], list[tuple[str, str]]] | None = None,
) -> Callable[[DriverFactory], DriverFactory]:
    """Décorateur : ``@register_driver("monlabo.platine", title=..., options=[...])``."""

    def deco(factory: DriverFactory) -> DriverFactory:
        _REGISTRY[name] = DriverInfo(
            name,
            factory,
            title or name,
            category,
            kind,
            description or (factory.__doc__ or ""),
            tuple(options),
            provides,
        )
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
            if ep.name in _REGISTRY:
                continue
            if isinstance(obj, DriverInfo):
                _REGISTRY[ep.name] = obj
            elif callable(obj):
                _REGISTRY[ep.name] = DriverInfo(
                    ep.name, obj, ep.name, "Autres", description=obj.__doc__ or ""
                )
        except Exception:  # pragma: no cover - dépend des paquets installés
            log.exception("Impossible de charger le pilote externe %s", ep.name)


def available_drivers() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def drivers_info() -> list[DriverInfo]:
    """Tous les pilotes, triés par catégorie puis par titre."""
    _load_all()
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return sorted(_REGISTRY.values(), key=lambda d: (order.get(d.category, 99), d.title))


def driver_info(name: str) -> DriverInfo | None:
    _load_all()
    return _REGISTRY.get(name)


def create_devices(spec: DeviceSpec, ctx: DriverContext) -> list[Device]:
    _load_all()
    try:
        info = _REGISTRY[spec.driver]
    except KeyError:
        raise KeyError(
            f"Pilote inconnu {spec.driver!r} pour {spec.name!r}. "
            f"Pilotes disponibles : {', '.join(sorted(_REGISTRY))}"
        ) from None
    result = info.factory(spec, ctx)
    return list(result) if isinstance(result, (list, tuple)) else [result]
