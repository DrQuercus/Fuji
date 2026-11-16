"""Profils de microscopes (fichiers YAML).

Un profil décrit **un** setup : ses périphériques (et quel pilote les commande),
ses objectifs (grossissement, ouverture numérique, calibration du pixel), ses
canaux (fond clair, GFP, DAPI...) et le rôle de chaque périphérique.

Le même logiciel s'adapte ainsi à un microscope manuel (Diaplan + ToupTek), à un
microscope motorisé piloté par Micro-Manager, ou à un montage maison.
"""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _YAMLLoader(yaml.SafeLoader):
    """YAML « 1.2 » : seuls true/false sont des booléens.

    En YAML 1.1 (PyYAML), ``on``, ``off``, ``yes``, ``no`` deviennent des booléens :
    ``{on: true}`` donnerait la clé ``True`` et un filtre nommé « NO » deviendrait
    ``False``. On évite ces pièges.
    """


_YAMLLoader.yaml_implicit_resolvers = {
    key: [r for r in resolvers if r[0] != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_YAMLLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_YAMLLoader)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DeviceSpec(_Model):
    """Un périphérique (ou un groupe de périphériques) et son pilote."""

    name: str
    #: Identifiant de pilote, ex. ``demo.camera``, ``touptek.camera``, ``manual.state``,
    #: ``micromanager`` (charge toute une configuration Micro-Manager).
    driver: str
    label: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


IMMERSIONS = ("air", "huile", "eau", "glycérol", "silicone")
CONTRASTS = ("fond clair", "phase", "DIC", "fond noir", "fluorescence", "polarisation")
CORRECTIONS = (
    "achromat",
    "plan-achromat",
    "fluorite (semi-apo)",
    "plan-fluor",
    "apochromat",
    "plan-apochromat",
    "autre",
)
STANDS = ("droit", "inversé", "stéréo", "macroscope", "autre")
TUBE_LENGTHS = ("infini", "160 mm", "170 mm", "autre")


class InstrumentInfo(_Model):
    """Le statif et l'optique fixe (informatif, utilisé par les assistants)."""

    manufacturer: str = ""
    model: str = ""
    stand: Literal["droit", "inversé", "stéréo", "macroscope", "autre"] = "droit"
    tube_length: Literal["infini", "160 mm", "170 mm", "autre"] = "infini"
    eyepiece_magnification: float | None = Field(default=10.0, gt=0)
    #: Indice de champ des oculaires (mm), gravé sur l'oculaire (ex. 18, 20, 22).
    field_number_mm: float | None = Field(default=None, gt=0)
    condenser: str = ""
    serial_number: str = ""
    notes: str = ""


class FilterSet(_Model):
    """Cube / bloc de filtres de fluorescence."""

    name: str
    excitation_nm: float | None = Field(default=None, gt=0)
    excitation_bandwidth_nm: float | None = Field(default=None, gt=0)
    dichroic_nm: float | None = Field(default=None, gt=0)
    emission_nm: float | None = Field(default=None, gt=0)
    #: Largeur de bande d'émission ; vide = filtre passe-haut (à partir de emission_nm).
    emission_bandwidth_nm: float | None = Field(default=None, gt=0)
    fluorophores: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def summary(self) -> str:
        """Résumé lisible, ex. « Ex 470/40 · DM 495 · Em 525/50 »."""

        def band(center: float | None, width: float | None, longpass: bool = False) -> str:
            if center is None:
                return "?"
            if width:
                return f"{center:g}/{width:g}"
            return f"LP {center:g}" if longpass else f"{center:g}"

        parts = [f"Ex {band(self.excitation_nm, self.excitation_bandwidth_nm)}"]
        if self.dichroic_nm:
            parts.append(f"DM {self.dichroic_nm:g}")
        parts.append(f"Em {band(self.emission_nm, self.emission_bandwidth_nm, longpass=True)}")
        return " · ".join(parts)


class Objective(_Model):
    label: str
    magnification: float = Field(gt=0)
    na: float = Field(gt=0, le=1.7, description="Ouverture numérique")
    immersion: Literal["air", "huile", "eau", "glycérol", "silicone"] = "air"
    #: Méthode de contraste prévue (informatif, utilisé par les assistants).
    contrast: Literal["fond clair", "phase", "DIC", "fond noir", "fluorescence", "polarisation"] = (
        "fond clair"
    )
    correction: (
        Literal[
            "achromat",
            "plan-achromat",
            "fluorite (semi-apo)",
            "plan-fluor",
            "apochromat",
            "plan-apochromat",
            "autre",
        ]
        | None
    ) = None
    working_distance_mm: float | None = Field(default=None, gt=0)
    #: Épaisseur de lamelle prévue (mm) ; 0 = sans lamelle.
    coverslip_mm: float | None = Field(default=None, ge=0)
    #: Anneau de phase (Ph1, Ph2, Ph3) pour le contraste de phase.
    phase_ring: str = ""
    #: Taille du pixel calibrée (µm/pixel), mesurée avec un micromètre objet.
    #: Si absente, elle est estimée à partir de la caméra et du grossissement.
    pixel_size_um: float | None = Field(default=None, gt=0)
    #: Décalage de mise au point par rapport aux autres objectifs (parfocalité), en µm.
    parfocal_offset_um: float = 0.0
    notes: str = ""

    @property
    def resolution_um(self) -> float:
        """Résolution latérale de Rayleigh à 550 nm (0,61 λ / ON)."""
        return 0.61 * 0.55 / self.na


class LightSetting(_Model):
    on: bool = True
    intensity: float | None = Field(default=None, ge=0, le=100)


class ChannelPreset(_Model):
    """Préréglage de canal : ce qu'il faut régler pour obtenir une image donnée."""

    name: str
    kind: Literal["transmission", "fluorescence", "réflexion"] = "transmission"
    exposure_ms: float | None = Field(default=None, gt=0)
    gain: float | None = None
    #: Positions des périphériques à états, ex. ``{"cubes": "GFP"}``.
    states: dict[str, str] = Field(default_factory=dict)
    #: Réglages des sources de lumière, ex. ``{"led_470": {"on": true, "intensity": 30}}``.
    lights: dict[str, LightSetting] = Field(default_factory=dict)
    #: Obturateurs à ouvrir pendant l'exposition.
    shutters: list[str] = Field(default_factory=list)
    #: Préréglage Micro-Manager équivalent (groupe, préréglage), si applicable.
    mm_config: tuple[str, str] | None = None
    #: Table de couleurs d'affichage (nom napari : gray, green, magenta, cyan...).
    color: str = "gray"
    #: Cube de filtres utilisé (nom d'un ``filter_sets`` du profil).
    filter_set: str | None = None
    excitation_nm: float | None = None
    emission_nm: float | None = None


class Roles(_Model):
    """Quel périphérique joue quel rôle. Laisser vide = détection automatique."""

    camera: str | None = None
    xy_stage: str | None = None
    focus: str | None = None
    objective_changer: str | None = None
    transmitted_light: str | None = None


class MicroscopeProfile(_Model):
    name: str
    description: str = ""
    instrument: InstrumentInfo = Field(default_factory=InstrumentInfo)
    devices: list[DeviceSpec] = Field(default_factory=list)
    roles: Roles = Field(default_factory=Roles)
    #: Grossissement de l'adaptateur caméra (monture C 0.5x, 0.63x, 1x...).
    camera_adapter_magnification: float = Field(default=1.0, gt=0)
    objectives: list[Objective] = Field(default_factory=list)
    filter_sets: list[FilterSet] = Field(default_factory=list)
    channels: list[ChannelPreset] = Field(default_factory=list)
    default_channel: str | None = None
    #: Ouverture numérique maximale du condenseur (pour l'assistant Köhler).
    condenser_na: float | None = Field(default=None, gt=0, le=1.4)
    save_directory: str | None = None

    @model_validator(mode="after")
    def _check_names(self) -> MicroscopeProfile:
        names = [d.name for d in self.devices]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"Noms de périphériques en double : {sorted(dup)}")
        labels = [o.label for o in self.objectives]
        if len(set(labels)) != len(labels):
            raise ValueError("Deux objectifs portent le même nom")
        chans = [c.name for c in self.channels]
        if len(set(chans)) != len(chans):
            raise ValueError("Deux canaux portent le même nom")
        if self.default_channel and self.default_channel not in chans:
            raise ValueError(f"Canal par défaut inconnu : {self.default_channel!r}")
        sets = [f.name for f in self.filter_sets]
        if len(set(sets)) != len(sets):
            raise ValueError("Deux cubes de filtres portent le même nom")
        return self

    def check(self) -> list[str]:
        """Problèmes de cohérence (sans bloquer l'enregistrement) : rôles, références..."""
        from fuji.drivers import driver_info

        issues: list[str] = []
        devices = {d.name: d for d in self.devices}
        has_mm = any(d.driver == "micromanager" for d in self.devices)
        kinds = {}
        for d in self.devices:
            info = driver_info(d.driver)
            if info is None:
                issues.append(f"Périphérique « {d.name} » : pilote inconnu « {d.driver} ».")
                continue
            kinds[d.name] = info.kind
            known = {o.name for o in info.options}
            extra = set(d.options) - known
            if info.options and extra and d.driver != "micromanager":
                issues.append(f"Périphérique « {d.name} » : options inconnues {sorted(extra)}.")
        for d in self.devices:
            if d.driver in ("manual.state", "demo.state") and not d.options.get("labels"):
                if self.roles.objective_changer != d.name and not d.name.lower().startswith("obj"):
                    issues.append(f"« {d.name} » : positions non définies (option « Positions »).")
                elif not self.objectives:
                    issues.append(f"« {d.name} » : revolver sans objectif (onglet Objectifs).")
        expected = {
            "camera": "camera",
            "xy_stage": "xy_stage",
            "focus": "focus",
            "objective_changer": "state",
            "transmitted_light": "light_source",
        }
        for role, kind in expected.items():
            name = getattr(self.roles, role)
            if not name:
                continue
            if name not in devices:
                if not has_mm:
                    issues.append(f"Rôle « {role} » : périphérique « {name} » inexistant.")
            elif kinds.get(name) not in (kind, None):
                issues.append(f"Rôle « {role} » : « {name} » n'est pas du bon type ({kinds.get(name)}).")
        if not any(k == "camera" for k in kinds.values()) and not has_mm:
            issues.append("Aucune caméra : la capture et le live seront indisponibles.")
        sets = {f.name for f in self.filter_sets}
        for ch in self.channels:
            for dev in [*ch.states, *ch.lights, *ch.shutters]:
                if dev not in devices and not has_mm:
                    issues.append(f"Canal « {ch.name} » : périphérique « {dev} » inexistant.")
            for dev, label in ch.states.items():
                spec = devices.get(dev)
                labels = (spec.options.get("labels") if spec else None) or (
                    [o.label for o in self.objectives] if dev == self.roles.objective_changer else None
                )
                if labels and label not in labels:
                    issues.append(f"Canal « {ch.name} » : « {label} » n'est pas une position de {dev}.")
            if ch.filter_set and ch.filter_set not in sets:
                issues.append(f"Canal « {ch.name} » : cube « {ch.filter_set} » inexistant.")
        return issues

    # utilitaires ---------------------------------------------------------------
    def objective(self, label: str) -> Objective | None:
        return next((o for o in self.objectives if o.label == label), None)

    def channel(self, name: str) -> ChannelPreset | None:
        return next((c for c in self.channels if c.name == name), None)

    # entrée / sortie -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> MicroscopeProfile:
        path = Path(path)
        data = load_yaml(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, text: str) -> MicroscopeProfile:
        return cls.model_validate(load_yaml(text) or {})

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_defaults=True)
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")


def user_profile_dir() -> Path:
    """Dossier des profils de l'utilisateur (calibrations enregistrées, profils perso).

    ``~/.fuji/profiles`` par défaut, modifiable avec la variable ``FUJI_HOME``.
    """
    home = Path(os.environ.get("FUJI_HOME", Path.home() / ".fuji")).expanduser()
    return home / "profiles"


def builtin_profiles() -> dict[str, Path]:
    """Profils fournis avec Fuji (``demo``, ``leitz-diaplan-touptek``...)."""
    folder = resources.files("fuji") / "profiles"
    out: dict[str, Path] = {}
    for entry in folder.iterdir():
        if entry.name.endswith((".yaml", ".yml")):
            out[entry.name.rsplit(".", 1)[0]] = Path(str(entry))
    return dict(sorted(out.items()))


def available_profiles() -> dict[str, Path]:
    """Profils utilisateur et profils fournis (l'utilisateur est prioritaire)."""
    out = builtin_profiles()
    folder = user_profile_dir()
    if folder.is_dir():
        for p in sorted(folder.glob("*.y*ml")):
            out[p.stem] = p
    return out


def is_builtin_profile(path: str | Path) -> bool:
    builtin_dir = Path(str(resources.files("fuji") / "profiles")).resolve()
    return Path(path).resolve().parent == builtin_dir


def resolve_profile(name_or_path: str | Path) -> Path:
    """Accepte un chemin de fichier ou le nom d'un profil (utilisateur ou fourni)."""
    path = Path(name_or_path).expanduser()
    if path.suffix in (".yaml", ".yml") and path.exists():
        return path
    profiles = available_profiles()
    key = str(name_or_path)
    if key in profiles:
        return profiles[key]
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Profil introuvable : {name_or_path!r}. Profils disponibles : {', '.join(profiles)}"
    )
