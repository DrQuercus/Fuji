"""Opérations sur un profil en cours d'édition (dictionnaire), sans interface.

L'éditeur travaille sur ``profile.model_dump(mode="json")`` : ces fonctions gardent
les références cohérentes (rôles, canaux, options) quand on renomme ou supprime
un périphérique.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from fuji.core.profile import MicroscopeProfile, is_builtin_profile, user_profile_dir
from fuji.drivers import driver_info

ROLE_KINDS = {
    "camera": "camera",
    "xy_stage": "xy_stage",
    "focus": "focus",
    "objective_changer": "state",
    "transmitted_light": "light_source",
}
ROLE_TITLES = {
    "camera": "Caméra",
    "xy_stage": "Platine XY",
    "focus": "Mise au point",
    "objective_changer": "Revolver d'objectifs",
    "transmitted_light": "Lampe de transmission",
}
KIND_ICONS = {
    "camera": "camera",
    "xy_stage": "move",
    "focus": "updown",
    "state": "turret",
    "light_source": "bulb",
    "shutter": "shutter",
    None: "chip",
}


def empty_profile(name: str = "Nouveau microscope") -> dict[str, Any]:
    return MicroscopeProfile(name=name).model_dump(mode="json")


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Complète un dictionnaire partiel avec toutes les clés du modèle."""
    return MicroscopeProfile.model_validate(data).model_dump(mode="json")


def slug(text: str) -> str:
    s = re.sub(r"[^\w]+", "_", text.strip().lower(), flags=re.UNICODE).strip("_")
    return s or "peripherique"


def devices(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.setdefault("devices", [])


def device(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((d for d in devices(data) if d["name"] == name), None)


def device_kind(data: dict[str, Any], name: str) -> str | None:
    d = device(data, name)
    info = driver_info(d["driver"]) if d else None
    return info.kind if info else None


def provided_devices(d: dict[str, Any]) -> list[tuple[str, str]]:
    """Périphériques supplémentaires créés par une entrée (ex. « moteurs_xy » pour « moteurs »)."""
    info = driver_info(d["driver"])
    return info.provided(d["name"], d.get("options") or {}) if info is not None else []


def devices_of_kind(data: dict[str, Any], kind: str) -> list[str]:
    names = []
    for d in devices(data):
        if (driver_info(d["driver"]) or _NO).kind == kind:
            names.append(d["name"])
        names += [name for name, k in provided_devices(d) if k == kind]
    return names


class _NoInfo:
    kind = None


_NO = _NoInfo()


def has_micromanager(data: dict[str, Any]) -> bool:
    return any(d["driver"] == "micromanager" for d in devices(data))


def state_labels(data: dict[str, Any], name: str) -> list[str]:
    """Positions connues d'un périphérique à états (vide si inconnues, ex. Micro-Manager)."""
    d = device(data, name)
    if d is None:
        return []
    labels = (d.get("options") or {}).get("labels")
    if labels:
        return [str(x) for x in labels]
    roles = data.get("roles") or {}
    if roles.get("objective_changer") == name or name.lower().startswith("obj"):
        return [o["label"] for o in data.get("objectives", [])]
    return []


def unique_name(data: dict[str, Any], base: str) -> str:
    base = slug(base)
    names = {d["name"] for d in devices(data)}
    if base not in names:
        return base
    i = 2
    while f"{base}_{i}" in names:
        i += 1
    return f"{base}_{i}"


def add_device(
    data: dict[str, Any], driver: str, *, name: str | None = None, label: str | None = None
) -> dict[str, Any]:
    info = driver_info(driver)
    if info is None:
        raise KeyError(driver)
    base = name or {
        "camera": "camera",
        "xy_stage": "platine",
        "focus": "mise_au_point",
        "state": "tourelle",
        "light_source": "lampe",
        "shutter": "obturateur",
    }.get(info.kind or "", "systeme")
    dev = {
        "name": unique_name(data, base),
        "driver": driver,
        "label": label or info.title,
        "options": copy.deepcopy(info.defaults()),
    }
    devices(data).append(dev)
    # rôle évident non attribué → on le donne au nouveau périphérique
    roles = data.setdefault("roles", {})
    for role, kind in ROLE_KINDS.items():
        if kind == info.kind and not roles.get(role) and role != "objective_changer":
            if role == "transmitted_light" and (info.options and dev["options"].get("wavelength_nm")):
                continue
            roles[role] = dev["name"]
            break
    return dev


def rename_device(data: dict[str, Any], old: str, new: str) -> None:
    if old == new:
        return
    if device(data, new) is not None:
        raise ValueError(f"Un périphérique s'appelle déjà « {new} »")
    d = device(data, old)
    if d is None:
        raise KeyError(old)
    before = [name for name, _ in provided_devices(d)]
    d["name"] = new
    # les périphériques créés par cette entrée changent de nom avec elle (moteurs_xy → platine_xy)
    renames = {old: new, **dict(zip(before, [name for name, _ in provided_devices(d)], strict=False))}
    roles = data.setdefault("roles", {})
    for role, value in list(roles.items()):
        if value in renames:
            roles[role] = renames[value]
    for ch in data.get("channels", []):
        for key in ("states", "lights"):
            mapping = ch.get(key) or {}
            if any(k in renames for k in mapping):
                ch[key] = {renames.get(k, k): v for k, v in mapping.items()}
        ch["shutters"] = [renames.get(s, s) for s in ch.get("shutters", [])]
    for other in devices(data):
        info = driver_info(other["driver"])
        if info is None:
            continue
        for opt in info.options:
            if opt.type == "device" and other.get("options", {}).get(opt.name) in renames:
                other["options"][opt.name] = renames[other["options"][opt.name]]
        fluo = (other.get("options") or {}).get("fluorescence")
        if isinstance(fluo, dict) and old in fluo:
            other["options"]["fluorescence"] = {(new if k == old else k): v for k, v in fluo.items()}


def remove_device(data: dict[str, Any], name: str) -> None:
    data["devices"] = [d for d in devices(data) if d["name"] != name]
    roles = data.setdefault("roles", {})
    for role, value in list(roles.items()):
        if value == name:
            roles[role] = None
    for ch in data.get("channels", []):
        for key in ("states", "lights"):
            (ch.get(key) or {}).pop(name, None)
        ch["shutters"] = [s for s in ch.get("shutters", []) if s != name]
    for other in devices(data):
        info = driver_info(other["driver"])
        for opt in info.options if info else ():
            if opt.type == "device" and other.get("options", {}).get(opt.name) == name:
                other["options"].pop(opt.name, None)


def rename_channel(data: dict[str, Any], old: str, new: str) -> None:
    for ch in data.get("channels", []):
        if ch["name"] == old:
            ch["name"] = new
    if data.get("default_channel") == old:
        data["default_channel"] = new


def remove_channel(data: dict[str, Any], name: str) -> None:
    data["channels"] = [c for c in data.get("channels", []) if c["name"] != name]
    if data.get("default_channel") == name:
        data["default_channel"] = data["channels"][0]["name"] if data["channels"] else None


def rename_filter_set(data: dict[str, Any], old: str, new: str) -> None:
    for ch in data.get("channels", []):
        if ch.get("filter_set") == old:
            ch["filter_set"] = new
    for d in devices(data):
        labels = (d.get("options") or {}).get("labels")
        if labels and old in labels:
            d["options"]["labels"] = [new if x == old else x for x in labels]
            for ch in data.get("channels", []):
                if (ch.get("states") or {}).get(d["name"]) == old:
                    ch["states"][d["name"]] = new


def setup_fluorescence(data: dict[str, Any]) -> list[str]:
    """Crée (ou met à jour) la tourelle de cubes et un canal par cube. Renvoie les actions."""
    from fuji.core.catalog import color_for_wavelength

    done: list[str] = []
    sets = data.get("filter_sets", [])
    if not sets:
        return done
    cubes = next((d for d in devices(data) if d["name"] == "cubes"), None)
    labels = ["Vide"] + [f["name"] for f in sets]
    if cubes is None:
        cubes = {
            "name": "cubes",
            "label": "Cubes de fluorescence",
            "driver": "manual.state",
            "options": {"labels": labels},
        }
        devices(data).append(cubes)
        done.append("Tourelle de cubes « cubes » créée (manuelle).")
    else:
        old = list(cubes.setdefault("options", {}).get("labels") or [])
        merged = old + [lbl for lbl in labels if lbl not in old]
        if merged != old:
            cubes["options"]["labels"] = merged
            done.append("Positions de la tourelle de cubes complétées.")
    lights = [d for d in devices(data) if (driver_info(d["driver"]) or _NO).kind == "light_source"]
    transmitted = (data.get("roles") or {}).get("transmitted_light")
    fluo_lamp = next((d["name"] for d in lights if d["name"] != transmitted), None)
    if fluo_lamp is None:
        lamp = {
            "name": unique_name(data, "lampe_fluo"),
            "label": "Lampe de fluorescence",
            "driver": "manual.light",
            "options": {"source_type": "mercure (HBO)", "initial_on": False},
        }
        devices(data).append(lamp)
        fluo_lamp = lamp["name"]
        done.append(f"Lampe de fluorescence « {fluo_lamp} » créée (manuelle).")
    channels = data.setdefault("channels", [])
    used = {c.get("filter_set") for c in channels}
    for fs in sets:
        if fs["name"] in used:
            continue
        name = fs["name"].split("/")[0].strip() or fs["name"]
        if any(c["name"] == name for c in channels):
            name = fs["name"]
        channels.append(
            {
                "name": name,
                "kind": "fluorescence",
                "exposure_ms": 200.0,
                "states": {"cubes": fs["name"]},
                "lights": {fluo_lamp: {"on": True}},
                "color": color_for_wavelength(fs.get("emission_nm")),
                "filter_set": fs["name"],
                "excitation_nm": fs.get("excitation_nm"),
                "emission_nm": fs.get("emission_nm"),
            }
        )
        done.append(f"Canal « {name} » créé.")
    for ch in channels:
        if ch.get("kind") == "transmission":
            ch.setdefault("states", {}).setdefault("cubes", "Vide")
            if transmitted:
                ch.setdefault("lights", {}).setdefault(transmitted, {"on": True})
    return done


DARK_FIELD = "Fond noir"


def _condenser(data: dict[str, Any]) -> dict[str, Any] | None:
    """Le périphérique « condenseur » (tourelle, porte-diaphragme), s'il y en a un."""
    for d in devices(data):
        text = f"{d.get('name', '')} {d.get('label', '')}".lower()
        if "condens" in text and (driver_info(d["driver"]) or _NO).kind == "state":
            return d
    return None


def setup_dark_field(data: dict[str, Any]) -> list[str]:
    """Prépare le fond noir : un canal « Fond noir » (lampe à fond, exposition plus longue)
    et, pour le réglage du condenseur, une tourelle à deux positions. Renvoie ce qui a été
    fait (rien si le canal existe déjà)."""
    done: list[str] = []
    channels = data.setdefault("channels", [])
    if any(c.get("kind") == "fond noir" for c in channels):
        return done
    condenser = _condenser(data)
    if condenser is None:
        condenser = {
            "name": unique_name(data, "condenseur"),
            "label": "Condenseur",
            "driver": "manual.state",
            "options": {"labels": ["Fond clair", DARK_FIELD]},
        }
        devices(data).append(condenser)
        done.append(f"Condenseur « {condenser['name']} » créé (manuel : Fuji vous dira quand le tourner).")
    labels = list((condenser.setdefault("options", {}).get("labels")) or [])
    dark = next((lbl for lbl in labels if "noir" in lbl.lower() or "dark" in lbl.lower()), None)
    if dark is None and labels:
        dark = DARK_FIELD
        condenser["options"]["labels"] = [*labels, dark]
        done.append(f"Position « {dark} » ajoutée au condenseur.")
    dark = dark or DARK_FIELD
    bright = next((lbl for lbl in labels if lbl != dark), None)
    name = DARK_FIELD if not any(c["name"] == DARK_FIELD for c in channels) else f"{DARK_FIELD} 2"
    channel: dict[str, Any] = {
        "name": name,
        "kind": "fond noir",
        "exposure_ms": 200.0,
        "states": {condenser["name"]: dark},
        "color": "gray",
    }
    for key in ("cubes",):  # fluorescence : le cube vide, comme en fond clair
        cubes = device(data, key)
        if cubes is not None and "Vide" in (state_labels(data, key) or []):
            channel["states"][key] = "Vide"
    lamp = (data.get("roles") or {}).get("transmitted_light")
    if lamp:
        channel["lights"] = {lamp: {"on": True, "intensity": 100.0}}
    channels.append(channel)
    done.append(f"Canal « {name} » créé : lampe à fond, exposition 200 ms, objectifs jusqu'à ON 0,65.")
    if bright:  # revenir en fond clair remet le condenseur en place
        for ch in channels:
            if ch.get("kind", "transmission") == "transmission":
                ch.setdefault("states", {}).setdefault(condenser["name"], bright)
    return done


def save_path_for(data: dict[str, Any], current: Path | None) -> Path:
    """Où enregistrer : le fichier utilisateur existant, sinon un nouveau fichier utilisateur."""
    if current is not None and not is_builtin_profile(current):
        return current
    folder = user_profile_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if current is not None:  # profil fourni : copie utilisateur du même nom
        return folder / current.name
    base = slug(data.get("name", "microscope")).replace("_", "-")
    path = folder / f"{base}.yaml"
    i = 2
    while path.exists():
        path = folder / f"{base}-{i}.yaml"
        i += 1
    return path
