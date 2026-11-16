"""Catalogue : objectifs courants, cubes de fluorescence et modèles de microscopes.

Sert à pré-remplir l'éditeur de configuration (on part d'un modèle puis on ajuste).
"""

from __future__ import annotations

from typing import Any

#: Objectifs courants (à adapter à la gravure réelle).
OBJECTIVES: list[dict[str, Any]] = [
    {"label": "2.5x/0.07", "magnification": 2.5, "na": 0.07},
    {"label": "4x/0.10", "magnification": 4, "na": 0.10, "correction": "achromat"},
    {"label": "4x/0.13 plan", "magnification": 4, "na": 0.13, "correction": "plan-achromat"},
    {"label": "10x/0.25", "magnification": 10, "na": 0.25, "correction": "achromat"},
    {"label": "10x/0.30 plan-fluor", "magnification": 10, "na": 0.30, "correction": "plan-fluor"},
    {"label": "10x/0.25 Ph1", "magnification": 10, "na": 0.25, "contrast": "phase", "phase_ring": "Ph1"},
    {"label": "20x/0.40", "magnification": 20, "na": 0.40, "correction": "plan-achromat"},
    {"label": "20x/0.50 plan-fluor", "magnification": 20, "na": 0.50, "correction": "plan-fluor"},
    {"label": "40x/0.65", "magnification": 40, "na": 0.65, "correction": "achromat"},
    {"label": "40x/0.75 plan-fluor", "magnification": 40, "na": 0.75, "correction": "plan-fluor"},
    {"label": "40x/0.65 Ph2", "magnification": 40, "na": 0.65, "contrast": "phase", "phase_ring": "Ph2"},
    {"label": "60x/0.85", "magnification": 60, "na": 0.85, "correction": "plan-achromat"},
    {
        "label": "60x/1.40 huile plan-apo",
        "magnification": 60,
        "na": 1.40,
        "immersion": "huile",
        "correction": "plan-apochromat",
    },
    {
        "label": "100x/1.25 huile",
        "magnification": 100,
        "na": 1.25,
        "immersion": "huile",
        "correction": "achromat",
    },
    {
        "label": "100x/1.30 huile fluor",
        "magnification": 100,
        "na": 1.30,
        "immersion": "huile",
        "correction": "fluorite (semi-apo)",
    },
]

#: Cubes de fluorescence courants. Les blocs Leitz (Ploemopak) correspondent au Diaplan.
FILTER_SETS: list[dict[str, Any]] = [
    {
        "name": "DAPI",
        "excitation_nm": 365,
        "excitation_bandwidth_nm": 50,
        "dichroic_nm": 400,
        "emission_nm": 460,
        "emission_bandwidth_nm": 50,
        "fluorophores": ["DAPI", "Hoechst"],
    },
    {
        "name": "GFP / FITC",
        "excitation_nm": 470,
        "excitation_bandwidth_nm": 40,
        "dichroic_nm": 495,
        "emission_nm": 525,
        "emission_bandwidth_nm": 50,
        "fluorophores": ["GFP", "FITC", "Alexa 488"],
    },
    {
        "name": "Cy3 / TRITC",
        "excitation_nm": 545,
        "excitation_bandwidth_nm": 25,
        "dichroic_nm": 565,
        "emission_nm": 605,
        "emission_bandwidth_nm": 70,
        "fluorophores": ["Cy3", "TRITC", "Alexa 555"],
    },
    {
        "name": "mCherry / Texas Red",
        "excitation_nm": 560,
        "excitation_bandwidth_nm": 40,
        "dichroic_nm": 585,
        "emission_nm": 630,
        "emission_bandwidth_nm": 75,
        "fluorophores": ["mCherry", "Texas Red", "Alexa 594"],
    },
    {
        "name": "Cy5",
        "excitation_nm": 620,
        "excitation_bandwidth_nm": 60,
        "dichroic_nm": 660,
        "emission_nm": 700,
        "emission_bandwidth_nm": 75,
        "fluorophores": ["Cy5", "Alexa 647"],
    },
    {
        "name": "Leitz A (UV)",
        "excitation_nm": 360,
        "excitation_bandwidth_nm": 40,
        "dichroic_nm": 400,
        "emission_nm": 430,
        "fluorophores": ["DAPI", "Hoechst"],
        "notes": "Bloc Leitz Ploemopak : BP 340–380, RKP 400, LP 430",
    },
    {
        "name": "Leitz I3 (bleu)",
        "excitation_nm": 470,
        "excitation_bandwidth_nm": 40,
        "dichroic_nm": 510,
        "emission_nm": 515,
        "fluorophores": ["FITC", "GFP"],
        "notes": "Bloc Leitz Ploemopak : BP 450–490, RKP 510, LP 515",
    },
    {
        "name": "Leitz N2.1 (vert)",
        "excitation_nm": 537,
        "excitation_bandwidth_nm": 45,
        "dichroic_nm": 580,
        "emission_nm": 590,
        "fluorophores": ["TRITC", "Cy3", "rhodamine"],
        "notes": "Bloc Leitz Ploemopak : BP 515–560, RKP 580, LP 590",
    },
]


def color_for_wavelength(nm: float | None) -> str:
    """Couleur d'affichage napari approximative pour une longueur d'onde d'émission."""
    if nm is None:
        return "gray"
    if nm < 480:
        return "blue"
    if nm < 500:
        return "cyan"
    if nm < 560:
        return "green"
    if nm < 590:
        return "yellow"
    if nm < 650:
        return "magenta"
    return "red"


def _objectives(*labels: str) -> list[dict[str, Any]]:
    by_label = {o["label"]: o for o in OBJECTIVES}
    return [dict(by_label[label]) for label in labels]


TEMPLATES: dict[str, dict[str, Any]] = {
    "manuel-touptek": {
        "title": "Microscope manuel + caméra ToupTek",
        "description": "Statif sans moteur (objectifs, lampe, mise au point à la main) et caméra "
        "ToupTek. Idéal pour un Leitz, Zeiss, Olympus... ancien.",
        "profile": {
            "name": "Mon microscope",
            "instrument": {"stand": "droit", "tube_length": "160 mm"},
            "devices": [
                {
                    "name": "camera",
                    "label": "Caméra ToupTek",
                    "driver": "touptek.camera",
                    "options": {"bit_depth": 12},
                },
                {"name": "objectifs", "label": "Revolver", "driver": "manual.state"},
                {
                    "name": "mise_au_point",
                    "label": "Mise au point fine",
                    "driver": "manual.focus",
                    "options": {"um_per_division": 1.0},
                },
                {
                    "name": "lampe",
                    "label": "Halogène",
                    "driver": "manual.light",
                    "options": {"source_type": "halogène"},
                },
            ],
            "roles": {
                "camera": "camera",
                "objective_changer": "objectifs",
                "focus": "mise_au_point",
                "transmitted_light": "lampe",
            },
            "objectives": _objectives("4x/0.10", "10x/0.25", "40x/0.65", "100x/1.25 huile"),
            "channels": [
                {
                    "name": "Fond clair",
                    "kind": "transmission",
                    "exposure_ms": 20,
                    "lights": {"lampe": {"on": True}},
                }
            ],
            "default_channel": "Fond clair",
        },
    },
    "manuel-fluo": {
        "title": "Microscope manuel avec fluorescence",
        "description": "Comme ci-dessus, avec une tourelle de cubes et une lampe de "
        "fluorescence (mercure) réglées à la main.",
        "profile": {
            "name": "Mon microscope (fluorescence)",
            "instrument": {"stand": "droit", "tube_length": "160 mm"},
            "devices": [
                {
                    "name": "camera",
                    "label": "Caméra",
                    "driver": "touptek.camera",
                    "options": {"bit_depth": 12},
                },
                {"name": "objectifs", "label": "Revolver", "driver": "manual.state"},
                {
                    "name": "cubes",
                    "label": "Cubes de fluorescence",
                    "driver": "manual.state",
                    "options": {"labels": ["Vide", "DAPI", "GFP / FITC", "Cy3 / TRITC"]},
                },
                {"name": "mise_au_point", "label": "Mise au point fine", "driver": "manual.focus"},
                {"name": "lampe", "label": "Halogène", "driver": "manual.light"},
                {
                    "name": "hbo",
                    "label": "Lampe à mercure",
                    "driver": "manual.light",
                    "options": {"source_type": "mercure (HBO)", "initial_on": False},
                },
            ],
            "roles": {
                "camera": "camera",
                "objective_changer": "objectifs",
                "focus": "mise_au_point",
                "transmitted_light": "lampe",
            },
            "objectives": _objectives("10x/0.25", "20x/0.40", "40x/0.65", "100x/1.25 huile"),
            "filter_sets": [f for f in FILTER_SETS if f["name"] in ("DAPI", "GFP / FITC", "Cy3 / TRITC")],
            "channels": [
                {
                    "name": "Fond clair",
                    "kind": "transmission",
                    "exposure_ms": 20,
                    "states": {"cubes": "Vide"},
                    "lights": {"lampe": {"on": True}},
                },
                {
                    "name": "DAPI",
                    "kind": "fluorescence",
                    "exposure_ms": 200,
                    "color": "blue",
                    "filter_set": "DAPI",
                    "states": {"cubes": "DAPI"},
                    "lights": {"hbo": {"on": True}},
                },
                {
                    "name": "GFP",
                    "kind": "fluorescence",
                    "exposure_ms": 300,
                    "color": "green",
                    "filter_set": "GFP / FITC",
                    "states": {"cubes": "GFP / FITC"},
                    "lights": {"hbo": {"on": True}},
                },
                {
                    "name": "Cy3",
                    "kind": "fluorescence",
                    "exposure_ms": 300,
                    "color": "magenta",
                    "filter_set": "Cy3 / TRITC",
                    "states": {"cubes": "Cy3 / TRITC"},
                    "lights": {"hbo": {"on": True}},
                },
            ],
            "default_channel": "Fond clair",
        },
    },
    "micromanager": {
        "title": "Microscope motorisé (Micro-Manager)",
        "description": "Tout le matériel est décrit par une configuration Micro-Manager (.cfg). "
        "Fuji détecte caméra, platines, tourelles et canaux.",
        "profile": {
            "name": "Microscope Micro-Manager",
            "instrument": {"stand": "inversé", "tube_length": "infini"},
            "devices": [
                {
                    "name": "mm",
                    "label": "Micro-Manager",
                    "driver": "micromanager",
                    "options": {"config_file": "MMConfig_demo.cfg"},
                }
            ],
            "objectives": _objectives("10x/0.30 plan-fluor", "20x/0.50 plan-fluor", "40x/0.75 plan-fluor"),
        },
    },
    "demo": {
        "title": "Microscope simulé",
        "description": "Tout est simulé : pour apprendre, tester les assistants ou développer.",
        "profile": None,  # copie du profil « demo » fourni
    },
    "vide": {
        "title": "Partir de zéro",
        "description": "Un profil vide : ajoutez vous-même chaque élément.",
        "profile": {"name": "Nouveau microscope"},
    },
}
