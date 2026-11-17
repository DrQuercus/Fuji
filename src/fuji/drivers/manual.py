"""Pilotes « manuels » : éléments non motorisés que le logiciel connaît quand même.

Exemple de profil ::

    - name: objectifs
      driver: manual.state          # positions = objectifs du profil si non précisées
    - name: mise_au_point
      driver: manual.focus
      options: {um_per_division: 1.0}
    - name: lampe
      driver: manual.light
"""

from __future__ import annotations

from fuji.core.manual import ManualFocusDrive, ManualLightSource, ManualStateDevice, ManualXYStage
from fuji.core.profile import DeviceSpec

from . import LIGHT_TYPES, DriverContext, OptionSpec, register_driver


def default_state_labels(spec: DeviceSpec, ctx: DriverContext) -> list[str]:
    """Positions d'une tourelle : celles du profil, ou les objectifs s'il s'agit
    de la tourelle d'objectifs."""
    labels = spec.options.get("labels")
    if labels:
        return [str(x) for x in labels]
    if ctx.profile.roles.objective_changer == spec.name or spec.name.lower().startswith("obj"):
        labels = [o.label for o in ctx.profile.objectives]
    if not labels:
        raise ValueError(f"{spec.name} : précisez options.labels (liste des positions)")
    return labels


@register_driver(
    "manual.state",
    title="Tourelle / glissière manuelle",
    category="Tourelles, filtres et cubes",
    kind="state",
    description="Revolver d'objectifs, cubes de fluorescence, roue à filtres ou condenseur à "
    "tourelle réglés à la main. Fuji mémorise la position et donne des consignes.",
    options=[
        OptionSpec(
            "labels",
            "list",
            "Positions",
            help="Laisser vide pour le revolver d'objectifs (positions = objectifs)",
        ),
        OptionSpec("initial", "int", "Position au démarrage", 0, min=0),
        OptionSpec(
            "prompt",
            "str",
            "Consigne",
            "Mettez en place « {label} » ({device}).",
            advanced=True,
            help="{label} = position demandée, {device} = nom de l'élément",
        ),
    ],
)
def _manual_state(spec: DeviceSpec, ctx: DriverContext) -> ManualStateDevice:
    opts = dict(spec.options)
    opts.pop("labels", None)
    return ManualStateDevice(spec.name, default_state_labels(spec, ctx), label=spec.label, **opts)


@register_driver(
    "manual.focus",
    title="Mise au point manuelle",
    category="Platines et mise au point",
    kind="focus",
    description="Vis micrométrique : Fuji indique de combien la tourner (piles Z).",
    options=[
        OptionSpec("um_per_division", "float", "Valeur d'une graduation", None, min=0.01, max=100, unit="µm")
    ],
)
def _manual_focus(spec: DeviceSpec, ctx: DriverContext) -> ManualFocusDrive:
    return ManualFocusDrive(spec.name, label=spec.label, **spec.options)


@register_driver(
    "manual.xy_stage",
    title="Platine manuelle",
    category="Platines et mise au point",
    kind="xy_stage",
    description="Platine sans moteur : Fuji donne les positions à atteindre (multi-positions).",
)
def _manual_xy(spec: DeviceSpec, ctx: DriverContext) -> ManualXYStage:
    return ManualXYStage(spec.name, label=spec.label, **spec.options)


@register_driver(
    "manual.light",
    title="Lampe manuelle",
    category="Éclairage",
    kind="light_source",
    description="Halogène, lampe à mercure, LED... allumée et réglée à la main.",
    options=[
        OptionSpec(
            "source_type",
            "choice",
            "Type",
            "halogène",
            choices=LIGHT_TYPES,
        ),
        OptionSpec("wavelength_nm", "float", "Longueur d'onde", None, min=200, max=1500, unit="nm"),
        OptionSpec("initial_on", "bool", "Allumée au démarrage", True),
        OptionSpec("initial_intensity", "float", "Intensité au démarrage", 100.0, min=0, max=100, unit="%"),
    ],
)
def _manual_light(spec: DeviceSpec, ctx: DriverContext) -> ManualLightSource:
    return ManualLightSource(spec.name, label=spec.label, **spec.options)
