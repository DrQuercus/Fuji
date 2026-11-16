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

from . import DriverContext, register_driver


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


@register_driver("manual.state")
def _manual_state(spec: DeviceSpec, ctx: DriverContext) -> ManualStateDevice:
    opts = dict(spec.options)
    opts.pop("labels", None)
    return ManualStateDevice(spec.name, default_state_labels(spec, ctx), label=spec.label, **opts)


@register_driver("manual.focus")
def _manual_focus(spec: DeviceSpec, ctx: DriverContext) -> ManualFocusDrive:
    return ManualFocusDrive(spec.name, label=spec.label, **spec.options)


@register_driver("manual.xy_stage")
def _manual_xy(spec: DeviceSpec, ctx: DriverContext) -> ManualXYStage:
    return ManualXYStage(spec.name, label=spec.label, **spec.options)


@register_driver("manual.light")
def _manual_light(spec: DeviceSpec, ctx: DriverContext) -> ManualLightSource:
    return ManualLightSource(spec.name, label=spec.label, **spec.options)
