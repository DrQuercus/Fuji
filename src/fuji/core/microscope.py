"""Le microscope : assemble les périphériques décrits par un profil.

C'est le point d'entrée unique pour l'interface, les assistants et le moteur
d'acquisition. Exemple sans interface ::

    from fuji.core import Microscope

    with Microscope.from_file("demo") as scope:
        frame = scope.snap()
        print(frame.data.shape, frame.metadata.pixel_size_um)
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar

from psygnal import Signal

from fuji.drivers import DriverContext, create_devices

from .devices import (
    Camera,
    Device,
    FocusDrive,
    LightSource,
    Shutter,
    StateDevice,
    XYStage,
)
from .flatfield import FlatFieldCorrection
from .frame import Frame, FrameMetadata
from .manual import ManualActionHandler
from .profile import (
    ChannelPreset,
    MicroscopeProfile,
    Objective,
    is_builtin_profile,
    resolve_profile,
    user_profile_dir,
)

log = logging.getLogger(__name__)
D = TypeVar("D", bound=Device)

_OBJECTIVE_HINTS = ("objecti", "nosepiece", "revolver", "tourelle")


class Microscope:
    objective_changed = Signal(object)  # Objective | None
    channel_changed = Signal(object)  # str | None
    connection_changed = Signal(bool)
    profile_changed = Signal()

    def __init__(
        self,
        profile: MicroscopeProfile,
        *,
        profile_path: str | Path | None = None,
        manual_handler: ManualActionHandler | None = None,
    ) -> None:
        self.profile = profile
        self.profile_path = Path(profile_path) if profile_path else None
        self.devices: dict[str, Device] = {}
        self._manual_handler = manual_handler
        self._channel: str | None = profile.default_channel
        self._connected = False
        self.camera: Camera | None = None
        self.xy_stage: XYStage | None = None
        self.focus: FocusDrive | None = None
        self.objective_changer: StateDevice | None = None
        self.transmitted_light: LightSource | None = None
        #: correction du fond par l'image de référence, appliquée à la sortie de la caméra
        self.flatfield = FlatFieldCorrection(self)

    @classmethod
    def from_file(cls, name_or_path: str | Path, **kwargs: Any) -> Microscope:
        path = resolve_profile(name_or_path)
        return cls(MicroscopeProfile.load(path), profile_path=path, **kwargs)

    # ------------------------------------------------------------------ connexion
    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        ctx = DriverContext(self.profile, self.devices)
        try:
            for spec in self.profile.devices:
                created = create_devices(spec, ctx)
                for i, dev in enumerate(created):
                    try:
                        if dev.name in self.devices:
                            raise ValueError(f"Nom de périphérique en double : {dev.name!r}")
                        if not dev.initialized:
                            dev.initialize()
                    except Exception:
                        # ne pas laisser de matériel ouvert (caméra « occupée » au prochain essai)
                        for orphan in created[i:]:
                            with contextlib.suppress(Exception):
                                orphan.shutdown()
                        raise
                    self.devices[dev.name] = dev
            self._resolve_roles()
        except Exception:
            self.disconnect()
            raise
        if self._channel is None:  # un pilote a pu importer des canaux (Micro-Manager)
            self._channel = self.profile.default_channel
        self.manual_handler = self._manual_handler
        if self.objective_changer is not None:
            self.objective_changer.state_changed.connect(self._on_objective_state)
        self._connected = True
        self.connection_changed.emit(True)

    def disconnect(self) -> None:
        for dev in reversed(list(self.devices.values())):
            try:
                dev.shutdown()
            except Exception:  # pragma: no cover - dépend du matériel
                log.exception("Erreur à la fermeture de %s", dev.name)
        self.devices.clear()
        self.camera = self.xy_stage = self.focus = None
        self.objective_changer = self.transmitted_light = None
        was = self._connected
        self._connected = False
        if was:
            self.connection_changed.emit(False)

    def __enter__(self) -> Microscope:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    # ----------------------------------------------------------------- rôles
    def _resolve_roles(self) -> None:
        r = self.profile.roles
        self.camera = self._role(r.camera, Camera, auto=True)
        self.xy_stage = self._role(r.xy_stage, XYStage, auto=True)
        self.focus = self._role(r.focus, FocusDrive, auto=True)
        self.transmitted_light = self._role(r.transmitted_light, LightSource, auto=False)
        changer = self._role(r.objective_changer, StateDevice, auto=False)
        if changer is None:
            changer = self._guess_objective_changer()
        self.objective_changer = changer

    def _role(self, name: str | None, cls: type[D], *, auto: bool) -> D | None:
        if name:
            return self.device(name, cls)
        if auto:
            return next((d for d in self.devices.values() if isinstance(d, cls)), None)
        return None

    def _guess_objective_changer(self) -> StateDevice | None:
        labels = {o.label for o in self.profile.objectives}
        for dev in self.devices_of(StateDevice):
            text = f"{dev.name} {dev.label}".lower()
            if any(h in text for h in _OBJECTIVE_HINTS) or (labels and labels <= set(dev.labels)):
                return dev
        return None

    def device(self, name: str, cls: type[D] = Device) -> D:  # type: ignore[assignment]
        try:
            dev = self.devices[name]
        except KeyError:
            raise KeyError(f"Périphérique {name!r} inconnu. Disponibles : {list(self.devices)}") from None
        if not isinstance(dev, cls):
            raise TypeError(f"{name!r} est un {type(dev).__name__}, pas un {cls.__name__}")
        return dev

    def devices_of(self, cls: type[D]) -> list[D]:
        return [d for d in self.devices.values() if isinstance(d, cls)]

    def require_camera(self) -> Camera:
        if self.camera is None:
            raise RuntimeError("Aucune caméra dans ce profil de microscope")
        return self.camera

    # ---------------------------------------------------------- actions manuelles
    @property
    def manual_handler(self) -> ManualActionHandler | None:
        return self._manual_handler

    @manual_handler.setter
    def manual_handler(self, handler: ManualActionHandler | None) -> None:
        self._manual_handler = handler
        for dev in self.devices.values():
            if dev.manual:
                dev.action_handler = handler  # type: ignore[attr-defined]

    def stop_motion(self) -> None:
        """Arrêt d'urgence : interrompt tous les moteurs (platine, mise au point)."""
        for device in self.devices.values():
            stop = getattr(device, "stop", None)
            if callable(stop) and not device.manual:
                try:
                    stop()
                except Exception:  # un moteur injoignable ne doit pas empêcher d'arrêter les autres
                    log.exception("Arrêt de %s impossible", device.name)

    def capabilities(self) -> dict[str, bool]:
        """Ce que ce microscope sait faire : l'interface s'y adapte."""
        cam = self.camera
        return {
            "camera": cam is not None,
            "color_camera": bool(cam and cam.is_color),
            "xy_stage": self.xy_stage is not None,
            "xy_motorized": bool(self.xy_stage and not self.xy_stage.manual),
            "focus": self.focus is not None,
            "focus_motorized": bool(self.focus and not self.focus.manual),
            "objectives": bool(self.profile.objectives),
            "objectives_motorized": bool(self.objective_changer and not self.objective_changer.manual),
            "fluorescence": any(c.kind == "fluorescence" for c in self.profile.channels),
            "gain": bool(cam and cam.supports_gain),
            "auto_white_balance": bool(cam and cam.supports_auto_white_balance),
        }

    # -------------------------------------------------------------- objectifs
    @property
    def objective(self) -> Objective | None:
        objs = self.profile.objectives
        if self.objective_changer is not None:
            label = self.objective_changer.get_label()
            found = self.profile.objective(label)
            if found is not None:
                return found
            # Micro-Manager : libellés parfois différents (« 10X », « Nikon 10X S Fluor »)
            return next((o for o in objs if o.label.lower() in label.lower()), None)
        return objs[0] if len(objs) == 1 else None

    def set_objective(self, label: str, *, declare: bool = False) -> None:
        """Change d'objectif. ``declare=True`` : l'utilisateur l'a déjà fait à la main."""
        changer = self.objective_changer
        if changer is None:
            raise RuntimeError("Aucune tourelle d'objectifs déclarée dans le profil")
        if declare and changer.manual:
            changer.declare_state(label)  # type: ignore[attr-defined]
        else:
            changer.set_label(label)

    def _on_objective_state(self, *_: object) -> None:
        self.objective_changed.emit(self.objective)

    # -------------------------------------------------------------- taille du pixel
    def pixel_size_um(self) -> float | None:
        """Taille du pixel dans le plan de l'échantillon (µm), ou ``None`` si inconnue."""
        return self._pixel_size()[0]

    def pixel_size_source(self) -> str | None:
        """« calibré » (micromètre objet), « configuré » (Micro-Manager), « estimé »
        (photosite / grossissement) ou ``None``."""
        return self._pixel_size()[1]

    def _pixel_size(self) -> tuple[float | None, str | None]:
        cam, obj = self.camera, self.objective
        binning = cam.binning if cam else 1
        if obj is not None and obj.pixel_size_um:
            return obj.pixel_size_um * binning, "calibré"
        hint = getattr(cam, "configured_pixel_size_um", None)
        if callable(hint):
            value = hint()
            if value:
                return value, "configuré"
        if obj is not None and cam is not None and cam.sensor_pixel_size_um:
            total_mag = obj.magnification * self.profile.camera_adapter_magnification
            return cam.sensor_pixel_size_um * binning / total_mag, "estimé"
        return None, None

    def set_pixel_calibration(self, pixel_size_um: float, objective: str | None = None) -> None:
        """Enregistre une calibration mesurée (au binning actuel) pour un objectif."""
        obj = self.profile.objective(objective) if objective else self.objective
        if obj is None:
            raise RuntimeError("Objectif inconnu : sélectionnez d'abord l'objectif en place")
        binning = self.camera.binning if self.camera else 1
        obj.pixel_size_um = float(pixel_size_um) / binning
        self.profile_changed.emit()

    # ------------------------------------------------------------------ canaux
    @property
    def channel(self) -> str | None:
        return self._channel

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.profile.channels]

    @property
    def channel_kind(self) -> str | None:
        """Type du canal en place (« transmission », « fond noir », « fluorescence »…)."""
        ch = self.profile.channel(self._channel) if self._channel else None
        return ch.kind if ch is not None else None

    def channel_issue(self) -> str | None:
        """Ce qui empêche le canal en place de donner une bonne image avec l'objectif en
        place (fond noir avec un objectif trop ouvert), ou None."""
        ch = self.profile.channel(self._channel) if self._channel else None
        return ch.objective_issue(self.objective) if ch is not None else None

    def channel_preset(self, name: str) -> ChannelPreset:
        ch = self.profile.channel(name)
        if ch is None:
            raise KeyError(f"Canal inconnu {name!r}. Canaux : {self.channel_names}")
        return ch

    def apply_channel(self, name: str) -> None:
        """Règle tout le chemin optique pour un canal (cubes, lampes, exposition)."""
        ch = self.channel_preset(name)
        for dev_name, label in ch.states.items():
            dev = self.device(dev_name, StateDevice)
            if dev.get_label() != label:
                dev.set_label(label)
        if ch.mm_config is not None:
            self._apply_mm_config(*ch.mm_config)
        # éteindre les sources utilisées par les autres canaux
        others = dict.fromkeys(
            n for c in self.profile.channels if c.name != name for n in c.lights if n not in ch.lights
        )
        for dev_name in others:  # ordre du profil : consignes toujours dans le même ordre
            dev = self.device(dev_name, LightSource)
            if dev.is_on():
                dev.set_on(False)
        for dev_name, setting in ch.lights.items():
            dev = self.device(dev_name, LightSource)
            if setting.intensity is not None and dev.supports_intensity:
                if abs(dev.get_intensity() - setting.intensity) > 1e-6:
                    dev.set_intensity(setting.intensity)
            if dev.is_on() != setting.on:
                dev.set_on(setting.on)
        cam = self.camera
        if cam is not None:
            if ch.exposure_ms is not None:
                cam.set_exposure(ch.exposure_ms)
            if ch.gain is not None and cam.supports_gain:
                cam.set_gain(ch.gain)
        self._channel = name
        self.channel_changed.emit(name)

    def _apply_mm_config(self, group: str, preset: str) -> None:
        for dev in self.devices.values():
            apply = getattr(dev, "apply_config", None)
            if callable(apply):
                apply(group, preset)
                return
        raise RuntimeError(f"Préréglage Micro-Manager {group}/{preset} : aucun cœur Micro-Manager")

    @contextlib.contextmanager
    def exposure_shutters(self) -> Iterator[None]:
        """Ouvre les obturateurs du canal courant le temps d'une exposition."""
        names = self.channel_preset(self._channel).shutters if self._channel else []
        shutters = [self.device(n, Shutter) for n in names]
        opened = [s for s in shutters if not s.is_open()]
        for s in opened:
            s.set_open(True)
        try:
            yield
        finally:
            for s in opened:
                s.set_open(False)

    # ---------------------------------------------------------------- acquisition
    def snap(self) -> Frame:
        cam = self.require_camera()
        with self.exposure_shutters():
            data = cam.snap()
        return Frame(data, self.current_metadata())

    def current_metadata(self) -> FrameMetadata:
        cam, obj = self.camera, self.objective
        md = FrameMetadata(
            camera=cam.label if cam else None,
            exposure_ms=cam.get_exposure() if cam else None,
            gain=cam.get_gain() if cam and cam.supports_gain else None,
            bit_depth=cam.bit_depth if cam else None,
            binning=cam.binning if cam else 1,
            pixel_size_um=self.pixel_size_um(),
            objective=obj.label if obj else None,
            objective_na=obj.na if obj else None,
            channel=self._channel,
        )
        preset = self.profile.channel(self._channel) if self._channel else None
        if preset is not None:
            md.contrast = preset.kind
            md.channel_color = preset.color
        if self.xy_stage is not None:
            md.x_um, md.y_um = self.xy_stage.get_position()
        if self.focus is not None:
            md.z_um = self.focus.get_position()
        for dev in self.devices.values():
            if isinstance(dev, StateDevice):
                md.devices[dev.name] = dev.get_label()
            elif isinstance(dev, LightSource):
                md.devices[dev.name] = {"on": dev.is_on(), "intensity": dev.get_intensity()}
        md.extra["microscope"] = self.profile.name
        if self.flatfield.active:
            md.extra["flat_field_corrected"] = True
        return md

    # ------------------------------------------------------------------ profil
    def save_profile(self, path: str | Path | None = None) -> Path:
        """Enregistre le profil (calibrations comprises).

        Un profil fourni avec Fuji n'est jamais modifié : sa copie est enregistrée
        dans le dossier utilisateur (``~/.fuji/profiles``) et sera utilisée ensuite.
        """
        target = Path(path) if path else self.profile_path
        if target is None:
            raise ValueError("Aucun chemin de profil : précisez où enregistrer")
        if is_builtin_profile(target):
            target = user_profile_dir() / target.name
        target.parent.mkdir(parents=True, exist_ok=True)
        self.profile.save(target)
        self.profile_path = target
        return target
