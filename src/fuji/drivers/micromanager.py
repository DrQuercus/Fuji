"""Compatibilité Micro-Manager (via pymmcore-plus).

Une seule entrée de profil charge toute une configuration Micro-Manager ; chaque
périphérique Micro-Manager devient un périphérique Fuji (caméra, platine XY,
focus, tourelles, obturateurs, autres). Les propriétés Micro-Manager apparaissent
dans le panneau « Propriétés », et les groupes de configuration (Channel,
Objective...) peuvent devenir des canaux Fuji.

Installation : ``pip install "fuji-microscopy[micromanager]"`` puis
``mmcore install`` (télécharge les adaptateurs Micro-Manager pour ta plateforme).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from fuji.core.devices import Camera, Device, FocusDrive, Shutter, StateDevice, XYStage
from fuji.core.profile import ChannelPreset, DeviceSpec
from fuji.core.properties import PropertyInfo, PropertyType

from . import DriverContext, register_driver

log = logging.getLogger(__name__)

# valeurs numériques des énumérations MMCore (stables depuis des années)
_CAMERA, _SHUTTER, _STATE, _STAGE, _XYSTAGE = 2, 3, 4, 5, 6
_MM_FLOAT, _MM_INT = 2, 3

_TRANSMISSION_WORDS = ("bf", "bright", "dic", "phase", "trans", "fond clair")
_COLOR_WORDS = {
    "dapi": "blue",
    "hoechst": "blue",
    "bfp": "blue",
    "fitc": "green",
    "gfp": "green",
    "alexa488": "green",
    "cy3": "magenta",
    "tritc": "magenta",
    "rfp": "magenta",
    "mcherry": "magenta",
    "cy5": "red",
    "far": "red",
}


def _require_pymmcore() -> Any:
    try:
        import pymmcore_plus
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise ImportError(
            "Micro-Manager n'est pas installé. Lance :\n"
            '    pip install "fuji-microscopy[micromanager]"\n'
            "    mmcore install"
        ) from exc
    return pymmcore_plus


class _MMMixin:
    """Partie commune : propriétés Micro-Manager → propriétés Fuji."""

    _core: Any
    _mm: str

    def _init_mm(self, core: Any, mm_label: str) -> None:
        self._core = core
        self._mm = mm_label
        for prop in core.getDevicePropertyNames(mm_label):
            try:
                self._register_mm_property(prop)
            except Exception:  # pragma: no cover - propriété exotique
                log.debug("Propriété %s.%s ignorée", mm_label, prop, exc_info=True)

    def _register_mm_property(self, prop: str) -> None:
        core, label = self._core, self._mm
        mm_type = int(core.getPropertyType(label, prop))
        allowed = tuple(core.getAllowedPropertyValues(label, prop))
        limits = None
        if core.hasPropertyLimits(label, prop):
            limits = (core.getPropertyLowerLimit(label, prop), core.getPropertyUpperLimit(label, prop))
        if allowed:
            ptype, conv = PropertyType.STRING, str
        elif mm_type == _MM_FLOAT:
            ptype, conv = PropertyType.FLOAT, float
        elif mm_type == _MM_INT:
            ptype, conv = PropertyType.INT, lambda s: int(float(s))
        else:
            ptype, conv = PropertyType.STRING, str
        info = PropertyInfo(
            name=prop,
            type=ptype,
            read_only=bool(core.isPropertyReadOnly(label, prop)),
            limits=limits if ptype is not PropertyType.STRING else None,
            choices=allowed or None,
            category="Micro-Manager",
            advanced=prop not in ("Exposure", "Binning", "Gain", "Label", "State", "PixelType"),
        )

        def getter() -> Any:
            raw = core.getProperty(label, prop)
            try:
                return conv(raw)
            except ValueError:
                return raw

        def setter(value: Any) -> None:
            core.setProperty(label, prop, value)

        self._add_property(info, getter, None if info.read_only else setter)  # type: ignore[attr-defined]

    def _mm_describe(self) -> dict[str, Any]:
        core, label = self._core, self._mm
        out: dict[str, Any] = {"mm_label": label}
        for key, fn in (
            ("library", "getDeviceLibrary"),
            ("adapter", "getDeviceName"),
            ("mm_description", "getDeviceDescription"),
        ):
            try:
                out[key] = getattr(core, fn)(label)
            except Exception:
                pass
        return out


class MMCamera(_MMMixin, Camera):
    def __init__(self, name: str, core: Any, mm_label: str, **kw: Any) -> None:
        Camera.__init__(self, name, **kw)
        self._init_mm(core, mm_label)
        self.supports_gain = "Gain" in core.getDevicePropertyNames(mm_label)

    def _select(self) -> None:
        if self._core.getCameraDevice() != self._mm:
            self._core.setCameraDevice(self._mm)

    def sensor_shape(self) -> tuple[int, int]:
        self._select()
        return int(self._core.getImageHeight()), int(self._core.getImageWidth())

    @property
    def is_color(self) -> bool:
        self._select()
        return int(self._core.getNumberOfComponents()) == 4

    @property
    def bit_depth(self) -> int:
        self._select()
        return int(self._core.getImageBitDepth())

    @property
    def binning(self) -> int:
        try:
            return int(str(self._core.getProperty(self._mm, "Binning")).split("x")[0])
        except Exception:
            return 1

    def configured_pixel_size_um(self) -> float | None:
        size = float(self._core.getPixelSizeUm())
        return size if size > 0 else None

    def get_exposure(self) -> float:
        return float(self._core.getExposure(self._mm))

    def set_exposure(self, exposure_ms: float) -> None:
        self._core.setExposure(self._mm, float(exposure_ms))

    def get_gain(self) -> float:
        return float(self._core.getProperty(self._mm, "Gain"))

    def set_gain(self, gain: float) -> None:
        self._core.setProperty(self._mm, "Gain", gain)

    def gain_limits(self) -> tuple[float, float]:
        core = self._core
        if core.hasPropertyLimits(self._mm, "Gain"):
            return (
                core.getPropertyLowerLimit(self._mm, "Gain"),
                core.getPropertyUpperLimit(self._mm, "Gain"),
            )
        return (0.0, 1000.0)

    def _snap(self) -> np.ndarray:
        self._select()
        self._core.snapImage()
        return np.asarray(self._core.getImage(fix=True))

    # live : acquisition en séquence continue de Micro-Manager
    def start_live(self) -> None:
        if self.is_live:
            return
        self._select()
        self._core.startContinuousSequenceAcquisition(0)
        self._live_stop.clear()
        self._live_thread = threading.Thread(target=self._mm_live_loop, daemon=True, name=f"live-{self.name}")
        self._live_thread.start()

    def _mm_live_loop(self) -> None:
        core = self._core
        while not self._live_stop.is_set():
            if core.getRemainingImageCount() > 0:
                img = None
                while core.getRemainingImageCount() > 0:
                    img = core.popNextImage(fix=True)
                if img is not None:
                    self._publish(np.asarray(img))
            else:
                time.sleep(0.005)

    def stop_live(self) -> None:
        if not self.is_live:
            return
        self._live_stop.set()
        if self._live_thread is not None:
            self._live_thread.join(timeout=5)
        self._live_thread = None
        if self._core.isSequenceRunning():
            self._core.stopSequenceAcquisition()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), **self._mm_describe()}


class MMXYStage(_MMMixin, XYStage):
    def __init__(self, name: str, core: Any, mm_label: str, **kw: Any) -> None:
        XYStage.__init__(self, name, **kw)
        self._init_mm(core, mm_label)

    def get_position(self) -> tuple[float, float]:
        x, y = self._core.getXYPosition(self._mm)
        return float(x), float(y)

    def _move_to(self, x: float, y: float) -> None:
        self._core.setXYPosition(self._mm, x, y)

    def is_busy(self) -> bool:
        return bool(self._core.deviceBusy(self._mm))

    def stop(self) -> None:
        self._core.stop(self._mm)

    def home(self) -> None:
        self._core.home(self._mm)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), **self._mm_describe()}


class MMFocus(_MMMixin, FocusDrive):
    def __init__(self, name: str, core: Any, mm_label: str, **kw: Any) -> None:
        FocusDrive.__init__(self, name, **kw)
        self._init_mm(core, mm_label)

    def get_position(self) -> float:
        return float(self._core.getPosition(self._mm))

    def _move_to(self, z: float) -> None:
        self._core.setPosition(self._mm, z)

    def is_busy(self) -> bool:
        return bool(self._core.deviceBusy(self._mm))

    def stop(self) -> None:
        self._core.stop(self._mm)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), **self._mm_describe()}


class MMState(_MMMixin, StateDevice):
    def __init__(self, name: str, core: Any, mm_label: str, **kw: Any) -> None:
        StateDevice.__init__(self, name, **kw)
        self._init_mm(core, mm_label)

    @property
    def labels(self) -> list[str]:
        return [str(x) for x in self._core.getStateLabels(self._mm)]

    def get_state(self) -> int:
        return int(self._core.getState(self._mm))

    def _set_state(self, index: int) -> None:
        self._core.setState(self._mm, index)
        self._core.waitForDevice(self._mm)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), **self._mm_describe()}


class MMShutter(_MMMixin, Shutter):
    def __init__(self, name: str, core: Any, mm_label: str, **kw: Any) -> None:
        Shutter.__init__(self, name, **kw)
        self._init_mm(core, mm_label)

    def is_open(self) -> bool:
        return bool(self._core.getShutterOpen(self._mm))

    def _set_open(self, open_: bool) -> None:
        self._core.setShutterOpen(self._mm, open_)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), **self._mm_describe()}


class MMGeneric(_MMMixin, Device):
    def __init__(self, name: str, core: Any, mm_label: str, **kw: Any) -> None:
        Device.__init__(self, name, **kw)
        self._init_mm(core, mm_label)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), **self._mm_describe()}


class MMCoreDevice(Device):
    """Le cœur Micro-Manager lui-même : groupes de configuration, déchargement."""

    def __init__(self, name: str, core: Any, **kw: Any) -> None:
        super().__init__(name, **kw)
        self.core = core
        for group in core.getAvailableConfigGroups():
            presets = tuple(core.getAvailableConfigs(group))
            if not presets:
                continue
            self._add_property(
                PropertyInfo(group, PropertyType.STRING, choices=presets, category="Groupes Micro-Manager"),
                lambda g=group: core.getCurrentConfig(g),
                lambda v, g=group: self.apply_config(g, v),
            )

    def apply_config(self, group: str, preset: str) -> None:
        self.core.setConfig(group, preset)
        self.core.waitForConfig(group, preset)

    def shutdown(self) -> None:
        try:
            self.core.unloadAllDevices()
        except Exception:  # pragma: no cover
            log.exception("Erreur au déchargement de Micro-Manager")
        super().shutdown()


_CLASSES = {_CAMERA: MMCamera, _XYSTAGE: MMXYStage, _STAGE: MMFocus, _STATE: MMState, _SHUTTER: MMShutter}


def _guess_channel(name: str) -> tuple[str, str]:
    low = name.lower()
    kind = "transmission" if any(w in low for w in _TRANSMISSION_WORDS) else "fluorescence"
    color = next((c for w, c in _COLOR_WORDS.items() if w in low), "gray")
    return kind, color


@register_driver("micromanager")
def _micromanager(spec: DeviceSpec, ctx: DriverContext) -> list[Device]:
    opts = dict(spec.options)
    core = opts.pop("core", None)
    prefix = str(opts.pop("prefix", ""))
    if core is None:
        pmp = _require_pymmcore()
        core = pmp.CMMCorePlus()
        adapter_path = opts.get("adapter_path")
        if adapter_path:
            core.setDeviceAdapterSearchPaths([str(Path(adapter_path).expanduser())])
        cfg = opts.get("config_file")
        cfg_path = Path(str(cfg)).expanduser() if cfg else None
        if cfg_path is not None and cfg_path.exists():
            core.loadSystemConfiguration(str(cfg_path))
        elif cfg in (None, "demo", "MMConfig_demo.cfg"):
            core.loadSystemConfiguration()  # configuration de démonstration de Micro-Manager
        else:
            raise FileNotFoundError(f"Configuration Micro-Manager introuvable : {cfg}")

    devices: list[Device] = [MMCoreDevice(spec.name, core, label=spec.label or "Micro-Manager")]
    for mm_label in core.getLoadedDevices():
        if mm_label == "Core":
            continue
        cls = _CLASSES.get(int(core.getDeviceType(mm_label)), MMGeneric)
        devices.append(cls(prefix + mm_label, core, mm_label))

    # rôles : ceux déclarés dans Micro-Manager, sauf si le profil les impose
    roles = ctx.profile.roles
    for role, getter in (
        ("camera", "getCameraDevice"),
        ("xy_stage", "getXYStageDevice"),
        ("focus", "getFocusDevice"),
    ):
        current = getattr(core, getter)()
        if current and getattr(roles, role) is None:
            setattr(roles, role, prefix + current)
    if roles.objective_changer is None and hasattr(core, "guessObjectiveDevices"):
        guesses = core.guessObjectiveDevices()
        if guesses:
            roles.objective_changer = prefix + guesses[0]

    # canaux : importer le groupe « Channel » si le profil n'en définit pas
    group = opts.get("channel_group", "Channel")
    if (
        opts.get("import_channels", True)
        and not ctx.profile.channels
        and group in core.getAvailableConfigGroups()
    ):
        for preset in core.getAvailableConfigs(group):
            kind, color = _guess_channel(preset)
            ctx.profile.channels.append(
                ChannelPreset(name=preset, kind=kind, color=color, mm_config=(group, preset))
            )
        if ctx.profile.channels and ctx.profile.default_channel is None:
            ctx.profile.default_channel = ctx.profile.channels[0].name
    return devices
