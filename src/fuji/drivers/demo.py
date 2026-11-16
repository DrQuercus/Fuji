"""Microscope simulé : pour développer et tester sans matériel.

La caméra de démonstration produit des images réalistes :

- échantillon : cellules colorées, micromètre objet (10 µm) ou lame vide ;
- défocalisation selon la position Z et l'ouverture numérique de l'objectif ;
- éclairage de Köhler simulé : diaphragme de champ (ouverture, netteté selon la
  hauteur du condenseur, centrage), vignettage, lampe décentrée ;
- fluorescence : si une source de fluorescence est allumée et la lampe de
  transmission éteinte, l'image devient une image de fluorescence ;
- bruit de photons et de lecture, saturation selon la profondeur de bits.

Les « molettes » du microscope (diaphragme, condenseur...) sont des propriétés de la
caméra (catégorie « Simulation ») : on peut ainsi s'entraîner avec l'assistant Köhler.
"""

from __future__ import annotations

import functools
import math
import time
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from fuji.core.devices import Camera, FocusDrive, LightSource, Shutter, StateDevice, XYStage
from fuji.core.profile import DeviceSpec, MicroscopeProfile
from fuji.core.properties import PropertyInfo, PropertyType

from . import LIGHT_TYPES, DriverContext, OptionSpec, register_driver
from .manual import default_state_labels

WORLD_RES_UM = 0.5  # résolution de l'échantillon virtuel (µm/pixel)
WORLD_SIZE = 2048  # 1 mm x 1 mm, répété à l'infini

SPECIMENS = ("cellules", "micromètre", "vide")

# absorbance (densité optique) par canal RGB : cytoplasme rose, noyaux violets
_CYTO_OD_RGB = np.array([0.15, 0.55, 0.35], dtype=np.float32)
_NUC_OD_RGB = np.array([0.75, 0.85, 0.35], dtype=np.float32)


# ------------------------------------------------------------------ échantillons
@functools.lru_cache(maxsize=4)
def _world(kind: str, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Deux cartes (cytoplasme, noyaux) en densité optique, ``WORLD_SIZE²``."""
    from skimage.draw import ellipse

    n = WORLD_SIZE
    cyto = np.zeros((n, n), np.float32)
    nuc = np.zeros((n, n), np.float32)
    if kind == "cellules":
        rng = np.random.default_rng(seed)
        for _ in range(700):
            r, c = rng.uniform(0, n, 2)
            a, b = rng.uniform(12, 30), rng.uniform(10, 22)  # demi-axes en pixels monde
            rot = rng.uniform(-math.pi, math.pi)
            rr, cc = ellipse(r, c, a, b, rotation=rot)
            cyto[rr % n, cc % n] += rng.uniform(0.25, 0.5)
            dr, dc = rng.normal(0, 2, 2)
            rr, cc = ellipse(r + dr, c + dc, a * 0.42, b * 0.42, rotation=rot)
            nuc[rr % n, cc % n] += rng.uniform(0.6, 1.0)
        texture = ndi.gaussian_filter(rng.normal(0, 1, (n, n)).astype(np.float32), 2, mode="wrap")
        cyto = ndi.gaussian_filter(cyto, 1.2, mode="wrap") * (1 + 0.25 * texture)
        nuc = ndi.gaussian_filter(nuc, 1.0, mode="wrap") * (1 + 0.15 * texture)
    elif kind == "micromètre":
        # traits tous les 10 µm (20 px monde), larges de 1.5 µm, sur une bande de 300 µm
        period = int(round(10 / WORLD_RES_UM))
        for i, x in enumerate(range(0, n, period)):
            length = 300 if i % 10 == 0 else (200 if i % 5 == 0 else 120)
            nuc[n // 2 - length : n // 2 + length, x : x + 3] = 1.6
        nuc[n // 2 - 2 : n // 2 + 2, :] = 1.6
        nuc = ndi.gaussian_filter(nuc, 0.7, mode="wrap")
    np.clip(cyto, 0, None, out=cyto)
    np.clip(nuc, 0, None, out=nuc)
    return cyto, nuc


def _regular_polygon(center: tuple[float, float], radius: float, n: int, rot: float) -> np.ndarray:
    ang = rot + np.arange(n) * 2 * math.pi / n
    return np.stack([center[0] + radius * np.sin(ang), center[1] + radius * np.cos(ang)], 1)


# ------------------------------------------------------------------ caméra
class DemoCamera(Camera):
    supports_gain = True
    supports_auto_white_balance = True
    supports_auto_exposure = True

    def __init__(
        self,
        name: str,
        *,
        profile: MicroscopeProfile | None = None,
        devices: dict[str, Any] | None = None,
        width: int = 1024,
        height: int = 768,
        color: bool = False,
        bit_depth: int = 12,
        sensor_pixel_size_um: float = 2.9,
        xy_stage: str | None = None,
        focus: str | None = None,
        objective_changer: str | None = None,
        transmitted_light: str | None = None,
        fluorescence: dict[str, dict[str, Any]] | None = None,
        focus_position_um: float = 0.0,
        specimen: str = "cellules",
        simulate_timing: bool = True,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._profile = profile
        self._devices = devices if devices is not None else {}
        self._shape = (int(height), int(width))
        self._color = bool(color)
        self._bits = int(bit_depth)
        self._sensor_px = float(sensor_pixel_size_um)
        self._refs = {
            "xy": xy_stage,
            "z": focus,
            "obj": objective_changer,
            "lamp": transmitted_light,
        }
        self._fluo = fluorescence or {}
        self._focus0 = float(focus_position_um)
        self._timing = simulate_timing
        self._rng = np.random.default_rng(seed)
        self._exposure = 10.0
        self._gain = 1.0
        self._wb = np.ones(3, np.float32)
        self._auto_exposure = False
        # état « mécanique » simulé (volontairement mal réglé au départ)
        self.sim = {
            "specimen": specimen if specimen in SPECIMENS else "cellules",
            "field_diaphragm": 100.0,
            "condenser_height_error_um": 350.0,
            "condenser_offset_x": 14.0,
            "condenser_offset_y": -9.0,
            "lamp_offset_x": 8.0,
            "lamp_offset_y": 0.0,
            "camera_rotation_deg": 1.5,
            "noise": True,
        }
        self._mask_cache: tuple[tuple, np.ndarray] | None = None
        self._register_props()

    # -- propriétés -----------------------------------------------------------
    def _register_props(self) -> None:
        sim = "Simulation"

        def prop(key: str, info: PropertyInfo) -> None:
            self._add_property(info, lambda: self.sim[key], lambda v: self.sim.__setitem__(key, v))

        prop("specimen", PropertyInfo("Échantillon", PropertyType.STRING, choices=SPECIMENS, category=sim))
        prop(
            "field_diaphragm",
            PropertyInfo("Diaphragme de champ", PropertyType.FLOAT, limits=(3, 100), unit="%", category=sim),
        )
        prop(
            "condenser_height_error_um",
            PropertyInfo(
                "Hauteur du condenseur (erreur)",
                PropertyType.FLOAT,
                limits=(-2000, 2000),
                unit="µm",
                category=sim,
                description="0 = condenseur au point",
            ),
        )
        prop(
            "condenser_offset_x",
            PropertyInfo(
                "Décentrage condenseur X", PropertyType.FLOAT, limits=(-60, 60), unit="%", category=sim
            ),
        )
        prop(
            "condenser_offset_y",
            PropertyInfo(
                "Décentrage condenseur Y", PropertyType.FLOAT, limits=(-60, 60), unit="%", category=sim
            ),
        )
        prop(
            "lamp_offset_x",
            PropertyInfo(
                "Décentrage lampe X",
                PropertyType.FLOAT,
                limits=(-40, 40),
                unit="%",
                category=sim,
                advanced=True,
            ),
        )
        prop(
            "lamp_offset_y",
            PropertyInfo(
                "Décentrage lampe Y",
                PropertyType.FLOAT,
                limits=(-40, 40),
                unit="%",
                category=sim,
                advanced=True,
            ),
        )
        prop(
            "camera_rotation_deg",
            PropertyInfo(
                "Rotation caméra", PropertyType.FLOAT, limits=(-15, 15), unit="°", category=sim, advanced=True
            ),
        )
        prop("noise", PropertyInfo("Bruit", PropertyType.BOOL, category=sim, advanced=True))
        self._add_property(
            PropertyInfo(
                "Profondeur de bits",
                PropertyType.STRING,
                choices=("8", "12", "16"),
                category="Image",
                advanced=True,
            ),
            lambda: str(self._bits),
            lambda v: setattr(self, "_bits", int(v)),
        )

    # -- interface Camera -------------------------------------------------------
    def sensor_shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def is_color(self) -> bool:
        return self._color

    @property
    def bit_depth(self) -> int:
        return self._bits

    @property
    def sensor_pixel_size_um(self) -> float:
        return self._sensor_px

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure_ms: float) -> None:
        lo, hi = self.exposure_limits()
        self._exposure = float(np.clip(exposure_ms, lo, hi))

    def exposure_limits(self) -> tuple[float, float]:
        return (0.05, 10_000.0)

    def get_gain(self) -> float:
        return self._gain

    def set_gain(self, gain: float) -> None:
        lo, hi = self.gain_limits()
        self._gain = float(np.clip(gain, lo, hi))

    def gain_limits(self) -> tuple[float, float]:
        return (1.0, 16.0)

    def auto_white_balance(self) -> None:
        """Égalise les canaux sur la zone la plus claire (fond de la lame)."""
        if not self._color:
            return
        self._wb[:] = 1
        img = self._render().astype(np.float32)
        bright = img.reshape(-1, 3)
        lum = bright.mean(1)
        sel = bright[lum >= np.percentile(lum, 90)]
        means = sel.mean(0) + 1e-6
        self._wb = (means.mean() / means).astype(np.float32)

    def set_auto_exposure(self, enabled: bool) -> None:
        self._auto_exposure = bool(enabled)

    def _snap(self) -> np.ndarray:
        t0 = time.monotonic()
        img = self._render()
        if self._auto_exposure:
            self._adjust_exposure(img)
        if self._timing:
            remaining = self._exposure / 1000 - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(min(remaining, 10.0))
        return img

    def _adjust_exposure(self, img: np.ndarray) -> None:
        maxval = 2**self._bits - 1
        p99 = float(np.percentile(img, 99.5))
        if p99 > 0:
            target = 0.7 * maxval
            self.set_exposure(self._exposure * float(np.clip(target / p99, 0.5, 2.0)))

    # -- simulation -------------------------------------------------------------
    def _noise(self, shape: tuple[int, ...]) -> np.ndarray:
        """Bruit gaussien tiré d'une réserve précalculée (bien plus rapide)."""
        n = int(np.prod(shape))
        pool = getattr(self, "_noise_pool", None)
        if pool is None or pool.size < 2 * n:
            pool = self._noise_pool = self._rng.standard_normal(2 * n + 4099, dtype=np.float32)
        start = int(self._rng.integers(0, pool.size - n))
        return pool[start : start + n].reshape(shape)

    def _dev(self, key: str) -> Any:
        name = self._refs.get(key)
        return self._devices.get(name) if name else None

    def _optics(self) -> tuple[float, float]:
        """(taille du pixel dans l'échantillon en µm, ouverture numérique)."""
        mag, na = 10.0, 0.25
        adapter = self._profile.camera_adapter_magnification if self._profile else 1.0
        changer = self._dev("obj")
        if changer is not None and self._profile is not None:
            obj = self._profile.objective(changer.get_label())
            if obj is not None:
                mag, na = obj.magnification, obj.na
        elif self._profile is not None and len(self._profile.objectives) == 1:
            mag, na = self._profile.objectives[0].magnification, self._profile.objectives[0].na
        return self._sensor_px / (mag * adapter), na

    def _illumination_mask(self, shape: tuple[int, int]) -> np.ndarray:
        s = self.sim
        key = (
            shape,
            s["field_diaphragm"],
            s["condenser_height_error_um"],
            s["condenser_offset_x"],
            s["condenser_offset_y"],
            s["lamp_offset_x"],
            s["lamp_offset_y"],
        )
        if self._mask_cache and self._mask_cache[0] == key:
            return self._mask_cache[1]
        from skimage.draw import polygon2mask

        h, w = shape
        half_diag = math.hypot(h, w) / 2
        center = (
            h / 2 + s["condenser_offset_y"] / 100 * h / 2,
            w / 2 + s["condenser_offset_x"] / 100 * w / 2,
        )
        radius = s["field_diaphragm"] / 100 * 1.25 * half_diag
        poly = _regular_polygon(center, radius, 10, 0.2)
        mask = polygon2mask(shape, poly).astype(np.float32)
        edge_sigma = 1.2 + abs(s["condenser_height_error_um"]) / 35
        mask = ndi.gaussian_filter(mask, edge_sigma, mode="nearest")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / half_diag**2
        vignette = 1 - 0.12 * r2
        tilt = (
            1
            + (s["lamp_offset_x"] / 100) * (xx - w / 2) / (w / 2) * 0.8
            + (s["lamp_offset_y"] / 100) * (yy - h / 2) / (h / 2) * 0.8
        )
        out = np.clip(mask * vignette * tilt, 0, None).astype(np.float32)
        self._mask_cache = (key, out)
        return out

    def _sample_world(
        self, shape: tuple[int, int], px_um: float, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        cyto_w, nuc_w = _world(self.sim["specimen"])
        h, w = shape
        x0, y0 = (0.0, 0.0)
        stage = self._dev("xy")
        if stage is not None:
            x0, y0 = stage.get_position()
        th = math.radians(self.sim["camera_rotation_deg"])
        co, si = math.cos(th), math.sin(th)
        k = px_um / WORLD_RES_UM
        matrix = np.array([[k * co, k * si], [-k * si, k * co]])
        center_world = np.array([y0 / WORLD_RES_UM + WORLD_SIZE / 2, x0 / WORLD_RES_UM + WORLD_SIZE / 2])
        offset = center_world - matrix @ np.array([h / 2, w / 2])
        out = []
        for m in (cyto_w, nuc_w):
            out.append(
                ndi.affine_transform(
                    m, matrix, offset, output_shape=shape, order=order, mode="grid-wrap", output=np.float32
                )
            )
        return out[0], out[1]

    def _render(self) -> np.ndarray:
        shape = self._shape
        px_um, na = self._optics()
        focus = self._dev("z")
        dz = (focus.get_position() - self._focus0) if focus is not None else 0.0
        # flou : diffraction (≈ 0.21 λ/ON, λ = 0.55 µm) + défocalisation
        sigma_um = math.hypot(abs(dz) * na * 0.5, 0.21 * 0.55 / na)
        sigma_px = min(sigma_um / px_um, 30.0)
        # si l'image est floue de toute façon, l'interpolation au plus proche suffit
        order = 0 if (sigma_px >= 1.5 and px_um < WORLD_RES_UM) else 1
        cyto, nuc = self._sample_world(shape, px_um, order)

        maxval = float(2**self._bits - 1)
        k = 0.6 * maxval / 10.0 * self._gain * self._exposure  # ~60 % de la dynamique à 10 ms
        lamp = self._dev("lamp")
        lamp_on = lamp is None or lamp.is_on()
        lamp_level = (lamp.get_intensity() / 100.0 if lamp is not None else 1.0) if lamp_on else 0.0
        active_fluo = [
            (self._devices[n], cfg)
            for n, cfg in self._fluo.items()
            if n in self._devices and self._devices[n].is_on()
        ]
        nch = 3 if self._color else 1

        if lamp_level > 0 or not active_fluo:
            # transmission (fond clair) ; flou appliqué aux cartes d'absorbance
            # (approximation suffisante pour un simulateur, 2 filtrages au lieu de 3)
            if sigma_px > 0.4:
                cyto = ndi.gaussian_filter(cyto, sigma_px, truncate=3.0)
                nuc = ndi.gaussian_filter(nuc, sigma_px, truncate=3.0)
            if self._color:
                od = cyto[..., None] * _CYTO_OD_RGB + nuc[..., None] * _NUC_OD_RGB
            else:
                od = (cyto * 0.45 + nuc * 0.9)[..., None]
            trans = np.exp(-od)
            illum = self._illumination_mask(shape)[..., None] * lamp_level
            signal = illum * trans * k
        else:
            # épifluorescence : fond noir, structures émettrices
            signal = np.zeros((*shape, nch), np.float32)
            for dev, cfg in active_fluo:
                struct = nuc if cfg.get("structure", "noyaux") == "noyaux" else cyto
                emission = np.asarray(cfg.get("emission_rgb", (0.2, 1.0, 0.3)), np.float32)
                weight = emission if self._color else np.array([emission.max()], np.float32)
                fl = struct * (dev.get_intensity() / 100.0) * float(cfg.get("brightness", 0.8))
                if sigma_px > 0.4:
                    fl = ndi.gaussian_filter(fl, sigma_px, truncate=3.0)
                signal += fl[..., None] * weight * k
            signal += 0.01 * k  # autofluorescence / fond

        if self._color:
            signal = signal * self._wb
        if self.sim["noise"]:
            noise_sd = np.sqrt(np.maximum(signal, 0) * 0.6 + 4.0)
            signal = signal + self._noise(signal.shape) * noise_sd
        signal += 0.004 * maxval  # offset noir
        img = np.clip(signal, 0, maxval).astype(np.uint8 if self._bits <= 8 else np.uint16)
        return img if self._color else img[..., 0]


# ----------------------------------------------------------- autres périphériques
class DemoXYStage(XYStage):
    def __init__(self, name: str, *, travel_um: float = 50_000.0, **kw: Any) -> None:
        super().__init__(name, **kw)
        self._pos = (0.0, 0.0)
        self._travel = travel_um

    def get_position(self) -> tuple[float, float]:
        return self._pos

    def _move_to(self, x: float, y: float) -> None:
        t = self._travel
        self._pos = (float(np.clip(x, -t, t)), float(np.clip(y, -t, t)))

    def limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        t = self._travel
        return ((-t, t), (-t, t))

    def home(self) -> None:
        self.move_to(0, 0)


class DemoFocus(FocusDrive):
    def __init__(self, name: str, *, initial_um: float = 0.0, **kw: Any) -> None:
        super().__init__(name, **kw)
        self._z = float(initial_um)

    def get_position(self) -> float:
        return self._z

    def _move_to(self, z: float) -> None:
        self._z = float(np.clip(z, -10_000, 10_000))


class DemoStateDevice(StateDevice):
    def __init__(self, name: str, labels: list[str], *, initial: int = 0, **kw: Any) -> None:
        super().__init__(name, **kw)
        self._labels = list(labels)
        self._state = int(initial)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def get_state(self) -> int:
        return self._state

    def _set_state(self, index: int) -> None:
        self._state = index


class DemoLightSource(LightSource):
    def __init__(
        self,
        name: str,
        *,
        wavelength_nm: float | None = None,
        on: bool = False,
        intensity: float = 100.0,
        source_type: str = "LED",
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        self.wavelength_nm = wavelength_nm
        self.source_type = source_type
        self._on = bool(on)
        self._intensity = float(intensity)

    def is_on(self) -> bool:
        return self._on

    def _set_on(self, on: bool) -> None:
        self._on = on

    def get_intensity(self) -> float:
        return self._intensity

    def _set_intensity(self, percent: float) -> None:
        self._intensity = percent


class DemoShutter(Shutter):
    def __init__(self, name: str, **kw: Any) -> None:
        super().__init__(name, **kw)
        self._open = False

    def is_open(self) -> bool:
        return self._open

    def _set_open(self, open_: bool) -> None:
        self._open = open_


# ------------------------------------------------------------------ fabriques
@register_driver(
    "demo.camera",
    title="Caméra simulée",
    category="Simulation",
    kind="camera",
    description="Caméra virtuelle qui simule un microscope complet (échantillon, Köhler, fluorescence).",
    options=[
        OptionSpec("color", "bool", "Couleur", True),
        OptionSpec("width", "int", "Largeur", 1024, min=64, max=8192, unit="px"),
        OptionSpec("height", "int", "Hauteur", 768, min=48, max=8192, unit="px"),
        OptionSpec("bit_depth", "int", "Profondeur", 12, min=8, max=16, unit="bits"),
        OptionSpec("sensor_pixel_size_um", "float", "Taille du photosite", 2.9, min=0.5, max=30, unit="µm"),
        OptionSpec("specimen", "choice", "Échantillon", "cellules", choices=SPECIMENS),
        OptionSpec(
            "xy_stage",
            "device",
            "Platine simulée",
            device_kind="xy_stage",
            advanced=True,
            help="Par défaut : le rôle « platine XY » du profil",
        ),
        OptionSpec("focus", "device", "Mise au point simulée", device_kind="focus", advanced=True),
        OptionSpec("objective_changer", "device", "Tourelle simulée", device_kind="state", advanced=True),
        OptionSpec("transmitted_light", "device", "Lampe simulée", device_kind="light_source", advanced=True),
        OptionSpec(
            "fluorescence",
            "mapping",
            "Fluorescence simulée",
            advanced=True,
            help="source → {structure: noyaux|cytoplasme, emission_rgb: [r, g, b]}",
        ),
        OptionSpec("focus_position_um", "float", "Plan net", 0.0, unit="µm", advanced=True),
        OptionSpec("simulate_timing", "bool", "Simuler le temps d'exposition", True, advanced=True),
    ],
)
def _demo_camera(spec: DeviceSpec, ctx: DriverContext) -> DemoCamera:
    opts = dict(spec.options)
    roles = ctx.profile.roles
    opts.setdefault("xy_stage", roles.xy_stage)
    opts.setdefault("focus", roles.focus)
    opts.setdefault("objective_changer", roles.objective_changer)
    opts.setdefault("transmitted_light", roles.transmitted_light)
    return DemoCamera(spec.name, label=spec.label, profile=ctx.profile, devices=ctx.devices, **opts)


@register_driver(
    "demo.xy_stage",
    title="Platine XY simulée",
    category="Simulation",
    kind="xy_stage",
    options=[OptionSpec("travel_um", "float", "Course", 50_000.0, min=100, unit="µm")],
)
def _demo_xy(spec: DeviceSpec, ctx: DriverContext) -> DemoXYStage:
    return DemoXYStage(spec.name, label=spec.label, **spec.options)


@register_driver(
    "demo.focus",
    title="Mise au point simulée",
    category="Simulation",
    kind="focus",
    options=[OptionSpec("initial_um", "float", "Position initiale", 0.0, unit="µm")],
)
def _demo_focus(spec: DeviceSpec, ctx: DriverContext) -> DemoFocus:
    return DemoFocus(spec.name, label=spec.label, **spec.options)


@register_driver(
    "demo.state",
    title="Tourelle / roue simulée",
    category="Simulation",
    kind="state",
    options=[
        OptionSpec(
            "labels",
            "list",
            "Positions",
            help="Laisser vide pour la tourelle d'objectifs (positions = objectifs)",
        ),
        OptionSpec("initial", "int", "Position initiale", 0, min=0),
    ],
)
def _demo_state(spec: DeviceSpec, ctx: DriverContext) -> DemoStateDevice:
    opts = dict(spec.options)
    opts.pop("labels", None)
    return DemoStateDevice(spec.name, default_state_labels(spec, ctx), label=spec.label, **opts)


@register_driver(
    "demo.light",
    title="Source de lumière simulée",
    category="Simulation",
    kind="light_source",
    options=[
        OptionSpec(
            "source_type",
            "choice",
            "Type",
            "LED",
            choices=LIGHT_TYPES,
        ),
        OptionSpec(
            "wavelength_nm",
            "float",
            "Longueur d'onde",
            None,
            min=200,
            max=1500,
            unit="nm",
            help="Pour une source de fluorescence",
        ),
        OptionSpec("on", "bool", "Allumée au démarrage", False),
        OptionSpec("intensity", "float", "Intensité", 100.0, min=0, max=100, unit="%"),
    ],
)
def _demo_light(spec: DeviceSpec, ctx: DriverContext) -> DemoLightSource:
    return DemoLightSource(spec.name, label=spec.label, **spec.options)


@register_driver("demo.shutter", title="Obturateur simulé", category="Simulation", kind="shutter")
def _demo_shutter(spec: DeviceSpec, ctx: DriverContext) -> DemoShutter:
    return DemoShutter(spec.name, label=spec.label, **spec.options)
