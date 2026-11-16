"""Interfaces abstraites des périphériques de microscope.

Toute l'application (interface, assistants, moteur d'acquisition) ne parle
**qu'à ces interfaces**, jamais directement à un SDK constructeur. Pour ajouter un
microscope, il suffit d'écrire (ou de réutiliser via Micro-Manager) un pilote qui
implémente ces classes.

Unités : positions en µm, temps d'exposition en ms, intensités en %.

Les signaux (``psygnal``) peuvent être émis depuis des threads matériels :
l'interface ne doit pas mettre à jour des widgets Qt directement depuis leurs
callbacks (elle interroge plutôt l'état avec un minuteur).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

import numpy as np
from psygnal import Signal

from .properties import PropertyInfo, PropertyRegistry


class DeviceKind(str, Enum):
    CAMERA = "camera"
    XY_STAGE = "xy_stage"
    FOCUS = "focus"
    STATE = "state"
    SHUTTER = "shutter"
    LIGHT_SOURCE = "light_source"
    GENERIC = "generic"


class Device(ABC):  # noqa: B024 - classe de base sans méthode abstraite obligatoire
    """Base commune à tous les périphériques."""

    kind: ClassVar[DeviceKind] = DeviceKind.GENERIC
    #: ``True`` pour un élément sans moteur, réglé à la main par l'utilisateur.
    manual: ClassVar[bool] = False

    property_changed = Signal(str, object)

    def __init__(self, name: str, *, label: str | None = None, description: str = "") -> None:
        self.name = name
        self.label = label or name
        self.description = description
        self._props = PropertyRegistry()
        self._initialized = False
        self._lock = threading.RLock()

    # cycle de vie ------------------------------------------------------------
    def initialize(self) -> None:
        """Ouvre la connexion au matériel."""
        self._initialized = True

    def shutdown(self) -> None:
        """Ferme la connexion au matériel."""
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    # propriétés génériques ---------------------------------------------------
    def property_infos(self) -> list[PropertyInfo]:
        return self._props.infos()

    def has_property(self, name: str) -> bool:
        return name in self._props

    def get_property(self, name: str) -> Any:
        return self._props.get(name)

    def set_property(self, name: str, value: Any) -> None:
        with self._lock:
            self._props.set(name, value)
        self.property_changed.emit(name, self._props.get(name))

    def _add_property(self, info: PropertyInfo, getter, setter=None) -> None:
        self._props.add(info, getter, setter)

    def describe(self) -> dict[str, Any]:
        """Résumé lisible (affiché dans l'interface et les métadonnées)."""
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind.value,
            "manual": self.manual,
            "driver": type(self).__name__,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


# --------------------------------------------------------------------------- caméra
class Camera(Device):
    """Caméra : capture d'images et flux live.

    Un pilote doit implémenter au minimum :meth:`_snap`, :meth:`get_exposure`,
    :meth:`set_exposure` et :meth:`sensor_shape`. Le flux live par défaut appelle
    ``_snap`` en boucle dans un thread ; un pilote qui dispose d'un vrai mode vidéo
    surcharge :meth:`start_live` / :meth:`stop_live` et publie les images avec
    :meth:`_publish`.
    """

    kind = DeviceKind.CAMERA
    #: Émis à chaque nouvelle image (depuis un thread d'acquisition).
    frame_ready = Signal(object)

    #: Cadence maximale du live logiciel (images/s).
    max_live_fps: float = 30.0

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._latest: np.ndarray | None = None
        self._latest_time = 0.0  # instant d'arrivée (time.monotonic) de la dernière image
        self._frame_count = 0
        self._frame_cond = threading.Condition()
        self._live_thread: threading.Thread | None = None
        self._live_stop = threading.Event()

    # --- à implémenter par les pilotes ---
    @abstractmethod
    def _snap(self) -> np.ndarray:
        """Acquiert une image (bloquant) et la renvoie."""

    @abstractmethod
    def get_exposure(self) -> float:
        """Temps d'exposition en ms."""

    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None: ...

    @abstractmethod
    def sensor_shape(self) -> tuple[int, int]:
        """Taille de l'image produite ``(hauteur, largeur)``."""

    # --- caractéristiques (valeurs par défaut prudentes) ---
    def exposure_limits(self) -> tuple[float, float]:
        return (0.01, 60_000.0)

    @property
    def is_color(self) -> bool:
        return False

    @property
    def bit_depth(self) -> int:
        return 8

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8 if self.bit_depth <= 8 else np.uint16)

    @property
    def sensor_pixel_size_um(self) -> float | None:
        """Taille physique d'un photosite (µm), si connue."""
        return None

    @property
    def binning(self) -> float:
        """Facteur par lequel le pixel image est plus grand que le photosite (binning
        ou résolution réduite du capteur). La calibration du pixel en tient compte."""
        return 1

    # gain (optionnel)
    supports_gain: ClassVar[bool] = False
    #: Unité du gain telle que la caméra l'exprime (« x », « % », « dB »...).
    gain_unit: str = "x"

    def get_gain(self) -> float:
        raise NotImplementedError

    def set_gain(self, gain: float) -> None:
        raise NotImplementedError

    def gain_limits(self) -> tuple[float, float]:
        return (1.0, 1.0)

    # automatismes (optionnels)
    supports_auto_white_balance: ClassVar[bool] = False
    supports_auto_exposure: ClassVar[bool] = False

    def auto_white_balance(self) -> None:
        raise NotImplementedError

    def set_auto_exposure(self, enabled: bool) -> None:
        raise NotImplementedError

    # --- acquisition ---
    def snap(self) -> np.ndarray:
        """Acquiert une image fraîche."""
        with self._lock:
            img = self._snap()
        self._publish(img)
        return img

    def auto_expose(self, target: float = 0.7, *, max_iterations: int = 8) -> float:
        """Règle l'exposition pour que les hautes lumières atteignent ``target`` (fraction
        de la dynamique). Renvoie l'exposition choisie (ms).

        Une image saturée ne dit pas de combien réduire, une image noire de combien
        augmenter : dans ces cas on corrige d'un facteur fixe, puis on vérifie sur une
        nouvelle image jusqu'à ce que la correction devienne négligeable.
        """
        maxval = 2**self.bit_depth - 1
        lo, hi = self.exposure_limits()
        changed = False
        for _ in range(max_iterations):
            if changed:
                self.snap()  # image encore prise avec l'ancien réglage sur certaines caméras
            frame = self.snap()
            p = float(np.percentile(frame[::4, ::4], 99.5))
            if p >= 0.98 * maxval:
                factor = 0.25
            elif p <= 0.02 * maxval:
                factor = 8.0
            else:
                factor = target * maxval / p
            current = self.get_exposure()
            new = float(np.clip(current * factor, lo, hi))
            if abs(new - current) <= 0.05 * current:
                break
            self.set_exposure(new)
            changed = True
        return self.get_exposure()

    def latest_frame(self) -> tuple[int, np.ndarray | None]:
        """Dernière image disponible et son numéro (pour le live sans copie)."""
        with self._frame_cond:
            return self._frame_count, self._latest

    def wait_for_frame(self, after: int, timeout: float = 10.0) -> np.ndarray:
        """Attend une image de numéro strictement supérieur à ``after``."""
        return self._wait_frame(after, timeout)[1]

    def _wait_frame(self, after: int, timeout: float) -> tuple[int, np.ndarray, float]:
        """Comme :meth:`wait_for_frame`, avec le numéro et l'instant d'arrivée de l'image."""
        deadline = time.monotonic() + timeout
        with self._frame_cond:
            while self._frame_count <= after:
                self._check_alive()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{self.name} : aucune image reçue en {timeout:.1f} s")
                self._frame_cond.wait(min(remaining, 0.25))
            assert self._latest is not None
            return self._frame_count, self._latest, self._latest_time

    def _check_alive(self) -> None:
        """Surchargée par les pilotes : lève une erreur claire si la caméra est perdue."""

    def _publish(self, img: np.ndarray) -> None:
        with self._frame_cond:
            self._latest = img
            self._latest_time = time.monotonic()
            self._frame_count += 1
            self._frame_cond.notify_all()
        self.frame_ready.emit(img)

    # --- live ---
    @property
    def is_live(self) -> bool:
        return self._live_thread is not None and self._live_thread.is_alive()

    def start_live(self) -> None:
        if self.is_live:
            return
        self._live_stop.clear()
        self._live_thread = threading.Thread(
            target=self._software_live_loop, name=f"live-{self.name}", daemon=True
        )
        self._live_thread.start()

    def stop_live(self) -> None:
        self._live_stop.set()
        thread = self._live_thread
        if thread is not None:
            thread.join(timeout=max(2.0, self.get_exposure() / 1000 * 2 + 1.0))
            if thread.is_alive():  # exposition très longue : on garde la référence
                return
        self._live_thread = None

    def _software_live_loop(self) -> None:
        min_period = 1.0 / self.max_live_fps
        while not self._live_stop.is_set():
            t0 = time.monotonic()
            try:
                self.snap()
            except Exception:  # pragma: no cover - dépend du matériel
                time.sleep(0.2)
                continue
            dt = time.monotonic() - t0
            if dt < min_period:
                self._live_stop.wait(min_period - dt)

    def shutdown(self) -> None:
        self.stop_live()
        super().shutdown()

    def describe(self) -> dict[str, Any]:
        h, w = self.sensor_shape()
        return {
            **super().describe(),
            "width": w,
            "height": h,
            "color": self.is_color,
            "bit_depth": self.bit_depth,
            "sensor_pixel_size_um": self.sensor_pixel_size_um,
        }


# ------------------------------------------------------------------------ platines
class _Positioner(Device):
    def is_busy(self) -> bool:
        return False

    def wait(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while self.is_busy():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.name} : mouvement non terminé après {timeout} s")
            time.sleep(0.01)

    def stop(self) -> None:
        """Arrêt d'urgence (si supporté)."""

    def home(self) -> None:
        raise NotImplementedError(f"{self.name} ne sait pas faire de prise d'origine")


class XYStage(_Positioner):
    """Platine XY motorisée (positions en µm)."""

    kind = DeviceKind.XY_STAGE
    position_changed = Signal(float, float)

    @abstractmethod
    def get_position(self) -> tuple[float, float]: ...

    @abstractmethod
    def _move_to(self, x: float, y: float) -> None: ...

    def move_to(self, x: float, y: float, *, wait: bool = True) -> None:
        self._move_to(float(x), float(y))
        if wait:
            self.wait()
        self.position_changed.emit(*self.get_position())

    def move_by(self, dx: float, dy: float, *, wait: bool = True) -> None:
        x, y = self.get_position()
        self.move_to(x + dx, y + dy, wait=wait)

    def limits(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return None


class FocusDrive(_Positioner):
    """Mise au point motorisée (axe Z, en µm)."""

    kind = DeviceKind.FOCUS
    position_changed = Signal(float)

    @abstractmethod
    def get_position(self) -> float: ...

    @abstractmethod
    def _move_to(self, z: float) -> None: ...

    def move_to(self, z: float, *, wait: bool = True) -> None:
        self._move_to(float(z))
        if wait:
            self.wait()
        self.position_changed.emit(self.get_position())

    def move_by(self, dz: float, *, wait: bool = True) -> None:
        self.move_to(self.get_position() + dz, wait=wait)


# ----------------------------------------------------------- tourelles, roues, cubes
class StateDevice(Device):
    """Périphérique à positions discrètes : tourelle d'objectifs, roue à filtres,
    tourelle de cubes de fluorescence, condenseur à tourelle..."""

    kind = DeviceKind.STATE
    state_changed = Signal(int, str)

    @property
    @abstractmethod
    def labels(self) -> list[str]: ...

    @abstractmethod
    def get_state(self) -> int: ...

    @abstractmethod
    def _set_state(self, index: int) -> None: ...

    def _check_index(self, index: int) -> int:
        index = int(index)
        if not 0 <= index < len(self.labels):
            raise IndexError(f"{self.name} : position {index} invalide (0..{len(self.labels) - 1})")
        return index

    def set_state(self, index: int) -> None:
        index = self._check_index(index)
        self._set_state(index)
        self.state_changed.emit(index, self.labels[index])

    def get_label(self) -> str:
        return self.labels[self.get_state()]

    def set_label(self, label: str) -> None:
        self.set_state(self.index_of(label))

    def index_of(self, label_or_index: str | int) -> int:
        if isinstance(label_or_index, int):
            return label_or_index
        try:
            return self.labels.index(label_or_index)
        except ValueError:
            raise KeyError(
                f"{self.name} : position {label_or_index!r} inconnue ; positions : {self.labels}"
            ) from None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "labels": self.labels}


class Shutter(Device):
    kind = DeviceKind.SHUTTER
    shutter_changed = Signal(bool)

    @abstractmethod
    def is_open(self) -> bool: ...

    @abstractmethod
    def _set_open(self, open_: bool) -> None: ...

    def set_open(self, open_: bool) -> None:
        self._set_open(bool(open_))
        self.shutter_changed.emit(self.is_open())


class LightSource(Device):
    """Source de lumière : halogène, LED de transmission, LED/lampe de fluorescence."""

    kind = DeviceKind.LIGHT_SOURCE
    light_changed = Signal(bool, float)

    supports_intensity: ClassVar[bool] = True
    #: Longueur d'onde centrale (nm) pour une source de fluorescence, sinon ``None``.
    wavelength_nm: float | None = None
    #: Type de source (halogène, LED, mercure...), informatif.
    source_type: str = ""

    @abstractmethod
    def is_on(self) -> bool: ...

    @abstractmethod
    def _set_on(self, on: bool) -> None: ...

    def get_intensity(self) -> float:
        return 100.0

    def _set_intensity(self, percent: float) -> None:
        raise NotImplementedError

    def set_on(self, on: bool) -> None:
        self._set_on(bool(on))
        self.light_changed.emit(self.is_on(), self.get_intensity())

    def set_intensity(self, percent: float) -> None:
        percent = float(np.clip(percent, 0.0, 100.0))
        self._set_intensity(percent)
        self.light_changed.emit(self.is_on(), self.get_intensity())
