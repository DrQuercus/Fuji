"""Image de référence (« champ plat ») : le fond de toutes les images est corrigé.

On photographie une zone **vide** de la lame : cette image de référence contient tout ce qui
n'est pas l'échantillon (éclairage inégal, vignettage, poussières sur le capteur ou
l'optique). Chaque image de la caméra est ensuite divisée par elle :

    image corrigée = (image − noir) × gain,   gain = moyenne(référence − noir) / (référence − noir)

La correction se fait **à la sortie de la caméra** (:attr:`fuji.core.devices.Camera.frame_filter`),
comme dans les logiciels fournis avec les caméras : le live, les captures, les compositions
en direct et les acquisitions voient tous la même image corrigée.

- :class:`FlatField` — une référence prête à appliquer (rapide : ≈ 15 ms pour une image 4K) ;
- :class:`FlatFieldStore` — les références enregistrées, une par objectif et par canal
  (``~/.fuji/flatfield/<microscope>/``) ;
- :class:`FlatFieldCorrection` — ce qui est appliqué au microscope : activer, prendre une
  référence, suivre les changements d'objectif et de canal.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from psygnal import Signal

if TYPE_CHECKING:
    from .microscope import Microscope

log = logging.getLogger(__name__)

#: Amplification maximale d'un pixel : au-delà, la référence y est trop sombre pour être
#: fiable (hors du champ éclairé) et l'on amplifierait surtout du bruit.
MAX_GAIN = 8.0
#: Nombre d'images moyennées pour une référence (le bruit est divisé par 4).
DEFAULT_FRAMES = 16
_WORKERS = max(1, min(4, os.cpu_count() or 1))
_POOL = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="fuji-fond")


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "x"


def _bands(height: int, count: int) -> list[slice]:
    step = max(1, height // count)
    return [slice(i * step, height if i == count - 1 else (i + 1) * step) for i in range(count)]


def max_value_of(image: np.ndarray, bit_depth: int | None) -> float:
    """Plus grande valeur possible des pixels (pour repérer la saturation). Certaines
    caméras étalent leurs 12 bits sur 16 : on se fie alors aux valeurs trouvées."""
    if not np.issubdtype(image.dtype, np.integer):
        return float(2**bit_depth - 1) if bit_depth else float(np.max(image) or 1.0)
    top = float(np.iinfo(image.dtype).max)
    if bit_depth:
        nominal = float(2**bit_depth - 1)
        if float(image[::8, ::8].max()) <= nominal:
            return min(nominal, top)
    return top


# --------------------------------------------------------------------- référence
class FlatField:
    """Une image de référence prête à appliquer : un gain par pixel, et le noir à retirer."""

    def __init__(
        self, flat: np.ndarray, dark: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> None:
        signal = np.asarray(flat, dtype=np.float32)
        self.dark: np.ndarray | None = None
        if dark is not None and dark.shape == signal.shape:
            self.dark = np.asarray(dark, dtype=np.float32)
            signal = signal - self.dark
        # la moyenne de chaque canal est conservée : la balance des blancs ne change pas
        mean = signal.mean(axis=(0, 1), keepdims=True)
        usable = signal > 1e-6
        gain = np.ones(signal.shape, np.float32)
        np.divide(mean, signal, out=gain, where=usable)
        np.clip(gain, 0.0, MAX_GAIN, out=gain)
        self.gain = gain
        self.info = dict(info or {})
        self._resized: dict[tuple[int, ...], FlatField | None] = {}

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.gain.shape)

    def for_shape(self, shape: tuple[int, ...]) -> FlatField | None:
        """La référence adaptée à des images de taille ``shape`` : elle-même, ou une version
        regroupée / agrandie d'un facteur entier (binning de la caméra). ``None`` si le
        cadrage est différent."""
        shape = tuple(shape)
        if shape == self.shape:
            return self
        if shape not in self._resized:
            self._resized[shape] = self._resize(shape)
        return self._resized[shape]

    def _resize(self, shape: tuple[int, ...]) -> FlatField | None:
        mine = self.shape
        if len(shape) != len(mine) or shape[2:] != mine[2:]:
            return None
        (h, w), (mh, mw) = shape[:2], mine[:2]
        out = object.__new__(FlatField)
        out.info, out._resized = self.info, {}
        if mh % h == 0 and mw % w == 0 and mh // h == mw // w:  # image regroupée (binning)
            k = mh // h

            def shrink(a: np.ndarray) -> np.ndarray:
                return a.reshape(h, k, w, k, *a.shape[2:]).mean(axis=(1, 3), dtype=np.float32)

            out.gain = shrink(self.gain)
            out.dark = None if self.dark is None else shrink(self.dark)
            return out
        if h % mh == 0 and w % mw == 0 and h // mh == w // mw:  # référence prise en binning
            k = h // mh

            def grow(a: np.ndarray) -> np.ndarray:
                return np.repeat(np.repeat(a, k, axis=0), k, axis=1)

            out.gain = grow(self.gain)
            out.dark = None if self.dark is None else grow(self.dark)
            return out
        return None

    def apply(self, frame: np.ndarray, max_value: float | None = None) -> np.ndarray:
        """Image corrigée, du même type que ``frame`` (les entiers sont arrondis et bornés
        à ``max_value``, par défaut le maximum du type)."""
        if frame.shape != self.gain.shape:
            raise ValueError(f"image {frame.shape} et référence {self.gain.shape} de tailles différentes")
        integer = np.issubdtype(frame.dtype, np.integer)
        top = float(np.iinfo(frame.dtype).max) if integer else 0.0
        if integer and max_value is not None:
            top = min(top, float(max_value))
        out = np.empty(frame.shape, frame.dtype if integer else np.float32)
        gain, dark = self.gain, self.dark

        def work(rows: slice) -> None:
            part = frame[rows].astype(np.float32)
            if dark is not None:
                part -= dark[rows]
            part *= gain[rows]
            if integer:
                part += 0.5
                np.clip(part, 0.0, top, out=part)
            out[rows] = part

        if frame.size < 400_000 or _WORKERS == 1:
            work(slice(None))
        else:  # numpy libère le verrou de Python : les bandes sont traitées en parallèle
            list(_POOL.map(work, _bands(frame.shape[0], 2 * _WORKERS)))
        return out


# ---------------------------------------------------------------------- stockage
class FlatFieldStore:
    """Références enregistrées : un fichier par objectif et par canal."""

    def __init__(self, microscope_name: str, root: str | Path | None = None) -> None:
        home = Path(os.environ.get("FUJI_HOME", Path.home() / ".fuji")).expanduser()
        self.folder = Path(root) if root else home / "flatfield" / _slug(microscope_name)

    def path(self, objective: str | None, channel: str | None) -> Path:
        return self.folder / f"{_slug(objective or 'objectif')}__{_slug(channel or 'canal')}.npz"

    def exists(self, objective: str | None, channel: str | None) -> bool:
        return self.path(objective, channel).exists()

    def save(
        self,
        objective: str | None,
        channel: str | None,
        flat: np.ndarray,
        dark: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> Path:
        """Enregistre la référence (16 bits, sans compression : rapide même en 4K)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(objective, channel)
        flat = np.asarray(flat, dtype=np.float32)
        top = float(flat.max()) or 1.0
        scale = 65535.0 / top
        arrays: dict[str, Any] = {
            "flat": np.clip(np.round(flat * scale), 0, 65535).astype(np.uint16),
            "scale": np.float32(scale),
            "info": np.array(json.dumps(info or {}, ensure_ascii=False)),
        }
        if dark is not None:
            arrays["dark"] = np.asarray(dark, dtype=np.float32)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)  # jamais de fichier à moitié écrit
        return path

    def load(self, objective: str | None, channel: str | None) -> tuple[np.ndarray, np.ndarray | None] | None:
        """(référence, noir ou None) en float32, ou None s'il n'y en a pas."""
        loaded = self._read(objective, channel)
        return None if loaded is None else loaded[:2]

    def info(self, objective: str | None, channel: str | None) -> dict[str, Any] | None:
        """Renseignements de la référence (date, exposition…), sans charger l'image."""
        path = self.path(objective, channel)
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                return json.loads(str(z["info"])) if "info" in z else {}
        except Exception:
            return {}

    def reference(self, objective: str | None, channel: str | None) -> FlatField | None:
        loaded = self._read(objective, channel)
        return None if loaded is None else FlatField(*loaded)

    def _read(
        self, objective: str | None, channel: str | None
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]] | None:
        path = self.path(objective, channel)
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                flat = z["flat"].astype(np.float32)
                if "scale" in z:  # sinon : ancien format, déjà en valeurs réelles
                    flat /= float(z["scale"])
                dark = z["dark"].astype(np.float32) if "dark" in z else None
                info = json.loads(str(z["info"])) if "info" in z else {}
        except Exception:
            log.exception("Référence illisible : %s", path)
            return None
        return flat, dark, info

    def delete(self, objective: str | None, channel: str | None) -> bool:
        path = self.path(objective, channel)
        if not path.exists():
            return False
        path.unlink()
        return True

    def correct(self, image: np.ndarray, objective: str | None, channel: str | None) -> np.ndarray | None:
        """Image corrigée avec la référence enregistrée, ou None s'il n'y en a pas pour cet
        objectif et ce canal (ou si le cadrage diffère)."""
        ref = self.reference(objective, channel)
        ref = ref.for_shape(image.shape) if ref is not None else None
        return None if ref is None else ref.apply(image)


# ------------------------------------------------------------- prise de référence
class FlatFieldError(RuntimeError):
    """La référence ne peut pas être prise (message à montrer tel quel)."""


@dataclass
class Candidate:
    """Une référence (ou un noir) qui vient d'être mesurée, avant d'être enregistrée."""

    kind: str  # « flat » ou « dark »
    image: np.ndarray
    objective: str | None
    channel: str | None
    info: dict[str, Any] = field(default_factory=dict)
    #: remarques à montrer avant d'enregistrer (la référence reste utilisable)
    warnings: list[str] = field(default_factory=list)


def check_reference(image: np.ndarray, max_value: float) -> list[str]:
    """Contrôle une image de référence. Lève :class:`FlatFieldError` si elle est inutilisable
    (saturée, trop sombre) ; renvoie des remarques si elle est douteuse (échantillon dans le
    champ)."""
    sample = np.asarray(image[::4, ::4], dtype=np.float32)
    saturated = float(np.mean(sample >= 0.98 * max_value))
    if saturated > 0.01:
        raise FlatFieldError(
            f"{saturated:.0%} de l'image est saturée : la référence serait fausse.\n\n"
            "Réduisez l'exposition ou la lampe, puis recommencez."
        )
    level = float(sample.mean()) / max_value
    if level < 0.05:
        raise FlatFieldError(
            f"L'image est trop sombre ({level:.0%} de la dynamique) : la référence serait bruitée.\n\n"
            "Augmentez l'exposition ou la lampe (visez la moitié de l'histogramme), puis recommencez."
        )
    warnings = []
    if level < 0.2:
        warnings.append(
            f"L'image est sombre ({level:.0%} de la dynamique) : la correction ajoutera un peu de bruit. "
            "Idéalement, le fond arrive vers la moitié de l'histogramme."
        )
    # un échantillon dans le champ : de grandes zones qui s'écartent du fond lissé
    from scipy import ndimage as ndi

    gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
    step = max(1, max(gray.shape) // 240)
    small = gray[::step, ::step]
    smooth = ndi.gaussian_filter(small, max(small.shape) / 12, mode="nearest")
    relative = np.abs(small - smooth) / np.maximum(smooth, 1e-6)
    objects = float(np.mean(ndi.uniform_filter(relative, 5, mode="nearest") > 0.06))
    if objects > 0.01:
        warnings.append(
            f"Il semble y avoir quelque chose dans le champ ({objects:.0%} de l'image) : la référence "
            "doit être prise sur une zone vide de la lame, sinon l'échantillon s'imprimera en négatif "
            "sur toutes les images."
        )
    return warnings


# ------------------------------------------------------------------- correction
@dataclass(frozen=True)
class FlatFieldStatus:
    #: « on » (corrige), « off » (désactivée), « missing » (activée mais pas de référence
    #: pour cet objectif et ce canal), « mismatch » (référence d'un autre cadrage)
    state: str
    text: str
    has_reference: bool = False


class FlatFieldCorrection:
    """Correction du fond du microscope : ce qui est appliqué aux images de la caméra."""

    #: l'état a changé (activation, nouvelle référence, autre objectif…) — émis depuis
    #: n'importe quel fil d'exécution
    changed = Signal()

    def __init__(self, microscope: Microscope, root: str | Path | None = None) -> None:
        self.m = microscope
        self.store = FlatFieldStore(microscope.profile.name, root)
        self._enabled = False
        self._lock = threading.RLock()
        self._loaded: tuple[tuple[str | None, str | None], float, FlatField] | None = None
        self._active: FlatField | None = None
        self._state = "off"
        self._pending_dark: np.ndarray | None = None
        self._raw = 0  # > 0 : la caméra livre des images brutes (prise d'une référence)
        self._warned = False
        microscope.objective_changed.connect(self.refresh)
        microscope.channel_changed.connect(self.refresh)
        microscope.connection_changed.connect(self.refresh)

    # ---------------------------------------------------------------- état
    def keys(self) -> tuple[str | None, str | None]:
        obj = self.m.objective
        return (obj.label if obj else None), self.m.channel

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)
        self.refresh()

    @property
    def active(self) -> bool:
        """Vrai si les images de la caméra sont corrigées en ce moment."""
        return self._active is not None and self._raw == 0

    def reference(self) -> FlatField | None:
        """La référence de l'objectif et du canal actuels (gardée en mémoire)."""
        keys = self.keys()
        path = self.store.path(*keys)
        with self._lock:
            if not path.exists():
                self._loaded = None
                return None
            mtime = path.stat().st_mtime
            if self._loaded is None or self._loaded[:2] != (keys, mtime):
                ref = self.store.reference(*keys)
                if ref is None:
                    return None
                self._loaded = (keys, mtime, ref)
            return self._loaded[2]

    def refresh(self, *_: object) -> None:
        """Recalcule ce qui s'applique (après un changement d'objectif, de canal, d'état)."""
        cam = self.m.camera
        with self._lock:
            active, state = None, "off"
            if self._enabled and cam is not None:
                ref = self.reference()
                if ref is None:
                    state = "missing"
                else:
                    frame = cam.latest_frame()[1]
                    shape = frame.shape if frame is not None else None
                    active = ref if shape is None else ref.for_shape(shape)
                    state = "on" if active is not None else "mismatch"
            self._active, self._state, self._warned = active, state, False
            if cam is not None:
                cam.frame_filter = self._filter if active is not None and self._raw == 0 else None
        self.changed.emit()

    def status(self) -> FlatFieldStatus:
        obj, ch = self.keys()
        what = " · ".join(x for x in (obj, ch) if x) or "ce réglage"
        info = self.store.info(obj, ch)
        taken = _when(info.get("date")) if info else ""
        taken = f", prise {taken}" if taken else ""
        state = self._state
        if info is None:
            if state == "missing":
                text = (
                    f"Pas de référence pour {what} : placez une zone vide de la lame sous "
                    "l'objectif, puis « Prendre la référence »."
                )
            else:
                text = f"Aucune référence pour {what}."
            return FlatFieldStatus(state, text, False)
        if state == "on":
            text = f"Fond corrigé — référence {what}{taken}."
        elif state == "mismatch":
            text = (
                "La référence n'a pas la taille des images de la caméra (résolution ou zone "
                "changée) : reprenez-la."
            )
        else:
            text = f"Référence {what}{taken} — correction désactivée."
        return FlatFieldStatus(state, text, True)

    # ------------------------------------------------------------- application
    def _filter(self, frame: np.ndarray) -> np.ndarray:
        """Appelée par la caméra pour chaque image (fil de la caméra)."""
        ref = self._active
        if ref is None:
            return frame
        if frame.shape != ref.gain.shape:
            fitted = ref.for_shape(frame.shape)
            if fitted is None:
                if not self._warned:
                    self._warned = True
                    self._state = "mismatch"
                    self.changed.emit()
                return frame
            ref = self._active = fitted
        cam = self.m.camera
        return ref.apply(frame, max_value_of(frame, cam.bit_depth if cam else None))

    @contextlib.contextmanager
    def raw_frames(self) -> Iterator[None]:
        """Le temps du bloc, la caméra livre des images brutes (non corrigées)."""
        cam = self.m.camera
        with self._lock:
            self._raw += 1
            if cam is not None:
                cam.frame_filter = None
        try:
            yield
        finally:
            with self._lock:
                self._raw -= 1
            self.refresh()

    # ------------------------------------------------------- prise de référence
    def measure(
        self,
        kind: str = "flat",
        frames: int = DEFAULT_FRAMES,
        progress: Callable[[int, int], None] | None = None,
    ) -> Candidate:
        """Moyenne de ``frames`` images brutes de la caméra, contrôlée. À enregistrer ensuite
        avec :meth:`save`. ``kind`` : « flat » (zone vide de la lame) ou « dark » (lumière
        coupée)."""
        cam = self.m.require_camera()
        objective, channel = self.keys()
        total: np.ndarray | None = None
        with self.raw_frames():
            live = cam.is_live
            count = cam.latest_frame()[0]
            timeout = max(5.0, cam.get_exposure() / 1000 * 4 + 2)
            if live:  # les images déjà en route ont pu être corrigées : on en laisse passer deux
                for _ in range(2):
                    count = cam._wait_frame(count, timeout)[0]
            for i in range(frames):
                if live:
                    count, image, _ = cam._wait_frame(count, timeout)
                else:
                    image = self.m.snap().data
                if total is None:
                    total = image.astype(np.float32)  # float32 : une image 4K en float64 = 600 Mo
                elif image.shape != total.shape:
                    raise FlatFieldError("La taille des images a changé pendant la mesure : recommencez.")
                else:
                    total += image
                if progress is not None:
                    progress(i + 1, frames)
            assert total is not None
            total /= frames
            top = max_value_of(image, cam.bit_depth)
        info = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "frames": frames,
            "exposure_ms": cam.get_exposure(),
            "objective": objective,
            "channel": channel,
            "max_value": top,
        }
        candidate = Candidate(kind, total, objective, channel, info)
        if kind == "dark":
            level = float(total[::4, ::4].mean()) / top
            if level > 0.1:
                raise FlatFieldError(
                    f"L'image n'est pas noire ({level:.0%} de la dynamique) : coupez la lumière "
                    "(ou fermez l'obturateur), puis recommencez."
                )
        else:
            candidate.warnings = check_reference(total, top)
        return candidate

    def save(self, candidate: Candidate) -> Path | None:
        """Enregistre la référence mesurée et active la correction. Un noir est joint à la
        référence existante, ou gardé pour la prochaine référence."""
        keys = (candidate.objective, candidate.channel)
        if candidate.kind == "dark":
            existing = self.store.load(*keys)
            if existing is None or existing[0].shape != candidate.image.shape:
                self._pending_dark = candidate.image
                return None
            path = self.store.save(*keys, existing[0], candidate.image, self.store.info(*keys))
        else:
            dark = self._pending_dark
            if dark is None or dark.shape != candidate.image.shape:
                previous = self.store.load(*keys)  # le noir déjà enregistré reste valable
                dark = previous[1] if previous is not None else None
                if dark is not None and dark.shape != candidate.image.shape:
                    dark = None
            path = self.store.save(*keys, candidate.image, dark, candidate.info)
            self._enabled = True
        self._pending_dark = None if candidate.kind == "flat" else self._pending_dark
        with self._lock:
            self._loaded = None
        self.refresh()
        return path

    def clear(self) -> bool:
        """Efface la référence de l'objectif et du canal actuels."""
        removed = self.store.delete(*self.keys())
        with self._lock:
            self._loaded = None
        self.refresh()
        return removed


def _when(iso: str | None) -> str:
    """« aujourd'hui à 14:02 », « le 17/11 à 14:02 »…"""
    if not iso:
        return ""
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    day = "aujourd'hui" if t.date() == datetime.now().date() else f"le {t:%d/%m/%Y}"
    return f"{day} à {t:%H:%M}"


__all__ = [
    "DEFAULT_FRAMES",
    "Candidate",
    "FlatField",
    "FlatFieldCorrection",
    "FlatFieldError",
    "FlatFieldStatus",
    "FlatFieldStore",
    "check_reference",
    "max_value_of",
]
