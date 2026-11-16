"""Moteur d'acquisition multidimensionnelle (temps, positions, canaux, Z).

Les séquences sont décrites avec ``useq-schema`` (le format utilisé aussi par
pymmcore-plus) : un protocole écrit pour Fuji est donc lisible par l'écosystème
Micro-Manager/Python, et inversement.

Le moteur ne parle qu'à :class:`~fuji.core.microscope.Microscope` : il fonctionne
de la même façon avec un microscope motorisé ou manuel (dans ce cas, chaque
changement nécessaire devient une consigne à l'utilisateur).

Exemple ::

    import useq
    seq = useq.MDASequence(
        channels=["Fond clair", "GFP"],
        time_plan={"interval": 60, "loops": 30},
        z_plan={"range": 10, "step": 2},
    )
    plan = plan_acquisition(scope, seq)       # durée, taille, avertissements
    engine = AcquisitionEngine(scope)
    engine.run(seq, writer=OMETiffWriter("~/acquisitions", "essai"))
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import useq
from psygnal import Signal

from .devices import StateDevice
from .frame import Frame
from .manual import ManualActionDeclined
from .microscope import Microscope

log = logging.getLogger(__name__)


class FrameWriter(Protocol):
    def open(self, sequence: useq.MDASequence) -> None: ...
    def write(self, event: useq.MDAEvent, frame: Frame) -> None: ...
    def close(self) -> None: ...


class AcquisitionCancelled(Exception):
    pass


# ----------------------------------------------------------------- planification
@dataclass
class AcquisitionPlan:
    """Estimation affichée avant de lancer : le logiciel « conseille »."""

    n_frames: int
    n_positions: int
    duration_s: float
    bytes_on_disk: int
    manual_actions: int
    warnings: list[str] = field(default_factory=list)

    @property
    def human_size(self) -> str:
        size = float(self.bytes_on_disk)
        for unit in ("o", "Ko", "Mo", "Go", "To"):
            if size < 1024 or unit == "To":
                return f"{size:.1f} {unit}" if unit != "o" else f"{int(size)} o"
            size /= 1024
        return ""  # pragma: no cover

    @property
    def human_duration(self) -> str:
        s = int(round(self.duration_s))
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        return f"{h} h {m:02d} min" if h else (f"{m} min {s:02d} s" if m else f"{s} s")


def plan_acquisition(
    microscope: Microscope,
    sequence: useq.MDASequence,
    save_dir: str | Path | None = None,
    *,
    overhead_s: float = 0.05,
) -> AcquisitionPlan:
    events = list(sequence)
    cam = microscope.require_camera()
    h, w = cam.sensor_shape()
    bytes_per_frame = h * w * (3 if cam.is_color else 1) * cam.dtype.itemsize
    warnings: list[str] = []

    def exposure_for(ev: useq.MDAEvent) -> float:
        if ev.exposure is not None:
            return ev.exposure
        if ev.channel is not None:
            preset = microscope.profile.channel(ev.channel.config)
            if preset is not None and preset.exposure_ms:
                return preset.exposure_ms
        return cam.get_exposure()

    # durée : temps d'exposition + surcoût, sans descendre sous le plan temporel
    busy = sum(exposure_for(e) / 1000 + overhead_s for e in events)
    last_start = max((e.min_start_time or 0.0) for e in events) if events else 0.0
    duration = max(busy, last_start + busy / max(1, sequence.sizes.get("t", 1)))

    # intervalle trop court ?
    t_plan = sequence.time_plan
    n_t = sequence.sizes.get("t", 0)
    if t_plan is not None and n_t > 1:
        per_timepoint = busy / n_t
        interval = getattr(t_plan, "interval", None)
        interval_s = interval.total_seconds() if hasattr(interval, "total_seconds") else None
        if interval_s is not None and 0 < interval_s < per_timepoint:
            warnings.append(
                f"L'intervalle ({interval_s:.2f} s) est plus court que le temps nécessaire "
                f"pour chaque point de temps (~{per_timepoint:.2f} s) : les images seront décalées."
            )

    # matériel requis
    has_z = sequence.sizes.get("z", 0) > 0
    if has_z and microscope.focus is None:
        warnings.append("Une pile Z est demandée mais ce microscope n'a pas de mise au point déclarée.")
    for ch in sequence.channels:
        if microscope.profile.channel(ch.config) is None:
            warnings.append(f"Canal inconnu dans ce profil : « {ch.config} ».")
    if sequence.stage_positions and microscope.xy_stage is None:
        warnings.append("Des positions XY sont demandées mais aucune platine n'est déclarée.")

    # actions manuelles à prévoir
    manual = _count_manual_actions(microscope, events)
    if manual:
        warnings.append(
            f"Cette acquisition demandera {manual} action(s) manuelle(s) "
            "(changement de cube, d'objectif, de mise au point...)."
        )

    # espace disque
    total_bytes = bytes_per_frame * len(events)
    if save_dir is not None:
        target = Path(save_dir).expanduser()
        probe = next((p for p in [target, *target.parents] if p.exists()), None)
        if probe is not None:
            free = shutil.disk_usage(probe).free
            if total_bytes > free * 0.95:
                warnings.append(
                    f"Espace disque insuffisant : {total_bytes / 1e9:.1f} Go nécessaires, "
                    f"{free / 1e9:.1f} Go libres."
                )
    n_pos = max(1, sequence.sizes.get("p", 0)) * max(1, sequence.sizes.get("g", 0))
    return AcquisitionPlan(len(events), n_pos, duration, total_bytes, manual, warnings)


def _count_manual_actions(microscope: Microscope, events: list[useq.MDAEvent]) -> int:
    count = 0
    current_ch = microscope.channel
    last_z: float | None = None
    last_xy: tuple[float | None, float | None] | None = None
    for ev in events:
        if ev.channel is not None and ev.channel.config != current_ch:
            preset = microscope.profile.channel(ev.channel.config)
            if preset is not None:
                for dev_name in preset.states:
                    dev = microscope.devices.get(dev_name)
                    if isinstance(dev, StateDevice) and dev.manual:
                        count += 1
                for dev_name in preset.lights:
                    dev = microscope.devices.get(dev_name)
                    if dev is not None and dev.manual:
                        count += 1
            current_ch = ev.channel.config
        if ev.z_pos is not None and microscope.focus is not None and microscope.focus.manual:
            if last_z is not None and abs(ev.z_pos - last_z) > 1e-9:
                count += 1
            last_z = ev.z_pos
        xy = (ev.x_pos, ev.y_pos)
        if xy != (None, None) and microscope.xy_stage is not None and microscope.xy_stage.manual:
            if last_xy is not None and xy != last_xy:
                count += 1
            last_xy = xy
    return count


# -------------------------------------------------------------------- exécution
@dataclass
class AcquisitionSummary:
    total: int
    acquired: int
    duration_s: float
    cancelled: bool = False
    error: str | None = None
    files: list[Path] = field(default_factory=list)


class AcquisitionEngine:
    started = Signal(object)  # MDASequence
    frame_acquired = Signal(object, object)  # MDAEvent, Frame
    progress = Signal(int, int)  # images faites, total
    finished = Signal(object)  # AcquisitionSummary
    paused_changed = Signal(bool)

    def __init__(self, microscope: Microscope) -> None:
        self.microscope = microscope
        self._cancel = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return not self._resume.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        self._resume.set()

    def pause(self) -> None:
        self._resume.clear()
        self.paused_changed.emit(True)

    def resume(self) -> None:
        self._resume.set()
        self.paused_changed.emit(False)

    def start(
        self, sequence: useq.MDASequence, *, writer: FrameWriter | None = None, **kw: Any
    ) -> threading.Thread:
        """Lance l'acquisition dans un thread (l'interface reste réactive)."""
        if self._running:
            raise RuntimeError("Une acquisition est déjà en cours")
        self._running = True  # réservé avant le démarrage du thread
        self._thread = threading.Thread(
            target=self.run,
            args=(sequence,),
            kwargs={"writer": writer, **kw},
            name="fuji-acquisition",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -------------------------------------------------------------- boucle
    def run(
        self, sequence: useq.MDASequence, *, writer: FrameWriter | None = None, restore: bool = True
    ) -> AcquisitionSummary:
        scope = self.microscope
        cam = scope.require_camera()
        self._running = True
        self._cancel.clear()
        self._resume.set()
        events = list(sequence)
        summary = AcquisitionSummary(total=len(events), acquired=0, duration_s=0.0)
        was_live = cam.is_live
        if was_live:
            cam.stop_live()
        start_channel = scope.channel
        focus = scope.focus
        z0 = focus.get_position() if focus is not None else 0.0
        relative_z = bool(sequence.z_plan is not None and sequence.z_plan.is_relative)
        positions_have_z = any(p.z is not None for p in sequence.stage_positions)
        last_xy: tuple[float, float] | None = None
        last_z: float | None = None
        t_start = time.monotonic()
        if writer is not None:
            writer.open(sequence)
        self.started.emit(sequence)
        try:
            for i, event in enumerate(events):
                self._checkpoint()
                self._wait_until(t_start + (event.min_start_time or 0.0))
                # position XY
                if event.x_pos is not None or event.y_pos is not None:
                    stage = scope.xy_stage
                    if stage is None:
                        raise RuntimeError("Position XY demandée sans platine XY")
                    cx, cy = stage.get_position()
                    target = (
                        event.x_pos if event.x_pos is not None else cx,
                        event.y_pos if event.y_pos is not None else cy,
                    )
                    if target != last_xy:
                        stage.move_to(*target)
                        last_xy = target
                # canal
                if event.channel is not None and event.channel.config != scope.channel:
                    scope.apply_channel(event.channel.config)
                # Z
                if event.z_pos is not None:
                    if focus is None:
                        raise RuntimeError("Pile Z demandée sans mise au point")
                    z = event.z_pos + (z0 if relative_z and not positions_have_z else 0.0)
                    if last_z is None or abs(z - last_z) > 1e-9:
                        focus.move_to(z)
                        last_z = z
                if event.exposure is not None:
                    cam.set_exposure(event.exposure)
                frame = scope.snap()
                frame.metadata.index = dict(event.index)
                frame.metadata.elapsed_s = time.monotonic() - t_start
                if event.pos_name:
                    frame.metadata.extra["position"] = event.pos_name
                if writer is not None:
                    writer.write(event, frame)
                summary.acquired = i + 1
                self.frame_acquired.emit(event, frame)
                self.progress.emit(i + 1, len(events))
        except (AcquisitionCancelled, ManualActionDeclined):
            summary.cancelled = True
        except Exception as exc:
            log.exception("Erreur pendant l'acquisition")
            summary.error = str(exc)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:  # pragma: no cover
                    log.exception("Erreur à la fermeture des fichiers")
                    summary.error = summary.error or f"écriture : {exc}"
                summary.files = list(getattr(writer, "paths", []))
            if restore:
                self._restore(start_channel, z0, last_z)
            if was_live:
                cam.start_live()
            summary.duration_s = time.monotonic() - t_start
            self._running = False
            self.finished.emit(summary)
        return summary

    def _restore(self, channel: str | None, z0: float, last_z: float | None) -> None:
        scope = self.microscope
        try:
            if scope.focus is not None and last_z is not None and not scope.focus.manual:
                scope.focus.move_to(z0)
            if channel is not None and channel != scope.channel:
                preset = scope.profile.channel(channel)
                manual = preset is not None and any(
                    getattr(scope.devices.get(n), "manual", False) for n in [*preset.states, *preset.lights]
                )
                if not manual:  # ne pas redemander une manipulation à la fin
                    scope.apply_channel(channel)
        except Exception:  # pragma: no cover
            log.exception("Impossible de restaurer l'état initial")

    def _checkpoint(self) -> None:
        while not self._resume.is_set():
            self._resume.wait(0.1)
        if self._cancel.is_set():
            raise AcquisitionCancelled

    def _wait_until(self, deadline: float) -> None:
        while True:
            self._checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cancel.wait(min(remaining, 0.1))
