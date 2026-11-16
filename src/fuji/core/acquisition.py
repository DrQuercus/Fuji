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

import itertools
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import useq
from psygnal import Signal

from .devices import LightSource, StateDevice
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


#: Au-delà, la planification extrapole à partir des premières images (rapidité de l'interface).
_PLAN_SAMPLE = 20_000


class _Resolver:
    """Convertit les positions d'un événement ``useq`` en cibles absolues.

    Partagé par la planification et l'exécution pour qu'elles ne divergent jamais.

    - pile Z relative : centrée sur la mise au point de départ, sauf pour les positions
      qui fixent leur propre Z (``useq`` fournit alors déjà une valeur absolue) ;
    - grille relative : centrée sur la position XY de départ, sauf si la position la fixe.
    """

    def __init__(self, sequence: useq.MDASequence, z0: float, xy0: tuple[float, float]) -> None:
        self.positions = list(sequence.stage_positions)
        self.relative_z = bool(sequence.z_plan is not None and sequence.z_plan.is_relative)
        grid = sequence.grid_plan
        self.relative_grid = bool(grid is not None and getattr(grid, "is_relative", False))
        self.z0, self.xy0 = z0, xy0

    def _position(self, event: useq.MDAEvent) -> Any:
        p = event.index.get("p")
        return self.positions[p] if p is not None and 0 <= p < len(self.positions) else None

    def xy(self, event: useq.MDAEvent, current: tuple[float, float]) -> tuple[float, float] | None:
        x, y = event.x_pos, event.y_pos
        if x is None and y is None:
            return None
        if self.relative_grid:
            pos = self._position(event)
            if pos is None or pos.x is None:
                x = (x or 0.0) + self.xy0[0]
            if pos is None or pos.y is None:
                y = (y or 0.0) + self.xy0[1]
        return (current[0] if x is None else float(x), current[1] if y is None else float(y))

    def z(self, event: useq.MDAEvent) -> float | None:
        if event.z_pos is None:
            return None
        if self.relative_z:
            pos = self._position(event)
            if pos is None or pos.z is None:
                return float(event.z_pos) + self.z0
        return float(event.z_pos)


def _sample_events(sequence: useq.MDASequence) -> tuple[list[useq.MDAEvent], float]:
    """Événements à simuler et facteur d'extrapolation (1 si la séquence est complète)."""
    estimate = 1
    for n in sequence.sizes.values():
        estimate *= max(1, n)
    if estimate <= _PLAN_SAMPLE:
        return list(sequence), 1.0
    events = list(itertools.islice(iter(sequence), _PLAN_SAMPLE))
    return events, (max(1.0, estimate / len(events)) if len(events) == _PLAN_SAMPLE else 1.0)


def plan_acquisition(
    microscope: Microscope,
    sequence: useq.MDASequence,
    save_dir: str | Path | None = None,
    *,
    overhead_s: float = 0.05,
) -> AcquisitionPlan:
    """Simule la séquence (comme le moteur l'exécutera) : durée, taille, avertissements."""
    cam = microscope.require_camera()
    events, scale = _sample_events(sequence)
    h, w = cam.sensor_shape()
    bytes_per_frame = h * w * (3 if cam.is_color else 1) * cam.dtype.itemsize
    warnings: list[str] = []
    profile = microscope.profile
    focus, stage = microscope.focus, microscope.xy_stage

    # état simulé des éléments que le moteur modifie
    states = {n: d.get_label() for n, d in microscope.devices.items() if isinstance(d, StateDevice)}
    lights = {
        n: [d.is_on(), d.get_intensity()] for n, d in microscope.devices.items() if isinstance(d, LightSource)
    }
    channel = microscope.channel
    exposure = cam.get_exposure()
    z = focus.get_position() if focus is not None else 0.0
    xy = stage.get_position() if stage is not None else (0.0, 0.0)
    resolver = _Resolver(sequence, z, xy)
    manual = 0
    clock = t_ref = 0.0
    busy = 0.0

    def is_manual(name: str) -> bool:
        return bool(getattr(microscope.devices.get(name), "manual", False))

    for ev in events:
        if ev.reset_event_timer:
            t_ref = clock
        clock = max(clock, t_ref + (ev.min_start_time or 0.0))
        target_xy = resolver.xy(ev, xy)
        if target_xy is not None and stage is not None and target_xy != xy:
            manual += stage.manual
            xy = target_xy
        if ev.channel is not None and ev.channel.config != channel:
            channel = ev.channel.config
            preset = profile.channel(channel)
            if preset is not None:
                for name, label in preset.states.items():
                    if name in states and states[name] != label:
                        states[name] = label
                        manual += is_manual(name)
                others = dict.fromkeys(
                    n
                    for c in profile.channels
                    if c.name != channel
                    for n in c.lights
                    if n not in preset.lights
                )
                for name in others:
                    if name in lights and lights[name][0]:
                        lights[name][0] = False
                        manual += is_manual(name)
                for name, setting in preset.lights.items():
                    if name not in lights:
                        continue
                    tolerance = 0.5 if is_manual(name) else 1e-6
                    if setting.intensity is not None and abs(lights[name][1] - setting.intensity) > tolerance:
                        lights[name][1] = setting.intensity
                        manual += is_manual(name)
                    if lights[name][0] != setting.on:
                        lights[name][0] = setting.on
                        manual += is_manual(name)
                if preset.exposure_ms is not None:
                    exposure = preset.exposure_ms
        target_z = resolver.z(ev)
        if target_z is not None and focus is not None and abs(target_z - z) > 1e-6:
            manual += focus.manual
            z = target_z
        if ev.exposure is not None:
            exposure = ev.exposure
        step = exposure / 1000 + overhead_s
        clock += step
        busy += step

    n_frames = round(len(events) * scale)
    duration = clock * scale
    manual = round(manual * scale)

    # intervalle trop court ?
    sizes = sequence.sizes
    n_t = sizes.get("t", 0)
    interval = getattr(sequence.time_plan, "interval", None) if sequence.time_plan is not None else None
    interval_s = interval.total_seconds() if hasattr(interval, "total_seconds") else None
    if n_t > 1 and interval_s:
        order = str(sequence.axis_order)
        time_first = "p" not in order or order.index("t") < order.index("p")
        # temps à l'extérieur : toutes les positions sont visitées à chaque point de temps
        visits = n_t if time_first else n_t * max(1, sizes.get("p", 0))
        per_timepoint = busy * scale / visits
        if interval_s < per_timepoint:
            warnings.append(
                f"L'intervalle ({interval_s:.2f} s) est plus court que le temps nécessaire "
                f"pour chaque point de temps (~{per_timepoint:.2f} s) : les images seront décalées."
            )

    # matériel requis
    if sizes.get("z", 0) > 0 and focus is None:
        warnings.append("Une pile Z est demandée mais ce microscope n'a pas de mise au point déclarée.")
    for ch in sequence.channels:
        if profile.channel(ch.config) is None:
            warnings.append(f"Canal inconnu dans ce profil : « {ch.config} ».")
    if (sequence.stage_positions or sequence.grid_plan is not None) and stage is None:
        warnings.append("Des positions XY sont demandées mais aucune platine n'est déclarée.")
    if manual:
        warnings.append(
            f"Cette acquisition demandera {manual} action(s) manuelle(s) "
            "(changement de cube, de lampe, de mise au point...)."
        )

    # espace disque
    total_bytes = bytes_per_frame * n_frames
    if save_dir is not None:
        target = Path(save_dir).expanduser()
        probe = next((q for q in [target, *target.parents] if q.exists()), None)
        if probe is not None:
            free = shutil.disk_usage(probe).free
            if total_bytes > free * 0.95:
                warnings.append(
                    f"Espace disque insuffisant : {total_bytes / 1e9:.1f} Go nécessaires, "
                    f"{free / 1e9:.1f} Go libres."
                )
    n_pos = max(1, sizes.get("p", 0)) * max(1, sizes.get("g", 0))
    return AcquisitionPlan(n_frames, n_pos, duration, total_bytes, manual, warnings)


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
        """Exécute la séquence (bloquant). Ne lève pas : l'issue est dans le résumé renvoyé,
        et ``finished`` est toujours émis, même si la préparation échoue."""
        scope = self.microscope
        self._running = True
        self._cancel.clear()
        self._resume.set()
        summary = AcquisitionSummary(total=0, acquired=0, duration_s=0.0)
        t_start = time.monotonic()
        cam = scope.camera
        was_live = False
        writer_opened = False
        start_channel = scope.channel
        focus, stage = scope.focus, scope.xy_stage
        z0: float | None = None
        xy0: tuple[float, float] | None = None
        moved_z = moved_xy = False
        try:
            cam = scope.require_camera()
            events = list(sequence)
            summary.total = len(events)
            was_live = cam.is_live
            if was_live:
                cam.stop_live()
            z0 = focus.get_position() if focus is not None else 0.0
            xy0 = stage.get_position() if stage is not None else (0.0, 0.0)
            resolver = _Resolver(sequence, z0, xy0)
            if writer is not None:
                writer.open(sequence)
                writer_opened = True
            self.started.emit(sequence)
            t_ref = time.monotonic()
            for i, event in enumerate(events):
                self._checkpoint()
                if event.reset_event_timer:  # ex. time-lapse complet position par position
                    t_ref = time.monotonic()
                self._wait_until(t_ref + (event.min_start_time or 0.0))
                # position XY
                if event.x_pos is not None or event.y_pos is not None:
                    if stage is None:
                        raise RuntimeError("Position XY demandée sans platine XY")
                    current = stage.get_position()
                    target = resolver.xy(event, current)
                    if target is not None and target != current:
                        stage.move_to(*target)
                        moved_xy = True
                # canal
                if event.channel is not None and event.channel.config != scope.channel:
                    scope.apply_channel(event.channel.config)
                # Z
                z = resolver.z(event)
                if z is not None:
                    if focus is None:
                        raise RuntimeError("Pile Z demandée sans mise au point")
                    if abs(z - focus.get_position()) > 1e-6:
                        focus.move_to(z)
                        moved_z = True
                if event.exposure is not None:
                    cam.set_exposure(event.exposure)
                frame = scope.snap()
                frame.metadata.index = {str(getattr(k, "value", k)): int(v) for k, v in event.index.items()}
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
            summary.error = str(exc) or type(exc).__name__
        finally:
            if writer is not None and writer_opened:
                try:
                    writer.close()
                except Exception as exc:
                    log.exception("Erreur à la fermeture des fichiers")
                    summary.error = summary.error or f"écriture : {exc}"
                summary.files = list(getattr(writer, "paths", []))
            if restore:
                self._restore(start_channel, z0 if moved_z else None, xy0 if moved_xy else None)
            if was_live and cam is not None:
                try:
                    cam.start_live()
                except Exception:  # pragma: no cover - dépend du matériel
                    log.exception("Impossible de relancer le live")
            summary.duration_s = time.monotonic() - t_start
            self._running = False
            try:
                self.finished.emit(summary)
            except Exception:  # pragma: no cover - un abonné défaillant ne doit rien bloquer
                log.exception("Erreur dans un abonné du signal de fin d'acquisition")
        return summary

    def _restore(self, channel: str | None, z0: float | None, xy0: tuple[float, float] | None) -> None:
        """Revient à l'état de départ, sans jamais redemander de manipulation manuelle."""
        scope = self.microscope
        try:
            if z0 is not None and scope.focus is not None and not scope.focus.manual:
                scope.focus.move_to(z0)
            if xy0 is not None and scope.xy_stage is not None and not scope.xy_stage.manual:
                scope.xy_stage.move_to(*xy0)
            if channel is not None and channel != scope.channel:
                preset = scope.profile.channel(channel)
                manual = preset is not None and any(
                    getattr(scope.devices.get(n), "manual", False) for n in [*preset.states, *preset.lights]
                )
                if not manual:
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
