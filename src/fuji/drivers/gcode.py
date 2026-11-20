"""Motorisation « maison » : contrôleurs de moteurs pas à pas pilotés en G-code.

Une carte Arduino ou ESP32 avec le micrologiciel **GRBL** (ou ses dérivés FluidNC et
grblHAL), ou une carte d'imprimante 3D sous **Marlin**, suffit à motoriser une platine
XY, la mise au point, une roue à filtres et à régler une LED. Ce pilote parle à la carte
par le port série (USB) et la présente à Fuji comme des périphériques ordinaires : le
reste du logiciel (acquisition multi-positions, autofocus, mosaïques, assistants) ne
voit pas la différence avec une platine du commerce.

Une entrée de profil crée plusieurs périphériques::

    devices:
      - name: moteurs
        driver: gcode
        options:
          port: /dev/tty.usbserial-1420      # ou COM3, socket://raspberrypi.local:2217, simulation
          um_per_unit_z: 100                 # 1 unité du contrôleur (mm) = 100 µm de mise au point
          backlash_z_um: 3

→ ``moteurs`` (le contrôleur), ``moteurs_xy`` (platine), ``moteurs_z`` (mise au point),
et sur demande ``moteurs_roue`` (roue à filtres) et ``moteurs_lampe`` (LED en PWM).

Sécurité : les limites de course du profil sont vérifiées avant chaque mouvement (un
mouvement hors limites est refusé, jamais tronqué en silence), et ``stop()`` interrompt
immédiatement les moteurs.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Any

from fuji.core.devices import Device, FocusDrive, LightSource, StateDevice, XYStage
from fuji.core.profile import DeviceSpec
from fuji.core.properties import PropertyInfo, PropertyType

from . import DriverContext, OptionSpec, register_driver

log = logging.getLogger(__name__)

_STATUS = re.compile(r"^<([^|,>]+)(.*)>$")
_MARLIN_POSITION = re.compile(r"\b([XYZABC]):\s*(-?\d+\.?\d*)")
_AXES = "XYZABC"

_GRBL_ERRORS = {
    1: "commande mal formée",
    2: "valeur numérique invalide",
    3: "commande « $ » inconnue",
    5: "la prise d'origine n'est pas activée sur ce contrôleur (réglage $22)",
    8: "commande refusée : le contrôleur n'est pas au repos",
    9: "contrôleur verrouillé (alarme) : déverrouillez-le ou faites une prise d'origine",
    15: "cible hors de la course autorisée (limites logicielles)",
    20: "commande G-code non reconnue",
    22: "vitesse d'avance non définie",
}
_GRBL_ALARMS = {
    1: "fin de course atteint pendant un mouvement : position perdue, refaites la prise d'origine",
    2: "cible hors de la course autorisée par le contrôleur",
    3: "réinitialisation en plein mouvement : position incertaine, refaites l'origine",
    8: "prise d'origine échouée : le fin de course n'a pas été libéré",
    9: "prise d'origine échouée : fin de course introuvable",
}


class GCodeError(RuntimeError):
    """Erreur signalée par le contrôleur de moteurs (ou absence de réponse)."""


class MotionStopped(GCodeError):
    """Le mouvement a été interrompu par un arrêt d'urgence."""


class GCodeLink:
    """Dialogue série avec un contrôleur G-code, partagé par tous ses axes.

    Un thread lit en permanence ce que la carte envoie : les réponses (« ok », erreurs)
    vont à la commande en attente, les rapports d'état mettent à jour la position connue.
    L'état peut ainsi être lu pendant qu'un mouvement est en cours, et un arrêt d'urgence
    passe immédiatement, même si une commande attend la fin d'un déplacement.
    """

    def __init__(self, port: str, baudrate: int = 115200, firmware: str = "grbl", *, io: Any = None) -> None:
        self.port, self.baudrate, self.firmware = port, int(baudrate), firmware
        self._io = io
        self._command_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._responses: queue.Queue[str] = queue.Queue()
        self._status_cond = threading.Condition()
        self._status_time = 0.0
        self._status_serial = 0
        self.state = "Inconnu"
        self.positions: dict[str, float] = dict.fromkeys(_AXES, 0.0)
        #: dernière cible commandée par axe : au repos c'est la position exacte, alors que le
        #: contrôleur ne relit la sienne qu'avec trois décimales (1 µm en millimètres)
        self._targets: dict[str, float] = {}
        self._offsets: dict[str, float] = dict.fromkeys(_AXES, 0.0)
        self.alarm: str | None = None
        self.banner = ""
        self._abort = threading.Event()
        self._closing = threading.Event()
        self._reader: threading.Thread | None = None
        self._last_motion = 0.0

    # ----------------------------------------------------------------- ouverture
    def open(self) -> None:
        if self._io is None:
            if self.port == "simulation":
                from .gcode_sim import SimulatedController

                self._io = SimulatedController(self.firmware)
            else:
                try:
                    import serial
                except ImportError as exc:  # pragma: no cover - dépend de l'installation
                    raise ImportError(
                        'Le pilote de moteurs a besoin de pyserial : pip install "fuji-microscopy[moteurs]"'
                    ) from exc
                try:
                    self._io = serial.serial_for_url(
                        self.port, baudrate=self.baudrate, timeout=0.1, write_timeout=2
                    )
                except Exception as exc:
                    raise GCodeError(
                        f"Impossible d'ouvrir le port {self.port!r} : {exc}. Carte branchée ? Port utilisé "
                        "par un autre logiciel ? « fuji moteurs ports » liste les ports disponibles."
                    ) from exc
        self._closing.clear()
        self._reader = threading.Thread(target=self._read_loop, name=f"gcode-{self.port}", daemon=True)
        self._reader.start()
        self._wake_up()

    def _wake_up(self) -> None:
        """Attend la bannière de démarrage (une carte Arduino redémarre à l'ouverture du port)."""
        if self.firmware == "grbl":
            deadline = time.monotonic() + 4.0
            self._write(b"\r\n\r\n")
            while not self.banner and time.monotonic() < deadline:
                time.sleep(0.02)
            self._drain()
            if not self.banner:  # carte déjà démarrée : on vérifie qu'elle répond
                try:
                    self.refresh_status(timeout=2.0)
                except GCodeError as exc:
                    raise GCodeError(
                        f"Aucun contrôleur GRBL ne répond sur {self.port} à {self.baudrate} bauds "
                        "(vitesse ? micrologiciel ?)"
                    ) from exc
            self.refresh_status(timeout=2.0)
            if self.state.startswith("Alarm"):
                self.alarm = "contrôleur verrouillé au démarrage"
            else:
                self.command("G21")  # millimètres
                self.command("G90")
        else:
            time.sleep(0.2)
            self._drain()
            try:
                self.command("G21", timeout=5.0)
            except GCodeError as exc:
                raise GCodeError(f"Aucun contrôleur Marlin ne répond sur {self.port}") from exc
            self.command("G90")
            self.refresh_status()

    def close(self) -> None:
        self._closing.set()
        if self._reader is not None:
            self._reader.join(timeout=2)
            self._reader = None
        if self._io is not None:
            try:
                self._io.close()
            except Exception:  # pragma: no cover - dépend du matériel
                log.debug("Fermeture du port %s", self.port, exc_info=True)
            self._io = None

    # ------------------------------------------------------------------ lecture
    def _read_loop(self) -> None:
        while not self._closing.is_set():
            try:
                raw = self._io.readline()
            except Exception as exc:  # carte débranchée
                if not self._closing.is_set():
                    self.alarm = f"liaison perdue avec le contrôleur ({exc})"
                    self._responses.put(f"!{self.alarm}")
                return
            if not raw:
                continue
            line = raw.decode("ascii", "replace").strip()
            if line:
                self._dispatch(line)

    def _dispatch(self, line: str) -> None:
        low = line.lower()
        if line.startswith("<"):
            self._parse_status(line)
        elif low == "ok" or low.startswith(("ok ", "error")):
            self._responses.put(line)
        elif low.startswith("alarm"):
            code = int(re.sub(r"\D", "", line) or 0)
            self.alarm = _GRBL_ALARMS.get(code, f"alarme {code}")
            self.state = "Alarm"
            self._responses.put(line)
        elif line.startswith("Grbl") or low == "start":
            self.banner = line
        elif self.firmware == "marlin" and _MARLIN_POSITION.search(line) and "count" in low:
            for axis, value in _MARLIN_POSITION.findall(line.split("Count")[0]):
                self.positions[axis] = float(value)
            self._mark_status("Idle")
        else:
            log.debug("%s : %s", self.port, line)

    def _parse_status(self, line: str) -> None:
        match = _STATUS.match(line)
        if not match:
            return
        fields = dict(f.split(":", 1) for f in match.group(2).strip("|,").split("|") if ":" in f)
        if "WCO" in fields:
            for axis, value in zip(_AXES, fields["WCO"].split(","), strict=False):
                self._offsets[axis] = float(value)
        if "MPos" in fields:
            for axis, value in zip(_AXES, fields["MPos"].split(","), strict=False):
                self.positions[axis] = float(value)
        elif "WPos" in fields:  # position « pièce » : on revient à la position machine
            for axis, value in zip(_AXES, fields["WPos"].split(","), strict=False):
                self.positions[axis] = float(value) + self._offsets[axis]
        self._mark_status(match.group(1))

    def _mark_status(self, state: str) -> None:
        with self._status_cond:
            self.state = state
            self._status_time = time.monotonic()
            self._status_serial += 1
            self._status_cond.notify_all()

    def _drain(self) -> None:
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                return

    # ----------------------------------------------------------------- écriture
    def _write(self, data: bytes) -> None:
        if self._io is None:
            raise GCodeError("Le contrôleur de moteurs n'est pas connecté")
        with self._write_lock:
            self._io.write(data)

    def command(self, line: str, timeout: float = 10.0) -> None:
        """Envoie une ligne de G-code et attend son accusé de réception.

        Lève :class:`GCodeError` si le contrôleur répond par une erreur ou ne répond pas,
        :class:`MotionStopped` si un arrêt d'urgence survient pendant l'attente.
        """
        with self._command_lock:
            if self._abort.is_set():
                raise MotionStopped("Mouvement interrompu")
            self._drain()
            self._write(line.encode("ascii") + b"\n")
            deadline = time.monotonic() + timeout
            while True:
                if self._abort.is_set():
                    raise MotionStopped("Mouvement interrompu par l'arrêt d'urgence")
                try:
                    answer = self._responses.get(timeout=0.05)
                except queue.Empty:
                    if time.monotonic() > deadline:
                        raise GCodeError(
                            f"Le contrôleur n'a pas répondu à « {line} » en {timeout:.0f} s"
                        ) from None
                    continue
                low = answer.lower()
                if low.startswith("ok"):
                    return
                if answer.startswith("!"):
                    raise GCodeError(answer[1:])
                code = int(re.sub(r"\D", "", answer) or 0)
                if low.startswith("alarm"):
                    raise GCodeError(f"Alarme du contrôleur : {_GRBL_ALARMS.get(code, f'code {code}')}")
                raise GCodeError(
                    f"Le contrôleur refuse « {line} » : {_GRBL_ERRORS.get(code, f'erreur {code}')}"
                )

    # --------------------------------------------------------------------- état
    def refresh_status(self, timeout: float = 1.0, max_age: float = 0.0) -> None:
        """Met à jour l'état et les positions (au plus vieux de ``max_age`` secondes)."""
        if max_age > 0 and time.monotonic() - self._status_time <= max_age:
            return
        if self.firmware != "grbl":
            if self._command_lock.acquire(blocking=False):  # pas pendant un mouvement bloquant
                try:
                    self.command("M114", timeout=timeout)
                finally:
                    self._command_lock.release()
            return
        with self._status_cond:
            serial = self._status_serial
            self._write(b"?")
            if not self._status_cond.wait_for(lambda: self._status_serial != serial, timeout=timeout):
                raise GCodeError("Le contrôleur ne répond plus aux demandes d'état")

    def position(self, axis: str, max_age: float = 0.1) -> float:
        self.refresh_status(max_age=max_age)
        reported = self.positions[axis]
        target = self._targets.get(axis)
        if target is not None and self.state == "Idle" and abs(reported - target) <= 0.00101:
            return target  # arrivé : la cible est connue plus finement que la relecture
        return reported

    def is_idle(self) -> bool:
        if time.monotonic() - self._last_motion < 0.05:
            return False  # le mouvement vient d'être envoyé : il n'a pas forcément commencé
        self.refresh_status(max_age=0.02)
        return self.state == "Idle"

    # ---------------------------------------------------------------- mouvements
    def move(self, targets: dict[str, float], feed: float) -> None:
        """Met en file un déplacement absolu (coordonnées machine), sans attendre sa fin."""
        if self.alarm and self.state.startswith("Alarm"):
            raise GCodeError(f"Contrôleur en alarme ({self.alarm}) : déverrouillez-le d'abord")
        words = " ".join(f"{axis}{value:.5f}" for axis, value in targets.items())
        prefix = "G53 G1" if self.firmware == "grbl" else "G1"
        self._last_motion = time.monotonic()
        self.command(f"{prefix} {words} F{feed:.1f}")
        self._targets.update({axis: round(value, 5) for axis, value in targets.items()})
        self._last_motion = time.monotonic()

    def wait_idle(self, timeout: float = 60.0) -> None:
        """Attend la fin de tous les mouvements en file."""
        self.command("G4 P0" if self.firmware == "grbl" else "M400", timeout=timeout)
        self._last_motion = 0.0
        self.refresh_status()

    def stop(self) -> None:
        """Arrêt d'urgence : les moteurs décélèrent, la file de mouvements est vidée."""
        self._abort.set()
        self._targets.clear()  # les cibles ne seront pas atteintes
        try:
            if self.firmware == "grbl":
                self._write(b"!")  # pause immédiate, sans perdre la position
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    try:
                        self.refresh_status(timeout=0.5)
                    except GCodeError:
                        break
                    if self.state == "Idle" or self.state.startswith("Hold:0") or self.state == "Hold":
                        break
                    time.sleep(0.02)
                self.banner = ""
                self._write(b"\x18")  # réinitialisation : vide la file des mouvements
                deadline = time.monotonic() + 2.0
                while not self.banner and time.monotonic() < deadline:
                    time.sleep(0.01)
            else:
                self._write(b"M410\n")
                time.sleep(0.1)
        finally:
            with self._command_lock:  # la commande interrompue a rendu la main
                self._abort.clear()
                self._drain()
                self._last_motion = 0.0
        if self.firmware == "grbl":
            self.refresh_status()
            if not self.state.startswith("Alarm"):
                self.alarm = None
                self.command("G21")
                self.command("G90")

    def unlock(self) -> None:
        """Lève l'alarme du contrôleur (``$X``) sans prise d'origine."""
        if self.firmware == "grbl":
            self.command("$X")
            self.command("G21")
            self.command("G90")
        self.alarm = None
        self.refresh_status()

    def home(self, timeout: float = 120.0) -> None:
        """Prise d'origine sur les fins de course du contrôleur."""
        try:
            self.command("$H" if self.firmware == "grbl" else "G28", timeout=timeout)
        except GCodeError as exc:
            if "prise d'origine" in str(exc):
                raise NotImplementedError(str(exc)) from exc
            raise
        self.alarm = None
        self._targets.clear()
        if self.firmware == "grbl":
            self.command("G21")
            self.command("G90")
        self.refresh_status()


class _Axis:
    """Un axe moteur vu en micromètres : échelle, sens, origine, limites, rattrapage de jeu."""

    def __init__(
        self,
        link: GCodeLink,
        letter: str,
        *,
        um_per_unit: float,
        invert: bool,
        feed: float,
        backlash_um: float,
        lo: float | None,
        hi: float | None,
    ) -> None:
        if um_per_unit <= 0:
            raise ValueError("Le nombre de µm par unité du contrôleur doit être positif")
        self.link, self.letter = link, letter.upper()
        self.scale = um_per_unit * (-1.0 if invert else 1.0)
        self.feed, self.backlash = float(feed), abs(float(backlash_um))
        self.lo, self.hi = lo, hi
        self.origin = 0.0  # position (µm) déclarée comme zéro par l'utilisateur

    def get(self) -> float:
        return self.link.position(self.letter) * self.scale - self.origin

    def check(self, target_um: float, name: str) -> None:
        if (self.lo is not None and target_um < self.lo - 1e-6) or (
            self.hi is not None and target_um > self.hi + 1e-6
        ):
            lo = "−∞" if self.lo is None else f"{self.lo:g}"
            hi = "+∞" if self.hi is None else f"{self.hi:g}"
            raise ValueError(
                f"{name} : la position {target_um:g} µm est hors de la course autorisée [{lo} ; {hi}] µm"
            )

    def to_units(self, um: float) -> float:
        return (um + self.origin) / self.scale

    def duration(self, from_um: float, to_um: float) -> float:
        """Durée approximative d'un déplacement (s), pour borner l'attente."""
        return abs(to_um - from_um) / abs(self.scale) / max(self.feed, 1e-6) * 60.0


def _move_axes(link: GCodeLink, moves: list[tuple[_Axis, float]], name: str) -> float:
    """Déplace un ou plusieurs axes ; renvoie la durée estimée.

    Rattrapage de jeu : un axe qui recule dépasse d'abord sa cible de la valeur du jeu,
    puis y revient. L'approche finale se fait ainsi toujours dans le même sens, et la
    position ne dépend plus du sens d'arrivée.
    """
    current = {axis.letter: axis.get() for axis, _ in moves}
    for axis, target in moves:
        axis.check(target, name)
    overshoot = {
        axis.letter: target - axis.backlash
        for axis, target in moves
        if axis.backlash > 0 and target < current[axis.letter] - 1e-9
    }
    feed = min(axis.feed for axis, _ in moves)
    duration = max(axis.duration(current[axis.letter], target) for axis, target in moves)
    if overshoot:
        first = {axis.letter: axis.to_units(overshoot.get(axis.letter, target)) for axis, target in moves}
        link.move(first, feed)
        duration += max(axis.backlash / abs(axis.scale) / max(axis.feed, 1e-6) * 120.0 for axis, _ in moves)
    link.move({axis.letter: axis.to_units(target) for axis, target in moves}, feed)
    return duration


class GCodeController(Device):
    """La carte de commande elle-même : état, déverrouillage, fermeture du port."""

    def __init__(self, name: str, link: GCodeLink, **kw: Any) -> None:
        super().__init__(name, **kw)
        self.link = link
        self._add_property(
            PropertyInfo(
                "État", PropertyType.STRING, read_only=True, description="repos, mouvement, alarme…"
            ),
            self._state,
        )
        self._add_property(
            PropertyInfo("Micrologiciel", PropertyType.STRING, read_only=True), lambda: link.banner
        )
        self._add_property(PropertyInfo("Port", PropertyType.STRING, read_only=True), lambda: link.port)

    def _state(self) -> str:
        try:
            self.link.refresh_status(max_age=0.5)
        except GCodeError as exc:
            return f"injoignable ({exc})"
        return f"alarme : {self.link.alarm}" if self.link.state.startswith("Alarm") else self.link.state

    def initialize(self) -> None:
        self.link.open()
        super().initialize()

    def shutdown(self) -> None:
        self.link.close()
        super().shutdown()

    def unlock(self) -> None:
        self.link.unlock()

    def stop(self) -> None:
        self.link.stop()

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "port": self.link.port,
            "firmware": self.link.banner or self.link.firmware,
        }


class GCodeXYStage(XYStage):
    def __init__(self, name: str, link: GCodeLink, x: _Axis, y: _Axis, **kw: Any) -> None:
        super().__init__(name, **kw)
        self.link, self._x, self._y = link, x, y
        self._eta = 0.0

    def get_position(self) -> tuple[float, float]:
        return self._x.get(), self._y.get()

    def _move_to(self, x: float, y: float) -> None:
        self._eta = _move_axes(self.link, [(self._x, x), (self._y, y)], self.label)

    def is_busy(self) -> bool:
        return not self.link.is_idle()

    def wait(self, timeout: float | None = None) -> None:
        self.link.wait_idle(timeout or max(30.0, self._eta * 2 + 10.0))

    def stop(self) -> None:
        self.link.stop()

    def home(self) -> None:
        self.link.home()
        self._x.origin = self._y.origin = 0.0
        self.position_changed.emit(*self.get_position())

    def set_origin(self) -> None:
        """Déclare la position actuelle comme (0, 0)."""
        x, y = self.get_position()
        self._x.origin += x
        self._y.origin += y
        self.position_changed.emit(0.0, 0.0)

    def limits(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if None in (self._x.lo, self._x.hi, self._y.lo, self._y.hi):
            return None
        return ((self._x.lo, self._x.hi), (self._y.lo, self._y.hi))  # type: ignore[return-value]


class GCodeFocus(FocusDrive):
    def __init__(self, name: str, link: GCodeLink, z: _Axis, **kw: Any) -> None:
        super().__init__(name, **kw)
        self.link, self._z = link, z
        self._eta = 0.0

    def get_position(self) -> float:
        return self._z.get()

    def _move_to(self, z: float) -> None:
        self._eta = _move_axes(self.link, [(self._z, z)], self.label)

    def is_busy(self) -> bool:
        return not self.link.is_idle()

    def wait(self, timeout: float | None = None) -> None:
        self.link.wait_idle(timeout or max(30.0, self._eta * 2 + 10.0))

    def stop(self) -> None:
        self.link.stop()

    def set_origin(self) -> None:
        """Déclare la position actuelle comme Z = 0 (par exemple la mise au point)."""
        self._z.origin += self.get_position()
        self.position_changed.emit(0.0)

    def limits(self) -> tuple[float, float] | None:
        if self._z.lo is None or self._z.hi is None:
            return None
        return (self._z.lo, self._z.hi)


class GCodeWheel(StateDevice):
    """Roue à filtres ou tourelle entraînée par un moteur pas à pas (positions fixes)."""

    def __init__(
        self,
        name: str,
        link: GCodeLink,
        axis: str,
        labels: list[str],
        positions: list[float],
        feed: float,
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        if not labels or len(labels) != len(positions):
            raise ValueError(f"{name} : il faut autant de positions que de noms (au moins une)")
        self.link, self._axis, self._feed = link, axis.upper(), float(feed)
        self._labels, self._positions = [str(lbl) for lbl in labels], [float(p) for p in positions]
        self._state = 0

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def get_state(self) -> int:
        return self._state

    def _set_state(self, index: int) -> None:
        self.link.move({self._axis: self._positions[index]}, self._feed)
        self.link.wait_idle(120.0)
        self._state = index


class GCodeLight(LightSource):
    """LED ou lampe réglée par la sortie PWM du contrôleur (broche ou ventilateur)."""

    def __init__(self, name: str, link: GCodeLink, output: str, maximum: float, **kw: Any) -> None:
        super().__init__(name, **kw)
        self.link, self._output, self._max = link, output, float(maximum)
        self._on, self._intensity = False, 100.0
        self.source_type = "LED"

    def is_on(self) -> bool:
        return self._on

    def get_intensity(self) -> float:
        return self._intensity

    def _apply(self) -> None:
        value = self._intensity / 100.0 * self._max if self._on else 0.0
        if self._output == "broche":
            self.link.command(f"M3 S{value:.0f}" if value > 0 else "M5")
        else:
            self.link.command(f"M106 S{value:.0f}" if value > 0 else "M107")

    def _set_on(self, on: bool) -> None:
        self._on = on
        self._apply()

    def _set_intensity(self, percent: float) -> None:
        self._intensity = percent
        self._apply()

    def shutdown(self) -> None:
        if self._on and self.link._io is not None:
            try:
                self._set_on(False)
            except GCodeError:
                log.debug("Extinction de %s impossible", self.name, exc_info=True)
        super().shutdown()


def list_serial_ports() -> list[dict[str, str]]:
    """Ports série disponibles sur cet ordinateur (pour remplir l'option ``port``)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [{"port": p.device, "description": p.description or ""} for p in list_ports.comports()]


def _limit(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _provides(name: str, options: dict[str, Any]) -> list[tuple[str, str]]:
    """Périphériques créés par une entrée « gcode », d'après ses options."""
    out: list[tuple[str, str]] = []
    if options.get("xy_stage", True):
        out.append((f"{name}_xy", "xy_stage"))
    if options.get("focus", True):
        out.append((f"{name}_z", "focus"))
    if options.get("wheel_axis"):
        out.append((f"{name}_roue", "state"))
    if options.get("light", "aucune") != "aucune":
        out.append((f"{name}_lampe", "light_source"))
    return out


@register_driver(
    "gcode",
    title="Moteurs pas à pas (GRBL, FluidNC, Marlin)",
    category="Platines et mise au point",
    kind=None,
    description="Motorisation maison : une carte Arduino ou ESP32 (GRBL / FluidNC / grblHAL) ou une carte "
    "d'imprimante 3D (Marlin) pilote la platine XY, la mise au point, une roue à filtres et une LED.",
    provides=_provides,
    options=[
        OptionSpec(
            "port",
            "str",
            "Port série",
            "simulation",
            help="ex. /dev/tty.usbserial-1420, COM3, socket://raspberrypi.local:2217 ; "
            "« simulation » pour essayer",
        ),
        OptionSpec("firmware", "choice", "Micrologiciel", "grbl", choices=("grbl", "marlin")),
        OptionSpec("baudrate", "int", "Vitesse", 115200, unit="bauds", advanced=True),
        OptionSpec("xy_stage", "bool", "Platine XY", True),
        OptionSpec("focus", "bool", "Mise au point", True),
        OptionSpec(
            "um_per_unit_xy",
            "float",
            "Platine : µm par unité",
            1000.0,
            min=0.001,
            unit="µm",
            help="1000 si le contrôleur compte en millimètres réels",
        ),
        OptionSpec(
            "um_per_unit_z",
            "float",
            "Mise au point : µm par unité",
            1000.0,
            min=0.001,
            unit="µm",
            help="déplacement réel de la mise au point pour 1 unité du contrôleur",
        ),
        OptionSpec("feed_xy", "float", "Vitesse de la platine", 600.0, min=1, unit="unités/min"),
        OptionSpec("feed_z", "float", "Vitesse de la mise au point", 60.0, min=0.1, unit="unités/min"),
        OptionSpec(
            "backlash_xy_um",
            "float",
            "Jeu de la platine",
            0.0,
            min=0,
            unit="µm",
            help="mesuré par l'assistant « Platine » ; rattrapé automatiquement",
        ),
        OptionSpec("backlash_z_um", "float", "Jeu de la mise au point", 0.0, min=0, unit="µm"),
        OptionSpec("invert_x", "bool", "Inverser X", False),
        OptionSpec("invert_y", "bool", "Inverser Y", False),
        OptionSpec("invert_z", "bool", "Inverser Z", False),
        OptionSpec("x_min_um", "float", "X minimum", None, unit="µm", help="vide = pas de limite"),
        OptionSpec("x_max_um", "float", "X maximum", None, unit="µm"),
        OptionSpec("y_min_um", "float", "Y minimum", None, unit="µm"),
        OptionSpec("y_max_um", "float", "Y maximum", None, unit="µm"),
        OptionSpec("z_min_um", "float", "Z minimum", None, unit="µm"),
        OptionSpec(
            "z_max_um",
            "float",
            "Z maximum",
            None,
            unit="µm",
            help="protège l'objectif : la mise au point ne montera jamais plus haut",
        ),
        OptionSpec("x_axis", "str", "Axe X du contrôleur", "X", advanced=True),
        OptionSpec("y_axis", "str", "Axe Y du contrôleur", "Y", advanced=True),
        OptionSpec("z_axis", "str", "Axe Z du contrôleur", "Z", advanced=True),
        OptionSpec(
            "unlock_on_connect",
            "bool",
            "Déverrouiller à la connexion",
            True,
            advanced=True,
            help="lève l'alarme de démarrage d'un contrôleur avec fins de course ($X)",
        ),
        OptionSpec("home_on_connect", "bool", "Prise d'origine à la connexion", False, advanced=True),
        OptionSpec(
            "light",
            "choice",
            "Lampe / LED",
            "aucune",
            choices=("aucune", "broche", "ventilateur"),
            help="broche : sortie PWM de GRBL (M3/M5) ; ventilateur : sortie de Marlin (M106/M107)",
        ),
        OptionSpec(
            "light_max",
            "float",
            "Valeur PWM maximale",
            1000.0,
            min=1,
            advanced=True,
            help="1000 pour GRBL (réglage $30), 255 pour Marlin",
        ),
        OptionSpec("wheel_axis", "str", "Axe de la roue à filtres", "", help="ex. A ; vide = pas de roue"),
        OptionSpec("wheel_labels", "list", "Positions de la roue", None),
        OptionSpec(
            "wheel_positions", "list", "Coordonnées des positions", None, help="en unités du contrôleur"
        ),
        OptionSpec(
            "wheel_feed", "float", "Vitesse de la roue", 600.0, min=1, unit="unités/min", advanced=True
        ),
    ],
)
def _gcode(spec: DeviceSpec, ctx: DriverContext) -> list[Device]:
    o = dict(spec.options)
    link = o.pop("link", None) or GCodeLink(
        str(o.get("port") or "simulation"), int(o.get("baudrate", 115200)), str(o.get("firmware", "grbl"))
    )
    controller = GCodeController(spec.name, link, label=spec.label or "Contrôleur de moteurs")
    controller._unlock_on_connect = bool(o.get("unlock_on_connect", True))  # type: ignore[attr-defined]
    controller._home_on_connect = bool(o.get("home_on_connect", False))  # type: ignore[attr-defined]
    original_initialize = controller.initialize

    def initialize() -> None:
        original_initialize()
        if controller._home_on_connect:  # type: ignore[attr-defined]
            link.home()
        elif link.state.startswith("Alarm") and controller._unlock_on_connect:  # type: ignore[attr-defined]
            link.unlock()

    controller.initialize = initialize  # type: ignore[method-assign]
    devices: list[Device] = [controller]
    roles = ctx.profile.roles

    def axis(letter_key: str, default: str, kind: str) -> _Axis:
        short = letter_key[0]
        return _Axis(
            link,
            str(o.get(letter_key) or default),
            um_per_unit=float(o.get(f"um_per_unit_{kind}", 1000.0)),
            invert=bool(o.get(f"invert_{short}", False)),
            feed=float(o.get(f"feed_{kind}", 600.0 if kind == "xy" else 60.0)),
            backlash_um=float(o.get(f"backlash_{kind}_um", 0.0)),
            lo=_limit(o.get(f"{short}_min_um")),
            hi=_limit(o.get(f"{short}_max_um")),
        )

    if o.get("xy_stage", True):
        stage = GCodeXYStage(
            f"{spec.name}_xy", link, axis("x_axis", "X", "xy"), axis("y_axis", "Y", "xy"), label="Platine XY"
        )
        devices.append(stage)
        if roles.xy_stage is None:
            roles.xy_stage = stage.name
    if o.get("focus", True):
        focus = GCodeFocus(f"{spec.name}_z", link, axis("z_axis", "Z", "z"), label="Mise au point")
        devices.append(focus)
        if roles.focus is None:
            roles.focus = focus.name
    if o.get("wheel_axis"):
        devices.append(
            GCodeWheel(
                f"{spec.name}_roue",
                link,
                str(o["wheel_axis"]),
                list(o.get("wheel_labels") or []),
                [float(p) for p in (o.get("wheel_positions") or [])],
                float(o.get("wheel_feed", 600.0)),
                label="Roue à filtres",
            )
        )
    if o.get("light", "aucune") != "aucune":
        lamp = GCodeLight(
            f"{spec.name}_lampe", link, str(o["light"]), float(o.get("light_max", 1000.0)), label="LED"
        )
        devices.append(lamp)
        if roles.transmitted_light is None:
            roles.transmitted_light = lamp.name
    return devices
