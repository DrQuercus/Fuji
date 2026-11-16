"""Faux contrôleur de moteurs (GRBL ou Marlin) : même dialogue série, moteurs imaginaires.

Sert à deux choses : essayer toute la chaîne de motorisation de Fuji avant d'acheter
le moindre moteur (``port: simulation`` dans le profil), et tester le pilote
:mod:`fuji.drivers.gcode` sans matériel.

Il reproduit ce qui compte pour le pilote : les mouvements prennent du temps, les
commandes sont mises en file, l'état (repos, mouvement, pause, alarme) se lit en temps
réel, et les erreurs ont les mêmes codes que le vrai micrologiciel.
"""

from __future__ import annotations

import queue
import re
import threading
import time

_WORD = re.compile(r"([A-Z$])\s*(-?\d*\.?\d+)")
AXES = "XYZABC"


class SimulatedController:
    """Objet « port série » (``write`` / ``readline`` / ``close``) simulant un contrôleur.

    ``speed`` : accélération du temps (1 = temps réel ; 1000 pour les tests).
    ``travel`` : course de chaque axe en unités du contrôleur (mm), ou ``None`` : sans limite.
    ``homing`` : le contrôleur a-t-il des fins de course (prise d'origine possible) ?
    ``backlash`` : jeu mécanique de chaque axe (mm). La position réelle (``mechanical``)
    traîne alors derrière la position commandée quand le moteur change de sens, comme une
    vis usée ; le contrôleur, lui, n'en sait rien.
    """

    def __init__(
        self,
        firmware: str = "grbl",
        *,
        speed: float = 1.0,
        travel: dict[str, tuple[float, float]] | None = None,
        homing: bool = False,
        rapid_feed: float = 3000.0,
        backlash: dict[str, float] | None = None,
    ) -> None:
        self.firmware = firmware
        self.speed = float(speed)
        self.travel = travel or {}
        self.homing = homing
        self.rapid_feed = rapid_feed
        self.timeout = 0.1
        self.is_open = True
        self._lock = threading.RLock()
        self._out: queue.Queue[bytes] = queue.Queue()
        self._buffer = b""
        self.position = dict.fromkeys(AXES, 0.0)
        self.backlash = backlash or {}
        self.mechanical = dict.fromkeys(AXES, 0.0)
        self._moves: list[tuple[dict[str, float], float]] = []  # (cible, avance)
        self._current: tuple[dict[str, float], dict[str, float], float, float] | None = None
        self._held = False
        self._hold_started = 0.0
        self._sync_pending = 0
        self._deferred: list[str] = []  # commandes reçues pendant une attente de fin de mouvement
        self.alarm: int | None = None
        self.absolute = True
        self.feed = 500.0
        self.spindle = 0.0
        self.fan = 0.0
        self.lines_received: list[str] = []
        self._emit("Grbl 1.1h ['$' for help]" if firmware == "grbl" else "start")

    # ------------------------------------------------------------ port série
    def write(self, data: bytes) -> int:
        with self._lock:
            for byte in data:
                char = bytes([byte])
                if self.firmware == "grbl" and char in (b"?", b"!", b"~", b"\x18", b"\x85"):
                    self._realtime(char)
                elif char in (b"\n", b"\r"):
                    line, self._buffer = self._buffer.decode("ascii", "replace").strip(), b""
                    if line:
                        self.lines_received.append(line)
                        self._execute(line)
                else:
                    self._buffer += char
        return len(data)

    def readline(self) -> bytes:
        deadline = time.monotonic() + self.timeout
        while True:
            with self._lock:
                self._advance()
            try:
                return self._out.get(timeout=0.005)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    return b""

    def close(self) -> None:
        self.is_open = False

    def reset_input_buffer(self) -> None:
        pass

    # --------------------------------------------------------------- moteurs
    def _emit(self, text: str) -> None:
        self._out.put(text.encode("ascii") + b"\r\n")

    def _now(self) -> float:
        return time.monotonic() * self.speed

    def _advance(self) -> None:
        """Fait avancer les moteurs jusqu'à maintenant, et répond aux attentes de fin de mouvement."""
        now = self._now()
        while not self._held:
            if self._current is None:
                if not self._moves:
                    break
                target, feed = self._moves.pop(0)
                start = dict(self.position)
                distance = max(abs(target[a] - start[a]) for a in AXES)
                duration = distance / max(feed, 1e-6) * 60.0
                self._current = (start, target, now, duration)
            start, target, t0, duration = self._current
            fraction = 1.0 if duration <= 0 else min((now - t0) / duration, 1.0)
            for a in AXES:
                self.position[a] = start[a] + (target[a] - start[a]) * fraction
                slack = self.backlash.get(a, 0.0)
                self.mechanical[a] = min(max(self.mechanical[a], self.position[a] - slack), self.position[a])
            if fraction < 1.0:
                break
            self._current = None
        if self._idle() and self._sync_pending:
            for _ in range(self._sync_pending):
                self._emit("ok")
            self._sync_pending = 0
            while self._deferred and not self._sync_pending:
                self._run(self._deferred.pop(0))

    def _idle(self) -> bool:
        return self._current is None and not self._moves and not self._held

    def _state(self) -> str:
        if self.alarm is not None:
            return "Alarm"
        if self._held:
            return "Hold:0"
        return "Idle" if self._idle() else "Run"

    def _realtime(self, char: bytes) -> None:
        self._advance()
        if char == b"?":
            pos = ",".join(f"{self.position[a]:.3f}" for a in "XYZA")
            self._emit(f"<{self._state()}|MPos:{pos}|FS:0,{self.spindle:.0f}>")
        elif char == b"!":
            if not self._idle():
                self._held = True
                self._hold_started = self._now()
        elif char == b"~":
            self._resume()
        elif char in (b"\x18", b"\x85"):
            moving = self._current is not None and not self._held
            self._moves.clear()
            self._current = None
            self._held = False
            self._sync_pending = 0
            self._deferred.clear()
            if char == b"\x18":
                self._buffer = b""
                if moving:
                    self.alarm = 3  # réinitialisation en plein mouvement : position incertaine
                self._emit("Grbl 1.1h ['$' for help]")

    def _resume(self) -> None:
        if self._held and self._current is not None:
            start, target, t0, duration = self._current
            self._current = (start, target, t0 + (self._now() - self._hold_started), duration)
        self._held = False

    def _queue_move(self, words: dict[str, float], *, rapid: bool, machine: bool) -> str | None:
        reference = (
            self._moves[-1][0] if self._moves else (self._current[1] if self._current else self.position)
        )
        target = dict(reference)
        for axis in AXES:
            if axis in words:
                target[axis] = words[axis] if (self.absolute or machine) else reference[axis] + words[axis]
        for axis, (lo, hi) in self.travel.items():
            if not lo - 1e-9 <= target[axis] <= hi + 1e-9:
                self.alarm = 2
                return "ALARM:2"
        if "F" in words:
            self.feed = words["F"]
        self._moves.append((target, self.rapid_feed if rapid else self.feed))
        return None

    # ------------------------------------------------------------- commandes
    def _execute(self, line: str) -> None:
        self._advance()
        if self._sync_pending:  # un vrai contrôleur traite les commandes dans l'ordre
            self._deferred.append(line)
            return
        self._run(line)

    def _run(self, line: str) -> None:
        text = line.upper().split(";")[0].strip()
        if self.firmware == "grbl":
            self._execute_grbl(text)
        else:
            self._execute_marlin(text)

    def _words(self, text: str) -> tuple[list[tuple[str, float]], dict[str, float]]:
        pairs = [(letter, float(value)) for letter, value in _WORD.findall(text)]
        return pairs, {letter: value for letter, value in pairs if letter not in "GM"}

    def _execute_grbl(self, text: str) -> None:
        if text == "$X":
            self.alarm = None
            self._emit("[MSG:Caution: Unlocked]")
            self._emit("ok")
            return
        if text == "$H":
            if not self.homing:
                self._emit("error:5")
                return
            self.alarm = None
            self._moves.clear()
            self._current = None
            self.position = dict.fromkeys(AXES, 0.0)
            self._emit("ok")
            return
        if text == "$$":
            self._emit(f"$22={int(self.homing)}")
            self._emit("ok")
            return
        if text == "$I":
            self._emit("[VER:1.1h.simulation:]")
            self._emit("ok")
            return
        jog = text.startswith("$J=")
        if jog:
            text = text[3:]
        elif text.startswith("$"):
            self._emit("error:3")
            return
        if self.alarm is not None:
            self._emit("error:9")
            return
        pairs, words = self._words(text)
        codes = [(letter, int(value)) for letter, value in pairs if letter in "GM"]
        unknown = [c for c in codes if c not in _GRBL_KNOWN]
        if unknown or (
            not codes and not any(a in words for a in AXES) and "F" not in words and "S" not in words
        ):
            self._emit("error:20")
            return
        if ("G", 90) in codes:
            self.absolute = True
        if ("G", 91) in codes:
            self.absolute = False
        if ("M", 3) in codes or ("M", 4) in codes:
            self.spindle = words.get("S", self.spindle)
        elif "S" in words and not any(c[0] == "M" for c in codes):
            self.spindle = words["S"] if self.spindle > 0 else self.spindle
        if ("M", 5) in codes:
            self.spindle = 0.0
        if ("G", 4) in codes:
            self._sync_pending += 1  # « ok » seulement une fois les mouvements terminés
            self._advance()
            return
        moving = any(a in words for a in AXES) and ("G", 10) not in codes and ("G", 92) not in codes
        if moving:
            error = self._queue_move(words, rapid=("G", 0) in codes, machine=("G", 53) in codes)
            if error:
                self._emit(error)
                return
        elif "F" in words:
            self.feed = words["F"]
        self._emit("ok")

    def _execute_marlin(self, text: str) -> None:
        pairs, words = self._words(text)
        codes = [(letter, int(value)) for letter, value in pairs if letter in "GM"]
        if not codes:
            self._emit('echo:Unknown command: "' + text + '"')
            self._emit("ok")
            return
        code = codes[0]
        if code in (("G", 0), ("G", 1)):
            self._queue_move(words, rapid=code == ("G", 0), machine=False)
            self.alarm = None  # Marlin borne les mouvements au lieu de lever une alarme
        elif code == ("G", 90):
            self.absolute = True
        elif code == ("G", 91):
            self.absolute = False
        elif code == ("G", 28):
            self._moves.clear()
            self._current = None
            self.position = dict.fromkeys(AXES, 0.0)
        elif code == ("M", 114):
            p = self.position
            self._emit(f"X:{p['X']:.2f} Y:{p['Y']:.2f} Z:{p['Z']:.2f} E:0.00 Count X:0 Y:0 Z:0")
        elif code == ("M", 400):
            self._sync_pending += 1
            self._advance()
            return
        elif code == ("M", 410):
            self._moves.clear()
            self._current = None
            self._sync_pending, pending = 0, self._sync_pending
            for _ in range(pending):
                self._emit("ok")
        elif code == ("M", 106):
            self.fan = words.get("S", 255.0)
        elif code == ("M", 107):
            self.fan = 0.0
        elif code == ("M", 115):
            self._emit("FIRMWARE_NAME:Marlin simulation")
        self._emit("ok")


_GRBL_KNOWN = {
    ("G", 0),
    ("G", 1),
    ("G", 4),
    ("G", 10),
    ("G", 21),
    ("G", 53),
    ("G", 54),
    ("G", 90),
    ("G", 91),
    ("G", 92),
    ("M", 3),
    ("M", 4),
    ("M", 5),
    ("M", 8),
    ("M", 9),
}
