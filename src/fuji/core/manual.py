"""Périphériques manuels.

Un microscope comme le Leitz Diaplan n'a aucun moteur, mais le logiciel peut quand
même le *connaître* : l'objectif en place, le cube de fluorescence, l'état de la
lampe... Ces périphériques mémorisent l'état déclaré par l'utilisateur.

Deux chemins :

- l'utilisateur **déclare** ce qu'il vient de faire (``declare_state``,
  ``declare_position``...) : aucun message, l'état est simplement enregistré ;
- un processus automatique (protocole, acquisition) **demande** un changement
  (``set_state``, ``move_to``...) : le logiciel affiche une consigne
  (« Placez l'objectif 40x ») et attend la confirmation.

Le jour où un élément est motorisé, on remplace le périphérique manuel par le
pilote motorisé : le reste du logiciel ne change pas.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fuji.text import number

from .devices import FocusDrive, LightSource, StateDevice, XYStage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManualAction:
    """Consigne à afficher à l'utilisateur."""

    device: str
    message: str


#: Reçoit une consigne, renvoie ``True`` quand l'utilisateur a confirmé.
ManualActionHandler = Callable[[ManualAction], bool]


class ManualActionDeclined(RuntimeError):
    """L'utilisateur a refusé ou annulé une action manuelle."""


def auto_confirm_handler(action: ManualAction) -> bool:
    """Gestionnaire par défaut (scripts, tests) : journalise et confirme."""
    log.warning("Action manuelle requise [%s] : %s", action.device, action.message)
    return True


class _ManualMixin:
    manual = True
    name: str

    action_handler: ManualActionHandler | None = None

    def _ask(self, message: str) -> None:
        handler = self.action_handler or auto_confirm_handler
        if not handler(ManualAction(self.name, message)):
            raise ManualActionDeclined(message)


class ManualStateDevice(_ManualMixin, StateDevice):
    """Tourelle, roue ou glissière réglée à la main."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        *,
        initial: int | str = 0,
        prompt: str = "Mettez en place « {label} » ({device}).",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        if not labels:
            raise ValueError(f"{name} : au moins une position est nécessaire")
        self._labels = [str(lbl) for lbl in labels]
        self._state = self._check_index(self.index_of(initial))
        self._prompt = prompt

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def get_state(self) -> int:
        return self._state

    def _set_state(self, index: int) -> None:
        if index != self._state:
            self._ask(self._prompt.format(label=self._labels[index], device=self.label))
        self._state = index

    def declare_state(self, index_or_label: int | str) -> None:
        """L'utilisateur indique la position actuelle (sans consigne)."""
        index = self._check_index(self.index_of(index_or_label))
        self._state = index
        self.state_changed.emit(index, self._labels[index])


class ManualFocusDrive(_ManualMixin, FocusDrive):
    """Mise au point manuelle : le logiciel indique de combien tourner la vis.

    ``um_per_division`` : valeur d'une graduation de la vis micrométrique (si connue),
    pour donner la consigne en graduations.
    """

    def __init__(self, name: str, *, um_per_division: float | None = None, **kw: Any) -> None:
        super().__init__(name, **kw)
        self._z = 0.0
        self.um_per_division = um_per_division

    def get_position(self) -> float:
        return self._z

    def _move_to(self, z: float) -> None:
        dz = z - self._z
        if abs(dz) > 1e-6:
            sens = "vers le haut" if dz > 0 else "vers le bas"
            msg = f"Tournez la mise au point fine de {number(abs(dz), 2)} µm {sens}"
            if self.um_per_division:
                msg += f" (≈ {number(abs(dz) / self.um_per_division, 1)} graduations)"
            self._ask(msg + ".")
        self._z = z

    def declare_position(self, z: float) -> None:
        self._z = float(z)
        self.position_changed.emit(self._z)


class ManualXYStage(_ManualMixin, XYStage):
    """Platine non motorisée : consignes de déplacement (ex. acquisitions multi-positions)."""

    def __init__(self, name: str, **kw: Any) -> None:
        super().__init__(name, **kw)
        self._xy = (0.0, 0.0)

    def get_position(self) -> tuple[float, float]:
        return self._xy

    def _move_to(self, x: float, y: float) -> None:
        if (x, y) != self._xy:
            self._ask(f"Déplacez la platine vers X = {number(x, 1)} µm, Y = {number(y, 1)} µm.")
        self._xy = (x, y)

    def declare_position(self, x: float, y: float) -> None:
        self._xy = (float(x), float(y))
        self.position_changed.emit(*self._xy)


class ManualLightSource(_ManualMixin, LightSource):
    """Lampe réglée à la main (halogène du Diaplan, lampe à mercure...)."""

    def __init__(
        self,
        name: str,
        *,
        wavelength_nm: float | None = None,
        initial_on: bool = True,
        initial_intensity: float = 100.0,
        source_type: str = "halogène",
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        self.wavelength_nm = wavelength_nm
        self.source_type = source_type
        self._on = initial_on
        self._intensity = float(initial_intensity)

    def is_on(self) -> bool:
        return self._on

    def _set_on(self, on: bool) -> None:
        if on != self._on:
            self._ask(f"{'Allumez' if on else 'Éteignez'} {self.label}.")
        self._on = on

    def get_intensity(self) -> float:
        return self._intensity

    def _set_intensity(self, percent: float) -> None:
        if abs(percent - self._intensity) > 0.5:
            self._ask(f"Réglez {self.label} à {percent:.0f} %.")
        self._intensity = percent

    def declare(self, *, on: bool | None = None, intensity: float | None = None) -> None:
        if on is not None:
            self._on = bool(on)
        if intensity is not None:
            self._intensity = float(intensity)
        self.light_changed.emit(self._on, self._intensity)
