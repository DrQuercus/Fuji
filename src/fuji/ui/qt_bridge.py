"""Ponts entre les threads matériels et l'interface Qt.

Les signaux du cœur (psygnal) sont émis depuis des threads d'acquisition : on les
relaie par des signaux Qt, qui passent automatiquement dans le thread de
l'interface.
"""

from __future__ import annotations

import threading

from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QMessageBox, QWidget

from fuji.core.manual import ManualAction


class _Request:
    """Une consigne en attente de réponse (une par appel : pas d'état partagé entre threads)."""

    __slots__ = ("action", "answer", "cancelled", "done", "generation")

    def __init__(self, action: ManualAction, generation: int) -> None:
        self.action = action
        self.generation = generation
        self.answer = False
        self.cancelled = False
        self.done = threading.Event()


class QtManualActionHandler(QObject):
    """Affiche les consignes manuelles (« Placez l'objectif 40x ») et attend la réponse.

    Utilisable depuis n'importe quel thread : la boîte de dialogue s'ouvre toujours
    dans le thread de l'interface, et le thread appelant attend la réponse.

    Le thread appelant n'attend jamais en tenant un verrou, et son attente peut être
    interrompue (:meth:`abort_pending`) : sans cela, fermer la fenêtre pendant qu'une
    acquisition attend une réponse bloquerait les deux threads l'un sur l'autre.
    """

    _request = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._parent = parent
        self._generation = 0
        # connexion automatique : en file d'attente quand l'appel vient d'un autre thread
        self._request.connect(self._ask)

    def __call__(self, action: ManualAction) -> bool:
        request = _Request(action, self._generation)
        if QThread.currentThread() is self.thread():
            self._ask(request)
            return request.answer
        self._request.emit(request)
        while not request.done.wait(0.1):
            if request.generation != self._generation:  # interface en cours de fermeture
                request.cancelled = True
                return False
        return request.answer

    def abort_pending(self) -> None:
        """Répond « annuler » à toutes les consignes en attente (fermeture, changement de
        microscope)."""
        self._generation += 1

    def _ask(self, request: _Request) -> None:
        try:
            if request.cancelled or request.generation != self._generation:
                return
            box = QMessageBox(self._parent)
            box.setWindowTitle("Action manuelle")
            box.setIcon(QMessageBox.Icon.Information)
            box.setText(request.action.message)
            box.setInformativeText("Cliquez sur « Continuer » une fois l'action effectuée.")
            cont = box.addButton("Continuer", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Annuler", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            request.answer = box.clickedButton() is cont
        finally:
            request.done.set()


class AcquisitionBridge(QObject):
    """Relaie les signaux du moteur d'acquisition vers l'interface."""

    started = Signal(object)
    frame = Signal(object, object)
    progress = Signal(int, int)
    finished = Signal(object)

    def connect_engine(self, engine) -> None:  # type: ignore[no-untyped-def]
        engine.started.connect(self.started.emit)
        engine.frame_acquired.connect(self.frame.emit)
        engine.progress.connect(self.progress.emit)
        engine.finished.connect(self.finished.emit)
