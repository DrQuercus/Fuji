"""Ponts entre les threads matériels et l'interface Qt.

Les signaux du cœur (psygnal) sont émis depuis des threads d'acquisition : on les
relaie par des signaux Qt, qui passent automatiquement dans le thread de
l'interface.
"""

from __future__ import annotations

import threading

from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import QMessageBox, QWidget

from fuji.core.manual import ManualAction


class QtManualActionHandler(QObject):
    """Affiche les consignes manuelles (« Placez l'objectif 40x ») et attend la réponse.

    Utilisable depuis n'importe quel thread : la boîte de dialogue s'ouvre toujours
    dans le thread de l'interface, et le thread appelant attend la réponse.
    """

    _request = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._parent = parent
        self._answer = False
        self._lock = threading.Lock()
        self._request.connect(self._ask, Qt.ConnectionType.BlockingQueuedConnection)

    def __call__(self, action: ManualAction) -> bool:
        with self._lock:
            if QThread.currentThread() is self.thread():
                self._ask(action)
            else:
                self._request.emit(action)
            return self._answer

    def _ask(self, action: ManualAction) -> None:
        box = QMessageBox(self._parent)
        box.setWindowTitle("Action manuelle")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(action.message)
        box.setInformativeText("Cliquez sur « Continuer » une fois l'action effectuée.")
        cont = box.addButton("Continuer", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Annuler", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        self._answer = box.clickedButton() is cont


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
