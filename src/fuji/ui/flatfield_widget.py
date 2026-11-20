"""Image de référence : activer la correction du fond, prendre la référence.

Comme dans le logiciel fourni avec la caméra : on place une zone vide de la lame sous
l'objectif, « Prendre la référence », et le fond de toutes les images devient uniforme — dans
le live d'abord, donc aussi dans les captures, les panoramas et les acquisitions.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QMenu, QMessageBox, QVBoxLayout, QWidget
from superqt.utils import create_worker

from fuji.core.flatfield import DEFAULT_FRAMES, Candidate
from fuji.core.microscope import Microscope

from .components import button, hbox, hint, icon_button
from .prefs import settings

TITLE = "Image de référence"


class FlatFieldWidget(QWidget):
    message = Signal(str)
    #: relais vers le fil de l'interface (la correction prévient depuis celui de la caméra)
    _changed = Signal()
    _progress = Signal(int, int)

    def __init__(self, microscope: Microscope, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.m = microscope
        self.ff = microscope.flatfield
        self._busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.enable = QCheckBox("Corriger le fond")
        self.enable.setToolTip(
            "Divise chaque image par l'image de référence : éclairage inégal, vignettage et "
            "poussières disparaissent du live, des captures et des panoramas."
        )
        self.enable.toggled.connect(self._toggled)
        self.take_btn = button(
            "Prendre la référence",
            None,
            tooltip="Sur une zone VIDE de la lame, nette et bien éclairée (histogramme vers le "
            f"milieu, rien de saturé). Moyenne de {DEFAULT_FRAMES} images.",
        )
        self.take_btn.clicked.connect(lambda: self.take("flat"))
        self.more_btn = icon_button("gear", "Noir, effacer la référence")
        self.more_btn.setFixedWidth(34)
        menu = QMenu(self.more_btn)
        menu.addAction("Prendre le noir (lumière coupée)…", lambda: self.take("dark"))
        self.clear_action = menu.addAction("Effacer cette référence", self.clear)
        self.more_btn.setMenu(menu)
        self.status = hint()
        lay.addWidget(self.enable)
        lay.addWidget(hbox(self.take_btn, self.more_btn))
        lay.addWidget(self.status)
        self._changed.connect(self._refresh)
        self._progress.connect(self._show_progress)
        self.ff.changed.connect(self._on_changed)
        self._refresh()

    def shutdown(self) -> None:
        try:
            self.ff.changed.disconnect(self._on_changed)
        except (ValueError, RuntimeError):
            pass

    # ------------------------------------------------------------------ état
    def _on_changed(self) -> None:
        try:
            self._changed.emit()
        except RuntimeError:  # panneau déjà détruit (changement de microscope)
            pass

    def _refresh(self) -> None:
        st = self.ff.status()
        self.enable.blockSignals(True)
        self.enable.setChecked(self.ff.enabled)
        self.enable.blockSignals(False)
        self.clear_action.setEnabled(st.has_reference)
        if not self._busy:
            self.status.setText(st.text)
            self.status.setObjectName("warning" if st.state in ("missing", "mismatch") else "hint")
            self.status.style().unpolish(self.status)
            self.status.style().polish(self.status)

    def _toggled(self, on: bool) -> None:
        settings().setValue("flatfield/enabled", on)
        self.ff.set_enabled(on)
        st = self.ff.status()
        self.message.emit(st.text if on else "Correction du fond désactivée")

    # --------------------------------------------------------------- référence
    def take(self, kind: str = "flat") -> None:
        """Mesure une référence (« flat ») ou un noir (« dark ») dans un fil de travail."""
        if self._busy or self.m.camera is None:
            return
        if kind == "dark":
            answer = QMessageBox.information(
                self,
                "Noir",
                "Coupez la lumière (lampe éteinte ou trajet fermé), puis validez.\n\n"
                "Le noir est le signal de la caméra sans lumière ; il est retiré de chaque image. "
                "Utile surtout en fluorescence et en pose longue. Gardez la même exposition ensuite.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Ok:
                return
        self._set_busy(True)
        self._show_progress(0, DEFAULT_FRAMES)
        worker = create_worker(
            self.ff.measure,
            kind,
            DEFAULT_FRAMES,
            self._progress.emit,
            _start_thread=False,
            _ignore_errors=True,
        )
        worker.returned.connect(self._measured)
        worker.errored.connect(self._failed)
        worker.finished.connect(lambda: self._set_busy(False))
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for w in (self.take_btn, self.more_btn, self.enable):
            w.setEnabled(not busy)
        if not busy:
            self._refresh()

    def _show_progress(self, done: int, total: int) -> None:
        if self._busy:
            self.status.setText(f"Mesure en cours… {done}/{total} images (ne touchez à rien)")

    def _failed(self, exc: Exception) -> None:
        QMessageBox.warning(self, TITLE, f"L'image de référence n'a pas pu être prise : {exc}")

    def _measured(self, candidate: Candidate) -> None:
        if candidate.warnings:
            answer = QMessageBox.question(
                self,
                TITLE,
                "\n\n".join(candidate.warnings) + "\n\nUtiliser quand même cette référence ?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.message.emit("Référence abandonnée")
                return
        self.accept(candidate)

    def accept(self, candidate: Candidate) -> None:
        """Enregistre la référence mesurée ; la correction s'active aussitôt."""
        self._busy = False
        path: Any = self.ff.save(candidate)
        if candidate.kind == "dark":
            self.message.emit(
                "Noir enregistré avec la référence"
                if path
                else "Noir mesuré : il sera enregistré avec la prochaine référence"
            )
            return
        settings().setValue("flatfield/enabled", True)
        self.message.emit("Référence enregistrée : le fond est corrigé (live, captures, panoramas)")

    def clear(self) -> None:
        if self.ff.clear():
            self.message.emit("Référence effacée")


__all__ = ["FlatFieldWidget"]
