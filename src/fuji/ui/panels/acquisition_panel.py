"""Panneau « Acquisition » : time-lapse, piles Z, canaux, positions → OME-TIFF.

Avant le lancement, le panneau affiche en direct le nombre d'images, la durée,
la taille sur disque et les avertissements (intervalle trop court, actions
manuelles à prévoir, disque plein...).
"""

from __future__ import annotations

from pathlib import Path

import useq
from qtpy.QtCore import QSettings, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fuji.core.acquisition import AcquisitionEngine, AcquisitionSummary, plan_acquisition
from fuji.core.microscope import Microscope
from fuji.io import OMETiffWriter

from ..qt_bridge import AcquisitionBridge
from ..widgets import hint_label, row, section


class AcquisitionPanel(QWidget):
    frame_ready = Signal(object, object)  # useq.MDAEvent, Frame
    acquisition_done = Signal(object)  # AcquisitionSummary
    message = Signal(str)

    def __init__(self, microscope: Microscope, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.m = microscope
        self.engine = AcquisitionEngine(microscope)
        self.bridge = AcquisitionBridge(self)
        self.bridge.connect_engine(self.engine)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.frame.connect(self.frame_ready.emit)
        self.bridge.finished.connect(self._on_finished)
        self._positions: list[tuple[float, float]] = []
        caps = microscope.capabilities()
        settings = QSettings("Fuji", "Fuji")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        # destination
        box, blay = section("Enregistrement")
        self.name = QLineEdit(settings.value("acq/name", "acquisition"))
        default_dir = microscope.profile.save_directory or str(Path.home() / "Fuji" / "acquisitions")
        self.folder = QLineEdit(settings.value("acq/folder", default_dir))
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        form = QFormLayout()
        form.addRow("Nom", self.name)
        form.addRow("Dossier", row(self.folder, browse, stretch_last=False))
        blay.addLayout(form)
        lay.addWidget(box)

        # temps
        box, blay = section("Time-lapse")
        self.use_time = QCheckBox("Activer")
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0, 86_400)
        self.interval.setValue(10)
        self.interval.setSuffix(" s")
        self.loops = QSpinBox()
        self.loops.setRange(1, 100_000)
        self.loops.setValue(10)
        form = QFormLayout()
        form.addRow(self.use_time)
        form.addRow("Intervalle", self.interval)
        form.addRow("Nombre de points", self.loops)
        blay.addLayout(form)
        lay.addWidget(box)

        # Z
        box, blay = section("Pile Z")
        self.use_z = QCheckBox("Activer")
        self.z_range = QDoubleSpinBox()
        self.z_range.setRange(0, 1000)
        self.z_range.setValue(10)
        self.z_range.setSuffix(" µm")
        self.z_step = QDoubleSpinBox()
        self.z_step.setRange(0.01, 100)
        self.z_step.setValue(1)
        self.z_step.setSuffix(" µm")
        form = QFormLayout()
        form.addRow(self.use_z)
        form.addRow("Étendue (centrée)", self.z_range)
        form.addRow("Pas", self.z_step)
        blay.addLayout(form)
        if not caps["focus"]:
            box.setEnabled(False)
            blay.addWidget(hint_label("Pas de mise au point déclarée dans ce profil."))
        elif not caps["focus_motorized"]:
            blay.addWidget(
                hint_label("Mise au point manuelle : Fuji indiquera de combien tourner la vis à chaque plan.")
            )
        self._z_nyquist = hint_label()
        blay.addWidget(self._z_nyquist)
        lay.addWidget(box)

        # canaux
        box, blay = section("Canaux")
        self.channels = QListWidget()
        self.channels.setMaximumHeight(110)
        for name in microscope.channel_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name == microscope.channel else Qt.CheckState.Unchecked
            )
            self.channels.addItem(item)
        blay.addWidget(self.channels)
        if not microscope.channel_names:
            blay.addWidget(hint_label("Aucun canal défini : réglages actuels utilisés."))
        lay.addWidget(box)

        # positions
        if caps["xy_motorized"]:
            box, blay = section("Positions")
            self.pos_list = QListWidget()
            self.pos_list.setMaximumHeight(90)
            add = QPushButton("Ajouter la position actuelle")
            add.clicked.connect(self._add_position)
            clear = QPushButton("Vider")
            clear.clicked.connect(self._clear_positions)
            blay.addWidget(self.pos_list)
            blay.addWidget(row(add, clear))
            lay.addWidget(box)

        # plan + commandes
        self.plan_label = QLabel()
        self.plan_label.setWordWrap(True)
        self.warn_label = QLabel()
        self.warn_label.setWordWrap(True)
        self.warn_label.setObjectName("warning")
        lay.addWidget(self.plan_label)
        lay.addWidget(self.warn_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.start_btn = QPushButton("Démarrer l'acquisition")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._pause)
        self.stop_btn = QPushButton("Arrêter")
        self.stop_btn.clicked.connect(self.engine.cancel)
        for b in (self.pause_btn, self.stop_btn):
            b.setEnabled(False)
        lay.addWidget(self.start_btn)
        lay.addWidget(row(self.pause_btn, self.stop_btn))
        lay.addStretch(1)

        for w in (self.use_time, self.use_z):
            w.toggled.connect(self.update_plan)
        for w in (self.interval, self.z_range, self.z_step):
            w.valueChanged.connect(self.update_plan)
        self.loops.valueChanged.connect(self.update_plan)
        self.channels.itemChanged.connect(self.update_plan)
        self.folder.editingFinished.connect(self.update_plan)
        self.update_plan()

    # ----------------------------------------------------------------- séquence
    def sequence(self) -> useq.MDASequence:
        kwargs: dict = {"axis_order": "tpcz"}
        chans = [
            self.channels.item(i).text()
            for i in range(self.channels.count())
            if self.channels.item(i).checkState() == Qt.CheckState.Checked
        ]
        if chans:
            kwargs["channels"] = chans
        if self.use_time.isChecked() and self.loops.value() > 1:
            kwargs["time_plan"] = {"interval": self.interval.value(), "loops": self.loops.value()}
        if self.use_z.isChecked() and self.m.focus is not None:
            kwargs["z_plan"] = {"range": self.z_range.value(), "step": self.z_step.value()}
        if self._positions:
            kwargs["stage_positions"] = [{"x": x, "y": y} for x, y in self._positions]
        return useq.MDASequence(**kwargs)

    def update_plan(self) -> None:
        try:
            seq = self.sequence()
            plan = plan_acquisition(self.m, seq, self.folder.text())
        except Exception as exc:
            self.plan_label.setText(f"Séquence invalide : {exc}")
            return
        self.plan_label.setText(
            f"<b>{plan.n_frames}</b> image(s) · ≈ {plan.human_duration} · {plan.human_size}"
        )
        self.warn_label.setText("\n".join(f"⚠ {w}" for w in plan.warnings))
        obj = self.m.objective
        if self.use_z.isChecked() and obj is not None:
            # Nyquist axial ≈ λ / (2 n (1 − cos θ)) / 2 ; approximation simple λ/(ON²)/2
            n_medium = 1.515 if obj.immersion == "huile" else (1.33 if obj.immersion == "eau" else 1.0)
            dz = 0.55 * n_medium / (obj.na**2) / 2
            self._z_nyquist.setText(f"Pas conseillé pour {obj.label} : ≈ {dz:.2f} µm (Nyquist).")
        else:
            self._z_nyquist.setText("")

    # ----------------------------------------------------------------- actions
    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier d'enregistrement", self.folder.text())
        if folder:
            self.folder.setText(folder)
            self.update_plan()

    def _add_position(self) -> None:
        if self.m.xy_stage is None:
            return
        x, y = self.m.xy_stage.get_position()
        self._positions.append((x, y))
        self.pos_list.addItem(f"P{len(self._positions)} : X {x:.1f}, Y {y:.1f} µm")
        self.update_plan()

    def _clear_positions(self) -> None:
        self._positions.clear()
        self.pos_list.clear()
        self.update_plan()

    def start(self) -> None:
        if self.engine.is_running:
            return
        seq = self.sequence()
        plan = plan_acquisition(self.m, seq, self.folder.text())
        if plan.warnings:
            answer = QMessageBox.question(
                self,
                "Avant de lancer",
                "\n\n".join(plan.warnings) + "\n\nLancer quand même ?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        settings = QSettings("Fuji", "Fuji")
        settings.setValue("acq/name", self.name.text())
        settings.setValue("acq/folder", self.folder.text())
        writer = OMETiffWriter(self.folder.text(), self.name.text())
        self.progress.setRange(0, plan.n_frames)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.engine.start(seq, writer=writer)
        self.message.emit("Acquisition démarrée")

    def _pause(self, paused: bool) -> None:
        if paused:
            self.engine.pause()
            self.pause_btn.setText("Reprendre")
        else:
            self.engine.resume()
            self.pause_btn.setText("Pause")

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)
        self.progress.setFormat(f"{done} / {total}")

    def _on_finished(self, summary: AcquisitionSummary) -> None:
        self.start_btn.setEnabled(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        files = ", ".join(p.name for p in summary.files)
        if summary.error:
            QMessageBox.warning(self, "Acquisition interrompue", f"Erreur : {summary.error}")
        state = "annulée" if summary.cancelled else "terminée"
        self.message.emit(f"Acquisition {state} : {summary.acquired}/{summary.total} images. {files}")
        self.acquisition_done.emit(summary)
