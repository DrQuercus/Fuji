"""Fenêtre principale : napari au centre, microscope à gauche, outils à droite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.qt import QtViewer
from qtpy.QtCore import QSettings, Qt, QTimer
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QTabWidget,
    QWidget,
)
from superqt.utils import create_worker

from fuji import __version__
from fuji.analysis import Image, Table
from fuji.analysis.imagej import ImageJBridge
from fuji.core.frame import Frame
from fuji.core.microscope import Microscope
from fuji.core.profile import available_profiles, user_profile_dir
from fuji.io import save_image

from .live import LiveController, contrast_for, units_for
from .panels.acquisition_panel import AcquisitionPanel
from .panels.analysis_panel import AnalysisPanel, layer_to_image
from .panels.assistants_panel import AssistantsPanel
from .panels.microscope_panel import MicroscopePanel
from .panels.properties_panel import PropertiesPanel
from .qt_bridge import QtManualActionHandler
from .widgets import ResultsWidget


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    return area


class MainWindow(QMainWindow):
    def __init__(self, microscope: Microscope) -> None:
        super().__init__()
        self.m = microscope
        self.settings = QSettings("Fuji", "Fuji")
        self.setWindowTitle(f"Fuji — {microscope.profile.name}")
        self.resize(1500, 950)
        self._captures = 0

        # visualisation napari
        self.viewer = ViewerModel(title="Fuji")
        self.viewer.scale_bar.visible = True
        self.qt_viewer = QtViewer(self.viewer)
        self.setCentralWidget(self.qt_viewer)

        # matériel
        self.manual_handler = QtManualActionHandler(self)
        microscope.manual_handler = self.manual_handler
        self.live = LiveController(microscope, self.viewer, self)

        # panneaux
        self.microscope_panel = MicroscopePanel(microscope)
        self.assistants = AssistantsPanel(microscope, self.viewer, self.live)
        self.acquisition = AcquisitionPanel(microscope)
        self.analysis = AnalysisPanel(self.viewer)
        self.properties = PropertiesPanel(microscope)

        left = self._dock("Microscope", _scroll(self.microscope_panel), Qt.DockWidgetArea.LeftDockWidgetArea)
        left.setMinimumWidth(300)
        self.tools = QTabWidget()
        self.tools.setDocumentMode(True)
        self.tools.addTab(_scroll(self.assistants), "Assistants")
        self.tools.addTab(_scroll(self.acquisition), "Acquisition")
        self.tools.addTab(_scroll(self.analysis), "Analyse")
        self._props_index = self.tools.addTab(self.properties, "Propriétés")
        right = self._dock("Outils", self.tools, Qt.DockWidgetArea.RightDockWidgetArea)
        right.setMinimumWidth(360)
        layers_dock, controls_dock = self.qt_viewer.dockLayerList, self.qt_viewer.dockLayerControls
        layers_dock.setWindowTitle("Calques")
        controls_dock.setWindowTitle("Affichage")
        for dock in (layers_dock, controls_dock):
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.splitDockWidget(left, layers_dock, Qt.Orientation.Vertical)
        self.tabifyDockWidget(layers_dock, controls_dock)
        layers_dock.raise_()
        self._left_docks = (left, layers_dock)
        self.results_tabs = QTabWidget()
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.tabCloseRequested.connect(self.results_tabs.removeTab)
        self.results_dock = self._dock("Résultats", self.results_tabs, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.results_dock.hide()

        # signaux
        for panel in (self.microscope_panel, self.assistants, self.acquisition, self.analysis):
            panel.message.connect(self.notify)
        self.analysis.table_ready.connect(self.show_table)
        self.acquisition.frame_ready.connect(self._on_acq_frame)
        self.acquisition.acquisition_done.connect(self._on_acq_done)
        self.assistants.calibration.profile_saved.connect(self.microscope_panel.refresh)
        self.live.fps_changed.connect(lambda f: self.fps_label.setText(f"{f:4.1f} im/s"))
        self.live.live_changed.connect(self._on_live_changed)
        self.viewer.events.status.connect(self._on_status)

        self._build_actions()
        self._build_status()
        self.set_expert(self.settings.value("expert", False, type=bool))
        state = self.settings.value("window_state")
        if state is not None:
            self.restoreState(state)
        else:  # première ouverture : microscope en haut à gauche, calques en dessous
            QTimer.singleShot(
                0, lambda: self.resizeDocks(list(self._left_docks), [620, 280], Qt.Orientation.Vertical)
            )

    # ------------------------------------------------------------------ cadre
    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title}")
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(area, dock)
        return dock

    def _action(
        self, text: str, slot: Any, shortcut: str | None = None, *, checkable: bool = False, tip: str = ""
    ) -> QAction:
        act = QAction(text, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.setCheckable(checkable)
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        return act

    def _build_actions(self) -> None:
        self.act_live = self._action(
            "Live", self.live.toggle, "Ctrl+L", checkable=True, tip="Démarrer / arrêter le live (Ctrl+L)"
        )
        self.act_snap = self._action(
            "Capturer", self.snap, "Ctrl+K", tip="Capturer une image avec ses métadonnées (Ctrl+K)"
        )
        act_save = self._action("Enregistrer le calque…", self.save_layer, "Ctrl+S")
        act_open = self._action("Ouvrir une image…", self.open_dialog, "Ctrl+O")
        self.act_expert = self._action(
            "Mode expert", self.set_expert, checkable=True, tip="Affiche tous les réglages avancés"
        )
        tb = self.addToolBar("Principal")
        tb.setObjectName("toolbar_main")
        tb.setMovable(False)
        for a in (self.act_live, self.act_snap, act_save):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_expert)

        mb = self.menuBar()
        m_file = mb.addMenu("&Fichier")
        m_file.addAction(act_open)
        m_file.addAction(act_save)
        m_file.addSeparator()
        m_file.addAction(self._action("Changer de microscope…", self.change_profile))
        m_file.addSeparator()
        m_file.addAction(self._action("Quitter", self.close, "Ctrl+Q"))
        m_scope = mb.addMenu("&Microscope")
        m_scope.addAction(self.act_live)
        m_scope.addAction(self.act_snap)
        m_scope.addSeparator()
        m_scope.addAction(self._action("Informations matériel…", self.show_hardware_info))
        m_scope.addAction(self._action("Dossier des profils", self.show_profile_folder))
        m_an = mb.addMenu("&Analyse")
        m_an.addAction(self._action("Démarrer Fiji (pont ImageJ)", self.start_imagej))
        m_an.addAction(self._action("Exécuter une macro ImageJ…", self.macro_dialog))
        m_help = mb.addMenu("&Aide")
        m_help.addAction(self._action("À propos de Fuji", self.about))

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.optics_label = QLabel()
        self.fps_label = QLabel()
        sb.addPermanentWidget(self.optics_label)
        sb.addPermanentWidget(self.fps_label)
        self.m.objective_changed.connect(lambda *_: self._update_optics())
        self.m.channel_changed.connect(lambda *_: self._update_optics())
        self.m.profile_changed.connect(self._update_optics)
        self._update_optics()

    def _update_optics(self) -> None:
        obj = self.m.objective
        px = self.m.pixel_size_um()
        parts = []
        if obj:
            parts.append(obj.label)
        if self.m.channel:
            parts.append(self.m.channel)
        parts.append(f"{px:.4f} µm/px ({self.m.pixel_size_source()})" if px else "non calibré")
        self.optics_label.setText("  ·  ".join(parts))

    def notify(self, text: str) -> None:
        self.statusBar().showMessage(text, 6000)

    def _on_status(self, event: Any) -> None:
        value = event.value
        if isinstance(value, dict):
            value = value.get("coordinates", "")
        if value:
            self.statusBar().showMessage(str(value))

    def _on_live_changed(self, on: bool) -> None:
        self.act_live.setChecked(on)
        if not on:
            self.fps_label.setText("")

    def set_expert(self, expert: bool) -> None:
        expert = bool(expert)
        self.act_expert.setChecked(expert)
        self.settings.setValue("expert", expert)
        self.tools.setTabVisible(self._props_index, expert)
        self.properties.set_expert(expert)
        self.analysis.set_expert(expert)

    # ------------------------------------------------------------------ images
    def snap(self) -> None:
        self.act_snap.setEnabled(False)
        worker = create_worker(self.m.snap, _start_thread=False)
        worker.returned.connect(self._on_snap)
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Capture", str(e)))
        worker.finished.connect(lambda: self.act_snap.setEnabled(True))
        worker.start()

    def _on_snap(self, frame: Frame) -> None:
        self._captures += 1
        data = self.assistants.illumination.correct(frame.data)
        self.add_frame(
            data, f"Capture {self._captures}", frame.metadata.to_dict(), frame.metadata.pixel_size_um
        )
        self.notify(
            f"Capture {self._captures} : {frame.metadata.channel or ''} {frame.metadata.exposure_ms:.1f} ms"
        )

    def add_frame(
        self,
        data: np.ndarray,
        name: str,
        metadata: dict[str, Any] | None = None,
        pixel_size_um: float | None = None,
    ) -> None:
        rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
        px = pixel_size_um or 1.0
        bits = self.m.camera.bit_depth if self.m.camera else None
        layer = self.viewer.add_image(
            data,
            name=name,
            rgb=rgb,
            scale=(px, px),
            units=units_for(pixel_size_um),
            contrast_limits=contrast_for(data, bits),
        )
        layer.metadata.update(metadata or {})
        layer.metadata["pixel_size_um"] = pixel_size_um

    def _on_acq_frame(self, _event: Any, frame: Frame) -> None:
        self.live.show_frame(frame.data, "Acquisition")

    def _on_acq_done(self, summary: Any) -> None:
        if summary.files and not summary.error:
            answer = QMessageBox.question(
                self, "Acquisition terminée", f"{summary.acquired} images enregistrées.\nOuvrir le résultat ?"
            )
            if answer == QMessageBox.StandardButton.Yes:
                for f in summary.files:
                    self.open_file(f)

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir une image",
            str(Path.home()),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;Tous les fichiers (*)",
        )
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> None:
        path = Path(path)
        try:
            if path.suffix.lower() in (".tif", ".tiff"):
                import tifffile

                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    axes, data = series.axes, series.asarray()
                img = Image.from_file(path)
                px = img.pixel_size_um
            else:
                from skimage import io

                data, axes, px = io.imread(path), "", None
        except Exception as exc:
            QMessageBox.warning(self, "Ouvrir", f"Impossible de lire {path.name} : {exc}")
            return
        rgb = axes.endswith("S") or (not axes and data.ndim == 3 and data.shape[-1] in (3, 4))
        kwargs: dict[str, Any] = {"name": path.stem.replace(".ome", ""), "rgb": rgb}
        nd = data.ndim - (1 if rgb else 0)
        scale = [1.0] * nd
        if px:
            scale[-1] = scale[-2] = px
        if "C" in axes and not rgb:
            c_axis = axes.index("C")
            kwargs["channel_axis"] = c_axis
            scale.pop(c_axis)
        kwargs["scale"] = scale
        if px:
            kwargs["units"] = ("µm",) * len(scale)
        layers = self.viewer.add_image(data, **kwargs)
        for layer in layers if isinstance(layers, list) else [layers]:
            layer.metadata["pixel_size_um"] = px
            layer.metadata["source"] = str(path)
        self.notify(f"Ouvert : {path.name} ({axes or data.shape})")

    def save_layer(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is None:
            QMessageBox.information(self, "Enregistrer", "Sélectionnez un calque à enregistrer.")
            return
        default = str(Path.home() / f"{layer.name}.ome.tif")
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer en OME-TIFF", default, "OME-TIFF (*.ome.tif)"
        )
        if not path:
            return
        img = layer_to_image(layer)
        save_image(path, img.data, pixel_size_um=img.pixel_size_um)
        self.notify(f"Enregistré : {path}")

    def show_table(self, table: Table) -> None:
        self.results_tabs.addTab(ResultsWidget(table), table.name[:30])
        self.results_tabs.setCurrentIndex(self.results_tabs.count() - 1)
        self.results_dock.show()

    # ------------------------------------------------------------------ menus
    def change_profile(self) -> None:
        profiles = available_profiles()
        names = list(profiles)
        name, ok = QInputDialog.getItem(self, "Changer de microscope", "Profil :", names, 0, False)
        if not ok:
            return
        self.settings.setValue("last_profile", name)
        QMessageBox.information(
            self,
            "Changer de microscope",
            f"Le profil « {name} » sera utilisé au prochain démarrage.\nFuji va se fermer : relancez-le.",
        )
        self.close()

    def show_hardware_info(self) -> None:
        import json

        info = {name: dev.describe() for name, dev in self.m.devices.items()}
        text = QPlainTextEdit(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        text.setReadOnly(True)
        box = QMessageBox(self)
        box.setWindowTitle("Matériel")
        box.setText(f"{len(info)} périphérique(s) — profil « {self.m.profile.name} »")
        box.setDetailedText(text.toPlainText())
        box.exec()

    def show_profile_folder(self) -> None:
        QMessageBox.information(
            self,
            "Profils",
            f"Profils utilisateur : {user_profile_dir()}\n"
            f"Profil actuel : {self.m.profile_path or '(non enregistré)'}",
        )

    def start_imagej(self) -> None:
        self.notify("Démarrage de Fiji (Java)… quelques secondes")
        worker = create_worker(ImageJBridge.get().version, _start_thread=False)
        worker.returned.connect(lambda v: self.notify(f"Fiji prêt : ImageJ {v}"))
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Fiji", str(e)))
        worker.start()

    def macro_dialog(self) -> None:
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Macro ImageJ",
            "Macro (appliquée au calque sélectionné) :",
            'run("Gaussian Blur...", "sigma=2");',
        )
        if not ok or not text.strip():
            return
        layer = self.viewer.layers.selection.active
        image = layer_to_image(layer) if layer is not None else None

        def job() -> Any:
            return ImageJBridge.get().run_macro(text, image)

        worker = create_worker(job, _start_thread=False)
        worker.returned.connect(self._on_macro)
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Macro ImageJ", str(e)))
        worker.start()

    def _on_macro(self, res: Any) -> None:
        if res.image is not None:
            self.add_frame(res.image.data, f"{res.image.name} (macro)", None, res.image.pixel_size_um)
        if res.table is not None:
            self.show_table(res.table)
        if res.log.strip():
            self.notify(res.log.strip().splitlines()[-1])

    def about(self) -> None:
        QMessageBox.about(
            self,
            "À propos de Fuji",
            f"<b>Fuji {__version__}</b><br>Acquisition multi-microscopes, assistants "
            "d'alignement et analyse d'images.<br><br>Construit sur napari, pymmcore-plus "
            "(Micro-Manager) et PyImageJ (Fiji).",
        )

    # ------------------------------------------------------------------ fermeture
    def closeEvent(self, event: Any) -> None:
        self.settings.setValue("window_state", self.saveState())
        try:
            if self.acquisition.engine.is_running:
                self.acquisition.engine.cancel()
                self.acquisition.engine.wait(5)
            self.live.stop()
            self.m.disconnect()
        finally:
            super().closeEvent(event)
