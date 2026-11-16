"""Fenêtre principale de Fuji.

::

    ┌──────────────────────── barre supérieure ─────────────────────────┐
    │ logo · microscope ▾                         Live  Capturer  Expert │
    ├────┬───────────────────────────────────────┬──────────────────────┤
    │ na │                                       │ panneau de la section│
    │ vi │        image (napari)                 │ (contrôle, assistants│
    │ ga │                                       │  acquisition, ...)   │
    │ ti ├───────────────────────────────────────┤                      │
    │ on │ barre rapide : exposition · objectif · canal · histogramme   │
    └────┴───────────────────────────────────────┴──────────────────────┘
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.qt import QtViewer
from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji import __version__
from fuji.analysis import Table
from fuji.analysis.imagej import ImageJBridge
from fuji.core.frame import Frame
from fuji.core.microscope import Microscope
from fuji.core.profile import MicroscopeProfile, available_profiles
from fuji.io import save_image

from .chrome import SECTIONS, Sidebar, TopBar
from .components import PanelHeader
from .config.editor import ConfigEditor
from .config.wizard import NewMicroscopeDialog
from .live import LiveController, contrast_for, units_for
from .panels.acquisition_panel import AcquisitionPanel
from .panels.analysis_panel import AnalysisPanel, layer_to_image
from .panels.assistants_panel import AssistantsPanel
from .panels.config_panel import ConfigPanel
from .panels.microscope_panel import MicroscopePanel
from .panels.properties_panel import PropertiesPanel
from .panels.settings_panel import SettingsPanel
from .qt_bridge import QtManualActionHandler
from .quickbar import QuickBar
from .theme import ThemeManager, napari_theme_id, palette
from .widgets import ResultsWidget

SUBTITLES = {
    "control": "Caméra, optique, éclairage, platine et mise au point.",
    "assistants": "Köhler, mise au point, calibration du pixel, éclairage.",
    "acquisition": "Time-lapse, piles Z, canaux, positions → OME-TIFF.",
    "analysis": "Opérations calibrées (définitions d'ImageJ) et pont vers Fiji.",
    "layers": "Images affichées et réglages d'affichage.",
    "config": "Configuration complète et choix du microscope.",
    "settings": "Apparence, Fiji, caméras ToupTek, dossiers.",
}
TITLES = {key: title for key, title, _icon, _bottom in SECTIONS}
TITLES.update(control="Contrôle du microscope", config="Microscopes")


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


class LayersPanel(QWidget):
    """Les widgets de calques de napari, regroupés dans le panneau latéral."""

    def __init__(self, qt_viewer: QtViewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(qt_viewer.layerButtons)
        lay.addWidget(qt_viewer.layers, 2)
        lay.addWidget(qt_viewer.viewerButtons)
        lay.addWidget(qt_viewer.controls, 3)


class MainWindow(QMainWindow):
    def __init__(self, microscope: Microscope) -> None:
        super().__init__()
        self.settings = QSettings("Fuji", "Fuji")
        self.resize(1560, 980)
        self._captures = 0
        self._frame_counter = 0
        self.m: Microscope | None = None
        self.live: LiveController | None = None

        # visualisation napari ---------------------------------------------------
        self.viewer = ViewerModel(title="Fuji")
        self.viewer.scale_bar.visible = True
        self.qt_viewer = QtViewer(self.viewer)
        self._apply_viewer_theme()
        ThemeManager.get().changed.connect(self._on_theme_changed)
        self.manual_handler = QtManualActionHandler(self)

        # cadre -------------------------------------------------------------------
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.topbar = TopBar()
        root.addWidget(self.topbar)
        self.sidebar = Sidebar()
        canvas_col = QWidget()
        canvas_col.setObjectName("CanvasFrame")
        canvas_col.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.canvas_layout = QVBoxLayout(canvas_col)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(0)
        self.canvas_layout.addWidget(self.qt_viewer, 1)
        self.quickbar: QuickBar | None = None

        self.side = QWidget()
        self.side.setObjectName("SidePanel")
        self.side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.side.setMinimumWidth(360)
        side_lay = QVBoxLayout(self.side)
        side_lay.setContentsMargins(18, 16, 14, 12)
        side_lay.setSpacing(8)
        self.panel_header = PanelHeader("", "")
        side_lay.addWidget(self.panel_header)
        self.stack = QStackedWidget()
        side_lay.addWidget(self.stack, 1)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.sidebar)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(canvas_col)
        self.splitter.addWidget(self.side)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setSizes([1100, 420])
        body.addWidget(self.splitter, 1)
        root.addLayout(body, 1)
        self.setCentralWidget(central)

        self.results_tabs = QTabWidget()
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.tabCloseRequested.connect(self.results_tabs.removeTab)
        self.results_dock = QDockWidget("Résultats", self)
        self.results_dock.setObjectName("dock_results")
        self.results_dock.setWidget(self.results_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.results_dock)
        self.results_dock.hide()

        # pages indépendantes du microscope --------------------------------------
        self._pages: dict[str, QWidget] = {}
        for key, *_ in SECTIONS:
            holder = QWidget()
            QVBoxLayout(holder).setContentsMargins(0, 0, 0, 0)
            self._pages[key] = holder
            self.stack.addWidget(holder)
        self.analysis = AnalysisPanel(self.viewer)
        self.analysis.table_ready.connect(self.show_table)
        self.analysis.message.connect(self.notify)
        self._set_page("analysis", _scroll(self.analysis))
        self._set_page("layers", LayersPanel(self.qt_viewer))
        self.settings_panel = SettingsPanel()
        self.settings_panel.expert_changed.connect(self.set_expert)
        self.settings_panel.auto_contrast_changed.connect(self._set_auto_contrast)
        self.settings_panel.message.connect(self.notify)
        self._set_page("settings", _scroll(self.settings_panel))

        # barre d'état -----------------------------------------------------------
        sb = self.statusBar()
        self.optics_label = QLabel()
        self.fps_label = QLabel()
        sb.addPermanentWidget(self.optics_label)
        sb.addPermanentWidget(self.fps_label)

        # signaux du cadre -------------------------------------------------------
        self.sidebar.selected.connect(self.show_section)
        self.sidebar.collapse_requested.connect(self.toggle_side_panel)
        self.topbar.live_toggled.connect(self._live_toggled)
        self.topbar.capture_clicked.connect(self.snap)
        self.topbar.expert_toggled.connect(self.set_expert)
        self.topbar.profile_selected.connect(self.switch_profile)
        self.topbar.manage_profiles.connect(lambda: self.sidebar.select("config"))
        self.viewer.events.status.connect(self._on_status)

        self.set_microscope(microscope)
        self._build_menus()
        self.set_expert(self.settings.value("expert", False, type=bool))
        geometry = self.settings.value("window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.sidebar.select(self.settings.value("section", "control"))

    # ================================================================ microscope
    def set_microscope(self, scope: Microscope) -> None:
        """(Re)construit tout ce qui dépend du microscope : panneaux, live, barre rapide."""
        if self.m is not None:
            self._teardown()
        self.m = scope
        scope.manual_handler = self.manual_handler
        self.live = LiveController(scope, self.viewer, self)
        self.live.fps_changed.connect(lambda f: self.fps_label.setText(f"{f:4.1f} im/s"))
        self.live.live_changed.connect(self._on_live_changed)
        self.live.frame_shown.connect(self._on_live_frame)

        self.control = MicroscopePanel(scope)
        self.properties = PropertiesPanel(scope)
        control_tabs = QTabWidget()
        control_tabs.setDocumentMode(True)
        control_tabs.addTab(_scroll(self.control), "Réglages")
        self._props_index = control_tabs.addTab(self.properties, "Toutes les propriétés")
        self.control_tabs = control_tabs
        self._set_page("control", control_tabs)

        self.assistants = AssistantsPanel(scope, self.viewer, self.live)
        self._set_page("assistants", _scroll(self.assistants))
        self.acquisition = AcquisitionPanel(scope)
        self.acquisition.frame_ready.connect(self._on_acq_frame)
        self.acquisition.acquisition_done.connect(self._on_acq_done)
        self.acquisition.bridge.progress.connect(lambda d, t: self.topbar.set_acquisition(d, t, True))
        self.acquisition.bridge.finished.connect(lambda *_: self.topbar.set_acquisition(0, 0, False))
        self._set_page("acquisition", _scroll(self.acquisition))
        self.config_panel = ConfigPanel(scope)
        self.config_panel.edit_requested.connect(self.edit_profile)
        self.config_panel.new_requested.connect(self.new_profile)
        self.config_panel.switch_requested.connect(self.switch_profile)
        self._set_page("config", _scroll(self.config_panel))
        for panel in (self.control, self.assistants, self.acquisition):
            panel.message.connect(self.notify)
        self.assistants.calibration.profile_saved.connect(self._on_profile_saved)

        self.quickbar = QuickBar(scope)
        self.quickbar.message.connect(self.notify)
        self.canvas_layout.addWidget(self.quickbar)

        scope.objective_changed.connect(lambda *_: self._update_optics())
        scope.channel_changed.connect(lambda *_: self._update_optics())
        scope.profile_changed.connect(self._update_optics)
        self.setWindowTitle(f"Fuji — {scope.profile.name}")
        self._refresh_profiles()
        self._update_optics()
        if hasattr(self, "_expert"):
            self.set_expert(self._expert)

    def _teardown(self) -> None:
        """Arrête proprement live et acquisition, puis déconnecte le matériel.

        Tolère les objets Qt déjà détruits (fermeture de l'application).
        """
        try:
            if hasattr(self, "acquisition") and self.acquisition.engine.is_running:
                self.acquisition.engine.cancel()
                self.acquisition.engine.wait(5)
        except RuntimeError:
            pass
        if self.live is not None:
            try:
                self.live.stop()
                self.live.deleteLater()
            except RuntimeError:
                pass
            self.live = None
        if self.quickbar is not None:
            try:
                self.canvas_layout.removeWidget(self.quickbar)
                self.quickbar.deleteLater()
            except RuntimeError:
                pass
            self.quickbar = None
        if self.m is not None:
            self.m.disconnect()

    def _set_page(self, key: str, widget: QWidget) -> None:
        holder = self._pages[key]
        lay = holder.layout()
        while lay.count():
            item = lay.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        lay.addWidget(widget)

    def _refresh_profiles(self) -> None:
        entries: dict[str, str] = {}
        current = None
        for _key, path in available_profiles().items():
            try:
                entries[str(path)] = MicroscopeProfile.load(path).name
            except Exception:
                continue
            if (
                self.m is not None
                and self.m.profile_path is not None
                and path.resolve() == self.m.profile_path.resolve()
            ):
                current = str(path)
        if self.m is not None and current is None:
            key = str(self.m.profile_path) if self.m.profile_path else "__courant__"
            entries[key] = self.m.profile.name
            current = key
        self.topbar.set_profiles(entries, current)
        self.topbar.status.setText("manuel" if any(d.manual for d in self.m.devices.values()) else "motorisé")

    def switch_profile(self, path: str) -> None:
        if not path or path == "__courant__":
            return
        if (
            self.m is not None
            and self.m.profile_path is not None
            and Path(path).resolve() == self.m.profile_path.resolve()
        ):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new = Microscope.from_file(path)
            new.connect()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self, "Connexion impossible", f"Impossible de se connecter à ce microscope :\n\n{exc}"
            )
            self._refresh_profiles()
            return
        QApplication.restoreOverrideCursor()
        self.set_microscope(new)
        self.settings.setValue("last_profile", str(new.profile_path))
        self.notify(f"Microscope : {new.profile.name}")

    def reload_microscope(self) -> None:
        if self.m is None or self.m.profile_path is None:
            return
        path = str(self.m.profile_path)
        self._teardown()
        self.m = None
        try:
            new = Microscope.from_file(path)
            new.connect()
        except Exception as exc:
            QMessageBox.warning(self, "Connexion impossible", str(exc))
            new = Microscope.from_file("demo")
            new.connect()
        self.set_microscope(new)

    def edit_profile(self, path: Path | None = None) -> None:
        assert self.m is not None
        if path is None:
            data = self.m.profile.model_dump(mode="json")
            path = self.m.profile_path
        else:
            data = MicroscopeProfile.load(path).model_dump(mode="json")
        editor = ConfigEditor(data, path, self)
        editor.saved.connect(self._after_edit)
        editor.exec()

    def new_profile(self) -> None:
        assert self.m is not None
        dialog = NewMicroscopeDialog(self.m.profile.model_dump(mode="json"), self)
        if dialog.exec() and dialog.data() is not None:
            editor = ConfigEditor(dialog.data(), None, self)
            editor.saved.connect(self._after_edit)
            editor.exec()

    def _after_edit(self, path: Path) -> None:
        self.notify(f"Profil enregistré : {path}")
        is_current = (
            self.m is not None
            and self.m.profile_path is not None
            and Path(path).name == self.m.profile_path.name
        )
        question = (
            "Reconnecter le microscope avec la nouvelle configuration ?"
            if is_current
            else "Utiliser ce microscope maintenant ?"
        )
        if (
            QMessageBox.question(self, "Configuration enregistrée", question)
            == QMessageBox.StandardButton.Yes
        ):
            if self.m is not None:
                self.m.profile_path = None  # forcer la reconnexion
            self.switch_profile(str(path))
        else:
            self._refresh_profiles()
            self.config_panel.refresh()

    def _on_profile_saved(self) -> None:
        self._refresh_profiles()
        self._update_optics()

    # ================================================================ sections
    def show_section(self, key: str) -> None:
        self.side.setVisible(True)
        self.stack.setCurrentWidget(self._pages[key])
        self.panel_header.title.setText(TITLES.get(key, key))
        self.panel_header.subtitle.setText(SUBTITLES.get(key, ""))
        self.settings.setValue("section", key)

    def toggle_side_panel(self) -> None:
        self.side.setVisible(not self.side.isVisible())

    def set_expert(self, expert: bool) -> None:
        expert = bool(expert)
        self._expert = expert
        self.settings.setValue("expert", expert)
        for toggle in (self.topbar.expert, self.settings_panel.expert):
            toggle.blockSignals(True)
            toggle.setChecked(expert)
            toggle.blockSignals(False)
        if hasattr(self, "control_tabs"):
            self.control_tabs.setTabVisible(self._props_index, expert)
            self.control_tabs.tabBar().setVisible(expert)
            self.properties.set_expert(expert)
        self.analysis.set_expert(expert)

    def _set_auto_contrast(self, on: bool) -> None:
        if self.live is not None:
            self.live.auto_contrast = on

    def _on_theme_changed(self, *_: object) -> None:
        from .components import retheme

        self._apply_viewer_theme()
        for top in QApplication.topLevelWidgets():
            retheme(top)

    def _apply_viewer_theme(self) -> None:
        try:
            self.viewer.theme = napari_theme_id(palette())
        except Exception:  # thème non enregistré (napari absent du style)
            pass

    # ================================================================ menus
    def _action(
        self, text: str, slot: Any, shortcut: str | None = None, *, checkable: bool = False
    ) -> QAction:
        act = QAction(text, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.setCheckable(checkable)
        self.addAction(act)
        return act

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m_file = mb.addMenu("&Fichier")
        m_file.addAction(self._action("Ouvrir une image…", self.open_dialog, "Ctrl+O"))
        m_file.addAction(self._action("Enregistrer le calque…", self.save_layer, "Ctrl+S"))
        m_file.addSeparator()
        m_file.addAction(self._action("Nouveau microscope…", self.new_profile, "Ctrl+N"))
        m_file.addAction(self._action("Modifier le microscope…", lambda: self.edit_profile(None), "Ctrl+E"))
        m_file.addSeparator()
        prefs = self._action("Réglages…", lambda: self.sidebar.select("settings"), "Ctrl+,")
        prefs.setMenuRole(QAction.MenuRole.PreferencesRole)
        m_file.addAction(prefs)
        quit_ = self._action("Quitter", self.close, "Ctrl+Q")
        quit_.setMenuRole(QAction.MenuRole.QuitRole)
        m_file.addAction(quit_)

        m_scope = mb.addMenu("&Microscope")
        m_scope.addAction(self._action("Live", lambda: self.live and self.live.toggle(), "Ctrl+L"))
        m_scope.addAction(self._action("Capturer", self.snap, "Ctrl+K"))
        m_scope.addSeparator()
        m_scope.addAction(self._action("Reconnecter", self.reload_microscope))
        m_scope.addAction(self._action("Informations matériel…", self.show_hardware_info))

        m_view = mb.addMenu("&Affichage")
        for i, (key, title, _icon, _b) in enumerate(SECTIONS, start=1):
            m_view.addAction(self._action(title, lambda _=False, k=key: self.sidebar.select(k), f"Ctrl+{i}"))
        m_view.addSeparator()
        m_view.addAction(self._action("Replier / déplier le panneau", self.toggle_side_panel, "Ctrl+B"))
        m_view.addAction(self._action("Thème clair / sombre", ThemeManager.get().toggle_mode, "Ctrl+Shift+T"))
        m_view.addAction(self._action("Recentrer l'image", self.viewer.reset_view, "Ctrl+R"))

        m_an = mb.addMenu("&Analyse")
        m_an.addAction(self._action("Démarrer Fiji (pont ImageJ)", self.start_imagej))
        m_an.addAction(self._action("Exécuter une macro ImageJ…", self.macro_dialog))
        m_help = mb.addMenu("&Aide")
        about = self._action("À propos de Fuji", self.about)
        about.setMenuRole(QAction.MenuRole.AboutRole)
        m_help.addAction(about)

    # ================================================================ état
    def notify(self, text: str) -> None:
        self.statusBar().showMessage(text, 6000)

    def _on_status(self, event: Any) -> None:
        value = event.value
        if isinstance(value, dict):
            value = value.get("coordinates", "")
        if value:
            self.statusBar().showMessage(str(value))

    def _update_optics(self) -> None:
        if self.m is None:
            return
        obj = self.m.objective
        px = self.m.pixel_size_um()
        parts = []
        if obj:
            parts.append(obj.label)
        if self.m.channel:
            parts.append(self.m.channel)
        parts.append(f"{px:.4f} µm/px ({self.m.pixel_size_source()})" if px else "non calibré")
        self.optics_label.setText("  ·  ".join(parts))

    def _live_toggled(self, on: bool) -> None:
        if self.live is None:
            return
        if on and not self.live.running:
            self.live.start()
        elif not on and self.live.running:
            self.live.stop()

    def _on_live_changed(self, on: bool) -> None:
        self.topbar.set_live(on)
        if not on:
            self.fps_label.setText("")

    def _on_live_frame(self, frame: np.ndarray) -> None:
        self._frame_counter += 1
        if self.quickbar is not None and self._frame_counter % 3 == 0:
            self.quickbar.show_frame(frame)

    # ================================================================ images
    def snap(self) -> None:
        if self.m is None:
            return
        self.topbar.capture.setEnabled(False)
        worker = create_worker(self.m.snap, _start_thread=False)
        worker.returned.connect(self._on_snap)
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Capture", str(e)))
        worker.finished.connect(lambda: self.topbar.capture.setEnabled(True))
        worker.start()

    def _on_snap(self, frame: Frame) -> None:
        self._captures += 1
        data = self.assistants.illumination.correct(frame.data)
        self.add_frame(
            data, f"Capture {self._captures}", frame.metadata.to_dict(), frame.metadata.pixel_size_um
        )
        if self.quickbar is not None:
            self.quickbar.show_frame(data)
        md = frame.metadata
        self.notify(f"Capture {self._captures} : {md.channel or ''} {md.exposure_ms or 0:.1f} ms")

    def add_frame(
        self,
        data: np.ndarray,
        name: str,
        metadata: dict[str, Any] | None = None,
        pixel_size_um: float | None = None,
    ) -> None:
        rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
        px = pixel_size_um or 1.0
        bits = self.m.camera.bit_depth if self.m is not None and self.m.camera else None
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
        if self.live is not None:
            self.live.show_frame(frame.data, "Acquisition")

    def _on_acq_done(self, summary: Any) -> None:
        self.topbar.set_acquisition(0, 0, False)
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
        from fuji.analysis import Image

        path = Path(path)
        try:
            if path.suffix.lower() in (".tif", ".tiff"):
                import tifffile

                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    axes, data = series.axes, series.asarray()
                px = Image.from_file(path).pixel_size_um
            else:
                from skimage import io

                data, axes, px = io.imread(path), "", None
        except Exception as exc:
            QMessageBox.warning(self, "Ouvrir", f"Impossible de lire {path.name} : {exc}")
            return
        rgb = axes.endswith("S") or (not axes and data.ndim == 3 and data.shape[-1] in (3, 4))
        kwargs: dict[str, Any] = {"name": path.stem.replace(".ome", ""), "rgb": rgb}
        scale = [1.0] * (data.ndim - (1 if rgb else 0))
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

    # ================================================================ divers
    def show_hardware_info(self) -> None:
        if self.m is None:
            return
        info = {name: dev.describe() for name, dev in self.m.devices.items()}
        box = QMessageBox(self)
        box.setWindowTitle("Matériel")
        box.setText(f"{len(info)} périphérique(s) — « {self.m.profile.name} »")
        box.setDetailedText(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        box.exec()

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
        worker = create_worker(lambda: ImageJBridge.get().run_macro(text, image), _start_thread=False)
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
            "d'alignement et analyse d'images.<br><br>Logiciel libre (BSD-3-Clause), construit "
            "sur napari, pymmcore-plus (Micro-Manager) et PyImageJ (Fiji).",
        )

    def closeEvent(self, event: Any) -> None:
        if not getattr(self, "_closed", False):
            self._closed = True
            self.settings.setValue("window_geometry", self.saveGeometry())
            self._teardown()
        super().closeEvent(event)
