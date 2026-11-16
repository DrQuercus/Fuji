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
from qtpy.QtCore import QSettings, Qt, Signal
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

#: Au-delà de cette taille, un TIFF est lu à la demande au lieu d'être chargé en mémoire.
_MEMMAP_BYTES = 1 << 30
_CHANNEL_COLORMAPS = ("green", "magenta", "cyan", "yellow", "red", "blue")


def _scale_bar(viewer: ViewerModel) -> Any:
    """La barre d'échelle (son emplacement a changé avec napari 0.9)."""
    overlays = getattr(getattr(viewer, "canvas", None), "overlays", None)
    bar = getattr(overlays, "scale_bar", None)
    return bar if bar is not None else viewer.scale_bar


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
    #: objectif, canal ou calibration modifiés. Les signaux du microscope peuvent venir du
    #: thread d'acquisition : ce signal Qt ramène la mise à jour dans le thread de l'interface.
    _optics_changed = Signal()
    #: message à afficher, émis depuis un thread de travail (mouvements des moteurs)
    _message = Signal(str)
    #: l'IA a produit une image ou un résultat (nom) / a fait une action (texte) — émis par le
    #: thread du serveur, traités dans celui de l'interface
    _ai_item = Signal(str)
    _ai_action = Signal(str)

    def __init__(self, microscope: Microscope) -> None:
        super().__init__()
        self.settings = QSettings("Fuji", "Fuji")
        self.resize(1560, 980)
        self._captures = 0
        self._frame_counter = 0
        self.m: Microscope | None = None
        self.live: LiveController | None = None
        self._acquiring = False
        self._live_before_acq = False
        self._optics_changed.connect(self._update_optics)
        self._message.connect(self.notify)
        self._ai_server: Any = None
        self._ai_controller: Any = None
        self._ai_item.connect(self._show_ai_item)
        self._ai_action.connect(lambda text: self.notify(f"IA : {text}"))

        # visualisation napari ---------------------------------------------------
        self.viewer = ViewerModel(title="Fuji")
        _scale_bar(self.viewer).visible = True
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
        self.settings_panel.ai_server_requested.connect(self.set_ai_server)
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
        # double-clic : centrer le point (platine motorisée) ; sinon le zoom habituel de napari
        self._default_double_click = list(self.viewer.mouse_double_click_callbacks)
        self.viewer.mouse_double_click_callbacks.clear()
        self.viewer.mouse_double_click_callbacks.append(self._on_double_click)

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
        self._acquiring = False
        self._live_before_acq = False
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
        self.acquisition.starting.connect(self._on_acq_starting)
        self.acquisition.frame_ready.connect(self._on_acq_frame)
        self.acquisition.acquisition_done.connect(self._on_acq_done)
        self.acquisition.mosaic_ready.connect(lambda image, name, px: self.add_frame(image, name, None, px))
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
        if self.assistants.stage is not None:
            self.assistants.stage.profile_saved.connect(self._on_profile_saved)

        self.quickbar = QuickBar(scope)
        self.quickbar.message.connect(self.notify)
        self.canvas_layout.addWidget(self.quickbar)

        scope.objective_changed.connect(self._emit_optics_changed)
        scope.channel_changed.connect(self._emit_optics_changed)
        scope.profile_changed.connect(self._emit_optics_changed)
        self.setWindowTitle(f"Fuji — {scope.profile.name}")
        self._refresh_profiles()
        self._update_optics()
        self._set_hardware_controls_enabled(True)
        if hasattr(self, "_expert"):
            self.set_expert(self._expert)

    def _teardown(self) -> None:
        """Arrête proprement live et acquisition, puis déconnecte le matériel.

        Tolère les objets Qt déjà détruits (fermeture de l'application).
        """
        # débloquer un thread qui attendrait une réponse à une consigne manuelle
        self.manual_handler.abort_pending()
        try:
            if hasattr(self, "acquisition") and self.acquisition.engine.is_running:
                self.acquisition.engine.cancel()
                self.acquisition.engine.wait(5)
        except RuntimeError:
            pass
        # arrêter les minuteurs des panneaux : ils interrogeraient un matériel déconnecté
        for name in ("control", "assistants"):
            panel = getattr(self, name, None)
            try:
                if panel is not None:
                    panel.shutdown()
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
                self.quickbar.shutdown()
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

    def switch_profile(self, path: str, *, force: bool = False) -> None:
        """Passe au microscope du profil ``path`` (``force`` : même si c'est déjà lui)."""
        if not path or path == "__courant__":
            return
        same = (
            self.m is not None
            and self.m.profile_path is not None
            and Path(path).resolve() == self.m.profile_path.resolve()
        )
        if same and not force:
            return
        if self._refuse_while_acquiring("changer de microscope"):
            self._refresh_profiles()
            return
        self._connect_profile(path)

    def reload_microscope(self) -> None:
        if self.m is None or self.m.profile_path is None:
            return
        if self._refuse_while_acquiring("reconnecter le microscope"):
            return
        self._connect_profile(str(self.m.profile_path))

    def _refuse_while_acquiring(self, what: str) -> bool:
        if not self._acquiring:
            return False
        QMessageBox.information(self, "Acquisition en cours", f"Arrêtez d'abord l'acquisition pour {what}.")
        return True

    def _connect_profile(self, path: str) -> None:
        """Ferme le microscope actuel, puis ouvre celui du profil ``path``.

        L'ancien est fermé **avant** d'ouvrir le nouveau : deux profils peuvent décrire le
        même appareil (la même caméra ne s'ouvre qu'une fois). Si la connexion échoue, on
        revient au microscope précédent et, à défaut, au simulateur.
        """
        previous = str(self.m.profile_path) if self.m is not None and self.m.profile_path else None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        failure: Exception | None = None
        new: Microscope | None = None
        try:
            self._teardown()
            self.m = None
            for candidate in dict.fromkeys(c for c in (path, previous, "demo") if c):
                try:
                    scope = Microscope.from_file(candidate)
                    scope.connect()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    continue
                new = scope
                break
        finally:
            QApplication.restoreOverrideCursor()
        if new is None:  # même le simulateur a échoué : installation cassée
            raise RuntimeError(f"Aucun microscope utilisable : {failure}")
        self.set_microscope(new)
        if failure is None:
            self.settings.setValue("last_profile", str(new.profile_path))
            self.notify(f"Microscope : {new.profile.name}")
        else:
            QMessageBox.warning(
                self,
                "Connexion impossible",
                f"Impossible de se connecter à ce microscope :\n\n{failure}\n\n"
                f"Fuji utilise « {new.profile.name} » en attendant.",
            )

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
            self.switch_profile(str(path), force=True)
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
        m_file.addAction(self._action("Exporter la vue (PNG)…", self.export_view, "Ctrl+Shift+E"))
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
        m_scope.addAction(self._action("Live", self._toggle_live, "Ctrl+L"))
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
        # mêmes gestes que dans ImageJ : mesurer, histogramme, profil sur l'image active
        m_an.addAction(self._action("Mesurer", self.quick_measure, "Ctrl+M"))
        m_an.addAction(self._action("Histogramme", lambda: self.quick_operation("Histogramme"), "Ctrl+H"))
        m_an.addAction(
            self._action(
                "Profil d'intensité", lambda: self.quick_operation("Profil d'intensité"), "Ctrl+Shift+K"
            )
        )
        m_an.addAction(self._action("Définir l'échelle…", self.set_scale_dialog))
        m_an.addSeparator()
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

    def _emit_optics_changed(self, *_: object) -> None:
        try:
            self._optics_changed.emit()
        except RuntimeError:  # fenêtre déjà détruite
            pass

    def _set_hardware_controls_enabled(self, on: bool) -> None:
        """(Dé)verrouille tout ce qui commande le matériel depuis l'interface."""
        widgets: list[Any] = [self.topbar.live, self.topbar.capture, self.topbar.profile_combo]
        widgets += [getattr(self, n, None) for n in ("control_tabs", "assistants", "config_panel")]
        widgets.append(self.quickbar)
        for w in widgets:
            try:
                if w is not None:
                    w.setEnabled(on)
            except RuntimeError:  # objet Qt déjà détruit
                pass

    # ================================================================ assistant IA
    def set_ai_server(self, on: bool, port: int = 8765, hardware: bool = True) -> None:
        """Démarre ou arrête le serveur MCP : une IA externe peut alors piloter ce Fuji.

        Le serveur n'écoute que sur cet ordinateur. L'IA partage le microscope de la
        fenêtre : ses images s'affichent ici, et les manipulations manuelles sont demandées
        à l'écran comme d'habitude."""
        if self._ai_server is not None:
            self._ai_server.stop()
            self._ai_server = self._ai_controller = None
        if not on:
            self.settings_panel.set_ai_state(False, "Arrêté")
            return
        try:
            from fuji.remote import MicroscopeController, SafetyPolicy
            from fuji.remote.mcp_server import HttpServer

            policy = SafetyPolicy(read_only=not hardware)
            policy.workspace.mkdir(parents=True, exist_ok=True)
            controller = MicroscopeController(
                lambda: self.m,
                policy,
                own_manual_handler=False,
                engine=lambda: self.acquisition.engine if hasattr(self, "acquisition") else None,
            )
            controller.item_added.connect(self._ai_item.emit)
            controller.action.connect(self._ai_action.emit)
            server = HttpServer(controller, "127.0.0.1", port)
            server.start()
        except Exception as exc:
            self.settings_panel.set_ai_state(False, f"Impossible de démarrer : {exc}")
            QMessageBox.warning(self, "Assistant IA", f"Le serveur pour IA n'a pas pu démarrer :\n\n{exc}")
            return
        self._ai_server, self._ai_controller = server, controller
        mode = "réglages autorisés" if hardware else "observation et analyse seulement"
        self.settings_panel.set_ai_state(True, f"Actif — {server.url} ({mode})")
        self.notify(f"Serveur pour IA actif : {server.url}")

    def _show_ai_item(self, name: str) -> None:
        """Affiche dans Fuji ce que l'IA vient de produire (image, masque, points, tableau)."""
        controller = self._ai_controller
        if controller is None:
            return
        try:
            item = controller.item(name)
        except Exception:
            return  # déjà remplacé ou oublié
        if isinstance(item, Table):
            self.show_table(item)
            return
        layer_name = f"IA — {name}"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
        try:
            self.analysis._add_layer(item, (0.0, 0.0), "IA — ")
        except Exception as exc:
            self.notify(f"IA : {name} (affichage impossible : {exc})")

    def _on_double_click(self, viewer: Any, event: Any) -> None:
        """Double-clic sur l'image du live : la platine motorisée amène ce point au centre."""
        from .live import LIVE_LAYER

        m = self.m
        box = getattr(getattr(self, "control", None), "click_to_center", None)
        motorized = m is not None and m.xy_stage is not None and not m.xy_stage.manual
        try:
            wanted = bool(motorized and box is not None and box.isChecked())
        except RuntimeError:  # panneau détruit
            wanted = False
        if not wanted or self._acquiring or LIVE_LAYER not in self.viewer.layers:
            for callback in self._default_double_click:
                callback(viewer, event)
            return
        layer = self.viewer.layers[LIVE_LAYER]
        coords = layer.world_to_data(event.position)
        self.center_on(float(coords[-2]), float(coords[-1]), tuple(layer.data.shape))

    def center_on(self, row: float, col: float, shape: tuple[int, ...]) -> None:
        """Amène un pixel de l'image au centre du champ (mouvement dans un thread de travail)."""
        import threading

        from fuji.core.stage import center_on_pixel

        scope = self.m

        def work() -> None:
            try:
                x, y = center_on_pixel(scope, row, col, shape)  # type: ignore[arg-type]
                self._message.emit(f"Platine : X {x:.1f} µm, Y {y:.1f} µm")
            except Exception as exc:
                self._message.emit(f"Centrage impossible : {exc}")

        threading.Thread(target=work, name="fuji-centrage", daemon=True).start()

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

    def _toggle_live(self) -> None:
        if self.live is not None and not self._acquiring:
            self.live.toggle()

    def _live_toggled(self, on: bool) -> None:
        if self.live is None or self._acquiring:
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
        if self._acquiring:
            self.notify("Acquisition en cours : capture indisponible")
            return
        self.topbar.capture.setEnabled(False)
        worker = create_worker(self.m.snap, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(self._on_snap)
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Capture", str(e)))
        worker.finished.connect(lambda: self.topbar.capture.setEnabled(not self._acquiring))
        worker.start()

    def _on_snap(self, frame: Frame) -> None:
        self._captures += 1
        data, corrected = self.assistants.illumination.correct(frame.data)
        metadata = frame.metadata.to_dict()
        note = ""
        if corrected:
            metadata["flat_field_corrected"] = True
            note = " · champ plat corrigé"
        elif corrected is False:
            note = " · ⚠ champ plat NON appliqué (pas de référence pour cet objectif et ce canal)"
        self.add_frame(data, f"Capture {self._captures}", metadata, frame.metadata.pixel_size_um)
        if self.quickbar is not None:
            self.quickbar.show_frame(data)
        md = frame.metadata
        self.notify(f"Capture {self._captures} : {md.channel or ''} {md.exposure_ms or 0:.1f} ms{note}")

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

    def _on_acq_starting(self) -> None:
        """Juste avant une acquisition : le live s'arrête et les réglages se verrouillent,
        pour que rien d'autre ne commande le matériel pendant la séquence."""
        self._acquiring = True
        self._live_before_acq = self.live is not None and self.live.running
        if self._live_before_acq and self.live is not None:
            self.live.stop()
        self._set_hardware_controls_enabled(False)

    def _on_acq_frame(self, _event: Any, frame: Frame) -> None:
        if self.live is not None:
            self.live.show_frame(frame.data, "Acquisition")

    def _on_acq_done(self, summary: Any) -> None:
        self.topbar.set_acquisition(0, 0, False)
        by_hand = self._acquiring  # sinon l'acquisition a été lancée par l'IA : pas de question
        self._acquiring = False
        self._set_hardware_controls_enabled(True)
        if self._live_before_acq and self.live is not None and self.m is not None and self.m.connected:
            self.live.start()
        self._live_before_acq = False
        if by_hand and summary.files and not summary.error:
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
        from fuji.analysis.image import read_pixel_size_um

        path = Path(path)
        try:
            if path.suffix.lower() in (".tif", ".tiff"):
                import tifffile

                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    axes = series.axes
                    px = read_pixel_size_um(tif)
                    data = None
                    if series.size * series.dtype.itemsize > _MEMMAP_BYTES:
                        try:  # grande acquisition : les plans sont lus à la demande
                            data = tifffile.memmap(path, mode="r")
                        except Exception:  # fichier compressé ou en mosaïque
                            data = None
                    if data is None:
                        data = series.asarray()
            else:
                from skimage import io

                data, axes, px = io.imread(path), "", None
        except Exception as exc:
            QMessageBox.warning(self, "Ouvrir", f"Impossible de lire {path.name} : {exc}")
            return
        rgb = axes.endswith("S") or (not axes and data.ndim == 3 and data.shape[-1] in (3, 4))
        name = path.stem.replace(".ome", "")
        scale = [1.0] * (data.ndim - (1 if rgb else 0))
        if px:
            scale[-1] = scale[-2] = px
        kwargs: dict[str, Any] = {"rgb": rgb}
        if "C" in axes and not rgb:
            c_axis = axes.index("C")
            scale.pop(c_axis)
            if isinstance(data, np.memmap):
                # un calque par canal, sans copie (napari chargerait tout en mémoire)
                planes = [data[(slice(None),) * c_axis + (i,)] for i in range(data.shape[c_axis])]
                layers = [
                    self.viewer.add_image(
                        plane,
                        name=f"{name} [C{i + 1}]",
                        scale=scale,
                        units=("µm",) * len(scale) if px else None,
                        blending="additive",
                        colormap=_CHANNEL_COLORMAPS[i % len(_CHANNEL_COLORMAPS)],
                    )
                    for i, plane in enumerate(planes)
                ]
            else:
                kwargs["channel_axis"] = c_axis
                layers = None
        else:
            layers = None
        if layers is None:
            if px:
                kwargs["units"] = ("µm",) * len(scale)
            layers = self.viewer.add_image(data, name=name, scale=scale, **kwargs)
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
        try:
            img = layer_to_image(layer)
            save_image(path, img.data, pixel_size_um=img.pixel_size_um)
        except Exception as exc:
            QMessageBox.warning(self, "Enregistrer", f"Enregistrement impossible : {exc}")
            return
        self.notify(f"Enregistré : {path}")

    def export_view(self) -> None:
        """Image de ce qui est affiché : calques superposés, ROI et barre d'échelle
        (l'équivalent de « Flatten » dans ImageJ, pour les figures)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter la vue", str(Path.home() / "vue.png"), "PNG (*.png)"
        )
        if not path:
            return
        try:
            self.qt_viewer.screenshot(path, flash=False)
        except Exception as exc:
            QMessageBox.warning(self, "Exporter la vue", f"Export impossible : {exc}")
            return
        self.notify(f"Vue exportée : {path}")

    # ================================================================ analyse rapide
    def _active_image_layer(self) -> Any | None:
        from napari.layers import Image as ImageLayer

        from .panels.analysis_panel import PREVIEW_PREFIX

        active = self.viewer.layers.selection.active
        if isinstance(active, ImageLayer) and not active.name.startswith(PREVIEW_PREFIX):
            return active
        images = [ly for ly in reversed(self.viewer.layers) if isinstance(ly, ImageLayer)]
        return images[0] if images else None

    def quick_operation(self, name: str) -> None:
        """Lance une opération sur l'image active, avec les ROI dessinées s'il y en a."""
        from .panels.analysis_panel import ROI_LAYER

        layer = self._active_image_layer()
        if layer is None:
            self.notify("Aucune image à analyser")
            return
        panel = self.analysis
        self.sidebar.select("analysis")
        panel.preview.setChecked(False)
        panel.select_operation(name)
        for key, combo in panel._inputs.items():
            if key in ("image", "image1") and combo.findText(layer.name) >= 0:
                combo.setCurrentText(layer.name)
            elif key == "roi" and combo.findText(ROI_LAYER) >= 0:
                combo.setCurrentText(ROI_LAYER)
        panel._run()

    def quick_measure(self) -> None:
        """« Mesurer » d'ImageJ : les ROI dessinées s'il y en a, sinon l'image entière."""
        from .panels.analysis_panel import ROI_LAYER

        has_rois = ROI_LAYER in self.viewer.layers and len(self.viewer.layers[ROI_LAYER].data) > 0
        self.quick_operation("Mesurer les ROI" if has_rois else "Mesurer")

    def set_scale_dialog(self) -> None:
        """« Set Scale » d'ImageJ : taille du pixel de l'image active, saisie directement
        ou déduite d'une ligne tracée sur une distance connue."""
        from napari.layers import Shapes as ShapesLayer

        from .panels.analysis_panel import ROI_LAYER, layer_to_rois

        layer = self._active_image_layer()
        if layer is None:
            self.notify("Aucune image à calibrer")
            return
        current = float(layer.metadata.get("pixel_size_um") or 0.0)
        length_px = 0.0
        if ROI_LAYER in self.viewer.layers and isinstance(self.viewer.layers[ROI_LAYER], ShapesLayer):
            rois = layer_to_rois(self.viewer.layers[ROI_LAYER], layer, self.viewer)
            lines = [i for i in range(len(rois)) if rois.is_line(i)]
            if lines:
                length_px = rois.length(lines[-1])
        if length_px > 0:
            known, ok = QInputDialog.getDouble(
                self,
                "Définir l'échelle",
                f"La dernière ligne tracée mesure {length_px:.1f} pixels.\n"
                "Distance réelle correspondante (µm) — 0 pour saisir la taille du pixel :",
                0.0,
                0.0,
                1e9,
                3,
            )
            if not ok:
                return
            if known > 0:
                self.set_pixel_size(layer, known / length_px)
                return
        px, ok = QInputDialog.getDouble(
            self, "Définir l'échelle", f"Taille du pixel de « {layer.name} » (µm) :", current, 0.0, 1e6, 5
        )
        if ok and px > 0:
            self.set_pixel_size(layer, px)

    def set_pixel_size(self, layer: Any, px: float) -> None:
        """Applique une taille de pixel à un calque (les mesures suivantes seront en µm)."""
        scale = list(layer.scale)
        scale[-2:] = [px, px]
        layer.scale = scale
        layer.metadata["pixel_size_um"] = float(px)
        try:
            layer.units = ("pixel",) * (len(scale) - 2) + ("µm", "µm")
        except Exception:  # anciennes versions de napari : pas d'unités par calque
            pass
        self.viewer.reset_view()
        self.notify(f"Échelle de « {layer.name} » : {px:.4f} µm/pixel")

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
        worker = create_worker(ImageJBridge.get().version, _start_thread=False, _ignore_errors=True)
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
        worker = create_worker(
            lambda: ImageJBridge.get().run_macro(text, image), _start_thread=False, _ignore_errors=True
        )
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
        if self._ai_server is not None and not self._acquiring:
            self._ai_server.stop()
            self._ai_server = None
        if self._acquiring and not getattr(self, "_closed", False):
            answer = QMessageBox.question(
                self, "Acquisition en cours", "Une acquisition est en cours. L'arrêter et quitter ?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if not getattr(self, "_closed", False):
            self._closed = True
            self.settings.setValue("window_geometry", self.saveGeometry())
            self._teardown()
        super().closeEvent(event)
