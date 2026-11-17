"""Fenêtre principale de Fuji.

::

    ┌──────────────────────── barre supérieure ─────────────────────────┐
    │ logo [Microscope|Analyse]  microscope ▾          Live  Capturer   │
    ├────┬───────────────────────────────────────┬──────────────────────┤
    │ na │                                       │ panneau de la section│
    │ vi │   image : live (espace Microscope)    │ (contrôle, images,   │
    │ ga │   ou images ouvertes (espace Analyse) │  outils d'analyse...)│
    │ ti ├───────────────────────────────────────┤                      │
    │ on │ barre rapide (Microscope seulement)   │                      │
    └────┴───────────────────────────────────────┴──────────────────────┘

Deux espaces de travail, chacun avec sa propre vue d'image : **Microscope** (régler, voir
le live, prendre des images) et **Analyse** (ouvrir les images prises, mesurer). Les images
prises arrivent dans la liste « Images », commune aux deux espaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.qt import QtViewer
from qtpy.QtCore import Qt, Signal
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
from fuji.analysis.guide import MENUS, RECIPES, help_of
from fuji.analysis.imagej import ImageJBridge
from fuji.core.frame import Frame
from fuji.core.microscope import Microscope
from fuji.core.profile import MicroscopeProfile, available_profiles
from fuji.io import save_image

from . import documents
from .chrome import PINNED, SECTIONS, Sidebar, TopBar, mode_of, sections_of
from .components import PanelHeader, icon_button
from .compose_panel import ComposePanel
from .config.editor import ConfigEditor
from .config.wizard import NewMicroscopeDialog
from .display_bar import DisplayBar
from .gallery import GalleryPanel
from .live import LiveController, contrast_for, units_for
from .panels.acquisition_panel import AcquisitionPanel
from .panels.analysis_panel import AnalysisPanel, layer_to_image
from .panels.assistants_panel import AssistantsPanel
from .panels.config_panel import ConfigPanel
from .panels.microscope_panel import MicroscopePanel
from .panels.properties_panel import PropertiesPanel
from .panels.settings_panel import SettingsPanel
from .prefs import settings
from .qt_bridge import QtManualActionHandler
from .quickbar import QuickBar
from .theme import ThemeManager, napari_theme_id, palette
from .widgets import ResultsWidget

SUBTITLES = {
    "images": "Enregistrées automatiquement. Double-clic : ouvrir dans Analyse.",
    "control": "Caméra, optique, éclairage, platine et mise au point.",
    "assistants": "Köhler, mise au point, calibration du pixel, éclairage.",
    "acquisition": "Time-lapse, piles Z, canaux, positions → OME-TIFF.",
    "analysis": "Rangés par tâche, chacun avec son explication. Démarches guidées pour débuter.",
    "layers": "Images affichées et réglages d'affichage.",
    "config": "Configuration complète et choix du microscope.",
    "settings": "Apparence, Fiji, caméras ToupTek, dossiers.",
}
TITLES = {key: title for key, title, *_ in SECTIONS}
TITLES.update(
    control="Contrôle du microscope",
    config="Microscopes",
    analysis="Outils d'analyse",
    images="Images prises",
)
#: section du panneau de droite affichée en entrant dans un espace pour la première fois
DEFAULT_SECTION = {"microscope": "acquisition", "analyse": "analysis"}

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


def fix_napari_buttons(qt_viewer: QtViewer) -> None:
    """Les boutons de napari qui passent par ses « actions » (supprimer un calque, recentrer,
    grille, 3D...) cherchent la fenêtre napari courante : Fuji n'en a pas (il intègre la vue
    seule), et ces boutons ne faisaient rien. On les relie directement à la bonne vue."""
    viewer = qt_viewer.viewer
    lb, vb = qt_viewer.layerButtons, qt_viewer.viewerButtons

    def toggle_3d() -> None:
        viewer.dims.ndisplay = 2 if viewer.dims.ndisplay == 3 else 3

    def toggle_grid() -> None:
        canvas = getattr(viewer, "canvas", None)  # napari ≥ 0.9 ; avant : viewer.grid
        grid = canvas.grid if canvas is not None and hasattr(canvas, "grid") else viewer.grid
        grid.enabled = not grid.enabled

    slots = (
        (lb.deleteButton, viewer.layers.remove_selected, "Fermer les calques sélectionnés"),
        (vb.resetViewButton, viewer.reset_view, "Recentrer l'image (Ctrl+R)"),
        (vb.ndisplayButton, toggle_3d, "Vue 2D / 3D (piles Z)"),
        (vb.rollDimsButton, viewer.dims.roll, "Faire tourner l'ordre des axes"),
        (vb.transposeDimsButton, viewer.dims.transpose, "Échanger les deux axes affichés"),
        (vb.gridViewButton, toggle_grid, "Calques côte à côte (grille)"),
    )
    for button, slot, tip in slots:
        try:
            button.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        button.clicked.connect(lambda _=False, f=slot: f())
        button.setToolTip(tip)
    vb.consoleButton.hide()  # la console Python de napari n'existe pas dans Fuji


class LayersPanel(QWidget):
    """Les widgets de calques de napari, regroupés dans le panneau latéral."""

    def __init__(self, qt_viewer: QtViewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        fix_napari_buttons(qt_viewer)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(qt_viewer.layerButtons)
        lay.addWidget(qt_viewer.layers, 2)
        lay.addWidget(qt_viewer.viewerButtons)
        # les réglages d'un calque (masque, formes) sont hauts : ils défilent plutôt que
        # d'imposer leur hauteur à la fenêtre (elle doit tenir sur un écran de portable)
        lay.addWidget(_scroll(qt_viewer.controls), 3)


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
        self.settings = settings()
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

        # visualisation napari : une vue par espace de travail -------------------
        # Microscope : live, aides des assistants, acquisitions, compositions en direct
        self.scope_viewer = ViewerModel(title="Fuji — microscope")
        _scale_bar(self.scope_viewer).visible = True
        self.scope_qt = QtViewer(self.scope_viewer)
        # Analyse : images ouvertes, ROI, résultats
        self.viewer = ViewerModel(title="Fuji — analyse")
        _scale_bar(self.viewer).visible = True
        self.qt_viewer = QtViewer(self.viewer)
        self._mode: str | None = None
        self._section: str | None = None
        self._live_paused = False
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
        self.canvas_stack = QStackedWidget()
        self.canvas_stack.setMinimumWidth(320)
        pages = []
        for qt in (self.scope_qt, self.qt_viewer):
            page = QWidget()
            page.setObjectName("CanvasFrame")
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            page_lay = QVBoxLayout(page)
            page_lay.setContentsMargins(0, 0, 0, 0)
            page_lay.setSpacing(0)
            page_lay.addWidget(qt, 1)
            self.canvas_stack.addWidget(page)
            pages.append(page_lay)
        self.canvas_layout = pages[0]  # la barre rapide s'y ajoute (espace Microscope)
        self.quickbar: QuickBar | None = None
        # espace Analyse : histogramme et contraste de l'image affichée, toujours sous l'image
        self.display_bar = DisplayBar(self.viewer)
        self.display_bar.message.connect(self.notify)
        self.display_bar.close_requested.connect(lambda: self.close_image())
        self.display_bar.document_chosen.connect(self.show_document)
        pages[1].addWidget(self.display_bar)

        # panneau de gauche : toujours ouvert (Contrôle, ou Images en analyse)
        self.left = QWidget()
        self.left.setObjectName("SidePanel")
        self.left.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.left.setMinimumWidth(280)
        left_lay = QVBoxLayout(self.left)
        left_lay.setContentsMargins(14, 16, 12, 12)
        left_lay.setSpacing(8)
        left_head = QHBoxLayout()
        self.left_header = PanelHeader("", "")
        fold = icon_button("left", "Replier ce panneau (le rouvrir : son bouton dans la barre de gauche)")
        fold.clicked.connect(self.toggle_left_panel)
        left_head.addWidget(self.left_header, 1)
        left_head.addWidget(fold, 0, Qt.AlignmentFlag.AlignTop)
        left_lay.addLayout(left_head)
        self.left_stack = QStackedWidget()
        self._left_pages: dict[str, QWidget] = {}
        for key in PINNED.values():
            holder = QWidget()
            QVBoxLayout(holder).setContentsMargins(0, 0, 0, 0)
            self._left_pages[key] = holder
            self.left_stack.addWidget(holder)
        left_lay.addWidget(self.left_stack, 1)

        self.side = QWidget()
        self.side.setObjectName("SidePanel")
        self.side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.side.setMinimumWidth(320)
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
        self.splitter.addWidget(self.left)
        self.splitter.addWidget(self.canvas_stack)
        self.splitter.addWidget(self.side)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([340, 840, 380])
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
        self.gallery = GalleryPanel()
        self.gallery.open_requested.connect(self.open_capture)
        self.gallery.deleted.connect(self._on_images_deleted)
        self.gallery.renamed.connect(self._on_image_renamed)
        self.gallery.open_file_requested.connect(self.open_dialog)
        self.gallery.message.connect(self.notify)
        self.gallery.count_changed.connect(lambda n: self.sidebar.set_badge("images", f"({n})" if n else ""))
        self._set_page("images", self.gallery)
        self.sidebar.set_badge("images", f"({len(self.gallery.entries())})" if self.gallery.entries() else "")
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
        self.sidebar.pin_toggled.connect(lambda _key: self.toggle_left_panel())
        self.topbar.live_toggled.connect(self._live_toggled)
        self.topbar.capture_clicked.connect(self.snap)
        self.topbar.expert_toggled.connect(self.set_expert)
        self.topbar.profile_selected.connect(self.switch_profile)
        self.topbar.manage_profiles.connect(lambda: self.sidebar.select("config"))
        self.topbar.mode_changed.connect(self.set_mode)
        self.viewer.events.status.connect(self._on_status)
        self.scope_viewer.events.status.connect(self._on_status)
        # double-clic sur le live : centrer le point (platine motorisée) ; sinon le zoom de napari
        self._default_double_click = list(self.scope_viewer.mouse_double_click_callbacks)
        self.scope_viewer.mouse_double_click_callbacks.clear()
        self.scope_viewer.mouse_double_click_callbacks.append(self._on_double_click)

        self.set_microscope(microscope)
        self._build_menus()
        self.set_expert(self.settings.value("expert", False, type=bool))
        geometry = self.settings.value("window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.set_mode(str(self.settings.value("mode", "microscope")))

    # ================================================================ microscope
    def set_microscope(self, scope: Microscope) -> None:
        """(Re)construit tout ce qui dépend du microscope : panneaux, live, barre rapide."""
        if self.m is not None:
            self._teardown()
        self._acquiring = False
        self._live_before_acq = False
        self.m = scope
        scope.manual_handler = self.manual_handler
        self.live = LiveController(scope, self.scope_viewer, self)
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
        # le panneau de gauche est assez large pour tout le contrôle de ce microscope
        self.left.setMinimumWidth(max(300, min(440, self.control.minimumSizeHint().width() + 44)))

        self.assistants = AssistantsPanel(scope, self.scope_viewer, self.live)
        self._set_page("assistants", _scroll(self.assistants))
        self.acquisition = AcquisitionPanel(scope)
        self.acquisition.starting.connect(self._on_acq_starting)
        self.acquisition.frame_ready.connect(self._on_acq_frame)
        self.acquisition.acquisition_done.connect(self._on_acq_done)
        self.acquisition.mosaic_ready.connect(self._on_mosaic)
        self.acquisition.bridge.progress.connect(lambda d, t: self.topbar.set_acquisition(d, t, True))
        self.acquisition.bridge.finished.connect(lambda *_: self.topbar.set_acquisition(0, 0, False))
        # en haut : compositions en direct (à la main) ; en dessous : acquisitions programmées
        self.compose = ComposePanel(scope, self.scope_viewer, self.live, corrector=self._live_corrector)
        self.compose.message.connect(self.notify)
        self.compose.done.connect(self._on_composed)
        acq_page = QWidget()
        acq_lay = QVBoxLayout(acq_page)
        acq_lay.setContentsMargins(0, 0, 0, 0)
        acq_lay.setSpacing(10)
        acq_lay.addWidget(self.compose)
        acq_lay.addWidget(self.acquisition)
        self._set_page("acquisition", _scroll(acq_page))
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
        self.quickbar.set_compact(self.settings.value("left/microscope", True, type=bool))

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
        for name in ("compose", "control", "assistants"):
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
        # « Contrôle » vit dans le panneau de gauche ; « Images » y passe en analyse (voir set_mode)
        holder = self._left_pages[key] if key == "control" else self._pages[key]
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
    def set_mode(self, mode: str, section: str | None = None) -> None:
        """Passe dans l'espace « microscope » ou « analyse » (et à une de ses sections).

        En passant dans Analyse, le live se met en pause (il reprend au retour) et les outils
        de dessin des ROI se désactivent en sortant d'Analyse."""
        if mode not in DEFAULT_SECTION:
            mode = "microscope"
        if mode != self._mode:
            if self._mode == "microscope" and self.live is not None and self.live.running:
                self._live_paused = True
                self.live.stop()
            if self._mode == "analyse":
                self.analysis.leave()
            self._mode = mode
            self.canvas_stack.setCurrentIndex(0 if mode == "microscope" else 1)
            self.sidebar.set_mode(mode)
            self.topbar.set_mode(mode)
            self._show_pinned(mode)
            self.settings.setValue("mode", mode)
            if mode == "microscope" and self._live_paused:
                self._live_paused = False
                if self.live is not None and not self._acquiring and self.m is not None and self.m.connected:
                    self.live.start()
        wanted = section or str(self.settings.value(f"section/{mode}", DEFAULT_SECTION[mode]))
        if wanted not in sections_of(mode):
            wanted = DEFAULT_SECTION[mode]
        if wanted == PINNED[mode]:  # déjà à gauche : à droite, la section habituelle
            self._set_left_visible(True)
            remembered = str(self.settings.value(f"section/{mode}", DEFAULT_SECTION[mode]))
            wanted = (
                remembered
                if remembered in sections_of(mode) and remembered != wanted
                else DEFAULT_SECTION[mode]
            )
        self.sidebar.select(wanted)

    def _show_pinned(self, mode: str) -> None:
        """Panneau de gauche de l'espace ``mode`` : Contrôle, ou la liste des images (la même
        liste que la section « Images » de l'espace Microscope, déplacée ici)."""
        key = PINNED[mode]
        target = self._left_pages["images"] if mode == "analyse" else self._pages["images"]
        if self.gallery.parentWidget() is not target:
            target.layout().addWidget(self.gallery)
        self.left_stack.setCurrentWidget(self._left_pages[key])
        self.left_header.set_text(TITLES.get(key, key), SUBTITLES.get(key, ""))
        self._set_left_visible(self.settings.value(f"left/{mode}", True, type=bool))

    def _set_left_visible(self, on: bool) -> None:
        self.left.setVisible(on)
        self.sidebar.set_pin_checked(on)
        if self._mode is not None:
            self.settings.setValue(f"left/{self._mode}", on)
        if self.mode == "microscope" and self.quickbar is not None:
            self.quickbar.set_compact(on)  # pas de réglages en double sous l'image

    def toggle_left_panel(self) -> None:
        self._set_left_visible(self.left.isHidden())

    @property
    def mode(self) -> str:
        return self._mode or "microscope"

    def current_viewer(self) -> ViewerModel:
        return self.scope_viewer if self.mode == "microscope" else self.viewer

    def show_section(self, key: str) -> None:
        mode = mode_of(key, self.mode)
        if mode != self._mode:
            self.set_mode(mode, key)
            return
        if key == PINNED[mode]:  # toujours ouverte, à gauche : on la montre simplement
            self._set_left_visible(True)
            return
        if self._section == "analysis" and key != "analysis":
            self.analysis.leave()  # les outils de dessin ne restent pas actifs ailleurs
        self._section = key
        self.side.setVisible(True)
        self.stack.setCurrentWidget(self._pages[key])
        self.panel_header.set_text(TITLES.get(key, key), SUBTITLES.get(key, ""))
        self.settings.setValue(f"section/{mode}", key)

    def toggle_side_panel(self) -> None:
        self.side.setVisible(not self.side.isVisible())

    def set_expert(self, expert: bool) -> None:
        expert = bool(expert)
        self._expert = expert
        self.settings.setValue("expert", expert)
        toggles = [self.topbar.expert, self.settings_panel.expert]
        if hasattr(self, "expert_action"):
            toggles.append(self.expert_action)
        for toggle in toggles:
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
        for viewer in (self.scope_viewer, self.viewer):
            try:
                viewer.theme = napari_theme_id(palette())
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
        m_file.addAction(self._action("Fermer l'image", lambda: self.close_image(), "Ctrl+W"))
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
        m_view.addAction(self._action("Espace Microscope", lambda: self.set_mode("microscope"), "Ctrl+1"))
        m_view.addAction(self._action("Espace Analyse", lambda: self.set_mode("analyse"), "Ctrl+2"))
        m_view.addAction(self._action("Images prises", lambda: self.sidebar.select("images"), "Ctrl+3"))
        m_view.addSeparator()
        self.expert_action = self._action("Mode expert", self.set_expert, checkable=True)
        m_view.addAction(self.expert_action)
        m_view.addAction(
            self._action("Replier / déplier le panneau de droite", self.toggle_side_panel, "Ctrl+B")
        )
        m_view.addAction(
            self._action("Replier / déplier le panneau de gauche", self.toggle_left_panel, "Ctrl+Shift+B")
        )
        m_view.addAction(self._action("Thème clair / sombre", ThemeManager.get().toggle_mode, "Ctrl+Shift+T"))
        m_view.addAction(
            self._action("Recentrer l'image", lambda: self.current_viewer().reset_view(), "Ctrl+R")
        )

        m_an = mb.addMenu("&Analyse")
        # les outils rangés par tâche, comme dans le panneau « Outils » : chacun ouvre sa fiche
        m_recipes = m_an.addMenu("Démarches guidées")
        for recipe in RECIPES:
            act = QAction(recipe.title, self)
            act.setStatusTip(recipe.intro)
            act.triggered.connect(lambda _=False, k=recipe.key: self.open_recipe(k))
            m_recipes.addAction(act)
        m_an.addSeparator()
        for menu in MENUS:
            sub = m_an.addMenu(menu.title)
            for name in menu.operations:
                act = QAction(name, self)
                act.setStatusTip(help_of(name).summary)
                act.triggered.connect(lambda _=False, n=name: self.open_tool(n))
                sub.addAction(act)
        m_an.addSeparator()
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
        if self.live is None or self._acquiring:
            return
        if self.mode != "microscope":  # le live se regarde dans l'espace Microscope
            self._live_paused = False
            self.set_mode("microscope")
            if not self.live.running:
                self.live.start()
            return
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
        """La capture rejoint la liste des images prises (enregistrée tout de suite) ; elle
        s'ouvre dans l'espace Analyse quand on en a besoin."""
        self._captures += 1
        data, corrected = self.assistants.illumination.correct(frame.data)
        metadata = frame.metadata.to_dict()
        note = ""
        if corrected:
            metadata["flat_field_corrected"] = True
            note = " · champ plat corrigé"
        elif corrected is False:
            note = " · ⚠ champ plat NON appliqué (pas de référence pour cet objectif et ce canal)"
        info = self.gallery.add_image(
            data, kind="capture", metadata=metadata, pixel_size_um=frame.metadata.pixel_size_um
        )
        if self.quickbar is not None:
            self.quickbar.show_frame(data)
        self.notify(f"{info.name} enregistrée — liste « Images »{note}")

    def _live_corrector(self) -> Any:
        """Correction de champ plat à appliquer aux images du live pendant une composition
        (si elle est demandée dans l'assistant « Éclairage ») : un panorama sans raccords."""
        illum = self.assistants.illumination
        if not illum.apply.isChecked():
            return None
        store, keys = illum.store, illum._keys()

        def correct(data: np.ndarray) -> np.ndarray:
            out = store.correct(data, *keys)
            return data if out is None else out

        return correct

    def _on_composed(self, image: np.ndarray, kind: str, metadata: Any, px: Any, summary: str) -> None:
        info = self.gallery.add_image(
            image, kind=kind, metadata=metadata, pixel_size_um=px, summary=summary or None
        )
        self.notify(f"{info.name} : dans la liste « Images » (double-clic pour l'ouvrir dans Analyse)")

    def _on_mosaic(self, image: np.ndarray, name: str, pixel_size_um: float | None) -> None:
        info = self.gallery.add_image(image, kind="mosaique", summary=name, pixel_size_um=pixel_size_um)
        self.notify(f"{name} assemblée : {info.name}, dans la liste « Images »")

    def open_capture(self, info: Any, data: np.ndarray | None = None, keep: bool = False) -> None:
        """Ouvre une image de la liste dans l'espace Analyse, à la place de l'image ouverte
        (``keep`` : à côté, les autres restent ouvertes et cachées). Déjà ouverte : on la
        remontre."""
        if info.name in documents.documents(self.viewer):
            self.show_document(info.name)
        elif data is None and not info.path.exists():
            QMessageBox.warning(self, "Ouvrir", f"Fichier introuvable :\n{info.path}")
            return
        else:
            if not self._make_room(keep):
                return
            if data is not None:
                metadata = {**info.metadata, "source": str(info.path)}
                self.add_frame(data, info.name, metadata, info.pixel_size_um)
            else:
                self.open_file(info.path, doc=info.name)
            self.show_document(info.name)
            self.viewer.reset_view()
        if self.mode != "analyse":
            self.set_mode("analyse", "analysis")

    # ------------------------------------------------------------ images ouvertes
    def show_document(self, doc: str) -> None:
        """Affiche une des images ouvertes ; les autres restent ouvertes, cachées."""
        if doc in documents.documents(self.viewer):
            documents.show_only(self.viewer, doc)

    def _make_room(self, keep: bool) -> bool:
        """Avant d'ouvrir une image : ferme celles qui sont ouvertes (ou les cache si
        ``keep``). Faux si l'utilisateur annule."""
        docs = documents.documents(self.viewer)
        if not docs and not self.viewer.layers:
            return True
        if not keep:
            choice = self._confirm_close(docs, opening=True)
            if choice == "cancel":
                return False
            keep = choice == "keep"
        if keep:
            for layer in self.viewer.layers:
                if documents.doc_of(layer) is not None:
                    layer.visible = False
        else:
            self.analysis.leave()
            documents.close(self.viewer, docs)
        return True

    def close_image(self, doc: str | None = None) -> bool:
        """« Fermer » : l'image affichée et ses résultats (masques, aperçus...). Avec la
        dernière image, les ROI dessinées sont fermées aussi. Les tableaux restent."""
        if doc is None:
            layer = self.display_bar.layer
            doc = documents.doc_of(layer) if layer is not None else None
        if doc is None:
            docs = documents.documents(self.viewer)
            if not docs:
                if self.viewer.layers:  # restent des ROI ou des calques sans image
                    self.viewer.layers.clear()
                return True
            doc = docs[-1]
        if self._confirm_close([doc]) == "cancel":
            return False
        self.analysis.leave()
        documents.close(self.viewer, [doc])
        self.notify(f"{doc} fermée")
        return True

    def _confirm_close(self, docs: list[str], opening: bool = False) -> str:
        """« close », « keep » (garder ouverte, seulement en ouvrant une autre image) ou
        « cancel ». On ne demande que si des résultats seraient perdus : les images, elles,
        sont enregistrées."""
        lost = documents.results_lost(self.viewer, docs)
        if not lost:
            return "close"
        return self._ask_close(docs, lost, opening)

    def _ask_close(self, docs: list[str], lost: list[Any], opening: bool) -> str:
        names = ", ".join(ly.name for ly in lost[:4]) + (" …" if len(lost) > 4 else "")
        which = docs[0] if len(docs) == 1 else f"{len(docs)} images"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Fermer l'image")
        box.setText(f"Fermer {which} et ses résultats ?")
        box.setInformativeText(
            f"{len(lost)} calque(s) de résultats seront fermés : {names}.\n"
            "Les tableaux de résultats restent ; enregistrez un masque avec Fichier → "
            "Enregistrer le calque si vous voulez le garder."
        )
        close_btn = box.addButton("Fermer", QMessageBox.ButtonRole.AcceptRole)
        keep_btn = box.addButton("Garder ouverte", QMessageBox.ButtonRole.ActionRole) if opening else None
        box.addButton("Annuler", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(close_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is close_btn:
            return "close"
        if keep_btn is not None and clicked is keep_btn:
            return "keep"
        return "cancel"

    def _on_images_deleted(self, infos: list[Any]) -> None:
        """Images supprimées de la liste : elles se ferment dans Analyse (sans question)."""
        open_docs = documents.documents(self.viewer)
        gone = [info.name for info in infos if info.name in open_docs]
        if gone:
            self.analysis.leave()
            documents.close(self.viewer, gone)

    def _on_image_renamed(self, old: str, info: Any) -> None:
        documents.rename(self.viewer, old, info.name)
        for layer in documents.layers_of(self.viewer, info.name):
            if documents.is_source(layer):
                layer.metadata["source"] = str(info.path)

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
        documents.tag(layer, name, source=True)

    def _on_acq_starting(self) -> None:
        """Juste avant une acquisition : le live s'arrête et les réglages se verrouillent,
        pour que rien d'autre ne commande le matériel pendant la séquence."""
        self._acquiring = True
        if self.compose.running:  # l'acquisition prend la caméra : la composition s'arrête
            self.compose.cancel()
        self._live_before_acq = self.live is not None and self.live.running
        if self._live_before_acq and self.live is not None:
            self.live.stop()
        self._set_hardware_controls_enabled(False)

    def _on_acq_frame(self, _event: Any, frame: Frame) -> None:
        if self.live is not None:
            self.live.show_frame(frame.data, "Acquisition")

    def _on_acq_done(self, summary: Any) -> None:
        self.topbar.set_acquisition(0, 0, False)
        self._acquiring = False
        self._set_hardware_controls_enabled(True)
        if self._live_before_acq and self.live is not None and self.m is not None and self.m.connected:
            if self.mode == "microscope":
                self.live.start()
            else:
                self._live_paused = True  # reprendra au retour dans l'espace Microscope
        self._live_before_acq = False
        if summary.files and not summary.error:
            for f in summary.files:
                self.gallery.add_existing(f, "acquisition", f"{summary.acquired} images")
            self.notify(f"Acquisition terminée : {summary.acquired} images, dans la liste « Images »")

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir une image",
            str(Path.home()),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;Tous les fichiers (*)",
        )
        if path and self._make_room(keep=False):
            doc = self.open_file(path)
            if doc is not None:
                self.show_document(doc)
                self.viewer.reset_view()
            self.set_mode("analyse")

    def open_file(self, path: str | Path, doc: str | None = None) -> str | None:
        """Ajoute un fichier image à l'espace Analyse ; renvoie le nom de l'image ouverte
        (tous ses canaux forment une seule image, qui se ferme d'un bloc)."""
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
            return None
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
        doc = doc or name
        for layer in layers if isinstance(layers, list) else [layers]:
            layer.metadata["pixel_size_um"] = px
            layer.metadata["source"] = str(path)
            documents.tag(layer, doc, source=True)
        self.notify(f"Ouvert : {path.name} ({axes or data.shape})")
        return doc

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
            qt = self.scope_qt if self.mode == "microscope" else self.qt_viewer
            qt.screenshot(path, flash=False)
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

    def open_tool(self, name: str) -> None:
        """Ouvre la fiche d'un outil d'analyse (menu « Analyse »)."""
        if self.mode != "analyse":
            self.set_mode("analyse", "analysis")
        self.sidebar.select("analysis")
        self.analysis.select_operation(name)

    def open_recipe(self, key: str) -> None:
        """Ouvre une démarche guidée (menu « Analyse »)."""
        if self.mode != "analyse":
            self.set_mode("analyse", "analysis")
        self.sidebar.select("analysis")
        self.analysis.open_recipe(key)

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
            self.gallery.wait_pending()  # une image en cours d'écriture ne doit pas être tronquée
            self._teardown()
        super().closeEvent(event)
