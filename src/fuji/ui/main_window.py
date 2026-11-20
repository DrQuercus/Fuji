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
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.qt import QtViewer
from qtpy.QtCore import QEvent, Qt, QTimer, Signal
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji import __version__
from fuji.analysis import Table
from fuji.analysis.guide import MENUS, RECIPES, help_of
from fuji.analysis.imagej import ImageJBridge
from fuji.core.frame import Frame, is_rgb
from fuji.core.microscope import Microscope
from fuji.core.profile import MicroscopeProfile, available_profiles
from fuji.io import save_image
from fuji.text import count, number, plural

from . import composite, documents
from .chrome import PINNED, SECTIONS, Sidebar, TopBar, mode_of, sections_of
from .components import EmptyState, PanelHeader, icon_button, shortcut
from .compose_panel import ComposePanel
from .config.editor import ConfigEditor
from .config.wizard import NewMicroscopeDialog
from .display_bar import DisplayBar, scale_bar_of
from .gallery import GalleryPanel
from .layers_panel import LayersPanel
from .live import LIVE_LAYER, LiveController, contrast_for, units_for
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
from .results import ResultsWidget, accumulate
from .theme import ThemeManager, napari_stylesheet, napari_theme_id, palette

log = logging.getLogger(__name__)
#: mots qui signalent un échec dans un message destiné à la barre d'état
_PROBLEM_WORDS = (
    "impossible",
    "échec",
    "erreur",
    "n'a pas pu",
    "ne répond pas",
    "non enregistré",
    "non appliqué",
)

SUBTITLES = {
    "images": "Enregistrées automatiquement. Double-clic : ouvrir dans Analyse.",
    "control": "Caméra, optique, éclairage, platine et mise au point.",
    "assistants": "Des aides pas à pas pour régler le microscope.",
    "acquisition": "Images composées en direct, séries dans le temps, piles Z, canaux.",
    "analysis": "Rangés par tâche, chacun avec son explication. Démarches guidées pour débuter.",
    "layers": "Ce qui est posé sur l'image : masques, formes, points.",
    "config": "Configuration complète et choix du microscope.",
    "settings": "Apparence, Fiji, caméras ToupTek, dossiers.",
}
TITLES = {key: title for key, title, *_ in SECTIONS}
TITLES.update(
    control="Contrôle",
    config="Configuration du microscope",
    analysis="Outils",
)
#: images d'une acquisition programmée en cours, montrées dans la vue du microscope
ACQUISITION_LAYER = "Acquisition"
#: section du panneau de droite affichée en entrant dans un espace pour la première fois
DEFAULT_SECTION = {"microscope": "acquisition", "analyse": "analysis"}

#: Au-delà de cette taille, un TIFF est lu à la demande au lieu d'être chargé en mémoire.


_scale_bar = scale_bar_of


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


def _follow_scale(drawing: Any, old_scale: np.ndarray, origin: np.ndarray, px: float) -> None:
    """Met un calque de dessin (« ROI », « Points ») à la nouvelle échelle de l'image, ses
    formes sur les mêmes pixels qu'avant. Sans cela, après « Définir l'échelle », la ligne
    tracée quittait la barre qu'elle mesurait et les mesures suivantes étaient fausses."""
    d_scale = np.asarray(drawing.scale[-2:], dtype=float)
    d_origin = np.asarray(drawing.translate[-2:], dtype=float)

    def to_pixels(vertices: Any) -> np.ndarray:
        v = np.array(vertices, dtype=float)
        v[:, -2:] = (v[:, -2:] * d_scale + d_origin - origin) / old_scale
        return v

    try:
        if len(drawing.data):
            if isinstance(drawing.data, list):  # formes : une liste de sommets
                drawing.data = [to_pixels(v) for v in drawing.data]
            else:
                drawing.data = to_pixels(drawing.data)
        scale, translate = list(drawing.scale), list(drawing.translate)
        scale[-2:], translate[-2:] = [px, px], [float(origin[0]), float(origin[1])]
        drawing.scale, drawing.translate = scale, translate
        drawing.metadata["pixel_size_um"] = px
        drawing.units = ("pixel",) * (len(scale) - 2) + ("µm", "µm")
    except Exception:  # un dessin que l'on ne sait pas déplacer ne doit pas empêcher l'étalonnage
        log.exception("Calque de dessin non remis à l'échelle")


def enable_napari_actions(current: Callable[[], ViewerModel]) -> None:
    """Rend utilisables les commandes de napari qui passent par ses « actions » : boutons de
    mode des réglages d'un calque (dessiner, sélectionner, peindre…), menu du clic droit sur
    la liste des calques. Elles cherchent « la vue courante » auprès de la fenêtre napari, que
    Fuji n'a pas (il intègre la vue seule) : sans cela, chaque clic levait une erreur."""
    try:
        import napari.viewer as napari_viewer
        from napari._qt._qapp_model.qactions import init_qactions

        init_qactions()
        napari_viewer.current_viewer = current
    except Exception:  # interne à napari : peut bouger d'une version à l'autre
        log.exception("Actions de napari indisponibles")


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
        (vb.resetViewButton, viewer.reset_view, f"Recentrer l'image ({shortcut('Ctrl+R')})"),
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
        enable_napari_actions(self.current_viewer)
        try:  # les messages de napari (il n'a pas sa fenêtre pour les montrer)
            from napari.utils.notifications import notification_manager

            notification_manager.notification_ready.connect(self._on_napari_notification)
        except Exception:
            log.debug("Notifications de napari indisponibles")
        # glisser un fichier sur l'image : c'est Fuji qui l'ouvre (napari n'a pas ses lecteurs ici)
        for qt in (self.scope_qt, self.qt_viewer):
            qt.setAcceptDrops(False)
        self.setAcceptDrops(True)
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
        # espace Analyse sans image : une invitation à en ouvrir une, par-dessus la vue vide
        self.empty_hint = EmptyState(
            "gallery",
            "Aucune image ouverte",
            "Double-cliquez une image de la liste « Images »,\nou glissez un fichier sur la fenêtre.",
            parent=self.canvas_stack.widget(1),
        )
        self.empty_hint.setObjectName("CanvasEmpty")
        self.empty_hint.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.qt_viewer.installEventFilter(self)
        self.viewer.layers.events.inserted.connect(self._update_empty_hint)
        self.viewer.layers.events.removed.connect(self._update_empty_hint)
        self.quickbar: QuickBar | None = None
        # espace Analyse : histogramme et contraste de l'image affichée, toujours sous l'image
        self.display_bar = DisplayBar(
            self.viewer, scale_bars=[_scale_bar(self.viewer), _scale_bar(self.scope_viewer)]
        )
        self.display_bar.other_views = [self.scope_viewer]
        for emitter in (self.scope_viewer.layers.events.inserted, self.scope_viewer.layers.events.removed):
            emitter.connect(self.display_bar.sync_scale_bars)
        self.display_bar.sync_scale_bars()
        self.display_bar.message.connect(self.notify)
        self.display_bar.apply_requested.connect(self._apply_display)
        self.display_bar.close_requested.connect(lambda: self.close_image())
        self.display_bar.document_chosen.connect(self.show_document)
        self.display_bar.color_image_requested.connect(lambda: self.color_image())
        self.display_bar.tabs_elsewhere = True  # les onglets au-dessus de l'image s'en chargent
        pages[1].addWidget(self.display_bar)
        # les images ouvertes, en onglets au-dessus de l'image : on passe de l'une à l'autre
        self.doc_tabs = QTabBar()
        self.doc_tabs.setObjectName("DocTabs")
        self.doc_tabs.setDocumentMode(True)
        self.doc_tabs.setTabsClosable(True)
        self.doc_tabs.setExpanding(False)
        self.doc_tabs.setUsesScrollButtons(True)
        self.doc_tabs.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.doc_tabs.setDrawBase(False)
        self.doc_tabs.hide()
        self.doc_tabs.currentChanged.connect(self._doc_tab_changed)
        self.doc_tabs.tabCloseRequested.connect(lambda i: self.close_image(self.doc_tabs.tabData(i)))
        pages[1].insertWidget(0, self.doc_tabs)
        self._shown_doc: str | None = None
        self._tabs_timer = QTimer(self)
        self._tabs_timer.setSingleShot(True)
        self._tabs_timer.setInterval(0)
        self._tabs_timer.timeout.connect(self._sync_doc_tabs)
        self.viewer.layers.events.inserted.connect(self._schedule_doc_tabs)
        self.viewer.layers.events.removed.connect(self._schedule_doc_tabs)

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
        self.results_tabs.tabCloseRequested.connect(self._close_results_tab)
        self.accumulate_box = QCheckBox("Cumuler")
        self.accumulate_box.setToolTip(
            "Ajouter les nouvelles mesures à la suite du tableau du même genre, au lieu d'ouvrir "
            "un onglet à chaque fois : un seul tableau pour toute une série d'images"
        )
        self.accumulate_box.setChecked(self.settings.value("results/accumulate", False, type=bool))
        self.accumulate_box.toggled.connect(lambda on: self.settings.setValue("results/accumulate", on))
        self.results_dock = QDockWidget("Résultats", self)
        self.results_dock.setObjectName("dock_results")
        self.results_dock.setWidget(self.results_tabs)
        # barre de titre aux couleurs de Fuji : titre, « Cumuler », détacher, fermer
        bar = QWidget()
        bar.setObjectName("DockTitle")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(14, 4, 8, 4)
        bar_lay.setSpacing(6)
        title = QLabel("Résultats")
        title.setObjectName("CardTitle")
        bar_lay.addWidget(title)
        bar_lay.addStretch(1)
        bar_lay.addWidget(self.accumulate_box)
        float_btn = icon_button("copy", "Détacher le volet dans sa propre fenêtre (ou le rattacher)")
        float_btn.clicked.connect(lambda: self.results_dock.setFloating(not self.results_dock.isFloating()))
        close_btn = icon_button(
            "close", "Fermer le volet (les tableaux restent : menu Affichage → Résultats)"
        )
        close_btn.clicked.connect(self.results_dock.hide)
        for b in (float_btn, close_btn):
            b.setFixedSize(30, 28)
            bar_lay.addWidget(b)
        self.results_dock.setTitleBarWidget(bar)
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
        fix_napari_buttons(self.qt_viewer)
        qt = self.qt_viewer
        # réglages complets de napari : hauts, ils défilent ; montrés en mode expert seulement
        controls = _scroll(qt.controls)
        controls.setMinimumHeight(260)
        self.layers_panel = LayersPanel(self.viewer, [qt.layerButtons, qt.layers, qt.viewerButtons, controls])
        self.layers_panel.message.connect(self.notify)
        self.layers_panel.save_requested.connect(self.save_layer)
        self.layers_panel.close_requested.connect(lambda doc: self.close_image(doc))
        self._set_page("layers", _scroll(self.layers_panel))
        self.gallery = GalleryPanel()
        self.gallery.open_requested.connect(self.open_capture)
        self.gallery.superpose_requested.connect(self.superpose_captures)
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
        sb.setSizeGripEnabled(False)
        self.cursor_label = QLabel()
        self.cursor_label.setObjectName("hint")
        self.optics_label = QLabel()
        self.fps_label = QLabel()
        sb.addPermanentWidget(self.cursor_label)
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
        for view in (self.viewer, self.scope_viewer):  # image fermée : sa valeur sous la souris s'efface
            view.layers.events.removed.connect(self._on_status)
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
        # image de référence : la correction du fond reprend comme on l'avait laissée
        scope.flatfield.set_enabled(self.settings.value("flatfield/enabled", False, type=bool))
        self.live = LiveController(scope, self.scope_viewer, self)
        self.live.fps_changed.connect(lambda f: self.fps_label.setText(f"{number(f, 1)} im/s"))
        self.live.live_changed.connect(self._on_live_changed)
        self.live.frame_shown.connect(self._on_live_frame)
        self.live.stalled.connect(self.warn)

        self.control = MicroscopePanel(scope)
        self.control.add_channel_requested.connect(lambda: self.edit_profile(page="ChannelsPage"))
        self.properties = PropertiesPanel(scope)
        control_tabs = QTabWidget()
        control_tabs.setDocumentMode(True)
        control_tabs.addTab(_scroll(self.control), "Réglages")
        self._props_index = control_tabs.addTab(self.properties, "Réglages avancés")
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
        self.compose = ComposePanel(scope, self.scope_viewer, self.live)
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

    def edit_profile(self, path: Path | None = None, page: str | None = None) -> None:
        assert self.m is not None
        if path is None:
            data = self.m.profile.model_dump(mode="json")
            path = self.m.profile_path
        else:
            data = MicroscopeProfile.load(path).model_dump(mode="json")
        editor = ConfigEditor(data, path, self)
        editor.saved.connect(self._after_edit)
        if page:
            editor.show_page(page)
        editor.exec()

    def new_profile(self) -> None:
        assert self.m is not None
        dialog = NewMicroscopeDialog(self.m.profile.model_dump(mode="json"), self)
        if dialog.exec() and dialog.data() is not None:
            editor = ConfigEditor(dialog.data(), None, self)
            editor.saved.connect(self._after_edit)
            editor.exec()

    def _after_edit(self, path: Path) -> None:
        self.notify(f"Configuration enregistrée : {path}")
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
        self.layers_panel.set_expert(expert)

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
        # le style de napari ne s'applique qu'à ses propres composants (vues, réglages de calque)
        qss = napari_stylesheet(palette())
        for widget in self._napari_widgets():
            widget.setStyleSheet(qss)

    def _napari_widgets(self) -> list[QWidget]:
        """Les composants qui viennent de napari : eux seuls reçoivent sa feuille de style."""
        qt = self.qt_viewer  # ses réglages de calque sont sortis de la vue, rangés dans « Calques »
        return [self.scope_qt, qt, qt.controls, qt.layers, qt.layerButtons, qt.viewerButtons]

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
        m_file.addAction(
            self._action("Superposer des images (canaux)…", self.superpose_dialog, "Ctrl+Shift+O")
        )
        m_file.addAction(self._action("Enregistrer le calque…", self.save_layer, "Ctrl+S"))
        m_file.addAction(self._action("Exporter la vue (PNG)…", self.export_view, "Ctrl+Shift+E"))
        m_file.addAction(self._action("Image couleur des canaux", lambda: self.color_image()))
        m_file.addSeparator()
        m_file.addAction(self._action("Fermer l'image", lambda: self.close_image(), "Ctrl+W"))
        m_file.addAction(self._action("Fermer toutes les images", self.close_all_images, "Ctrl+Shift+W"))
        m_file.addAction(self._action("Image suivante", lambda: self.next_document(1), "Ctrl+Alt+Right"))
        m_file.addAction(self._action("Image précédente", lambda: self.next_document(-1), "Ctrl+Alt+Left"))
        m_file.addSeparator()
        m_file.addAction(self._action("Nouveau microscope…", self.new_profile, "Ctrl+N"))
        m_file.addAction(self._action("Modifier le microscope…", lambda: self.edit_profile(None), "Ctrl+E"))
        m_file.addSeparator()
        prefs = self._action("Préférences…", lambda: self.sidebar.select("settings"), "Ctrl+,")
        prefs.setMenuRole(QAction.MenuRole.PreferencesRole)
        m_file.addAction(prefs)
        quit_ = self._action("Quitter", self.close, "Ctrl+Q")
        quit_.setMenuRole(QAction.MenuRole.QuitRole)
        m_file.addAction(quit_)

        m_scope = mb.addMenu("&Microscope")
        m_scope.addAction(self._action("Live", self._toggle_live, "Ctrl+L"))
        m_scope.addAction(self._action("Capturer", self.snap, "Ctrl+K"))
        m_scope.addSeparator()
        # compositions en direct et image de référence : comme dans le logiciel de la caméra
        m_scope.addAction(
            self._action("Panorama en direct", lambda: self.start_composition("panorama"), "Ctrl+Shift+P")
        )
        m_scope.addAction(
            self._action(
                "Profondeur de champ étendue en direct", lambda: self.start_composition("edf"), "Ctrl+Shift+D"
            )
        )
        m_scope.addAction(self._action("Prendre l'image de référence", self.take_reference))
        m_scope.addSeparator()
        m_scope.addAction(self._action("Reconnecter", self.reload_microscope))
        m_scope.addAction(self._action("Informations matériel…", self.show_hardware_info))

        m_view = mb.addMenu("&Affichage")
        m_view.addAction(self._action("Espace Microscope", lambda: self.set_mode("microscope"), "Ctrl+1"))
        m_view.addAction(self._action("Espace Analyse", lambda: self.set_mode("analyse"), "Ctrl+2"))
        m_view.addAction(self._action("Liste des images", lambda: self.sidebar.select("images"), "Ctrl+3"))
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
        # sur Mac, ⌘M (réduire la fenêtre) et ⌘H (masquer l'application) sont pris par le système
        mac = sys.platform == "darwin"
        m_an.addAction(self._action("Mesurer", self.quick_measure, "Ctrl+Shift+M" if mac else "Ctrl+M"))
        m_an.addAction(
            self._action(
                "Histogramme",
                lambda: self.quick_operation("Histogramme"),
                "Ctrl+Shift+H" if mac else "Ctrl+H",
            )
        )
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

    def _on_napari_notification(self, notification: Any) -> None:
        text = str(getattr(notification, "message", "") or "")
        if text:
            self._message.emit(text.splitlines()[0][:200])

    def _close_results_tab(self, index: int) -> None:
        widget = self.results_tabs.widget(index)
        self.results_tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()  # sinon le tableau (et ses données) reste en mémoire
        if not self.results_tabs.count():  # plus aucun tableau : le volet vide rend sa place à l'image
            self.results_dock.hide()

    # ------------------------------------------------------------ glisser-déposer
    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:
        """Fichiers déposés sur la fenêtre : ouverts dans Analyse (le premier remplace l'image
        ouverte, les autres s'ouvrent à côté)."""
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        event.acceptProposedAction()
        self.open_paths(paths)

    # ================================================================ état
    def notify(self, text: str) -> None:
        """Un message dans la barre d'état. Celui qui annonce un échec (moteurs, caméra,
        enregistrement…) y reste jusqu'au suivant, au lieu de s'effacer en six secondes."""
        if any(word in text.lower() for word in _PROBLEM_WORDS):
            self.warn(text)
            return
        self.statusBar().showMessage(text, 6000)

    def warn(self, text: str) -> None:
        """Un problème à ne pas manquer : le message reste affiché jusqu'au suivant."""
        log.warning(text)
        self.statusBar().showMessage(f"⚠ {text}")

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if obj is self.qt_viewer and event.type() in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show):
            self._update_empty_hint()
        return super().eventFilter(obj, event)

    def _update_empty_hint(self, *_: Any) -> None:
        """L'invitation couvre la vue d'Analyse tant qu'aucune image n'y est ouverte."""
        empty = len(self.viewer.layers) == 0
        if empty:
            self.empty_hint.setGeometry(self.qt_viewer.geometry())
            self.empty_hint.raise_()
        self.empty_hint.setVisible(empty)

    def _on_status(self, event: Any) -> None:
        """Coordonnées et valeur sous la souris : dans leur propre case, pour ne pas effacer
        les messages (dont les erreurs) affichés à gauche."""
        del event  # napari donne un texte brut (« [1.000 1.000]: [1643, 1654] ») : on écrit le nôtre
        self.cursor_label.setText(self._cursor_text())

    def _cursor_text(self) -> str:
        """« x 512 · y 300 — 1643, 1654, 1628 » : le pixel sous la souris et sa valeur."""
        from napari.layers import Image as ImageLayer

        viewer = self.current_viewer()
        if self.mode == "analyse":
            layer = self.display_bar.layer
        else:
            images = [ly for ly in reversed(viewer.layers) if isinstance(ly, ImageLayer) and ly.visible]
            layer = images[0] if images else None
        if layer is None:
            return ""
        try:
            position = viewer.cursor.position
            data = layer.world_to_data(position)
            y, x = int(round(float(data[-2]))), int(round(float(data[-1])))
            shape = layer.data.shape
            height, width = shape[-3:-1] if layer.rgb else shape[-2:]
            if not (0 <= y < height and 0 <= x < width):
                return ""
            value = layer.get_value(position, world=True)
        except Exception:  # calque en cours de changement
            return ""
        if value is None:
            return f"x {x} · y {y}"
        values = np.atleast_1d(np.asarray(value)).ravel()[:4]
        if np.issubdtype(values.dtype, np.integer):
            shown = ", ".join(str(int(v)) for v in values)
        else:
            shown = ", ".join(number(float(v)) for v in values)
        return f"x {x} · y {y} — {shown}"

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
            self.analysis.add_result(item, "IA — ")
        except Exception as exc:
            self.notify(f"IA : {name} (affichage impossible : {exc})")

    def _on_double_click(self, viewer: Any, event: Any) -> None:
        """Double-clic sur l'image du live : la platine motorisée amène ce point au centre."""
        m = self.m
        box = getattr(getattr(self, "control", None), "click_to_center", None)
        motorized = m is not None and m.xy_stage is not None and not m.xy_stage.manual
        try:
            wanted = bool(motorized and box is not None and box.isChecked())
        except RuntimeError:  # panneau détruit
            wanted = False
        if not wanted or self._acquiring or LIVE_LAYER not in viewer.layers:
            for callback in self._default_double_click:
                callback(viewer, event)
            return
        layer = viewer.layers[LIVE_LAYER]
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
                self._message.emit(f"Platine : X {number(x, 1)} µm, Y {number(y, 1)} µm")
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
        parts.append(f"{number(px)} µm/px ({self.m.pixel_size_source()})" if px else "non calibré")
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
            self.notify("Capture impossible pendant une acquisition : attendez la fin, ou « Arrêter »")
            return
        self.topbar.capture.setEnabled(False)
        worker = create_worker(self.m.snap, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(self._on_snap)
        worker.errored.connect(
            lambda e: QMessageBox.warning(self, "Capture", f"La capture n'a pas abouti : {e}")
        )
        worker.finished.connect(lambda: self.topbar.capture.setEnabled(not self._acquiring))
        worker.start()

    def _on_snap(self, frame: Frame) -> None:
        """La capture rejoint la liste des images prises (enregistrée tout de suite) ; elle
        s'ouvre dans l'espace Analyse quand on en a besoin."""
        data, metadata = frame.data, frame.metadata.to_dict()
        # le fond est corrigé à la sortie de la caméra (image de référence) : rien à faire ici
        note = ""
        if frame.metadata.extra.get("flat_field_corrected"):
            note = " · fond corrigé"
        elif self.m is not None and self.m.flatfield.enabled:
            note = " · ⚠ fond NON corrigé (pas de référence pour cet objectif et ce canal)"
        info = self.gallery.add_image(
            data, kind="capture", metadata=metadata, pixel_size_um=frame.metadata.pixel_size_um
        )
        if self.quickbar is not None:
            self.quickbar.show_frame(data)
        self.notify(f"{info.name} enregistrée — liste « Images »{note}")

    def _on_composed(self, image: np.ndarray, kind: str, metadata: Any, px: Any, summary: str) -> None:
        info = self.gallery.add_image(
            image, kind=kind, metadata=metadata, pixel_size_um=px, summary=summary or None
        )
        self.notify(f"{info.name} : dans la liste « Images » (double-clic pour l'ouvrir dans Analyse)")

    def _on_mosaic(self, image: np.ndarray, name: str, pixel_size_um: float | None) -> None:
        info = self.gallery.add_image(image, kind="mosaique", summary=name, pixel_size_um=pixel_size_um)
        self.notify(f"{name} assemblée : {info.name}, dans la liste « Images »")

    def open_capture(self, info: Any, data: np.ndarray | None = None, keep: bool = True) -> None:
        """Ouvre une image de la liste dans l'espace Analyse, dans un nouvel onglet : les images
        déjà ouvertes le restent. Déjà ouverte : on la remontre. (``keep`` : gardé pour
        compatibilité, on ne remplace plus l'image ouverte.)"""
        doc = documents.find_path(self.viewer, info.path)
        if doc is None:
            if data is None and not info.path.exists():
                QMessageBox.warning(self, "Ouvrir", f"Fichier introuvable :\n{info.path}")
                return
            self._make_room()
            name = documents.unique_name(self.viewer, info.name)
            if data is not None:
                metadata = {**info.metadata, "source": str(info.path)}
                self.add_frame(data, name, metadata, info.pixel_size_um)
                doc = name
            else:
                doc = self.open_file(info.path, doc=name, metadata=dict(info.metadata))
            if doc is None:
                return
            self.show_document(doc)
            self.viewer.reset_view()
        else:
            self.show_document(doc)
        if self.mode != "analyse":
            self.set_mode("analyse", "analysis")

    # ------------------------------------------------------------ images ouvertes
    def show_document(self, doc: str) -> None:
        """Affiche une des images ouvertes ; les autres restent ouvertes, cachées."""
        if doc in documents.documents(self.viewer):
            documents.show_only(self.viewer, doc)
            self._shown_doc = doc
            self.display_bar.refresh_target()
        self._schedule_doc_tabs()

    def _schedule_doc_tabs(self, *_: Any) -> None:
        self._tabs_timer.start()

    def _displayed_doc(self) -> str | None:
        docs = documents.documents(self.viewer)
        if self._shown_doc in docs:
            return self._shown_doc
        self._shown_doc = documents.shown(self.viewer)
        return self._shown_doc

    def _sync_doc_tabs(self) -> None:
        """Les onglets suivent les images ouvertes ; la liste « Images » marque celles-ci."""
        try:
            tabs = self.doc_tabs
        except RuntimeError:  # fenêtre fermée
            return
        docs = documents.documents(self.viewer)
        tabs.blockSignals(True)
        try:
            if [tabs.tabData(i) for i in range(tabs.count())] != docs:
                while tabs.count():
                    tabs.removeTab(0)
                for doc in docs:
                    i = tabs.addTab(doc)
                    tabs.setTabData(i, doc)
            for i, doc in enumerate(docs):
                channels = documents.channels_of(self.viewer, doc)
                path = documents.path_of(self.viewer, doc)
                tip = [doc, path or "image calculée dans Fuji"]
                if len(channels) > 1:
                    tip.append("Canaux : " + ", ".join(documents.channel_name(c) for c in channels))
                tabs.setTabToolTip(i, "\n".join(tip))
            current = self._displayed_doc()
            if current in docs:
                tabs.setCurrentIndex(docs.index(current))
        finally:
            tabs.blockSignals(False)
        tabs.setVisible(bool(docs))
        opened = {p for d in docs if (p := documents.path_of(self.viewer, d))}
        self.gallery.set_open(opened)

    def _doc_tab_changed(self, index: int) -> None:
        doc = self.doc_tabs.tabData(index)
        if doc:
            self.show_document(str(doc))

    def _make_room(self, keep: bool = True) -> bool:
        """Avant d'ouvrir une image : au-delà de :data:`documents.MAX_OPEN` images ouvertes,
        la plus ancienne qui n'a pas de résultat est refermée (elle se rouvre d'un
        double-clic). Les autres restent ouvertes."""
        docs = documents.documents(self.viewer)
        while len(docs) >= documents.MAX_OPEN:
            displayed = documents.shown(self.viewer)
            idle = next(
                (d for d in docs if d != displayed and not documents.results_lost(self.viewer, [d])), None
            )
            if idle is None:
                break
            documents.close(self.viewer, [idle])
            self.notify(f"{idle} refermée ({documents.MAX_OPEN} images ouvertes au plus)")
            docs = documents.documents(self.viewer)
        return True

    # ------------------------------------------------------------ superposition
    def superpose_captures(self, infos: list[Any]) -> str | None:
        """Superpose des images de la liste (une par canal : DAPI, GFP, fond clair…) en une
        image à plusieurs canaux, chacun dans sa couleur. Renvoie son nom."""
        from fuji.analysis.channels import ChannelImage, channel_plane

        if len(infos) < 2:
            self.notify("Choisissez au moins deux images à superposer (⌘ ou Ctrl + clic)")
            return None
        channels: list[ChannelImage] = []
        px: float | None = None
        for info in infos:
            data = self.gallery.data_of(info)
            md = dict(info.metadata or {})
            if data is None:
                try:
                    loaded = composite.read_file(info.path)
                except Exception as exc:
                    QMessageBox.warning(self, "Superposer", f"« {info.name} » ne peut pas être lue : {exc}")
                    return None
                split = composite.split_channels(loaded, self._channel_lookup, md)
                if split is not None:  # une acquisition à plusieurs canaux : tous y passent
                    channels.extend(split)
                    px = px or loaded.pixel_size_um
                    continue
                data = loaded.data
                px = px or loaded.pixel_size_um
            px = px or info.pixel_size_um
            name = str(md.get("channel") or info.name)
            kind, color, emission = self._channel_lookup(name)
            if emission and "emission_nm" not in md:
                md["emission_nm"] = emission
            channels.append(
                channel_plane(
                    data,
                    name,
                    kind=md.get("contrast") or kind,
                    color=md.get("channel_color") or color,
                    taken=[c.color for c in channels],
                    metadata={**md, "channel_source": str(info.path)},
                )
            )
        return self._add_superposition(channels, px)

    def superpose_paths(self, paths: list[str]) -> str | None:
        """« Superposer des fichiers… » : comme depuis la liste, pour des fichiers choisis."""
        from fuji.io.captures import CaptureInfo

        infos = []
        for path in paths:
            p = Path(path)
            infos.append(CaptureInfo(p, metadata=composite.sidecar_metadata(p)))
        return self.superpose_captures(infos)

    def _add_superposition(self, channels: list[Any], px: float | None) -> str | None:
        shapes = {(c.plane_shape, c.data.ndim - (3 if c.rgb else 2)) for c in channels}
        if len(shapes) > 1:
            sizes = " ; ".join(f"{c.name} : {c.plane_shape[1]}×{c.plane_shape[0]}" for c in channels)
            QMessageBox.warning(
                self,
                "Superposer",
                "Ces images n'ont pas la même taille : elles ne peuvent pas être superposées.\n\n"
                f"{sizes}\n\nPrenez-les avec la même résolution de caméra.",
            )
            return None
        seen: dict[str, int] = {}
        for c in channels:  # deux « Fond clair » : on les distingue
            seen[c.name] = seen.get(c.name, 0) + 1
            if seen[c.name] > 1:
                c.name = f"{c.name} {seen[c.name]}"
        names = " + ".join(c.name for c in channels)
        self._make_room()
        doc = documents.unique_name(self.viewer, names if len(names) <= 40 else "Superposition")
        composite.add_channels(self.viewer, doc, channels, pixel_size_um=px)
        self.show_document(doc)
        self.viewer.reset_view()
        if self.mode != "analyse":
            self.set_mode("analyse", "analysis")
        self.notify(
            f"{doc} : {count(len(channels), 'canal', 'canaux')} superposés — réglez chacun sous l'image"
        )
        return doc

    def color_image(self, doc: str | None = None) -> str | None:
        """Fait de l'image à plusieurs canaux affichée une image couleur, telle qu'on la voit
        (pour une figure) : elle s'ouvre à côté, à enregistrer ou à exporter."""
        doc = doc or documents.shown(self.viewer)
        if doc is None:
            return None
        try:
            rgb = composite.flatten(self.viewer, doc)
        except Exception as exc:
            QMessageBox.warning(self, "Image couleur", f"L'image couleur n'a pas pu être faite : {exc}")
            return None
        first = documents.channels_of(self.viewer, doc)[0]
        name = documents.unique_name(self.viewer, f"{doc} (couleur)")
        self.add_frame(rgb, name, {"from": doc}, first.metadata.get("pixel_size_um"))
        self.viewer.layers[name].contrast_limits = (0, 255)
        self.show_document(name)
        self.notify(f"{name} : image couleur prête (Fichier → Enregistrer le calque, ou Exporter la vue)")
        return name

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
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Fermer l'image")
        if not docs:  # un dessin fait sans image
            box.setText("Fermer le dessin en cours ?")
        elif len(docs) == 1:
            box.setText(f"Fermer {docs[0]} et ses résultats ?")
        else:
            box.setText(f"Fermer {len(docs)} images et leurs résultats ?")
        box.setInformativeText(
            f"{count(len(lost), 'résultat')} non {plural(len(lost), 'enregistré')} "
            f"{plural(len(lost), 'sera', 'seront')} {plural(len(lost), 'perdu')} : {names}.\n\n"
            "Les tableaux de mesures restent ouverts. Pour garder un masque : Annuler, puis "
            "Fichier → Enregistrer le calque."
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
        rgb = is_rgb(data)
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
        if self.live is None:
            return
        self.live.show_frame(frame.data, ACQUISITION_LAYER)
        layers = self.scope_viewer.layers
        if LIVE_LAYER in layers and layers[LIVE_LAYER].visible:
            # l'image figée du live cacherait les images acquises : elles passent devant
            layers[LIVE_LAYER].visible = False
            layers.move(layers.index(ACQUISITION_LAYER), len(layers))  # tout au-dessus

    def _end_acquisition_view(self) -> None:
        layers = self.scope_viewer.layers
        if ACQUISITION_LAYER in layers:
            layers.remove(ACQUISITION_LAYER)
        if LIVE_LAYER in layers:
            layers[LIVE_LAYER].visible = True

    def _on_acq_done(self, summary: Any) -> None:
        self.topbar.set_acquisition(0, 0, False)
        self._acquiring = False
        self._set_hardware_controls_enabled(True)
        self._end_acquisition_view()
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

    def superpose_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Superposer des images (une par canal)",
            str(self.gallery.store.root if self.gallery.store.root.exists() else Path.home()),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;Tous les fichiers (*)",
        )
        if paths:
            self.superpose_paths(paths)

    def close_all_images(self) -> None:
        docs = documents.documents(self.viewer)
        if not docs or self._confirm_close(docs) == "cancel":
            return
        self.analysis.leave()
        documents.close(self.viewer, docs)
        self.notify(f"{count(len(docs), 'image')} {plural(len(docs), 'fermée')}")

    def next_document(self, step: int) -> None:
        docs = documents.documents(self.viewer)
        if len(docs) < 2:
            return
        current = self._displayed_doc()
        i = docs.index(current) if current in docs else 0
        self.show_document(docs[(i + step) % len(docs)])

    def open_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Ouvrir une image",
            str(Path.home()),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;Tous les fichiers (*)",
        )
        if paths:
            self.open_paths(paths)

    def open_paths(self, paths: list[str]) -> None:
        """Ouvre des fichiers image dans Analyse, chacun dans son onglet (un fichier déjà
        ouvert est remontré)."""
        doc = None
        for path in paths:
            existing = documents.find_path(self.viewer, path)
            if existing is not None:
                doc = existing
                continue
            self._make_room()
            doc = self.open_file(path) or doc
        if doc is not None:
            self.show_document(doc)
            self.viewer.reset_view()
            self.set_mode("analyse")

    def _channel_lookup(self, name: str) -> tuple[str | None, str | None, float | None]:
        """Ce que la configuration dit d'un canal : (type, couleur, émission)."""
        ch = self.m.profile.channel(name) if self.m is not None else None
        if ch is None:
            return None, None, None
        return ch.kind, ch.color, ch.emission_nm

    def open_file(
        self, path: str | Path, doc: str | None = None, metadata: dict[str, Any] | None = None
    ) -> str | None:
        """Ajoute un fichier image à l'espace Analyse ; renvoie le nom de l'image ouverte.
        Une image à plusieurs canaux s'ouvre en canaux superposés, chacun dans sa couleur
        (ils forment une seule image, qui se ferme d'un bloc). ``metadata`` : renseignements
        connus par ailleurs (liste « Images »), joints à l'image."""
        path = Path(path)
        try:
            loaded = composite.read_file(path)
        except Exception as exc:
            log.warning("Lecture de %s impossible : %s", path, exc)
            QMessageBox.warning(
                self,
                "Ouvrir",
                f"Fuji ne peut pas ouvrir « {path.name} ».\n\nFormats lus : TIFF (dont OME-TIFF), PNG, "
                f"JPEG, BMP ; pour les autres, passez par Fiji.\n\nDétail : {exc}",
            )
            return None
        data, axes, px, rgb = loaded.data, loaded.axes, loaded.pixel_size_um, loaded.rgb
        doc = doc or documents.unique_name(self.viewer, path.stem.replace(".ome", ""))
        md = {**composite.sidecar_metadata(path), **(metadata or {})}
        channels = composite.split_channels(loaded, self._channel_lookup, md)
        if channels is not None:
            composite.add_channels(
                self.viewer, doc, channels, pixel_size_um=px, metadata=md, source=str(path)
            )
            self.notify(f"Ouvert : {path.name} — {count(len(channels), 'canal', 'canaux')} superposés")
            return doc
        if "C" in axes:  # un seul canal : l'axe ne sert à rien
            data = np.take(data, 0, axis=axes.index("C"))
        scale = [1.0] * (data.ndim - (1 if rgb else 0))
        if px:
            scale[-1] = scale[-2] = px
        kwargs: dict[str, Any] = {"rgb": rgb}
        if px:
            kwargs["units"] = ("µm",) * len(scale)
        if data.ndim == (3 if rgb else 2) and not isinstance(data, np.memmap):
            # même contraste que pour une image qui vient d'être prise (centiles robustes)
            kwargs["contrast_limits"] = contrast_for(data, md.get("bit_depth"))
        layer = self.viewer.add_image(data, name=doc, scale=scale, **kwargs)
        layer.metadata.update(md)
        layer.metadata["pixel_size_um"] = px
        layer.metadata["source"] = str(path)
        documents.tag(layer, doc, source=True)
        self.notify(f"Ouvert : {path.name}")
        return doc

    def save_layer(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is None:
            QMessageBox.information(
                self,
                "Enregistrer",
                "Cliquez d'abord, dans « Calques », sur l'image ou le masque à enregistrer.",
            )
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

    def start_composition(self, kind: str) -> None:
        """Lance un panorama (« panorama ») ou une profondeur de champ étendue (« edf ») en
        direct, depuis n'importe où (menu Microscope)."""
        if self.m is None or self.m.camera is None:
            self.notify("Aucune caméra connectée")
            return
        if self._acquiring:
            self.notify("Impossible pendant une acquisition")
            return
        if self.mode != "microscope":
            self.set_mode("microscope")
        self.sidebar.select("acquisition")
        if self.compose.running:
            self.notify("Une composition est déjà en cours : terminez-la ou annulez-la")
            return
        self.compose.start(kind)

    def take_reference(self) -> None:
        """Prend l'image de référence (fond) pour l'objectif et le canal actuels."""
        if self.m is None or self.m.camera is None:
            self.notify("Aucune caméra connectée")
            return
        if self.mode != "microscope":
            self.set_mode("microscope")
        self.control.flatfield.take("flat")

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
            self.notify("Ouvrez d'abord une image (double-clic dans la liste « Images »)")
            return
        self.sidebar.select("analysis")
        self.analysis.run_operation(name, {"image": layer.name, "image1": layer.name, "roi": ROI_LAYER})

    def _apply_display(self, name: str, black: float, white: float, gamma: float) -> None:
        """« Appliquer le contraste aux pixels » : une nouvelle image, telle qu'on la voit.

        Les bornes sont prises telles quelles. En passant par la fiche « Niveaux », un noir
        négatif (courant dès que l'on monte la luminosité) devenait 0 et des bornes non entières
        étaient arrondies : le résultat n'était pas ce que montrait l'écran."""
        from fuji.analysis.ops.adjust import stretch, white_of

        from .display_bar import _INVERTED
        from .panels.analysis_panel import layer_to_image

        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        try:
            image = layer_to_image(layer, self.viewer)
        except Exception as exc:
            self.notify(f"Contraste non appliqué : {exc}")
            return
        doc = documents.doc_of(layer)
        inverted = not layer.rgb and str(layer.colormap.name).endswith(_INVERTED)
        palette_name = str(layer.colormap.name).removesuffix(_INVERTED) if not layer.rgb else None
        gamma = min(max(float(gamma), 0.1), 10.0)

        def done(result: Any) -> None:
            if name not in self.viewer.layers:  # image fermée pendant le calcul
                return
            new = self.analysis.add_result(result, doc=doc)
            if new is None:
                return
            try:  # les pixels sont déjà étirés : affichés du noir au blanc, sans autre réglage
                full = float(white_of(result))
                new.contrast_limits_range = (0.0, full)
                new.contrast_limits = (0.0, full)
                new.gamma = 1.0
                if palette_name:
                    new.colormap = palette_name
            except Exception:
                log.exception("Affichage du résultat non réglé")
            self.notify(f"Contraste appliqué aux pixels : nouvelle image « {new.name} »")

        self.notify("Contraste appliqué aux pixels : calcul…")
        worker = create_worker(
            stretch, image, black, white, gamma, inverted, _start_thread=False, _ignore_errors=True
        )
        worker.returned.connect(done)
        worker.errored.connect(lambda exc: self.notify(f"Contraste non appliqué : {exc}"))
        worker.start()

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
            self.notify("Ouvrez d'abord une image (double-clic dans la liste « Images »)")
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
                f"La dernière ligne tracée mesure {number(length_px, 1)} pixels.\n"
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
        """Applique une taille de pixel à un calque (les mesures suivantes seront en µm). Les
        formes et les points déjà dessinés restent sur les mêmes pixels de l'image."""
        old_scale = np.asarray(layer.scale[-2:], dtype=float)
        origin = np.asarray(layer.translate[-2:], dtype=float)
        scale = list(layer.scale)
        scale[-2:] = [px, px]
        layer.scale = scale
        layer.metadata["pixel_size_um"] = float(px)
        try:
            layer.units = ("pixel",) * (len(scale) - 2) + ("µm", "µm")
        except Exception:  # anciennes versions de napari : pas d'unités par calque
            pass
        if layer in self.viewer.layers:
            for name in (documents.ROI_LAYER, documents.POINTS_LAYER):
                if name in self.viewer.layers:
                    _follow_scale(self.viewer.layers[name], old_scale, origin, float(px))
        self.viewer.reset_view()
        self.notify(f"Échelle de « {layer.name} » : {number(px)} µm/px")

    def show_table(self, table: Table) -> None:
        """Montre un tableau de résultats : dans un nouvel onglet, ou — avec « Cumuler » — à
        la suite du dernier tableau du même genre (mesures d'une série d'images)."""
        tabs = self.results_tabs
        if self.accumulate_box.isChecked():
            for index in reversed(range(tabs.count())):
                widget = tabs.widget(index)
                merged = accumulate(widget.table, table)
                if merged is not None:
                    widget.set_table(merged)
                    tabs.setTabText(index, merged.name[:30])
                    tabs.setCurrentIndex(index)
                    self.results_dock.show()
                    return
        tabs.addTab(ResultsWidget(table), table.name[:30])
        tabs.setCurrentIndex(tabs.count() - 1)
        if self.results_dock.isHidden():
            # à l'ouverture, les résultats prennent un tiers de la hauteur (plus avec un graphique) ;
            # l'image, si elle était ajustée à la fenêtre, se réajuste à la place qui reste
            refit = self.mode == "analyse" and self.display_bar.is_fitted()
            self.results_dock.show()
            height = max(240, min(330 if table.plot is not None else 290, self.height() // 3))
            self.resizeDocks([self.results_dock], [height], Qt.Orientation.Vertical)
            if refit:
                QTimer.singleShot(60, self.display_bar.zoom_fit)

    # ================================================================ divers
    def show_hardware_info(self) -> None:
        if self.m is None:
            return
        info = {name: dev.describe() for name, dev in self.m.devices.items()}
        box = QMessageBox(self)
        box.setWindowTitle("Matériel")
        box.setText(f"{count(len(info), 'périphérique')} — « {self.m.profile.name} »")
        box.setDetailedText(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        box.exec()

    def start_imagej(self) -> None:
        self.notify("Démarrage de Fiji (Java)… quelques secondes")
        worker = create_worker(ImageJBridge.get().version, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(lambda v: self.notify(f"Fiji prêt : ImageJ {v}"))
        worker.errored.connect(lambda e: QMessageBox.warning(self, "Fiji", f"Fiji n'a pas pu démarrer : {e}"))
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
        worker.errored.connect(
            lambda e: QMessageBox.warning(self, "Macro ImageJ", f"La macro n'a pas abouti : {e}")
        )
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
            self.analysis.drawing.shutdown()
        super().closeEvent(event)
