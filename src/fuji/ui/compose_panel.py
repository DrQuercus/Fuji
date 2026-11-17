"""Compositions en direct à partir du live : profondeur de champ étendue et panorama.

Pensé pour un microscope **manuel** : on tourne la mise au point, ou on déplace la lame à
la main, et l'image se construit à l'écran (comme le logiciel fourni avec la caméra).
Fonctionne aussi avec des moteurs.

Les calculs se font dans un fil de travail : l'interface et le live restent fluides, et une
image du live qui arrive pendant un calcul est simplement sautée.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from napari.components import ViewerModel
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QProgressBar, QStackedWidget, QVBoxLayout, QWidget
from superqt.utils import create_worker

from fuji.core.compose import FocusStacker, LiveMosaic
from fuji.core.microscope import Microscope

from .components import Card, button, hbox, hint, label
from .live import LIVE_LAYER, LiveController, contrast_for, units_for

COMPOSITION_LAYER = "Composition"
FIELD_LAYER = "Champ actuel"
#: couleur du cadre du champ actuel selon l'état du suivi
STATE_COLORS = {
    "attente": "#f5a524",
    "suivi": "#22c55e",
    "ajouté": "#22c55e",
    "couvert": "#f5a524",
    "perdu": "#ef4444",
}
#: au-delà, le panorama final est assemblé à une résolution réduite (mémoire de l'ordinateur)
MAX_PANORAMA_PIXELS = 80_000_000


class ComposePanel(QWidget):
    message = Signal(str)
    #: image terminée : (image, type « edf » ou « panorama », métadonnées, taille du pixel, résumé)
    done = Signal(object, str, object, object, str)
    _processed = Signal(object)

    def __init__(
        self,
        microscope: Microscope,
        viewer: ViewerModel,
        live: LiveController,
        corrector: Callable[[], Callable[[np.ndarray], np.ndarray] | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.m, self.viewer, self.live = microscope, viewer, live
        self._corrector_factory = corrector
        self._kind: str | None = None
        self._engine: FocusStacker | LiveMosaic | None = None
        #: dépôt de la dernière image du live (tenu très brièvement : le live ne doit pas attendre)
        self._cond = threading.Condition()
        #: le calcul en cours (« retirer le dernier champ » attend qu'il soit fini)
        self._engine_lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._px: float | None = None
        self._metadata: dict[str, Any] = {}
        self._canvas_version = -1
        self._processed.connect(self._show)

        caps = microscope.capabilities()
        manual = not (caps["focus_motorized"] or caps["xy_motorized"])
        card = Card(
            "En direct, à la main",
            icon_name="sparkle",
            subtitle=(
                "Pas besoin de moteurs : l'image se construit pendant que vous tournez la mise au point "
                "ou déplacez la lame."
                if manual
                else "Avec ou sans les moteurs : l'image se construit à partir du live."
            ),
        )
        self.pages = QStackedWidget()
        idle = QWidget()
        il = QVBoxLayout(idle)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(6)
        self.edf_btn = button("Profondeur de champ étendue", "stack", primary=True)
        self.edf_btn.clicked.connect(lambda: self.start("edf"))
        il.addWidget(self.edf_btn)
        il.addWidget(
            hint(
                "Tournez lentement la mise au point fine à travers l'échantillon : chaque zone garde "
                "son état le plus net. Résultat : une image nette partout."
            )
        )
        self.pano_btn = button("Panorama", "mosaic", primary=True)
        self.pano_btn.clicked.connect(lambda: self.start("panorama"))
        il.addWidget(self.pano_btn)
        il.addWidget(
            hint(
                "Déplacez la lame à la main et arrêtez-vous un instant sur chaque champ : il s'ajoute "
                "au panorama (cadre vert). Cadre rouge : revenez sur une zone déjà couverte."
            )
        )
        self.pages.addWidget(idle)

        running = QWidget()
        rl = QVBoxLayout(running)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.title = label("", "CardTitle")
        self.state = label("")
        self.state.setWordWrap(True)
        self.detail = hint()
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setFormat("apport de la dernière image : %p %")
        self.finish_btn = button("Terminer", "check", primary=True)
        self.finish_btn.clicked.connect(self.finish)
        self.cancel_btn = button("Annuler", "close")
        self.cancel_btn.clicked.connect(self.cancel)
        self.undo_btn = button("Retirer le dernier champ", "trash")
        self.undo_btn.clicked.connect(self.undo)
        for w in (self.title, self.state, self.bar, self.detail):
            rl.addWidget(w)
        rl.addWidget(hbox(self.finish_btn, self.cancel_btn))
        rl.addWidget(self.undo_btn)
        self.pages.addWidget(running)
        card.add(self.pages)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(card)

    # ------------------------------------------------------------------ cycle
    @property
    def running(self) -> bool:
        return self._kind is not None

    def start(self, kind: str) -> None:
        """Commence une composition (« edf » ou « panorama ») à partir du live."""
        if self.running or self.m.camera is None:
            return
        self._kind = kind
        self._px = self.m.pixel_size_um()
        self._metadata = self.m.current_metadata().to_dict()
        self._engine = FocusStacker() if kind == "edf" else LiveMosaic()
        correct = self._corrector_factory() if self._corrector_factory else None
        self._correct = correct
        self._canvas_version = -1
        self._stop.clear()
        self._latest = None
        self._thread = threading.Thread(target=self._work, name=f"fuji-{kind}", daemon=True)
        self._thread.start()
        self.live.frame_shown.connect(self._on_frame)
        if not self.live.running:
            self.live.start()
        edf = kind == "edf"
        self.title.setText("Profondeur de champ étendue" if edf else "Panorama")
        self.state.setText(
            "Tournez lentement la mise au point fine à travers tout l'échantillon"
            if edf
            else "Immobilisez la lame sur le premier champ"
        )
        self.detail.setText("")
        self.bar.setVisible(edf)
        self.bar.setValue(0)
        self.undo_btn.setVisible(not edf)
        self.undo_btn.setEnabled(False)
        self.finish_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.pages.setCurrentIndex(1)

    def finish(self) -> None:
        """Termine : l'image composée rejoint la liste des images prises."""
        engine, kind = self._engine, self._kind
        if engine is None or kind is None:
            return
        self._halt()
        md = dict(self._metadata)
        px = self._px
        if isinstance(engine, FocusStacker):
            result = engine.result
            md["note"] = f"{engine.status.used} images fusionnées"
            self._cleanup()
            if result is not None:
                self.done.emit(result.copy(), "edf", md, px, "")
            return
        assert isinstance(engine, LiveMosaic)
        n = len(engine.tiles)
        self.state.setText(f"Assemblage des {n} champs en pleine résolution…")
        self.finish_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        worker = create_worker(
            lambda: engine.render(max_pixels=MAX_PANORAMA_PIXELS), _start_thread=False, _ignore_errors=True
        )

        def rendered(out: tuple[np.ndarray, int]) -> None:
            image, scale = out
            md["note"] = f"panorama de {n} champs" + (f", réduit ×{scale}" if scale > 1 else "")
            engine.close()
            self._cleanup()
            self.done.emit(image, "panorama", md, (px or 0) * scale or None, "")

        def failed(exc: Exception) -> None:
            engine.close()
            self._cleanup()
            self.message.emit(f"Panorama non assemblé : {exc}")

        worker.returned.connect(rendered)
        worker.errored.connect(failed)
        worker.start()

    def cancel(self) -> None:
        self._halt()
        if isinstance(self._engine, LiveMosaic):
            self._engine.close()
        self._cleanup()
        self.message.emit("Composition annulée")

    def undo(self) -> None:
        engine = self._engine
        if not isinstance(engine, LiveMosaic):
            return
        with self._engine_lock:  # pas pendant que le fil de travail ajoute un champ
            engine.undo()
        self._canvas_version = -1
        self._show({"state": "suivi", "message": "Dernier champ retiré", "tiles": len(engine.tiles)})

    def shutdown(self) -> None:
        """À la fermeture ou au changement de microscope."""
        if self.running:
            self.cancel()

    def _halt(self) -> None:
        try:
            self.live.frame_shown.disconnect(self._on_frame)
        except (TypeError, RuntimeError):
            pass
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(10)
            self._thread = None

    def _cleanup(self) -> None:
        self._kind, self._engine = None, None
        for name in (COMPOSITION_LAYER, FIELD_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        if LIVE_LAYER in self.viewer.layers:
            layer = self.viewer.layers[LIVE_LAYER]
            layer.visible = True
            layer.translate = (0.0, 0.0)
        self.viewer.reset_view()
        self.pages.setCurrentIndex(0)

    # ------------------------------------------------------------------ calcul
    def _on_frame(self, frame: np.ndarray) -> None:
        with self._cond:
            self._latest = frame
            self._cond.notify()

    def _work(self) -> None:
        while not self._stop.is_set():
            with self._cond:  # prendre la plus récente image du live, sans bloquer le live
                while self._latest is None and not self._stop.is_set():
                    self._cond.wait(0.5)
                frame, self._latest = self._latest, None
            if frame is None:
                continue
            engine = self._engine
            if engine is None:
                return
            with self._engine_lock:
                try:
                    if self._correct is not None:
                        frame = self._correct(frame)
                    status = engine.add(frame)
                    update = self._snapshot(engine, status)
                except Exception as exc:  # une image illisible ne doit pas tout arrêter
                    update = {"state": "perdu", "message": f"Image ignorée : {exc}"}
            self._processed.emit(update)

    def _snapshot(self, engine: FocusStacker | LiveMosaic, status: Any) -> dict[str, Any]:
        """Ce qu'il faut afficher (copies : le fil de travail continue de modifier les images)."""
        if isinstance(engine, FocusStacker):
            out: dict[str, Any] = {
                "frames": status.frames,
                "used": status.used,
                "contribution": status.contribution,
                "accepted": status.accepted,
                "message": status.message,
            }
            result = engine.result
            if result is not None and (status.contribution > 0 or status.frames <= 1):
                step = max(1, math.ceil(max(result.shape[:2]) / 1600))
                out["image"] = np.ascontiguousarray(result[::step, ::step])
                out["step"] = step
            return out
        canvas, p, origin = engine.preview
        out = {
            "state": status.state,
            "message": status.message,
            "tiles": status.tiles,
            "position": status.position,
            "shape": engine.frame_shape,
        }
        if canvas is not None and status.tiles != self._canvas_version:
            self._canvas_version = status.tiles
            out["canvas"] = canvas.copy()
            out["p"], out["origin"] = p, origin
        return out

    # ------------------------------------------------------------------ affichage
    def _show(self, update: dict[str, Any]) -> None:
        if not self.running:
            return
        px = self._px or 1.0
        self.state.setText(update.get("message", ""))
        if self._kind == "edf":
            self._show_edf(update, px)
        else:
            self._show_panorama(update, px)

    def _show_edf(self, update: dict[str, Any], px: float) -> None:
        self.bar.setValue(int(round(100 * update.get("contribution", 0.0))))
        self.detail.setText(f"{update.get('used', 0)} images utiles sur {update.get('frames', 0)} reçues")
        self.finish_btn.setEnabled(update.get("used", 0) >= 1)
        image = update.get("image")
        if image is None:
            return
        step = update["step"]
        rgb = image.ndim == 3 and image.shape[-1] in (3, 4)
        if COMPOSITION_LAYER in self.viewer.layers:
            layer = self.viewer.layers[COMPOSITION_LAYER]
            if layer.data.shape == image.shape:
                layer.data = image
                return
            self.viewer.layers.remove(COMPOSITION_LAYER)
        cam = self.m.camera
        self.viewer.add_image(
            image,
            name=COMPOSITION_LAYER,
            rgb=rgb,
            scale=(px * step, px * step),
            units=units_for(self._px),
            contrast_limits=contrast_for(image, cam.bit_depth if cam else None),
        )
        if LIVE_LAYER in self.viewer.layers:
            self.viewer.layers[LIVE_LAYER].visible = False
        self.viewer.reset_view()

    def _show_panorama(self, update: dict[str, Any], px: float) -> None:
        tiles = update.get("tiles", 0)
        self.detail.setText(f"{tiles} champ(s) dans le panorama")
        self.finish_btn.setEnabled(tiles >= 1)
        self.undo_btn.setEnabled(tiles >= 1)
        canvas = update.get("canvas")
        if canvas is not None:
            p, origin = update["p"], update["origin"]
            scale = (px * p, px * p)
            translate = (origin[0] * px, origin[1] * px)
            if COMPOSITION_LAYER in self.viewer.layers:
                layer = self.viewer.layers[COMPOSITION_LAYER]
                grew = layer.data.shape != canvas.shape
                layer.data, layer.scale, layer.translate = canvas, scale, translate
            else:
                cam = self.m.camera
                layer = self.viewer.add_image(
                    canvas,
                    name=COMPOSITION_LAYER,
                    rgb=canvas.ndim == 3 and canvas.shape[-1] in (3, 4),
                    scale=scale,
                    translate=translate,
                    units=units_for(self._px),
                    contrast_limits=contrast_for(canvas, cam.bit_depth if cam else None),
                )
                grew = True
            # le panorama sous le live, le cadre du champ actuel par-dessus
            self.viewer.layers.move(self.viewer.layers.index(layer), 0)
            if grew:
                self.viewer.reset_view()
        position, shape = update.get("position"), update.get("shape")
        if position is None or shape is None:
            return
        top, left = position[0] * px, position[1] * px
        bottom, right = top + shape[0] * px, left + shape[1] * px
        rect = np.array([[top, left], [top, right], [bottom, right], [bottom, left]])
        color = STATE_COLORS.get(update.get("state", "suivi"), "#22c55e")
        if FIELD_LAYER in self.viewer.layers:
            frame_layer = self.viewer.layers[FIELD_LAYER]
            frame_layer.data = [rect]
            frame_layer.edge_color = color
        else:
            self.viewer.add_shapes(
                [rect],
                shape_type="rectangle",
                name=FIELD_LAYER,
                edge_color=color,
                face_color="transparent",
                edge_width=max(3 * px, 1e-6),
                units=units_for(self._px),
            )
        if LIVE_LAYER in self.viewer.layers:  # le live à sa place dans le panorama
            self.viewer.layers[LIVE_LAYER].translate = (top, left)
