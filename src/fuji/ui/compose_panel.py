"""Compositions en direct à partir du live : profondeur de champ étendue et panorama.

Pensé pour un microscope **manuel** : on tourne la mise au point, ou on déplace la lame à
la main, et l'image se construit à l'écran (comme le logiciel fourni avec la caméra).
Fonctionne aussi avec des moteurs.

Les yeux sont sur l'image, pas sur le panneau : l'état est écrit **sur l'image** (bandeau en
haut à gauche), le cadre du champ actuel change de couleur (vert : tout va bien, jaune :
trop vite, rouge : position perdue) et la vue suit le panorama qui grandit.

Les calculs se font dans un fil de travail : l'interface et le live restent fluides, et une
image du live qui arrive pendant un calcul est simplement sautée.
"""

from __future__ import annotations

import math
import threading
import time
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
#: couleurs du retour visuel
GREEN, YELLOW, RED, GREY = "#22c55e", "#f5a524", "#ef4444", "#9ca3af"
#: couleur du cadre du champ actuel (et du bandeau) selon l'état du panorama
STATE_COLORS = {
    "attente": YELLOW,
    "suivi": GREEN,
    "ajouté": GREEN,
    "affiné": GREEN,
    "vite": YELLOW,
    "trop_vite": YELLOW,
    "couvert": GREY,
    "perdu": RED,
}
#: … et selon l'état de la profondeur de champ étendue
EDF_COLORS = {"attente": YELLOW, "mieux": GREEN, "rien": GREY, "bougé": RED}
#: au-delà, le panorama final est assemblé à une résolution réduite (mémoire de l'ordinateur)
MAX_PANORAMA_PIXELS = 80_000_000
#: taille de la vignette du live pendant une profondeur de champ étendue (part de l'image)
INSET = 0.3
TITLES = {"edf": "Profondeur de champ étendue", "panorama": "Panorama"}
START_HINTS = {
    "edf": "Tournez lentement la mise au point fine à travers tout l'échantillon",
    "panorama": "Déplacez la lame : le panorama se construit tout seul",
}


class ComposePanel(QWidget):
    message = Signal(str)
    #: image terminée : (image, type « edf » ou « panorama », métadonnées, taille du pixel, résumé)
    done = Signal(object, str, object, object, str)
    _processed = Signal(object)
    _render_progress = Signal(int, int)

    def __init__(
        self,
        microscope: Microscope,
        viewer: ViewerModel,
        live: LiveController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.m, self.viewer, self.live = microscope, viewer, live
        self._kind: str | None = None
        self._engine: FocusStacker | LiveMosaic | None = None
        #: dépôt de la dernière image du live (tenu très brièvement : le live ne doit pas attendre)
        self._cond = threading.Condition()
        #: le calcul en cours (« retirer le dernier champ » attend qu'il soit fini)
        self._engine_lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._px: float | None = None
        self._exposure_s: float | None = None
        self._exposure_read = 0.0
        self._metadata: dict[str, Any] = {}
        self._canvas_version = -1
        self._rendering = False
        self._processed.connect(self._show)
        self._render_progress.connect(self._show_render_progress)

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
        self.pano_btn = button("Panorama", "mosaic", primary=True)
        self.pano_btn.clicked.connect(lambda: self.start("panorama"))
        il.addWidget(self.pano_btn)
        il.addWidget(
            hint(
                "Déplacez la lame à la main, sans vous arrêter : les champs s'assemblent au passage. "
                "Cadre vert : tout va bien ; jaune : un peu vite (l'image sera moins nette) ; rouge : "
                "revenez sur une zone déjà assemblée."
            )
        )
        self.edf_btn = button("Profondeur de champ étendue", "stack", primary=True)
        self.edf_btn.clicked.connect(lambda: self.start("edf"))
        il.addWidget(self.edf_btn)
        il.addWidget(
            hint(
                "Tournez lentement la mise au point fine à travers l'échantillon : chaque zone garde "
                "son état le plus net. Le live reste visible en vignette."
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
        self.finish_btn = button(
            "Terminer", "check", primary=True, tooltip="Enregistre l'image dans la liste « Images »"
        )
        self.finish_btn.clicked.connect(self.finish)
        self.restart_btn = button("Recommencer", "refresh", tooltip="Efface tout et repart de zéro")
        self.restart_btn.clicked.connect(self.restart)
        self.cancel_btn = button("Annuler", "close")
        self.cancel_btn.clicked.connect(self.cancel)
        self.undo_btn = button("Retirer le dernier champ", "trash")
        self.undo_btn.clicked.connect(self.undo)
        for w in (self.title, self.state, self.bar, self.detail):
            rl.addWidget(w)
        rl.addWidget(self.finish_btn)
        rl.addWidget(hbox(self.restart_btn, self.cancel_btn))
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

    @property
    def kind(self) -> str | None:
        """« edf », « panorama », ou None si rien n'est en cours."""
        return self._kind

    def start(self, kind: str) -> None:
        """Commence une composition (« edf » ou « panorama ») à partir du live."""
        if self.running or self.m.camera is None:
            return
        self._kind = kind
        self._px = self.m.pixel_size_um()
        self._metadata = self.m.current_metadata().to_dict()
        self._engine = FocusStacker() if kind == "edf" else LiveMosaic()
        self._canvas_version = -1
        self._exposure_read = 0.0
        self._read_exposure()
        self._stop.clear()
        self._latest = None
        self._thread = threading.Thread(target=self._work, name=f"fuji-{kind}", daemon=True)
        self._thread.start()
        self.live.frame_shown.connect(self._on_frame)
        if not self.live.running:
            self.live.start()
        edf = kind == "edf"
        self.title.setText(TITLES[kind])
        self.state.setText(START_HINTS[kind])
        self.detail.setText("")
        self.bar.setFormat("apport de la dernière image : %p %")
        self.bar.setVisible(edf)
        self.bar.setValue(0)
        self.undo_btn.setVisible(not edf)
        self.undo_btn.setEnabled(False)
        self.finish_btn.setEnabled(False)
        for b in (self.cancel_btn, self.restart_btn):
            b.setEnabled(True)
        self.pages.setCurrentIndex(1)
        self._banner(START_HINTS[kind], YELLOW)

    def restart(self) -> None:
        """Efface la composition en cours et la recommence."""
        kind = self._kind
        if kind is None or self._rendering:
            return
        self._discard()
        self.start(kind)
        self.message.emit(f"{TITLES[kind]} : on recommence")

    def finish(self) -> None:
        """Termine : l'image composée rejoint la liste des images prises."""
        engine, kind = self._engine, self._kind
        if engine is None or kind is None or self._rendering:
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
        self._rendering = True
        self.state.setText(f"Assemblage des {n} champs en pleine résolution…")
        self._banner(f"Assemblage des {n} champs…", GREY)
        self.bar.setFormat("assemblage : %p %")
        self.bar.setValue(0)
        self.bar.setVisible(True)
        for b in (self.finish_btn, self.cancel_btn, self.restart_btn, self.undo_btn):
            b.setEnabled(False)
        worker = create_worker(
            lambda: engine.render(max_pixels=MAX_PANORAMA_PIXELS, progress=self._render_progress.emit),
            _start_thread=False,
            _ignore_errors=True,
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
        if self._rendering:
            return
        self._discard()
        self.message.emit("Composition annulée")

    def _discard(self) -> None:
        self._halt()
        if isinstance(self._engine, LiveMosaic):
            self._engine.close()
        self._cleanup()

    def undo(self) -> None:
        engine = self._engine
        if not isinstance(engine, LiveMosaic):
            return
        with self._engine_lock:  # pas pendant que le fil de travail ajoute un champ
            engine.undo()
        self._canvas_version = -1
        self._show(self._snapshot(engine, engine.status) | {"message": "Dernier champ retiré"})

    def shutdown(self) -> None:
        """À la fermeture ou au changement de microscope."""
        if self.running:
            self._rendering = False
            self._discard()

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
        self._kind, self._engine, self._rendering = None, None, False
        for name in (COMPOSITION_LAYER, FIELD_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self.live.set_placement()  # le live reprend toute la vue
        if LIVE_LAYER in self.viewer.layers:
            self.viewer.layers[LIVE_LAYER].visible = True
        self._banner("")
        self.viewer.reset_view()
        self.pages.setCurrentIndex(0)

    # ------------------------------------------------------------------ calcul
    def _read_exposure(self) -> None:
        """Temps de pose de la caméra (relu de temps en temps : il sert à estimer le flou)."""
        now = time.monotonic()
        if now - self._exposure_read < 1.0:
            return
        self._exposure_read = now
        cam = self.m.camera
        try:
            self._exposure_s = cam.get_exposure() / 1000.0 if cam is not None else None
        except Exception:  # caméra occupée : on garde la dernière valeur
            pass

    def _on_frame(self, frame: np.ndarray) -> None:
        self._read_exposure()
        with self._cond:
            self._latest = (frame, time.monotonic())
            self._cond.notify()

    def _work(self) -> None:
        while not self._stop.is_set():
            with self._cond:  # prendre la plus récente image du live, sans bloquer le live
                while self._latest is None and not self._stop.is_set():
                    self._cond.wait(0.5)
                item, self._latest = self._latest, None
            if item is None:
                continue
            frame, stamp = item
            engine = self._engine
            if engine is None:
                return
            with self._engine_lock:
                try:
                    status = engine.add(frame, stamp, self._exposure_s)
                    update = self._snapshot(engine, status)
                except Exception as exc:  # une image illisible ne doit pas tout arrêter
                    update = {"state": "perdu", "message": f"Image ignorée : {exc}"}
            self._processed.emit(update)

    def _snapshot(self, engine: FocusStacker | LiveMosaic, status: Any) -> dict[str, Any]:
        """Ce qu'il faut afficher (copies : le fil de travail continue de modifier les images)."""
        if isinstance(engine, FocusStacker):
            out: dict[str, Any] = {
                "state": status.state,
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
                out["shape"] = result.shape
            return out
        canvas, p, origin = engine.preview
        out = {
            "state": status.state,
            "message": status.message,
            "tiles": len(engine.tiles),
            "position": status.position,
            "shape": engine.frame_shape,
            "bounds": engine.bounds(),
        }
        if canvas is not None and status.version != self._canvas_version:
            self._canvas_version = status.version
            out["canvas"] = canvas.copy()
            out["p"], out["origin"] = p, origin
        return out

    # ------------------------------------------------------------------ affichage
    def _banner(self, text: str, color: str = GREEN) -> None:
        """Écrit l'état sur l'image (les yeux sont sur l'image, pas sur le panneau)."""
        overlay = getattr(self.viewer, "text_overlay", None)
        if overlay is None:
            return
        try:
            overlay.text = text
            overlay.color = color
            overlay.font_size = 11
            overlay.position = "top_left"
            overlay.visible = bool(text)
        except Exception:  # version de napari sans ce réglage : le panneau suffit
            pass

    def _show(self, update: dict[str, Any]) -> None:
        if not self.running or self._rendering:
            return
        px = self._px or 1.0
        state = update.get("state", "")
        self.state.setText(update.get("message", ""))
        if self._kind == "edf":
            self._banner(_edf_banner(update), EDF_COLORS.get(state, GREEN))
            self._show_edf(update, px)
        else:
            self._banner(_panorama_banner(update), STATE_COLORS.get(state, GREEN))
            self._show_panorama(update, px)

    def _show_render_progress(self, done: int, total: int) -> None:
        if self._rendering:
            self.bar.setValue(int(round(100 * done / max(1, total))))

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
        layer = self.viewer.add_image(
            image,
            name=COMPOSITION_LAYER,
            rgb=rgb,
            scale=(px * step, px * step),
            units=units_for(self._px),
            contrast_limits=contrast_for(image, cam.bit_depth if cam else None),
        )
        # l'image composée prend toute la vue ; le live reste visible en vignette, dans un coin
        self.viewer.layers.move(self.viewer.layers.index(layer), 0)
        h, w = update["shape"][:2]
        margin = 0.015 * w * px
        self.live.set_placement(INSET, (margin, margin))
        self._frame_rect(
            (margin, margin, margin + INSET * h * px, margin + INSET * w * px), "#ffffff", px * w / 600
        )
        if hasattr(self.viewer, "fit_to_view"):
            self.viewer.fit_to_view(layers=[layer])
        else:
            self.viewer.reset_view()

    def _frame_rect(self, rect: tuple[float, float, float, float], color: str, width: float) -> None:
        """Cadre (haut, gauche, bas, droite) en unités de la vue : champ actuel ou vignette."""
        top, left, bottom, right = rect
        data = np.array([[top, left], [top, right], [bottom, right], [bottom, left]])
        try:  # trait de trois pixels d'écran, quel que soit le zoom
            width = 3.0 / float(self.viewer.camera.zoom)
        except Exception:
            pass
        if FIELD_LAYER in self.viewer.layers:
            layer = self.viewer.layers[FIELD_LAYER]
            layer.data = [data]
            layer.edge_color = color
            layer.edge_width = max(width, 1e-6)
            return
        self.viewer.add_shapes(
            [data],
            shape_type="rectangle",
            name=FIELD_LAYER,
            edge_color=color,
            face_color="transparent",
            edge_width=max(width, 1e-6),
            units=units_for(self._px),
        )

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
            # le panorama sous le live, le cadre du champ actuel par-dessus
            self.viewer.layers.move(self.viewer.layers.index(layer), 0)
        position, shape = update.get("position"), update.get("shape")
        if position is None or shape is None:
            return
        top, left = position[0] * px, position[1] * px
        bottom, right = top + shape[0] * px, left + shape[1] * px
        color = STATE_COLORS.get(update.get("state", "suivi"), GREEN)
        self._frame_rect((top, left, bottom, right), color, 3 * px * max(shape[:2]) / 900)
        self.live.set_placement(1.0, (top, left))  # le live à sa place dans le panorama
        # la vue suit : tout le panorama et le champ actuel restent visibles
        seen = (top, left, bottom, right)
        bounds = update.get("bounds")
        if bounds is not None:
            b = tuple(v * px for v in bounds)
            seen = (min(top, b[0]), min(left, b[1]), max(bottom, b[2]), max(right, b[3]))
        self._follow(seen, (bottom - top, right - left))

    def _follow(self, rect: tuple[float, float, float, float], field: tuple[float, float]) -> None:
        """Recadre la vue quand ``rect`` (haut, gauche, bas, droite) en sort — avec une marge
        d'un quart de champ, pour ne pas recadrer à chaque image — ou quand il n'en occupe
        plus qu'une petite partie."""
        try:
            height, width = self.viewer.canvas.size
            cam = self.viewer.camera
            cy, cx = (float(v) for v in cam.center[-2:])
            zoom = float(cam.zoom)
        except Exception:  # autre version de napari : recadrage simple
            self.viewer.reset_view()
            return
        if height <= 0 or width <= 0 or zoom <= 0:
            return
        top, left, bottom, right = rect
        half_h, half_w = height / zoom / 2, width / zoom / 2
        inside = top >= cy - half_h and bottom <= cy + half_h and left >= cx - half_w and right <= cx + half_w
        mh, mw = 0.25 * field[0], 0.25 * field[1]
        wanted = min(height / (bottom - top + 2 * mh), width / (right - left + 2 * mw))
        if inside and zoom <= 1.8 * wanted:
            return
        center = ((top + bottom) / 2, (left + right) / 2)
        cam.center = (*tuple(cam.center)[:-2], *center)
        cam.zoom = wanted


def _panorama_banner(update: dict[str, Any]) -> str:
    """Texte court écrit sur l'image pendant un panorama (le détail est dans le panneau)."""
    n = update.get("tiles", 0)
    return {
        "attente": "Ralentissez pour poser le premier champ",
        "vite": f"{n} champs · un peu vite",
        "trop_vite": "Trop vite : ralentissez",
        "couvert": f"{n} champs · avancez vers du nouveau",
        "perdu": "Position perdue : revenez en arrière",
    }.get(update.get("state", ""), f"{n} champs")


def _edf_banner(update: dict[str, Any]) -> str:
    """Texte court écrit sur l'image pendant une profondeur de champ étendue."""
    gain = update.get("contribution", 0.0)
    return {
        "attente": "Tournez lentement la mise au point",
        "mieux": f"+{gain:.0%} de net",
        "rien": "Plus rien de nouveau ici",
        "bougé": "La lame a bougé : remettez-la en place",
    }.get(update.get("state", ""), update.get("message", ""))
