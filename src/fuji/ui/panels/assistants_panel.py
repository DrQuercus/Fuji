"""Panneau « Assistants » : Köhler, mise au point, calibration, éclairage / image de référence.

Les assistants lisent les images du live et dessinent leurs repères par-dessus
l'image dans napari (contour du diaphragme, flèche de centrage, zones nettes...).
"""

from __future__ import annotations

import numpy as np
from napari.components import ViewerModel
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji.assistants import (
    FocusTracker,
    KohlerAssistant,
    KohlerStep,
    calibrate_micrometer,
    focus_peaking,
    illumination_uniformity,
)
from fuji.assistants.focus import METRICS, autofocus
from fuji.core.microscope import Microscope
from fuji.text import number, signed

from ..components import button, flow, label, set_icon
from ..flatfield_widget import FlatFieldWidget
from ..live import LiveController
from ..widgets import hint_label, row, section
from .tool_menu import ClickableRow

#: nom montré pour chaque façon de mesurer la netteté
_METRIC_NAMES = {
    "tenengrad": "Contours (conseillée)",
    "laplacien": "Détails fins",
    "variance normalisée": "Contraste global",
    "brenner": "Contours horizontaux",
}
KOHLER_LAYER = "Köhler (repères)"
PEAKING_LAYER = "Zones nettes"
UNIFORMITY_LAYER = "Uniformité de l'éclairage"


class _LiveAssistant(QWidget):
    """Base : un minuteur qui lit la dernière image du live."""

    message = Signal(str)

    def __init__(
        self,
        microscope: Microscope,
        viewer: ViewerModel,
        live: LiveController,
        interval_ms: int = 250,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.m, self.viewer, self.live = microscope, viewer, live
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._last = -1

    def start(self) -> None:
        if not self.live.running:
            self.live.start()
        self._last = -1
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def _tick(self) -> None:
        cam = self.m.camera
        if cam is None:
            return
        count, frame = cam.latest_frame()
        if frame is None or count == self._last:
            return
        self._last = count
        self.process(frame)

    def process(self, frame: np.ndarray) -> None:  # pragma: no cover - surchargée
        raise NotImplementedError

    def _scale(self) -> float:
        return self.m.pixel_size_um() or 1.0

    def _units(self) -> tuple[str, str] | None:
        return ("µm", "µm") if self.m.pixel_size_um() else None


# ------------------------------------------------------------------- Köhler
class KohlerWidget(_LiveAssistant):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        obj = self.m.objective
        self.assistant = KohlerAssistant(obj.na if obj else None, self.m.profile.condenser_na)
        lay = QVBoxLayout(self)
        lay.addWidget(
            hint_label(
                "Placez une lame avec peu d'objets (ou une zone vide) et faites la mise au point "
                "sur l'échantillon avant de commencer."
            )
        )
        self.toggle = button("Démarrer l'assistant", primary=True)
        self.toggle.clicked.connect(self._toggle)
        lay.addWidget(self.toggle)
        self.title = QLabel()
        self.title.setObjectName("step")
        self.title.setWordWrap(True)
        self.text = QLabel()
        self.text.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.text)
        self.sharp_bar = QProgressBar()
        self.sharp_bar.setRange(0, 100)
        self.sharp_bar.setFormat("Netteté du bord : %p %")
        self.sharp_bar.setVisible(False)
        lay.addWidget(self.sharp_bar)
        prev = button("Précédente", "left", tooltip="Revenir à l'étape précédente")
        prev.clicked.connect(lambda: (self.assistant.back(), self._render_step()))
        self.next = button("Étape suivante", "right")
        self.next.setLayoutDirection(Qt.LayoutDirection.RightToLeft)  # la flèche après le texte
        self.next.clicked.connect(lambda: (self.assistant.advance(), self._render_step()))
        lay.addWidget(flow(prev, self.next))  # panneau étroit : le second bouton passe dessous
        lay.addStretch(1)
        self._render_step()

    def _toggle(self) -> None:
        if self.running:
            self.stop()
            self.toggle.setText("Démarrer l'assistant")
            self._clear_overlay()
        else:
            obj = self.m.objective
            self.assistant.objective_na = obj.na if obj else None
            self.assistant.reset()
            self.start()
            self.toggle.setText("Arrêter l'assistant")

    def _render_step(self, guidance=None) -> None:  # type: ignore[no-untyped-def]
        from fuji.assistants.kohler import STEP_TITLES

        self.title.setText(STEP_TITLES[self.assistant.step])
        if guidance is None:
            self.text.setText("")

    def process(self, frame: np.ndarray) -> None:
        g = self.assistant.update(frame)
        self.title.setText(g.title)
        self.text.setText(g.message)
        if self.next.property("ready") != bool(g.can_advance):  # l'étape est réussie : bouton mis en avant
            self.next.setProperty("ready", bool(g.can_advance))
            self.next.setObjectName("primary" if g.can_advance else "")
            self.next.style().unpolish(self.next)
            self.next.style().polish(self.next)
        if g.step is KohlerStep.FOCUS_CONDENSER and g.edge_width_px and g.best_edge_width_px:
            self.sharp_bar.setVisible(True)
            self.sharp_bar.setValue(int(100 * g.best_edge_width_px / g.edge_width_px))
        else:
            self.sharp_bar.setVisible(False)
        self._draw(g, frame.shape[:2])

    def _draw(self, g, shape: tuple[int, int]) -> None:  # type: ignore[no-untyped-def]
        rep = g.report
        s = self._scale()
        h, w = shape
        shapes, types, colors = [], [], []
        # croix au centre de l'image
        c = np.array([h / 2, w / 2])
        arm = min(h, w) * 0.04
        shapes += [np.array([c - [arm, 0], c + [arm, 0]]), np.array([c - [0, arm], c + [0, arm]])]
        types += ["line", "line"]
        colors += ["yellow", "yellow"]
        if rep is not None and rep.visible and rep.contour is not None and len(rep.contour) > 3:
            shapes.append(rep.contour)
            types.append("polygon")
            colors.append("cyan")
            if rep.center is not None:
                ctr = np.array(rep.center)
                shapes.append(np.array([ctr, c]))
                types.append("line")
                colors.append("orange" if g.step is KohlerStep.CENTER_CONDENSER else "gray")
        data = [sh * s for sh in shapes]
        if KOHLER_LAYER in self.viewer.layers:
            layer = self.viewer.layers[KOHLER_LAYER]
            layer.data = []
            layer.add(
                data,
                shape_type=types,
                edge_color=colors,
                face_color="transparent",
                edge_width=max(2.0 * s, 1e-3),
            )
        else:
            self.viewer.add_shapes(
                data,
                shape_type=types,
                edge_color=colors,
                face_color="transparent",
                edge_width=max(2.0 * s, 1e-3),
                name=KOHLER_LAYER,
                units=self._units(),
            )

    def _clear_overlay(self) -> None:
        if KOHLER_LAYER in self.viewer.layers:
            self.viewer.layers.remove(KOHLER_LAYER)


# ------------------------------------------------------------- mise au point
class FocusWidget(_LiveAssistant):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, interval_ms=150, **kwargs)
        self.tracker = FocusTracker()
        lay = QVBoxLayout(self)
        motorized = self.m.focus is not None and not self.m.focus.manual
        lay.addWidget(
            hint_label(
                "Une jauge suit la netteté de l'image en direct. "
                + (
                    "« Autofocus » cherche le meilleur point tout seul."
                    if motorized
                    else "Tournez la vis de mise au point jusqu'à dépasser le maximum, puis revenez "
                    "au point où la jauge est pleine."
                )
            )
        )
        self.toggle = button("Démarrer l'aide à la mise au point")
        self.toggle.setObjectName("primary")
        self.toggle.clicked.connect(self._toggle)
        lay.addWidget(self.toggle)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFormat("Netteté : %p % du meilleur point")
        lay.addWidget(self.bar)
        self.trend = hint_label("")
        lay.addWidget(self.trend)
        reset = button(
            "Repartir de zéro",
            "refresh",
            tooltip="Oublie la meilleure netteté rencontrée (après un changement de zone ou d'objectif)",
        )
        reset.clicked.connect(self.tracker.reset)
        self.peaking = QCheckBox("Surligner les zones nettes")
        self.peaking.setToolTip("Colore en rouge, sur le live, ce qui est net à cet instant")
        self.peaking.toggled.connect(lambda on: None if on else self._remove_peaking())
        lay.addWidget(self.peaking)
        self.metric = QComboBox()
        for name in METRICS:
            self.metric.addItem(_METRIC_NAMES.get(name, name), name)
        self.metric.setToolTip("Façon de chiffrer la netteté ; la première convient presque toujours")
        self.metric.currentIndexChanged.connect(lambda _i: self._set_metric(self.metric.currentData()))
        lay.addWidget(row(QLabel("Méthode"), self.metric, stretch_last=True))
        lay.addWidget(row(reset))
        self._af_was_live = False
        if motorized:
            self.af_button = button(
                "Autofocus", "focus", tooltip="Balaye la mise au point et s'arrête au plus net"
            )
            self.af_button.clicked.connect(self._autofocus)
            lay.addWidget(self.af_button)
        lay.addStretch(1)

    def _set_metric(self, name: str) -> None:
        self.tracker.method = name
        self.tracker.reset()

    def _toggle(self) -> None:
        if self.running:
            self.stop()
            self.toggle.setText("Démarrer l'aide à la mise au point")
            self._remove_peaking()
        else:
            self.tracker.reset()
            self.start()
            self.toggle.setText("Arrêter l'aide à la mise au point")

    def process(self, frame: np.ndarray) -> None:
        st = self.tracker.update(frame)
        self.bar.setValue(int(100 * st.relative))
        self.trend.setText(f"Netteté {st.trend}." if st.trend != "meilleure" else "Meilleur point atteint.")
        if self.peaking.isChecked():
            mask, step = focus_peaking(frame)
            self._show_peaking(mask.astype(np.uint8), step)

    def _show_peaking(self, mask: np.ndarray, step: int) -> None:
        """Le masque est calculé sur une image réduite : un de ses pixels couvre
        ``step`` × ``step`` pixels de l'image."""
        px = self._scale()
        scale = (px * step, px * step)
        translate = (px * (step - 1) / 2, px * (step - 1) / 2)
        if PEAKING_LAYER in self.viewer.layers:
            layer = self.viewer.layers[PEAKING_LAYER]
            if layer.data.shape == mask.shape:
                layer.data = mask
                if tuple(layer.scale) != scale:  # changement d'objectif
                    layer.scale = scale
                    layer.translate = translate
                return
            self.viewer.layers.remove(PEAKING_LAYER)
        self.viewer.add_image(
            mask,
            name=PEAKING_LAYER,
            colormap="red",
            blending="additive",
            opacity=0.8,
            contrast_limits=(0, 1),
            scale=scale,
            translate=translate,
            units=self._units(),
        )

    def _remove_peaking(self) -> None:
        if PEAKING_LAYER in self.viewer.layers:
            self.viewer.layers.remove(PEAKING_LAYER)

    def _autofocus(self) -> None:
        """Balayage en Z dans un thread de travail : l'interface reste utilisable."""
        self._af_was_live = self.live.running
        if self._af_was_live:
            self.live.stop()
        self.af_button.setEnabled(False)
        self.message.emit("Autofocus en cours…")
        worker = create_worker(autofocus, self.m, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(self._autofocus_done)
        worker.errored.connect(self._autofocus_failed)
        worker.finished.connect(self._autofocus_finished)
        worker.start()

    def _autofocus_done(self, z: float) -> None:
        self.message.emit(f"Mise au point automatique faite : Z = {number(z, 2)} µm")

    def _autofocus_failed(self, exc: Exception) -> None:
        QMessageBox.warning(self, "Autofocus", f"La mise au point automatique n'a pas abouti : {exc}")

    def _autofocus_finished(self) -> None:
        self.af_button.setEnabled(True)
        if self._af_was_live and self.m.connected:
            self.live.start()


# ------------------------------------------------------------- calibration
class CalibrationWidget(QWidget):
    message = Signal(str)
    profile_saved = Signal()

    def __init__(
        self, microscope: Microscope, viewer: ViewerModel, live: LiveController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.m, self.viewer, self.live = microscope, viewer, live
        self._result = None
        lay = QVBoxLayout(self)
        lay.addWidget(
            hint_label(
                "Placez un micromètre objet, faites la mise au point sur les graduations "
                "(traits nets, sur toute la largeur de l'image si possible)."
            )
        )
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.1, 1000)
        self.spacing.setValue(10.0)
        self.spacing.setSuffix(" µm")
        lay.addWidget(flow(QLabel("Écart entre graduations"), self.spacing))
        self.measure = button("Mesurer sur l'image actuelle")
        self.measure.setObjectName("primary")
        self.measure.clicked.connect(self._measure)
        lay.addWidget(self.measure)
        self.result = QLabel()
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        self.save = button("Enregistrer pour l'objectif en place")
        self.save.setEnabled(False)
        self.save.clicked.connect(self._save)
        lay.addWidget(self.save)
        lay.addStretch(1)
        #: objectif et binning au moment de la mesure : la calibration leur appartient
        self._measured_for: tuple[str | None, float] = (None, 1)

    def _grab_and_measure(self, spacing_um: float, use_live: bool):  # type: ignore[no-untyped-def]
        cam = self.m.require_camera()
        # image fraîche : la dernière image mémorisée peut dater d'un autre objectif
        frame = cam.latest_frame()[1] if use_live else None
        if frame is None:
            frame = self.m.snap().data
        return calibrate_micrometer(frame, spacing_um)

    def _measure(self) -> None:
        obj = self.m.objective
        cam = self.m.require_camera()
        self._measured_for = (obj.label if obj else None, cam.binning)
        self._result = None
        self.measure.setEnabled(False)
        self.save.setEnabled(False)
        self.result.setText("Mesure en cours…")
        worker = create_worker(
            self._grab_and_measure,
            self.spacing.value(),
            self.live.running,
            _start_thread=False,
            _ignore_errors=True,
        )
        worker.returned.connect(self._measured)
        worker.errored.connect(self._measure_failed)
        worker.finished.connect(self._measure_finished)
        worker.start()

    def _measure_failed(self, exc: Exception) -> None:
        self.result.setText(f"Échec : {exc}")

    def _measure_finished(self) -> None:
        self.measure.setEnabled(True)

    def _measured(self, r) -> None:  # type: ignore[no-untyped-def]
        self._result = r
        current = self.m.pixel_size_um()
        cmp = ""
        if current:
            cmp = (
                f"<br>Valeur actuelle : {number(current)} µm/px ({self.m.pixel_size_source()}), "
                f"écart {signed(100 * (r.pixel_size_um / current - 1))} %"
            )
        warn = ""
        if not r.reliable:
            warn = (
                "<br><b>⚠ Mesure peu fiable</b> : il faut au moins 5 graduations nettes et "
                "régulières. Cadrez plus de traits, refaites la mise au point, puis recommencez."
            )
        self.result.setText(
            f"<b>{number(r.pixel_size_um)} µm/px</b><br>{r.n_lines} traits, "
            f"période {number(r.period_px, 2)} px, rotation de la caméra {number(r.angle_deg, 2)}°, "
            f"incertitude ± {number(r.relative_uncertainty * 100, 2)} %{cmp}{warn}"
        )
        label = self._measured_for[0]
        self.save.setEnabled(label is not None)
        self.save.setText(
            f"Enregistrer pour l'objectif {label}" if label else "Sélectionnez d'abord l'objectif en place"
        )

    def _save(self) -> None:
        r = self._result
        label, binning = self._measured_for
        if r is None or label is None:
            return
        cam = self.m.camera
        if cam is not None and cam.binning != binning:
            QMessageBox.information(
                self,
                "Calibration",
                "La résolution de la caméra a changé depuis la mesure : refaites la mesure.",
            )
            return
        if not r.reliable:
            answer = QMessageBox.question(
                self,
                "Mesure peu fiable",
                f"Seulement {r.n_lines} traits reconnus (incertitude "
                f"± {number(r.relative_uncertainty * 100, 1)} %). Toutes les mesures faites avec cet "
                "objectif en dépendront.\n\nEnregistrer quand même ?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.m.set_pixel_calibration(r.pixel_size_um, label)
        except Exception as exc:
            QMessageBox.warning(
                self, "Calibration", f"La taille du pixel n'a pas pu être enregistrée : {exc}"
            )
            return
        try:
            path = self.m.save_profile()
            self.message.emit(f"Calibration de {label} enregistrée dans {path}")
        except Exception as exc:
            self.message.emit(f"Calibration appliquée (non enregistrée : {exc})")
        self.profile_saved.emit()


# ------------------------------------------------- éclairage, image de référence
class IlluminationWidget(QWidget):
    """Uniformité de l'éclairage, et image de référence qui la corrige."""

    message = Signal(str)

    def __init__(
        self, microscope: Microscope, viewer: ViewerModel, live: LiveController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.m, self.viewer, self.live = microscope, viewer, live
        lay = QVBoxLayout(self)
        box, blay = section(
            "Uniformité",
            hint="Sur une zone vide de la lame, diaphragme de champ ouvert juste au-delà du champ.",
        )
        analyse = button("Analyser l'éclairage", primary=True)
        analyse.clicked.connect(self._uniformity)
        blay.addWidget(analyse)
        self.uni_text = QLabel()
        self.uni_text.setWordWrap(True)
        blay.addWidget(self.uni_text)
        lay.addWidget(box)
        box, blay = section(
            "Image de référence",
            hint="Corrige l'éclairage inégal, le vignettage et les poussières dans toutes les images "
            "(live, captures, panoramas). Une référence par objectif et par canal : sur une zone "
            "vide de la lame, « Prendre la référence ». Les mêmes commandes sont dans Contrôle → Caméra.",
        )
        self.reference = FlatFieldWidget(microscope)
        self.reference.message.connect(self.message.emit)
        blay.addWidget(self.reference)
        lay.addWidget(box)
        lay.addStretch(1)

    def shutdown(self) -> None:
        self.reference.shutdown()

    def _uniformity(self) -> None:
        try:
            frame = self.m.snap().data
        except Exception as exc:
            QMessageBox.warning(self, "Éclairage", f"L'éclairage n'a pas pu être analysé : {exc}")
            return
        rep = illumination_uniformity(frame)
        corrected = " (fond corrigé par l'image de référence)" if self.m.flatfield.active else ""
        self.uni_text.setText(
            f"Min/max : {number(rep.min_max_ratio, 2)} · coins/centre : "
            f"{number(rep.corner_center_ratio, 2)}{corrected}"
            f"<br>{rep.message}"
        )
        h, w = frame.shape[:2]
        s = self.m.pixel_size_um() or 1.0
        gh, gw = rep.map.shape
        if UNIFORMITY_LAYER in self.viewer.layers:
            self.viewer.layers.remove(UNIFORMITY_LAYER)
        self.viewer.add_image(
            rep.map.astype(np.float32),
            name=UNIFORMITY_LAYER,
            colormap="turbo",
            opacity=0.45,
            contrast_limits=(max(0.0, rep.map.min()), 1.0),
            scale=(h / gh * s, w / gw * s),
            translate=(h / gh * s / 2 - s / 2, w / gw * s / 2 - s / 2),
            units=("µm", "µm") if self.m.pixel_size_um() else None,
        )


# ------------------------------------------------------------- platine motorisée
class StageWidget(QWidget):
    """Réglages d'une platine motorisée : orientation par rapport à la caméra, jeu mécanique."""

    message = Signal(str)
    profile_saved = Signal()

    def __init__(
        self, microscope: Microscope, viewer: ViewerModel, live: LiveController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.m, self.viewer, self.live = microscope, viewer, live
        self._calibration = None
        lay = QVBoxLayout(self)
        box, blay = section(
            "Orientation de la platine",
            hint="Sur une lame avec des détails nets dans tout le champ. La platine fait deux petits pas "
            "et revient. Nécessaire pour centrer par double-clic et pour assembler les mosaïques.",
        )
        self.calibrate = button("Mesurer l'orientation")
        self.calibrate.setObjectName("primary")
        self.calibrate.clicked.connect(self._calibrate)
        self.orientation_text = QLabel()
        self.orientation_text.setWordWrap(True)
        self.save = button("Enregistrer dans la configuration")
        self.save.setEnabled(False)
        self.save.clicked.connect(self._save)
        for w in (self.calibrate, self.orientation_text, self.save):
            blay.addWidget(w)
        lay.addWidget(box)
        box, blay = section(
            "Jeu mécanique",
            hint="Écart entre la même position atteinte par la gauche ou par la droite. Reportez la valeur "
            "dans les options du pilote des moteurs : elle sera rattrapée automatiquement.",
        )
        self.backlash_x = button("Mesurer le jeu en X")
        self.backlash_y = button("Mesurer le jeu en Y")
        self.backlash_x.clicked.connect(lambda: self._backlash("x"))
        self.backlash_y.clicked.connect(lambda: self._backlash("y"))
        self.backlash_text = QLabel()
        self.backlash_text.setWordWrap(True)
        blay.addWidget(flow(self.backlash_x, self.backlash_y))
        blay.addWidget(self.backlash_text)
        lay.addWidget(box)
        lay.addStretch(1)
        if microscope.profile.stage_orientation:
            self.orientation_text.setText("Orientation déjà enregistrée dans la configuration.")

    def _buttons(self, enabled: bool) -> None:
        for b in (self.calibrate, self.backlash_x, self.backlash_y):
            b.setEnabled(enabled)

    def _start(self, fn, on_result) -> None:  # type: ignore[no-untyped-def]
        self._buttons(False)
        worker = create_worker(fn, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(on_result)
        worker.errored.connect(self._failed)
        worker.finished.connect(lambda: self._buttons(True))
        worker.start()

    def _failed(self, exc: Exception) -> None:
        for text in (self.orientation_text, self.backlash_text):
            if text.text().startswith("Mesure en cours"):
                text.setText("Mesure refusée : rien n'a été modifié.")
        QMessageBox.warning(self, "Platine", f"La mesure sur la platine n'a pas abouti : {exc}")

    def _calibrate(self) -> None:
        from fuji.core.stage import calibrate_stage

        self.orientation_text.setText("Mesure en cours…")
        self._start(lambda: calibrate_stage(self.m), self._calibrated)

    def _calibrated(self, calibration) -> None:  # type: ignore[no-untyped-def]
        self._calibration = calibration
        warn = ""
        if calibration.scale_error is not None and abs(calibration.scale_error) > 0.05:
            warn = (
                "<br><b>⚠ Le pixel mesuré par la platine diffère du pixel calibré</b> : vérifiez le nombre "
                "de µm par unité dans les options des moteurs, ou la calibration de l'objectif."
            )
        if calibration.skew > 0.05:
            warn += (
                "<br><b>⚠ Les deux axes ne semblent pas perpendiculaires</b> : recommencez sur une zone "
                "plus nette."
            )
        self.orientation_text.setText(calibration.summary[0].upper() + calibration.summary[1:] + "." + warn)
        self.save.setEnabled(True)

    def _save(self) -> None:
        from fuji.core.stage import save_stage_calibration

        if self._calibration is None:
            return
        save_stage_calibration(self.m, self._calibration)
        try:
            path = self.m.save_profile()
            self.message.emit(f"Orientation de la platine enregistrée dans {path}")
        except Exception as exc:
            self.message.emit(f"Orientation appliquée (non enregistrée : {exc})")
        self.profile_saved.emit()

    def _backlash(self, axis: str) -> None:
        from fuji.core.stage import measure_backlash

        self.backlash_text.setText("Mesure en cours…")
        self._start(
            lambda: measure_backlash(self.m, axis),
            lambda value: self.backlash_text.setText(f"Jeu en {axis.upper()} : <b>{number(value, 1)} µm</b>"),
        )


class AssistantsPanel(QWidget):
    """Les assistants : une liste où chacun est expliqué en une ligne, puis sa page (même
    principe que le panneau « Outils »)."""

    message = Signal(str)

    def __init__(
        self, microscope: Microscope, viewer: ViewerModel, live: LiveController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.kohler = KohlerWidget(microscope, viewer, live)
        self.focus = FocusWidget(microscope, viewer, live)
        self.calibration = CalibrationWidget(microscope, viewer, live)
        self.illumination = IlluminationWidget(microscope, viewer, live)
        self.stage: StageWidget | None = None
        if microscope.xy_stage is not None and not microscope.xy_stage.manual:
            self.stage = StageWidget(microscope, viewer, live)
        #: clé → (titre, icône, explication, page)
        self.assistants: dict[str, tuple[str, str, str, QWidget]] = {
            "kohler": (
                "Éclairage de Köhler",
                "bulb",
                "Régler le diaphragme et le condenseur, étape par étape",
                self.kohler,
            ),
            "focus": (
                "Mise au point",
                "focus",
                "Une jauge de netteté en direct pour trouver le meilleur point",
                self.focus,
            ),
            "calibration": (
                "Taille du pixel",
                "ruler",
                "Étalonner les mesures en µm avec un micromètre objet",
                self.calibration,
            ),
            "illumination": (
                "Uniformité de l'éclairage",
                "sun",
                "Vérifier le champ et prendre l'image de référence",
                self.illumination,
            ),
        }
        if self.stage is not None:
            self.assistants["stage"] = (
                "Platine motorisée",
                "move",
                "Mesurer l'orientation et le jeu mécanique de la platine",
                self.stage,
            )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        # la liste
        self.menu_page = QWidget()
        ml = QVBoxLayout(self.menu_page)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(2)
        self.rows: dict[str, ClickableRow] = {}
        for key, (title, icon_name, summary, _page) in self.assistants.items():
            entry = ClickableRow(title, summary, icon_name)
            entry.setObjectName("MenuHeader")
            chevron = QLabel()
            set_icon(chevron, "right", "muted", 14)
            entry.layout().addWidget(chevron, 0, Qt.AlignmentFlag.AlignVCenter)
            entry.clicked.connect(lambda k=key: self.open(k))
            self.rows[key] = entry
            ml.addWidget(entry)
        ml.addStretch(1)
        lay.addWidget(self.menu_page)
        # la page d'un assistant
        self.page = QWidget()
        pl = QVBoxLayout(self.page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(8)
        self.back_btn = button("Tous les assistants", "left", object_name="ghost")
        self.back_btn.clicked.connect(self.show_menu)
        pl.addWidget(self.back_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.page_title = label("", "step")
        pl.addWidget(self.page_title)
        self.stack = QStackedWidget()
        for _title, _icon, _summary, widget in self.assistants.values():
            widget.message.connect(self.message.emit)
            widget.layout().setContentsMargins(0, 0, 0, 0)
            self.stack.addWidget(widget)
        pl.addWidget(self.stack, 1)
        lay.addWidget(self.page, 1)
        self._current: str | None = None
        self.show_menu()

    def current(self) -> str | None:
        """La clé de l'assistant ouvert (``None`` : la liste)."""
        return self._current

    def open(self, key: str) -> None:
        """Ouvre la page d'un assistant : « kohler », « focus », « calibration »…"""
        title, _icon, _summary, widget = self.assistants[key]
        self._current = key
        self.page_title.setText(title)
        self.stack.setCurrentWidget(widget)
        self.menu_page.hide()
        self.page.show()

    def show_menu(self) -> None:
        self._current = None
        self.page.hide()
        self.menu_page.show()

    def shutdown(self) -> None:
        """Arrête les minuteurs avant la déconnexion du microscope."""
        for assistant in (self.kohler, self.focus):
            assistant.stop()
        self.illumination.shutdown()
