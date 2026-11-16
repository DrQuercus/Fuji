"""Panneau « Assistants » : Köhler, mise au point, calibration, éclairage / champ plat.

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
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from fuji.assistants import (
    FlatFieldStore,
    FocusTracker,
    KohlerAssistant,
    KohlerStep,
    average_frames,
    calibrate_micrometer,
    focus_peaking,
    illumination_uniformity,
)
from fuji.assistants.focus import METRICS, autofocus
from fuji.core.microscope import Microscope

from ..live import LiveController
from ..widgets import hint_label, row, section

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
        self.toggle = QPushButton("Démarrer l'assistant")
        self.toggle.setObjectName("primary")
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
        prev = QPushButton("◀ Étape précédente")
        prev.clicked.connect(lambda: (self.assistant.back(), self._render_step()))
        self.next = QPushButton("Étape suivante ▶")
        self.next.clicked.connect(lambda: (self.assistant.advance(), self._render_step()))
        lay.addWidget(row(prev, self.next))
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
        self.next.setStyleSheet("font-weight: bold;" if g.can_advance else "")
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
        self.toggle = QPushButton("Démarrer l'aide à la mise au point")
        self.toggle.setObjectName("primary")
        self.toggle.clicked.connect(self._toggle)
        lay.addWidget(self.toggle)
        self.metric = QComboBox()
        self.metric.addItems(METRICS)
        self.metric.currentTextChanged.connect(self._set_metric)
        lay.addWidget(row(QLabel("Mesure"), self.metric))
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFormat("Netteté : %p % du meilleur")
        lay.addWidget(self.bar)
        self.trend = QLabel()
        lay.addWidget(self.trend)
        reset = QPushButton("Oublier le meilleur")
        reset.clicked.connect(self.tracker.reset)
        lay.addWidget(reset)
        self.peaking = QCheckBox("Surligner les zones nettes (focus peaking)")
        self.peaking.toggled.connect(lambda on: None if on else self._remove_peaking())
        lay.addWidget(self.peaking)
        self._af_was_live = False
        if self.m.focus is not None and not self.m.focus.manual:
            self.af_button = QPushButton("Autofocus")
            self.af_button.clicked.connect(self._autofocus)
            lay.addWidget(self.af_button)
        else:
            lay.addWidget(
                hint_label(
                    "Mise au point manuelle : tournez la vis jusqu'à dépasser le maximum, puis "
                    "revenez au point où la barre est pleine."
                )
            )
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
            self.toggle.setText("Arrêter")

    def process(self, frame: np.ndarray) -> None:
        st = self.tracker.update(frame)
        self.bar.setValue(int(100 * st.relative))
        self.trend.setText(f"Tendance : {st.trend}")
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
        worker = create_worker(autofocus, self.m, _start_thread=False)
        worker.returned.connect(self._autofocus_done)
        worker.errored.connect(self._autofocus_failed)
        worker.finished.connect(self._autofocus_finished)
        worker.start()

    def _autofocus_done(self, z: float) -> None:
        self.message.emit(f"Autofocus : Z = {z:.2f} µm")

    def _autofocus_failed(self, exc: Exception) -> None:
        QMessageBox.warning(self, "Autofocus", str(exc))

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
        lay.addWidget(row(QLabel("Écart entre graduations"), self.spacing))
        self.measure = QPushButton("Mesurer sur l'image actuelle")
        self.measure.setObjectName("primary")
        self.measure.clicked.connect(self._measure)
        lay.addWidget(self.measure)
        self.result = QLabel()
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        self.save = QPushButton("Enregistrer pour l'objectif en place")
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
            self._grab_and_measure, self.spacing.value(), self.live.running, _start_thread=False
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
                f"<br>Valeur actuelle : {current:.4f} µm ({self.m.pixel_size_source()}), "
                f"écart {100 * (r.pixel_size_um / current - 1):+.1f} %"
            )
        warn = ""
        if not r.reliable:
            warn = (
                "<br><b>⚠ Mesure peu fiable</b> : il faut au moins 5 graduations nettes et "
                "régulières. Cadrez plus de traits, refaites la mise au point, puis recommencez."
            )
        self.result.setText(
            f"<b>{r.pixel_size_um:.4f} µm/pixel</b><br>{r.n_lines} traits, "
            f"période {r.period_px:.2f} px, rotation caméra {r.angle_deg:+.2f}°, "
            f"incertitude ±{r.relative_uncertainty * 100:.2f} %{cmp}{warn}"
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
                f"±{r.relative_uncertainty * 100:.1f} %). Toutes les mesures faites avec cet "
                "objectif en dépendront.\n\nEnregistrer quand même ?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.m.set_pixel_calibration(r.pixel_size_um, label)
        except Exception as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        try:
            path = self.m.save_profile()
            self.message.emit(f"Calibration de {label} enregistrée dans {path}")
        except Exception as exc:
            self.message.emit(f"Calibration appliquée (non enregistrée : {exc})")
        self.profile_saved.emit()


# --------------------------------------------------------- éclairage, champ plat
class IlluminationWidget(QWidget):
    message = Signal(str)
    #: objectif ou canal changé (émis depuis n'importe quel thread, traité dans celui de l'interface)
    _keys_changed = Signal()

    def __init__(
        self, microscope: Microscope, viewer: ViewerModel, live: LiveController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.m, self.viewer, self.live = microscope, viewer, live
        self.store = FlatFieldStore(microscope.profile.name)
        lay = QVBoxLayout(self)
        box, blay = section(
            "Uniformité",
            hint="Sur une zone vide de la lame, diaphragme de champ ouvert juste au-delà du champ.",
        )
        analyse = QPushButton("Analyser l'éclairage")
        analyse.clicked.connect(self._uniformity)
        blay.addWidget(analyse)
        self.uni_text = QLabel()
        self.uni_text.setWordWrap(True)
        blay.addWidget(self.uni_text)
        lay.addWidget(box)
        box, blay = section(
            "Champ plat",
            hint="Corrige le vignettage et l'éclairage inégal : indispensable pour mesurer des intensités.",
        )
        self.flat_btn = QPushButton("Acquérir le champ plat (lame vide)")
        self.flat_btn.clicked.connect(lambda: self._acquire("flat"))
        self.dark_btn = QPushButton("Acquérir le noir (lumière coupée)")
        self.dark_btn.clicked.connect(lambda: self._acquire("dark"))
        self.apply = QCheckBox("Corriger les captures")
        self.apply.toggled.connect(self._refresh_status)
        self.status = QLabel()
        self.status.setWordWrap(True)
        for w in (self.flat_btn, self.dark_btn, self.apply, self.status):
            blay.addWidget(w)
        lay.addWidget(box)
        lay.addStretch(1)
        self._dark: np.ndarray | None = None
        self._pending: tuple[str, tuple[str | None, str | None]] | None = None
        self._keys_changed.connect(self._refresh_status)
        microscope.objective_changed.connect(self._on_keys_changed)
        microscope.channel_changed.connect(self._on_keys_changed)
        self._refresh_status()

    def _keys(self) -> tuple[str | None, str | None]:
        obj = self.m.objective
        return (obj.label if obj else None), self.m.channel

    def _on_keys_changed(self, *_: object) -> None:
        try:
            self._keys_changed.emit()
        except RuntimeError:  # panneau déjà détruit (changement de microscope)
            pass

    def _refresh_status(self, *_: object) -> None:
        ref = self.store.load(*self._keys())
        obj, ch = self._keys()
        self.status.setText(
            f"Référence enregistrée pour {obj or '—'} / {ch or '—'}"
            if ref
            else f"Aucune référence pour {obj or '—'} / {ch or '—'}"
        )

    def _uniformity(self) -> None:
        try:
            frame = self.m.snap().data
        except Exception as exc:
            QMessageBox.warning(self, "Éclairage", str(exc))
            return
        rep = illumination_uniformity(frame)
        self.uni_text.setText(
            f"Min/max : {rep.min_max_ratio:.2f} · coins/centre : {rep.corner_center_ratio:.2f}"
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

    def _acquire(self, kind: str) -> None:
        """Moyenne de 8 images, dans un thread de travail (plusieurs secondes en pose longue)."""
        if self._pending is not None:
            return
        self._pending = (kind, self._keys())
        for b in (self.flat_btn, self.dark_btn):
            b.setEnabled(False)
        self.message.emit("Acquisition de la référence (8 images)…")
        worker = create_worker(average_frames, self.m, 8, _start_thread=False)
        worker.returned.connect(self._acquired)
        worker.errored.connect(self._acquire_failed)
        worker.finished.connect(self._acquire_finished)
        worker.start()

    def _acquire_failed(self, exc: Exception) -> None:
        QMessageBox.warning(self, "Champ plat", str(exc))

    def _acquire_finished(self) -> None:
        self._pending = None
        for b in (self.flat_btn, self.dark_btn):
            b.setEnabled(True)

    def _acquired(self, image: np.ndarray) -> None:
        if self._pending is None:
            return
        kind, keys = self._pending
        if kind == "dark":
            self._dark = image
            ref = self.store.load(*keys)
            if ref is not None and ref[0].shape == image.shape:
                self.store.save(*keys, ref[0], image)
                self.message.emit("Image de noir enregistrée avec le champ plat existant")
            else:
                self.message.emit("Image de noir acquise (elle sera enregistrée avec le champ plat)")
            return
        cam = self.m.camera
        if cam is not None:
            saturated = float(np.mean(image >= 0.98 * (2**cam.bit_depth - 1)))
            if saturated > 0.01:
                QMessageBox.warning(
                    self,
                    "Champ plat",
                    f"{saturated * 100:.0f} % de l'image est saturée : la référence serait fausse.\n\n"
                    "Réduisez l'exposition ou la lampe, puis recommencez.",
                )
                return
        dark = self._dark if self._dark is not None and self._dark.shape == image.shape else None
        path = self.store.save(*keys, image, dark)
        self.message.emit(f"Champ plat enregistré : {path}")
        self._refresh_status()

    def correct(self, data: np.ndarray) -> tuple[np.ndarray, bool | None]:
        """Applique la correction de champ plat si elle est demandée.

        Renvoie l'image et : ``None`` si la correction n'est pas demandée, ``True`` si
        elle est appliquée, ``False`` si elle est demandée mais impossible (pas de
        référence pour cet objectif et ce canal, ou taille d'image différente).
        """
        if not self.apply.isChecked():
            return data, None
        out = self.store.correct(data, *self._keys())
        return (data, False) if out is None else (out, True)


class AssistantsPanel(QTabWidget):
    message = Signal(str)

    def __init__(
        self, microscope: Microscope, viewer: ViewerModel, live: LiveController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.kohler = KohlerWidget(microscope, viewer, live)
        self.focus = FocusWidget(microscope, viewer, live)
        self.calibration = CalibrationWidget(microscope, viewer, live)
        self.illumination = IlluminationWidget(microscope, viewer, live)
        self.addTab(self.kohler, "Köhler")
        self.addTab(self.focus, "Netteté")
        self.addTab(self.calibration, "Calibration")
        self.addTab(self.illumination, "Éclairage")
        for w in (self.kohler, self.focus, self.calibration, self.illumination):
            w.message.connect(self.message.emit)
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setDocumentMode(True)
        self.setUsesScrollButtons(False)
        self.setElideMode(Qt.TextElideMode.ElideNone)
        self.tabBar().setExpanding(False)

    def shutdown(self) -> None:
        """Arrête les minuteurs avant la déconnexion du microscope."""
        for assistant in (self.kohler, self.focus):
            assistant.stop()
