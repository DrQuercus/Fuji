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
        self._draw(g)

    def _draw(self, g) -> None:  # type: ignore[no-untyped-def]
        rep = g.report
        s = self._scale()
        h, w = self.m.require_camera().sensor_shape()
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
        if self.m.focus is not None and not self.m.focus.manual:
            af = QPushButton("Autofocus")
            af.clicked.connect(self._autofocus)
            lay.addWidget(af)
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
            mask = focus_peaking(frame).astype(np.uint8)
            s = self._scale()
            if PEAKING_LAYER in self.viewer.layers:
                self.viewer.layers[PEAKING_LAYER].data = mask
            else:
                self.viewer.add_image(
                    mask,
                    name=PEAKING_LAYER,
                    colormap="red",
                    blending="additive",
                    opacity=0.8,
                    contrast_limits=(0, 1),
                    scale=(s, s),
                    units=self._units(),
                )

    def _remove_peaking(self) -> None:
        if PEAKING_LAYER in self.viewer.layers:
            self.viewer.layers.remove(PEAKING_LAYER)

    def _autofocus(self) -> None:
        was_live = self.live.running
        if was_live:
            self.live.stop()
        try:
            z = autofocus(self.m)
            self.message.emit(f"Autofocus : Z = {z:.2f} µm")
        except Exception as exc:
            QMessageBox.warning(self, "Autofocus", str(exc))
        finally:
            if was_live:
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
        measure = QPushButton("Mesurer sur l'image actuelle")
        measure.setObjectName("primary")
        measure.clicked.connect(self._measure)
        lay.addWidget(measure)
        self.result = QLabel()
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        self.save = QPushButton("Enregistrer pour l'objectif en place")
        self.save.setEnabled(False)
        self.save.clicked.connect(self._save)
        lay.addWidget(self.save)
        lay.addStretch(1)

    def _measure(self) -> None:
        cam = self.m.require_camera()
        _, frame = cam.latest_frame()
        if frame is None:
            frame = self.m.snap().data
        try:
            self._result = calibrate_micrometer(frame, self.spacing.value())
        except Exception as exc:
            self.result.setText(f"Échec : {exc}")
            self.save.setEnabled(False)
            return
        r = self._result
        current = self.m.pixel_size_um()
        cmp = ""
        if current:
            cmp = (
                f"<br>Valeur actuelle : {current:.4f} µm ({self.m.pixel_size_source()}), "
                f"écart {100 * (r.pixel_size_um / current - 1):+.1f} %"
            )
        self.result.setText(
            f"<b>{r.pixel_size_um:.4f} µm/pixel</b><br>{r.n_lines} traits, "
            f"période {r.period_px:.2f} px, rotation caméra {r.angle_deg:+.2f}°, "
            f"incertitude ±{r.relative_uncertainty * 100:.2f} %{cmp}"
        )
        obj = self.m.objective
        self.save.setEnabled(obj is not None)
        self.save.setText(
            f"Enregistrer pour l'objectif {obj.label}" if obj else "Sélectionnez d'abord l'objectif en place"
        )

    def _save(self) -> None:
        if self._result is None:
            return
        self.m.set_pixel_calibration(self._result.pixel_size_um)
        try:
            path = self.m.save_profile()
            self.message.emit(f"Calibration enregistrée dans {path}")
        except Exception as exc:
            self.message.emit(f"Calibration appliquée (non enregistrée : {exc})")
        self.profile_saved.emit()


# --------------------------------------------------------- éclairage, champ plat
class IlluminationWidget(QWidget):
    message = Signal(str)

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
        flat = QPushButton("Acquérir le champ plat (lame vide)")
        flat.clicked.connect(self._acquire_flat)
        dark = QPushButton("Acquérir le noir (lumière coupée)")
        dark.clicked.connect(self._acquire_dark)
        self.apply = QCheckBox("Corriger les captures")
        self.status = QLabel()
        self.status.setWordWrap(True)
        for w in (flat, dark, self.apply, self.status):
            blay.addWidget(w)
        lay.addWidget(box)
        lay.addStretch(1)
        self._dark: np.ndarray | None = None
        self._refresh_status()

    def _keys(self) -> tuple[str | None, str | None]:
        obj = self.m.objective
        return (obj.label if obj else None), self.m.channel

    def _refresh_status(self) -> None:
        ref = self.store.load(*self._keys())
        obj, ch = self._keys()
        self.status.setText(
            f"Référence enregistrée pour {obj or '—'} / {ch or '—'}"
            if ref
            else f"Aucune référence pour {obj or '—'} / {ch or '—'}"
        )

    def _uniformity(self) -> None:
        frame = self.m.snap().data
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

    def _acquire_dark(self) -> None:
        self._dark = average_frames(self.m, 8)
        self.message.emit("Image de noir acquise (elle sera enregistrée avec le champ plat)")

    def _acquire_flat(self) -> None:
        flat = average_frames(self.m, 8)
        path = self.store.save(*self._keys(), flat, self._dark)
        self.message.emit(f"Champ plat enregistré : {path}")
        self._refresh_status()

    def correct(self, data: np.ndarray) -> np.ndarray:
        """Applique la correction si demandée et disponible."""
        if not self.apply.isChecked():
            return data
        out = self.store.correct(data, *self._keys())
        return data if out is None else out


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
