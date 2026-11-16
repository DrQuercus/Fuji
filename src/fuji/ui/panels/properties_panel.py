"""Panneau « Propriétés » : tous les réglages de tous les périphériques.

Généré automatiquement à partir des propriétés déclarées par les pilotes : il
fonctionne avec n'importe quel matériel (ToupTek, Micro-Manager, pilote maison).
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fuji.core.microscope import Microscope

from ..widgets import property_editor, section


class PropertiesPanel(QWidget):
    def __init__(self, microscope: Microscope, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.m = microscope
        self.expert = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        refresh = QPushButton("Actualiser")
        refresh.clicked.connect(self.rebuild)
        outer.addWidget(refresh)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll, 1)
        self.rebuild()

    def set_expert(self, expert: bool) -> None:
        self.expert = expert
        self.rebuild()

    def rebuild(self) -> None:
        content = QWidget()
        lay = QVBoxLayout(content)
        for dev in self.m.devices.values():
            infos = [i for i in dev.property_infos() if self.expert or not i.advanced]
            if not infos:
                continue
            title = f"{dev.label}" + (" — manuel" if dev.manual else "")
            box, blay = section(title)
            by_cat: dict[str, list] = {}
            for info in infos:
                by_cat.setdefault(info.category, []).append(info)
            for cat, items in by_cat.items():
                if len(by_cat) > 1:
                    blay.addWidget(QLabel(f"<i>{cat}</i>"))
                form = QFormLayout()
                for info in items:
                    try:
                        value = dev.get_property(info.name)
                    except Exception as exc:  # propriété illisible : on l'affiche quand même
                        value = f"(erreur : {exc})"
                        form.addRow(info.name, QLabel(str(value)))
                        continue

                    def on_change(v, d=dev, n=info.name):  # type: ignore[no-untyped-def]
                        try:
                            d.set_property(n, v)
                        except Exception as exc:
                            self.window().statusBar().showMessage(f"{n} : {exc}", 5000)

                    editor = property_editor(info, value, on_change)
                    if info.description:
                        editor.setToolTip(info.description)
                    form.addRow(info.name, editor)
                blay.addLayout(form)
            lay.addWidget(box)
        lay.addStretch(1)
        self._scroll.setWidget(content)
