"""Pages de l'éditeur de configuration : général, statif, périphériques, rôles,
objectifs, cubes de filtres, canaux, YAML."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from fuji.core import catalog
from fuji.core.profile import (
    CONTRASTS,
    CORRECTIONS,
    IMMERSIONS,
    STANDS,
    TUBE_LENGTHS,
    MicroscopeProfile,
    load_yaml,
)
from fuji.drivers import CATEGORIES, driver_info, drivers_info

from .. import icons
from ..components import (
    Card,
    ColorChooser,
    PathEdit,
    badge,
    button,
    hint,
    icon_button,
    label,
)
from ..theme import palette
from . import model as M
from .forms import OptionForm

if TYPE_CHECKING:
    from .editor import ConfigEditor

NONE = "(aucun)"
UNCHANGED = "— inchangé —"


def _opt_spin(
    value: float | None,
    lo: float,
    hi: float,
    *,
    decimals: int = 2,
    suffix: str = "",
    none_text: str = "—",
    step: float = 1.0,
) -> QDoubleSpinBox:
    """Nombre facultatif : la valeur minimale affiche ``none_text`` et signifie « non défini »."""
    w = QDoubleSpinBox()
    w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setSpecialValueText(none_text)
    if suffix:
        w.setSuffix(f" {suffix}")
    w.setValue(lo if value is None else float(value))
    w.setKeyboardTracking(False)
    return w


def _spin_value(w: QDoubleSpinBox) -> float | None:
    return None if w.value() == w.minimum() and w.specialValueText() else w.value()


class Page(QWidget):
    title = ""
    subtitle = ""
    icon = "sliders"
    changed = Signal()

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__()
        self.editor = editor
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(12)

    def load(self, data: dict[str, Any]) -> None:  # pragma: no cover - surchargé
        raise NotImplementedError

    def store(self, data: dict[str, Any]) -> None:  # pragma: no cover - surchargé
        raise NotImplementedError


# ======================================================================= général
class GeneralPage(Page):
    title = "Général"
    subtitle = "Nom du microscope, adaptateur caméra et dossier d'enregistrement."
    icon = "info"

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        card = Card("Identité", icon_name="microscope")
        form = QFormLayout()
        self.name = QLineEdit()
        self.description = QPlainTextEdit()
        self.description.setFixedHeight(80)
        form.addRow("Nom", self.name)
        form.addRow("Description", self.description)
        card.add(layout=form)
        self.lay.addWidget(card)

        card = Card("Optique de la caméra", icon_name="camera")
        form = QFormLayout()
        self.adapter = QDoubleSpinBox()
        self.adapter.setRange(0.1, 5)
        self.adapter.setDecimals(3)
        self.adapter.setSingleStep(0.05)
        self.adapter.setSuffix(" x")
        form.addRow("Adaptateur caméra", self.adapter)
        form.addRow(
            "",
            hint(
                "Gravé sur l'adaptateur monture C (0.35x, 0.5x, 0.63x, 1x...). "
                "Sert à estimer la taille du pixel avant calibration."
            ),
        )
        self.condenser_na = _opt_spin(None, 0, 1.4, decimals=2, none_text="inconnue", step=0.05)
        form.addRow("ON du condenseur", self.condenser_na)
        form.addRow("", hint("Ouverture numérique maximale gravée sur le condenseur (assistant Köhler)."))
        card.add(layout=form)
        self.lay.addWidget(card)

        card = Card("Enregistrement", icon_name="save")
        form = QFormLayout()
        self.save_dir = PathEdit("dir")
        self.default_channel = QComboBox()
        form.addRow("Dossier des acquisitions", self.save_dir)
        form.addRow("Canal au démarrage", self.default_channel)
        card.add(layout=form)
        self.lay.addWidget(card)
        self.lay.addStretch(1)

    def load(self, data: dict[str, Any]) -> None:
        self.name.setText(data.get("name", ""))
        self.description.setPlainText(data.get("description", ""))
        self.adapter.setValue(float(data.get("camera_adapter_magnification") or 1.0))
        na = data.get("condenser_na")
        self.condenser_na.setValue(0 if na is None else float(na))
        self.save_dir.setText(data.get("save_directory") or "")
        self.default_channel.clear()
        self.default_channel.addItem(NONE)
        self.default_channel.addItems([c["name"] for c in data.get("channels", [])])
        self.default_channel.setCurrentText(data.get("default_channel") or NONE)

    def store(self, data: dict[str, Any]) -> None:
        data["name"] = self.name.text().strip() or "Microscope"
        data["description"] = self.description.toPlainText().strip()
        data["camera_adapter_magnification"] = self.adapter.value()
        data["condenser_na"] = _spin_value(self.condenser_na)
        data["save_directory"] = self.save_dir.text().strip() or None
        dc = self.default_channel.currentText()
        data["default_channel"] = None if dc == NONE else dc


# ======================================================================= statif
class InstrumentPage(Page):
    title = "Statif et optique"
    subtitle = "Le microscope lui-même : marque, type, tube, oculaires, condenseur."
    icon = "microscope"

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        card = Card("Statif", icon_name="microscope")
        form = QFormLayout()
        self.manufacturer = QLineEdit()
        self.manufacturer.setPlaceholderText("Leitz, Zeiss, Nikon, Olympus...")
        self.model = QLineEdit()
        self.model.setPlaceholderText("Diaplan, Axioskop, Eclipse...")
        self.stand = QComboBox()
        self.stand.addItems(STANDS)
        self.tube = QComboBox()
        self.tube.addItems(TUBE_LENGTHS)
        self.serial = QLineEdit()
        form.addRow("Fabricant", self.manufacturer)
        form.addRow("Modèle", self.model)
        form.addRow("Type", self.stand)
        form.addRow("Longueur de tube", self.tube)
        form.addRow(
            "",
            hint(
                "Gravée sur les objectifs : « 160/0.17 », « 170/0.17 » ou « ∞/0.17 ». "
                "Les objectifs ne sont interchangeables qu'à longueur égale."
            ),
        )
        form.addRow("N° de série", self.serial)
        card.add(layout=form)
        self.lay.addWidget(card)

        card = Card("Observation", icon_name="focus")
        form = QFormLayout()
        self.eyepiece = _opt_spin(10, 0, 40, decimals=1, suffix="x", none_text="—")
        self.fn = _opt_spin(None, 0, 32, decimals=1, suffix="mm", none_text="inconnu")
        self.condenser = QLineEdit()
        self.condenser.setPlaceholderText("ex. condenseur achromatique 0.90, tourelle phase")
        self.notes = QPlainTextEdit()
        self.notes.setFixedHeight(70)
        form.addRow("Oculaires", self.eyepiece)
        form.addRow("Indice de champ", self.fn)
        form.addRow("Condenseur", self.condenser)
        form.addRow("Notes", self.notes)
        card.add(layout=form)
        self.lay.addWidget(card)

        self.fov_card = Card("Champ et résolution par objectif", icon_name="objective")
        self.fov_grid = QGridLayout()
        self.fov_card.add(layout=self.fov_grid)
        self.lay.addWidget(self.fov_card)
        self.fn.valueChanged.connect(lambda *_: self._refresh_fov())
        self.lay.addStretch(1)
        self._objectives: list[dict[str, Any]] = []

    def load(self, data: dict[str, Any]) -> None:
        ins = data.get("instrument") or {}
        self.manufacturer.setText(ins.get("manufacturer", ""))
        self.model.setText(ins.get("model", ""))
        self.stand.setCurrentText(ins.get("stand", "droit"))
        self.tube.setCurrentText(ins.get("tube_length", "infini"))
        self.serial.setText(ins.get("serial_number", ""))
        self.eyepiece.setValue(float(ins.get("eyepiece_magnification") or 0))
        self.fn.setValue(float(ins.get("field_number_mm") or 0))
        self.condenser.setText(ins.get("condenser", ""))
        self.notes.setPlainText(ins.get("notes", ""))
        self._objectives = data.get("objectives", [])
        self._refresh_fov()

    def _refresh_fov(self) -> None:
        while self.fov_grid.count():
            item = self.fov_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        heads = ("Objectif", "Champ aux oculaires", "Résolution", "Profondeur de champ")
        for c, h in enumerate(heads):
            self.fov_grid.addWidget(label(h, "SectionTitle", wrap=False), 0, c)
        fn = _spin_value(self.fn)
        for r, o in enumerate(self._objectives, start=1):
            na, mag = float(o.get("na", 0) or 0), float(o.get("magnification", 0) or 0)
            n = {"huile": 1.515, "eau": 1.33, "glycérol": 1.47, "silicone": 1.40}.get(
                o.get("immersion", "air"), 1.0
            )
            fov = f"⌀ {fn / mag * 1000:.0f} µm" if fn and mag else "—"
            res = f"{0.61 * 0.55 / na:.2f} µm" if na else "—"
            dof = f"{0.55 * n / na**2:.1f} µm" if na else "—"
            for c, text in enumerate((o.get("label", "?"), fov, res, dof)):
                self.fov_grid.addWidget(QLabel(text), r, c)
        if not self._objectives:
            self.fov_grid.addWidget(hint("Ajoutez des objectifs dans l'onglet Objectifs."), 1, 0, 1, 4)

    def store(self, data: dict[str, Any]) -> None:
        data["instrument"] = {
            "manufacturer": self.manufacturer.text().strip(),
            "model": self.model.text().strip(),
            "stand": self.stand.currentText(),
            "tube_length": self.tube.currentText(),
            "serial_number": self.serial.text().strip(),
            "eyepiece_magnification": _spin_value(self.eyepiece),
            "field_number_mm": _spin_value(self.fn),
            "condenser": self.condenser.text().strip(),
            "notes": self.notes.toPlainText().strip(),
        }


# ================================================================= périphériques
class DevicesPage(Page):
    title = "Périphériques"
    subtitle = "Caméras, tourelles, lampes, platines... et le pilote qui commande chacun."
    icon = "chip"

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        row = QHBoxLayout()
        row.setSpacing(12)
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.setMinimumWidth(250)
        self.list.setMaximumWidth(300)
        self.list.setIconSize(QSize(18, 18))
        self.list.currentRowChanged.connect(self._select)
        left.addWidget(self.list, 1)
        tools = QHBoxLayout()
        tools.setSpacing(4)
        self.add_btn = button("Ajouter", "plus", primary=True)
        self.add_btn.setMenu(self._add_menu())
        dup = icon_button("copy", "Dupliquer", object_name="")
        dup.clicked.connect(self._duplicate)
        rm = icon_button("trash", "Supprimer", object_name="")
        rm.clicked.connect(self._remove)
        up = icon_button("up", "Monter", object_name="")
        up.clicked.connect(lambda: self._move(-1))
        down = icon_button("down", "Descendre", object_name="")
        down.clicked.connect(lambda: self._move(1))
        tools.addWidget(self.add_btn, 1)
        for b in (dup, rm, up, down):
            tools.addWidget(b)
        left.addLayout(tools)
        left.addWidget(
            hint("L'ordre compte : un périphérique peut faire référence à ceux déclarés au-dessus de lui.")
        )
        row.addLayout(left)
        self.detail_host = QVBoxLayout()
        row.addLayout(self.detail_host, 1)
        self.lay.addLayout(row, 1)
        self._data: dict[str, Any] = {}
        self._detail: QWidget | None = None
        self._form: OptionForm | None = None

    # ------------------------------------------------------------ liste
    def _add_menu(self) -> QMenu:
        menu = QMenu(self)
        infos = drivers_info()
        for cat in CATEGORIES:
            group = [i for i in infos if i.category == cat]
            if not group:
                continue
            menu.addSection(cat)
            for info in group:
                act = menu.addAction(
                    icons.icon(M.KIND_ICONS.get(info.kind, "chip"), palette().muted, 16), info.title
                )
                act.setToolTip(info.description)
                act.triggered.connect(lambda _=False, n=info.name: self._add(n))
        menu.setToolTipsVisible(True)
        return menu

    def _item_text(self, d: dict[str, Any]) -> str:
        info = driver_info(d["driver"])
        return f"{d.get('label') or d['name']}\n{info.title if info else d['driver']}"

    def _refresh_list(self, select: int | None = None) -> None:
        self.list.blockSignals(True)
        current = self.list.currentRow() if select is None else select
        self.list.clear()
        roles = {v: k for k, v in (self._data.get("roles") or {}).items() if v}
        for d in M.devices(self._data):
            info = driver_info(d["driver"])
            kind = info.kind if info else None
            item = QListWidgetItem(
                icons.icon(M.KIND_ICONS.get(kind, "chip"), palette().text, 18), self._item_text(d)
            )
            tip = f"Identifiant : {d['name']}"
            if d["name"] in roles:
                tip += f"\nRôle : {M.ROLE_TITLES[roles[d['name']]]}"
            item.setToolTip(tip)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.list.count():
            self.list.setCurrentRow(max(0, min(current, self.list.count() - 1)))
        else:
            self._show_empty()

    def _show_empty(self) -> None:
        from ..components import EmptyState

        self._set_detail(
            EmptyState(
                "chip", "Aucun périphérique", "Ajoutez une caméra, un revolver d'objectifs, une lampe..."
            )
        )

    def _set_detail(self, w: QWidget) -> None:
        if self._detail is not None:
            self.detail_host.removeWidget(self._detail)
            self._detail.deleteLater()
        self._detail = w
        self.detail_host.addWidget(w)

    def _add(self, driver: str) -> None:
        M.add_device(self._data, driver)
        self._refresh_list(select=len(M.devices(self._data)) - 1)
        self.changed.emit()

    def _duplicate(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        src = copy.deepcopy(M.devices(self._data)[row])
        src["name"] = M.unique_name(self._data, src["name"])
        M.devices(self._data).insert(row + 1, src)
        self._refresh_list(select=row + 1)
        self.changed.emit()

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        name = M.devices(self._data)[row]["name"]
        answer = QMessageBox.question(
            self,
            "Supprimer",
            f"Supprimer « {name} » ? Ses rôles et ses réglages dans les canaux seront aussi retirés.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        M.remove_device(self._data, name)
        self._refresh_list(select=row)
        self.changed.emit()

    def _move(self, delta: int) -> None:
        row = self.list.currentRow()
        devs = M.devices(self._data)
        new = row + delta
        if row < 0 or not 0 <= new < len(devs):
            return
        devs[row], devs[new] = devs[new], devs[row]
        self._refresh_list(select=new)
        self.changed.emit()

    # ------------------------------------------------------------ détail
    def _select(self, row: int) -> None:
        devs = M.devices(self._data)
        if not 0 <= row < len(devs):
            return
        d = devs[row]
        info = driver_info(d["driver"])
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)
        card = Card(
            info.title if info else d["driver"],
            icon_name=M.KIND_ICONS.get(info.kind if info else None, "chip"),
        )
        if info is not None:
            card.header.addWidget(badge(info.category, "accent"))
            if info.description:
                card.add(hint(info.description))
        form = QFormLayout()
        name = QLineEdit(d["name"])
        name.setToolTip("Identifiant utilisé dans les canaux et les rôles (sans espace)")
        name.editingFinished.connect(lambda: self._rename(d, name))
        lbl = QLineEdit(d.get("label") or "")
        lbl.setPlaceholderText(d["name"])
        lbl.textChanged.connect(lambda t: (d.__setitem__("label", t or None), self._update_item()))
        drv = QComboBox()
        for i in drivers_info():
            drv.addItem(i.title, i.name)
        if drv.findData(d["driver"]) < 0:
            drv.addItem(d["driver"], d["driver"])
        drv.setCurrentIndex(drv.findData(d["driver"]))
        drv.activated.connect(lambda _i: self._change_driver(d, drv.currentData()))
        form.addRow("Nom affiché", lbl)
        form.addRow("Identifiant", name)
        form.addRow("Pilote", drv)
        card.add(layout=form)
        v.addWidget(card)

        if info is not None:
            opts = Card("Réglages du pilote", icon_name="sliders")
            by_kind: dict[str | None, list[str]] = {}
            for other in M.devices(self._data):
                if other is d:
                    continue
                oi = driver_info(other["driver"])
                by_kind.setdefault(oi.kind if oi else None, []).append(other["name"])
            self._form = OptionForm(info, d.get("options") or {}, by_kind)
            self._form.changed.connect(lambda: self._store_options(d))
            opts.add(self._form)
            self._extras(d, info, opts)
            v.addWidget(opts)
        v.addStretch(1)
        self._set_detail(host)

    def _extras(self, d: dict[str, Any], info: Any, card: Card) -> None:
        roles = self._data.get("roles") or {}
        if info.name in ("manual.state", "demo.state") and (
            roles.get("objective_changer") == d["name"] or d["name"].lower().startswith("obj")
        ):
            card.add(
                hint(
                    "Revolver d'objectifs : laissez « Positions » vide, les positions "
                    "seront les objectifs de l'onglet Objectifs, dans l'ordre."
                )
            )
        if info.name == "touptek.camera":
            detect = button("Détecter les caméras branchées", "search")
            detect.clicked.connect(lambda: self._detect_touptek(d))
            card.add(detect)
        if info.name == "micromanager":
            card.add(
                hint(
                    "Le fichier .cfg se crée avec l'assistant de configuration matérielle "
                    "de Micro-Manager. Rôles et canaux sont détectés à la connexion."
                )
            )

    def _detect_touptek(self, d: dict[str, Any]) -> None:
        from fuji.drivers.touptek import list_cameras

        try:
            cams = list_cameras((d.get("options") or {}).get("sdk_path"))
        except Exception as exc:
            QMessageBox.warning(self, "ToupTek", f"Détection impossible :\n{exc}")
            return
        if not cams:
            QMessageBox.information(self, "ToupTek", "Aucune caméra ToupTek détectée.")
            return
        text = "\n".join(f"• {c['nom']} — {', '.join(c['résolutions'])}" for c in cams)
        QMessageBox.information(self, "ToupTek", f"Caméras détectées :\n{text}")
        if len(cams) == 1:
            d.setdefault("options", {})["camera_id"] = cams[0]["nom"]
            self._select(self.list.currentRow())

    def _store_options(self, d: dict[str, Any]) -> None:
        if self._form is None:
            return
        try:
            d["options"] = self._form.values()
        except Exception:  # saisie en cours (YAML incomplet...) : on ignore
            return
        self.changed.emit()

    def _rename(self, d: dict[str, Any], field: QLineEdit) -> None:
        new = M.slug(field.text())
        if new == d["name"]:
            field.setText(new)
            return
        try:
            M.rename_device(self._data, d["name"], new)
        except ValueError as exc:
            QMessageBox.warning(self, "Identifiant", str(exc))
            field.setText(d["name"])
            return
        field.setText(new)
        self._update_item()
        self.changed.emit()

    def _change_driver(self, d: dict[str, Any], driver: str) -> None:
        if driver == d["driver"]:
            return
        answer = QMessageBox.question(
            self, "Changer de pilote", "Changer de pilote remet ses réglages par défaut. Continuer ?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._select(self.list.currentRow())
            return
        info = driver_info(driver)
        d["driver"] = driver
        d["options"] = copy.deepcopy(info.defaults()) if info else {}
        self._update_item()
        self._select(self.list.currentRow())
        self.changed.emit()

    def _update_item(self) -> None:
        row = self.list.currentRow()
        devs = M.devices(self._data)
        if 0 <= row < len(devs):
            self.list.item(row).setText(self._item_text(devs[row]))

    # ------------------------------------------------------------ chargement
    def load(self, data: dict[str, Any]) -> None:
        self._data = data
        self._refresh_list(select=max(0, self.list.currentRow()))

    def store(self, data: dict[str, Any]) -> None:
        row = self.list.currentRow()
        devs = M.devices(data)
        if self._form is not None and 0 <= row < len(devs):
            try:
                devs[row]["options"] = self._form.values()
            except Exception as exc:
                raise ValueError(f"Options de « {devs[row]['name']} » invalides : {exc}") from exc


# ======================================================================= rôles
class RolesPage(Page):
    title = "Rôles"
    subtitle = "Quel périphérique sert de caméra, de revolver, de lampe de transmission..."
    icon = "link"

    HINTS: ClassVar[dict[str, str]] = {
        "camera": "Caméra utilisée pour le live et les acquisitions.",
        "xy_stage": "Platine pour les positions multiples et les mosaïques.",
        "focus": "Mise au point (motorisée ou manuelle) pour les piles Z et l'autofocus.",
        "objective_changer": "Ses positions doivent correspondre aux objectifs.",
        "transmitted_light": "Lampe du fond clair (éteinte automatiquement en fluorescence).",
    }

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        card = Card(
            "Rôles des périphériques",
            icon_name="link",
            subtitle="« automatique » : Fuji choisit le premier périphérique compatible.",
        )
        self.form = QFormLayout()
        self.form.setVerticalSpacing(6)
        card.add(layout=self.form)
        self.mm_hint = hint(
            "Avec Micro-Manager, les rôles déclarés dans la configuration "
            "Micro-Manager sont utilisés si vous laissez « automatique »."
        )
        card.add(self.mm_hint)
        self.lay.addWidget(card)
        self.lay.addStretch(1)
        self.combos: dict[str, QComboBox] = {}

    def load(self, data: dict[str, Any]) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.combos = {}
        roles = data.get("roles") or {}
        for role, kind in M.ROLE_KINDS.items():
            combo = QComboBox()
            combo.addItem("(automatique)", None)
            for name in M.devices_of_kind(data, kind):
                combo.addItem(name, name)
            current = roles.get(role)
            if current and combo.findData(current) < 0:
                combo.addItem(current, current)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            self.combos[role] = combo
            self.form.addRow(M.ROLE_TITLES[role], combo)
            self.form.addRow("", hint(self.HINTS[role]))
        self.mm_hint.setVisible(M.has_micromanager(data))

    def store(self, data: dict[str, Any]) -> None:
        data["roles"] = {role: combo.currentData() for role, combo in self.combos.items()}


# ===================================================================== tableaux
class _TablePage(Page):
    """Base des pages à tableau (objectifs, cubes)."""

    COLUMNS: tuple[tuple[str, str], ...] = ()
    KEY = ""

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in self.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.horizontalHeader().setStretchLastSection(True)
        tools = QHBoxLayout()
        tools.setSpacing(4)
        self.add_btn = button("Ajouter", "plus", primary=True)
        self.add_btn.setMenu(self._catalog_menu())
        rm = icon_button("trash", "Supprimer la ligne", object_name="")
        rm.clicked.connect(self._remove_row)
        up = icon_button("up", "Monter", object_name="")
        up.clicked.connect(lambda: self._move(-1))
        down = icon_button("down", "Descendre", object_name="")
        down.clicked.connect(lambda: self._move(1))
        tools.addWidget(self.add_btn)
        for b in (rm, up, down):
            tools.addWidget(b)
        tools.addStretch(1)
        self.tools = tools
        self.lay.addLayout(tools)
        self.lay.addWidget(self.table, 1)

    def _catalog_menu(self) -> QMenu:  # pragma: no cover - surchargé
        raise NotImplementedError

    def _row_widgets(self, row: int) -> list[QWidget]:
        return [self.table.cellWidget(row, c) for c in range(self.table.columnCount())]

    def _remove_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.changed.emit()

    def _move(self, delta: int) -> None:
        row = self.table.currentRow()
        new = row + delta
        if row < 0 or not 0 <= new < self.table.rowCount():
            return
        a, b = self._read_row(row), self._read_row(new)
        self._write_row(row, b)
        self._write_row(new, a)
        self.table.selectRow(new)
        self.changed.emit()

    def _append(self, values: dict[str, Any]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._write_row(row, values)
        self.table.selectRow(row)
        self.changed.emit()

    def _read_row(self, row: int) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def _write_row(self, row: int, values: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    WIDTHS: tuple[int, ...] = ()

    def load(self, data: dict[str, Any]) -> None:
        self.table.setRowCount(0)
        for values in data.get(self.KEY, []):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._write_row(row, values)
        for col, width in enumerate(self.WIDTHS):
            self.table.setColumnWidth(col, width)

    def rows(self) -> list[dict[str, Any]]:
        return [self._read_row(r) for r in range(self.table.rowCount())]


def _line(text: Any = "") -> QLineEdit:
    w = QLineEdit("" if text is None else str(text))
    return w


def _combo(options: tuple[str, ...], value: Any, *, allow_none: bool = False) -> QComboBox:
    w = QComboBox()
    if allow_none:
        w.addItem("—")
    w.addItems(options)
    w.setCurrentText(str(value) if value else ("—" if allow_none else options[0]))
    return w


# ===================================================================== objectifs
class ObjectivesPage(_TablePage):
    title = "Objectifs"
    subtitle = (
        "Dans l'ordre du revolver. La taille du pixel calibrée se remplit avec l'assistant de calibration."
    )
    icon = "objective"
    KEY = "objectives"
    COLUMNS = (
        ("label", "Nom"),
        ("magnification", "Gross."),
        ("na", "ON"),
        ("immersion", "Immersion"),
        ("contrast", "Contraste"),
        ("correction", "Correction"),
        ("phase_ring", "Phase"),
        ("working_distance_mm", "Dist. travail"),
        ("pixel_size_um", "Pixel calibré"),
        ("parfocal_offset_um", "Parfocalité"),
        ("notes", "Notes"),
    )
    WIDTHS = (150, 80, 70, 92, 112, 150, 62, 98, 112, 100, 160)

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        self.lay.insertWidget(
            0,
            hint(
                "Astuce : la gravure « 40/0.65 160/0.17 » signifie grossissement 40, ouverture "
                "numérique 0.65, tube de 160 mm, lamelle de 0.17 mm."
            ),
        )

    def _catalog_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.addAction("Objectif vide").triggered.connect(
            lambda: self._append({"label": "Nouvel objectif", "magnification": 10, "na": 0.25})
        )
        menu.addSection("Modèles courants")
        for o in catalog.OBJECTIVES:
            menu.addAction(o["label"]).triggered.connect(lambda _=False, o=o: self._append(dict(o)))
        return menu

    def _write_row(self, row: int, v: dict[str, Any]) -> None:
        mag = QDoubleSpinBox()
        mag.setRange(0.1, 250)
        mag.setDecimals(1)
        mag.setValue(float(v.get("magnification") or 10))
        na = QDoubleSpinBox()
        na.setRange(0.01, 1.7)
        na.setDecimals(2)
        na.setSingleStep(0.05)
        na.setValue(float(v.get("na") or 0.25))
        widgets = [
            _line(v.get("label", "")),
            mag,
            na,
            _combo(IMMERSIONS, v.get("immersion") or "air"),
            _combo(CONTRASTS, v.get("contrast") or "fond clair"),
            _combo(CORRECTIONS, v.get("correction"), allow_none=True),
            _line(v.get("phase_ring", "")),
            _opt_spin(v.get("working_distance_mm"), 0, 50, decimals=2, suffix="mm"),
            _opt_spin(v.get("pixel_size_um"), 0, 50, decimals=4, suffix="µm", step=0.01),
            _opt_spin(v.get("parfocal_offset_um") or 0, -5000, 5000, decimals=1, suffix="µm", none_text=""),
            _line(v.get("notes", "")),
        ]
        for c, w in enumerate(widgets):
            self.table.setCellWidget(row, c, w)

    def _read_row(self, row: int) -> dict[str, Any]:
        w = self._row_widgets(row)
        corr = w[5].currentText()
        return {
            "label": w[0].text().strip() or f"Objectif {row + 1}",
            "magnification": w[1].value(),
            "na": w[2].value(),
            "immersion": w[3].currentText(),
            "contrast": w[4].currentText(),
            "correction": None if corr == "—" else corr,
            "phase_ring": w[6].text().strip(),
            "working_distance_mm": _spin_value(w[7]),
            "pixel_size_um": _spin_value(w[8]),
            "parfocal_offset_um": w[9].value(),
            "notes": w[10].text().strip(),
        }

    def store(self, data: dict[str, Any]) -> None:
        data["objectives"] = self.rows()


# ======================================================================= cubes
class FiltersPage(_TablePage):
    title = "Cubes de fluorescence"
    subtitle = "Filtres d'excitation, miroir dichroïque, filtre d'émission (en nm)."
    icon = "cube"
    KEY = "filter_sets"
    COLUMNS = (
        ("name", "Nom"),
        ("excitation_nm", "Excitation"),
        ("excitation_bandwidth_nm", "Largeur"),
        ("dichroic_nm", "Dichroïque"),
        ("emission_nm", "Émission"),
        ("emission_bandwidth_nm", "Larg. émission"),
        ("fluorophores", "Fluorophores"),
        ("notes", "Notes"),
    )
    WIDTHS = (150, 96, 86, 96, 96, 118, 170, 220)

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        setup = button("Créer la tourelle de cubes et les canaux", "sparkle")
        setup.setToolTip(
            "Ajoute un périphérique « cubes » (manuel) dont les positions sont ces "
            "cubes, une lampe de fluorescence si besoin, et un canal par cube."
        )
        setup.clicked.connect(self._setup)
        self.tools.addWidget(setup)
        self.lay.insertWidget(
            0,
            hint(
                "Largeur d'émission vide = filtre passe-haut (LP). Les blocs "
                "Leitz Ploemopak (A, I3, N2.1) du Diaplan sont dans les "
                "modèles."
            ),
        )

    def _catalog_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.addAction("Cube vide").triggered.connect(lambda: self._append({"name": "Nouveau cube"}))
        menu.addSection("Modèles courants")
        for f in catalog.FILTER_SETS:
            menu.addAction(f["name"]).triggered.connect(lambda _=False, f=f: self._append(dict(f)))
        return menu

    def _write_row(self, row: int, v: dict[str, Any]) -> None:
        widgets = [
            _line(v.get("name", "")),
            _opt_spin(v.get("excitation_nm"), 0, 1500, decimals=0, suffix="nm"),
            _opt_spin(v.get("excitation_bandwidth_nm"), 0, 500, decimals=0, suffix="nm"),
            _opt_spin(v.get("dichroic_nm"), 0, 1500, decimals=0, suffix="nm"),
            _opt_spin(v.get("emission_nm"), 0, 1500, decimals=0, suffix="nm"),
            _opt_spin(
                v.get("emission_bandwidth_nm"), 0, 500, decimals=0, suffix="nm", none_text="passe-haut"
            ),
            _line(", ".join(v.get("fluorophores") or [])),
            _line(v.get("notes", "")),
        ]
        for c, w in enumerate(widgets):
            self.table.setCellWidget(row, c, w)

    def _read_row(self, row: int) -> dict[str, Any]:
        w = self._row_widgets(row)
        return {
            "name": w[0].text().strip() or f"Cube {row + 1}",
            "excitation_nm": _spin_value(w[1]),
            "excitation_bandwidth_nm": _spin_value(w[2]),
            "dichroic_nm": _spin_value(w[3]),
            "emission_nm": _spin_value(w[4]),
            "emission_bandwidth_nm": _spin_value(w[5]),
            "fluorophores": [x.strip() for x in w[6].text().split(",") if x.strip()],
            "notes": w[7].text().strip(),
        }

    def store(self, data: dict[str, Any]) -> None:
        old = [f["name"] for f in data.get("filter_sets", [])]
        new_rows = self.rows()
        if len(old) == len(new_rows):  # renommages : on suit les références
            for a, b in zip(old, [r["name"] for r in new_rows], strict=True):
                if a != b:
                    M.rename_filter_set(data, a, b)
        data["filter_sets"] = new_rows

    def _setup(self) -> None:
        data = self.editor.data
        self.store(data)
        done = M.setup_fluorescence(data)
        QMessageBox.information(self, "Fluorescence", "\n".join(done) if done else "Tout est déjà en place.")
        self.changed.emit()


# ======================================================================= canaux
class ChannelsPage(Page):
    title = "Canaux"
    subtitle = "Ce qu'il faut régler pour chaque type d'image : cubes, lampes, exposition."
    icon = "channels"

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        row = QHBoxLayout()
        row.setSpacing(12)
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.setMinimumWidth(220)
        self.list.setMaximumWidth(260)
        self.list.currentRowChanged.connect(self._select)
        left.addWidget(self.list, 1)
        tools = QHBoxLayout()
        tools.setSpacing(4)
        add = button("Ajouter", "plus", primary=True)
        add.clicked.connect(self._add)
        dup = icon_button("copy", "Dupliquer", object_name="")
        dup.clicked.connect(self._duplicate)
        rm = icon_button("trash", "Supprimer", object_name="")
        rm.clicked.connect(self._remove)
        up = icon_button("up", "Monter", object_name="")
        up.clicked.connect(lambda: self._move(-1))
        down = icon_button("down", "Descendre", object_name="")
        down.clicked.connect(lambda: self._move(1))
        tools.addWidget(add, 1)
        for b in (dup, rm, up, down):
            tools.addWidget(b)
        left.addLayout(tools)
        row.addLayout(left)
        self.detail_host = QVBoxLayout()
        row.addLayout(self.detail_host, 1)
        self.lay.addLayout(row, 1)
        self._data: dict[str, Any] = {}
        self._detail: QWidget | None = None
        self._collect = None

    def load(self, data: dict[str, Any]) -> None:
        # le formulaire affiché a été enregistré en quittant la page ; le relire maintenant
        # écraserait ce que les autres pages ont changé entre-temps (périphérique renommé...)
        self._collect = None
        self._data = data
        self._refresh(max(0, self.list.currentRow()))

    def _channels(self) -> list[dict[str, Any]]:
        return self._data.setdefault("channels", [])

    def _refresh(self, select: int) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        from ..components import CHANNEL_COLORS

        for ch in self._channels():
            color = CHANNEL_COLORS.get(ch.get("color", "gray"), "#cccccc")
            item = QListWidgetItem(icons.icon("record", color, 14), ch["name"])
            if self._data.get("default_channel") == ch["name"]:
                item.setText(f"{ch['name']}  (au démarrage)")
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.list.count():
            self.list.setCurrentRow(min(select, self.list.count() - 1))
        else:
            from ..components import EmptyState

            self._set_detail(
                EmptyState("channels", "Aucun canal", "Ajoutez « Fond clair », puis vos fluorochromes.")
            )

    def _set_detail(self, w: QWidget) -> None:
        if self._detail is not None:
            self.detail_host.removeWidget(self._detail)
            self._detail.deleteLater()
        self._detail = w
        self.detail_host.addWidget(w)

    def _commit(self) -> None:
        if self._collect is not None:
            self._collect()

    def _add(self) -> None:
        self._commit()
        names = {c["name"] for c in self._channels()}
        name = "Fond clair" if "Fond clair" not in names else "Nouveau canal"
        i = 2
        while name in names:
            name = f"Nouveau canal {i}"
            i += 1
        ch: dict[str, Any] = {"name": name, "kind": "transmission", "exposure_ms": 20.0}
        lamp = (self._data.get("roles") or {}).get("transmitted_light")
        if lamp:
            ch["lights"] = {lamp: {"on": True}}
        self._channels().append(ch)
        if not self._data.get("default_channel"):
            self._data["default_channel"] = name
        self._refresh(len(self._channels()) - 1)
        self.changed.emit()

    def _duplicate(self) -> None:
        self._commit()
        row = self.list.currentRow()
        if row < 0:
            return
        ch = copy.deepcopy(self._channels()[row])
        ch["name"] += " (copie)"
        self._channels().insert(row + 1, ch)
        self._refresh(row + 1)
        self.changed.emit()

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        self._collect = None
        M.remove_channel(self._data, self._channels()[row]["name"])
        self._refresh(row)
        self.changed.emit()

    def _move(self, delta: int) -> None:
        self._commit()
        row = self.list.currentRow()
        chans = self._channels()
        new = row + delta
        if row < 0 or not 0 <= new < len(chans):
            return
        chans[row], chans[new] = chans[new], chans[row]
        self._refresh(new)
        self.changed.emit()

    def _select(self, row: int) -> None:
        if self._collect is not None and self._detail is not None:
            self._collect()
        chans = self._channels()
        if not 0 <= row < len(chans):
            return
        ch = chans[row]
        data = self._data
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        card = Card("Canal", icon_name="channels")
        form = QFormLayout()
        name = QLineEdit(ch["name"])
        kind = _combo(("transmission", "fluorescence", "réflexion"), ch.get("kind", "transmission"))
        sets = [f["name"] for f in data.get("filter_sets", [])]
        fset = _combo(tuple(sets), ch.get("filter_set"), allow_none=True) if sets else None
        exposure = _opt_spin(
            ch.get("exposure_ms"), 0, 600_000, decimals=1, suffix="ms", none_text="inchangée"
        )
        gain = _opt_spin(ch.get("gain"), -1, 100_000, decimals=1, none_text="inchangé")
        color = ColorChooser(ch.get("color", "gray"))
        default = QCheckBox("Canal sélectionné au démarrage")
        default.setChecked(data.get("default_channel") == ch["name"])
        form.addRow("Nom", name)
        form.addRow("Type", kind)
        if fset is not None:
            form.addRow("Cube", fset)
        form.addRow("Exposition", exposure)
        form.addRow("Gain", gain)
        form.addRow("Couleur d'affichage", color)
        form.addRow("", default)
        card.add(layout=form)
        v.addWidget(card)

        # longueurs d'onde
        wl = Card("Longueurs d'onde", icon_name="bulb")
        wform = QFormLayout()
        ex = _opt_spin(ch.get("excitation_nm"), 0, 1500, decimals=0, suffix="nm")
        em = _opt_spin(ch.get("emission_nm"), 0, 1500, decimals=0, suffix="nm")
        wform.addRow("Excitation", ex)
        wform.addRow("Émission", em)
        wl.add(layout=wform)
        v.addWidget(wl)

        # chemin optique
        path = Card(
            "Chemin optique",
            icon_name="turret",
            subtitle="Positions à prendre et sources à allumer pour ce canal.",
        )
        pform = QFormLayout()
        state_combos: dict[str, QComboBox] = {}
        for dev in M.devices_of_kind(data, "state"):
            if dev == (data.get("roles") or {}).get("objective_changer"):
                continue
            combo = QComboBox()
            combo.setEditable(not M.state_labels(data, dev))
            combo.addItem(UNCHANGED)
            combo.addItems(M.state_labels(data, dev))
            current = (ch.get("states") or {}).get(dev)
            if current and combo.findText(current) < 0:
                combo.addItem(current)
            combo.setCurrentText(current or UNCHANGED)
            state_combos[dev] = combo
            pform.addRow(dev, combo)
        light_widgets: dict[str, tuple[QComboBox, QDoubleSpinBox]] = {}
        for dev in M.devices_of_kind(data, "light_source"):
            setting = (ch.get("lights") or {}).get(dev)
            mode = QComboBox()
            mode.addItems([UNCHANGED, "allumée", "éteinte"])
            mode.setCurrentText(
                UNCHANGED if setting is None else ("allumée" if setting.get("on", True) else "éteinte")
            )
            inten = _opt_spin(
                None if setting is None else setting.get("intensity"),
                -1,
                100,
                decimals=0,
                suffix="%",
                none_text="intensité inchangée",
            )
            light_widgets[dev] = (mode, inten)
            row = QHBoxLayout()
            row.addWidget(mode, 1)
            row.addWidget(inten, 1)
            pform.addRow(dev, row)
        shutter_boxes: dict[str, QCheckBox] = {}
        for dev in M.devices_of_kind(data, "shutter"):
            cb = QCheckBox("ouvrir pendant l'exposition")
            cb.setChecked(dev in (ch.get("shutters") or []))
            shutter_boxes[dev] = cb
            pform.addRow(dev, cb)
        if not (state_combos or light_widgets or shutter_boxes):
            pform.addRow(hint("Aucune tourelle, lampe ou obturateur déclaré (onglet Périphériques)."))
        path.add(layout=pform)
        v.addWidget(path)

        mm_group = mm_preset = None
        if M.has_micromanager(data):
            mm = Card(
                "Micro-Manager", icon_name="chip", subtitle="Préréglage Micro-Manager appliqué pour ce canal."
            )
            mform = QFormLayout()
            cfg = ch.get("mm_config") or ("", "")
            mm_group, mm_preset = QLineEdit(cfg[0]), QLineEdit(cfg[1])
            mm_group.setPlaceholderText("Channel")
            mform.addRow("Groupe", mm_group)
            mform.addRow("Préréglage", mm_preset)
            mm.add(layout=mform)
            v.addWidget(mm)
        v.addStretch(1)

        def on_set(text: str) -> None:
            fs = next((f for f in data.get("filter_sets", []) if f["name"] == text), None)
            if fs is None:
                return
            kind.setCurrentText("fluorescence")
            if fs.get("excitation_nm"):
                ex.setValue(float(fs["excitation_nm"]))
            if fs.get("emission_nm"):
                em.setValue(float(fs["emission_nm"]))
                from fuji.core.catalog import color_for_wavelength

                color.set_value(color_for_wavelength(fs["emission_nm"]))
            for combo in state_combos.values():
                if combo.findText(text) >= 0:
                    combo.setCurrentText(text)

        if fset is not None:
            fset.currentTextChanged.connect(on_set)

        def collect() -> None:
            old_name = ch["name"]
            new_name = name.text().strip() or old_name
            if new_name != old_name and any(c["name"] == new_name for c in self._channels()):
                new_name = old_name
            M.rename_channel(data, old_name, new_name)
            ch["kind"] = kind.currentText()
            ch["filter_set"] = None if fset is None or fset.currentText() == "—" else fset.currentText()
            ch["exposure_ms"] = _spin_value(exposure)
            ch["gain"] = _spin_value(gain)
            ch["color"] = color.value()
            ch["excitation_nm"] = _spin_value(ex)
            ch["emission_nm"] = _spin_value(em)
            ch["states"] = {
                d: c.currentText() for d, c in state_combos.items() if c.currentText() not in (UNCHANGED, "")
            }
            lights = {}
            for dev, (mode, inten) in light_widgets.items():
                if mode.currentText() == UNCHANGED:
                    continue
                lights[dev] = {"on": mode.currentText() == "allumée", "intensity": _spin_value(inten)}
            ch["lights"] = lights
            ch["shutters"] = [d for d, cb in shutter_boxes.items() if cb.isChecked()]
            if mm_group is not None and mm_preset is not None:
                g, p = mm_group.text().strip(), mm_preset.text().strip()
                ch["mm_config"] = [g, p] if g and p else None
            if default.isChecked():
                data["default_channel"] = ch["name"]
            elif data.get("default_channel") == ch["name"]:
                data["default_channel"] = None
            item = self.list.item(self._channels().index(ch))
            if item is not None:
                item.setText(
                    ch["name"] + ("  (au démarrage)" if data.get("default_channel") == ch["name"] else "")
                )

        self._collect = collect
        for w in (name,):
            w.editingFinished.connect(collect)
        self._set_detail(host)

    def store(self, data: dict[str, Any]) -> None:
        self._commit()


# ======================================================================= YAML
class YamlPage(Page):
    title = "Fichier YAML"
    subtitle = "Le profil complet, tel qu'il sera enregistré. Pour les réglages très avancés."
    icon = "code"

    def __init__(self, editor: ConfigEditor) -> None:
        super().__init__(editor)
        self.text = QPlainTextEdit()
        f = self.text.font()
        f.setFamily("Menlo")
        f.setStyleHint(f.StyleHint.Monospace)
        self.text.setFont(f)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.status = label("", "hint")
        self.lay.addWidget(self.text, 1)
        self.lay.addWidget(self.status)
        self._dirty = False
        self.text.textChanged.connect(lambda: setattr(self, "_dirty", True))

    def load(self, data: dict[str, Any]) -> None:
        try:
            text = MicroscopeProfile.model_validate(data).to_yaml()
            self.status.setText("Les modifications faites ici sont appliquées en quittant la page.")
        except Exception as exc:
            text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
            self.status.setText(f"Profil incomplet : {exc}")
        self.text.setPlainText(text)
        self._dirty = False

    def store(self, data: dict[str, Any]) -> None:
        if not self._dirty:
            return
        parsed = load_yaml(self.text.toPlainText()) or {}
        normalized = MicroscopeProfile.model_validate(parsed).model_dump(mode="json")
        data.clear()
        data.update(normalized)
        self._dirty = False


PAGES = (
    GeneralPage,
    InstrumentPage,
    DevicesPage,
    RolesPage,
    ObjectivesPage,
    FiltersPage,
    ChannelsPage,
    YamlPage,
)

__all__ = ["PAGES", "Page"]
