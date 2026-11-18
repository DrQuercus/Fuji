"""Tableau de résultats (la fenêtre « Results » d'ImageJ) et ses réglages.

Au-dessus du tableau : les colonnes à montrer, le nombre de décimales, la virgule décimale
(pour Excel en français), les statistiques de chaque colonne, et un graphique — celui du
tableau (profil, histogramme) ou la distribution d'une colonne au choix. Un clic sur un
titre de colonne trie le tableau. Les réglages sont retenus pour les tableaux suivants.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QLocale, Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fuji.analysis.image import Plot, Table

from .flow import FlowLayout
from .prefs import settings
from .widgets import PlotWidget, _fmt, hint_label

#: Au-delà, seules les premières lignes sont affichées (l'export contient tout).
MAX_ROWS_SHOWN = 2000
#: statistiques d'une colonne : (nom affiché, calcul sur les valeurs finies)
STATISTICS = (
    ("Moyenne", lambda v: float(np.mean(v)) if v.size else np.nan),
    ("Écart type", lambda v: float(np.std(v, ddof=1)) if v.size > 1 else np.nan),
    ("Médiane", lambda v: float(np.median(v)) if v.size else np.nan),
    ("Minimum", lambda v: float(np.min(v)) if v.size else np.nan),
    ("Maximum", lambda v: float(np.max(v)) if v.size else np.nan),
    ("Somme", lambda v: float(np.sum(v)) if v.size else np.nan),
    ("Nombre", lambda v: int(v.size)),
)
#: choix du nombre de décimales (−1 : au mieux pour chaque valeur)
DECIMALS = (-1, 0, 1, 2, 3, 4, 6)
#: colonne ajoutée quand on cumule les mesures de plusieurs images
SOURCE_COLUMN = "Image"
_ROW = 24  # hauteur d'une ligne des tableaux (pixels)
_DISTRIBUTION = "distribution:"  # début du choix « distribution d'une colonne » du graphique
_SEPARATOR = " — "  # entre le genre d'un tableau et le nom de son image : « Mesure — lame 3 »


# ------------------------------------------------------------------ calculs
def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Les colonnes de nombres (celles que l'on peut résumer ou tracer)."""
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne, écart type (n − 1, comme ImageJ), médiane, min, max, somme et nombre de
    valeurs de chaque colonne de nombres. Les cases vides ne comptent pas."""
    columns = numeric_columns(df)
    out = pd.DataFrame(index=[name for name, _ in STATISTICS], columns=columns, dtype=object)
    for c in columns:
        values = _finite(df[c])
        for name, compute in STATISTICS:
            out.loc[name, c] = compute(values)
    return out


def distribution(table: Table, column: str, bins: int = 0) -> Table:
    """Histogramme des valeurs d'une colonne (``bins`` à 0 : nombre de classes choisi d'après
    le nombre de valeurs), sous la forme d'un tableau avec son graphique en barres."""
    values = _finite(table.data[column])
    if not values.size:
        raise ValueError(f"La colonne « {column} » ne contient aucun nombre")
    n = int(bins) if bins > 0 else int(np.clip(np.ceil(np.sqrt(values.size)), 5, 50))
    counts, edges = np.histogram(values, bins=n)
    unit = table.units.get(column, "")
    label = f"{column} ({unit})" if unit else column
    df = pd.DataFrame({column: (edges[:-1] + edges[1:]) / 2, "Nombre": counts})
    return Table(
        df,
        name=f"Distribution{_SEPARATOR}{column}",
        units={column: unit} if unit else {},
        plot=Plot(column, ("Nombre",), "barres", label, "Nombre"),
    )


def format_value(value: Any, decimals: int = -1, comma: bool = False) -> str:
    """Texte d'une case : ``decimals`` chiffres après la virgule (−1 : au mieux)."""
    if isinstance(value, (bool, np.bool_)):
        return "oui" if value else "non"
    if isinstance(value, (float, np.floating)):
        if value != value:  # case vide
            return ""
        text = _fmt(value) if decimals < 0 else f"{float(value):.{decimals}f}"
        return text.replace(".", ",") if comma else text
    return str(value)


def kind_of(table: Table) -> str:
    """Le genre d'un tableau : « Mesure », « Particules », « ROI »…"""
    return table.name.split(_SEPARATOR)[0]


def accumulate(old: Table, new: Table) -> Table | None:
    """Ajoute les lignes de ``new`` à la suite de ``old`` (mesures d'une série d'images).

    Possible quand les deux tableaux sont du même genre, avec les mêmes colonnes, et sans
    graphique ; sinon ``None``. Une colonne « Image » dit d'où vient chaque ligne."""
    if old.plot is not None or new.plot is not None or kind_of(old) != kind_of(new):
        return None

    def columns(t: Table) -> list[str]:
        return [str(c) for c in t.data.columns if c != SOURCE_COLUMN]

    if columns(old) != columns(new):
        return None

    def with_source(t: Table) -> pd.DataFrame:
        if SOURCE_COLUMN in t.data.columns:
            return t.data
        df = t.data.copy()
        df.insert(0, SOURCE_COLUMN, t.name.split(_SEPARATOR, 1)[-1])
        return df

    df = pd.concat([with_source(old), with_source(new)], ignore_index=True)
    df.index = pd.RangeIndex(1, len(df) + 1)
    return Table(
        df,
        name=f"{kind_of(new)}{_SEPARATOR}cumul",
        units={**old.units, **new.units},
        summary={"Lignes": len(df), "Images": int(df[SOURCE_COLUMN].nunique())},
    )


def default_comma() -> bool:
    """Virgule décimale si l'ordinateur est réglé ainsi (Excel en français l'attend)."""
    return QLocale().decimalPoint() == ","


# ------------------------------------------------------------------ interface
class _Cell(QTableWidgetItem):
    """Case triée d'après sa valeur, et non d'après son texte (« 10 » après « 9 »)."""

    def __init__(self, text: str, value: Any) -> None:
        super().__init__(text)
        self.value = value
        self.label = ""  # le numéro de sa ligne dans le tableau d'origine

    def __lt__(self, other: QTableWidgetItem) -> bool:
        a, b = self.value, getattr(other, "value", None)
        try:
            if a != a or a is None:  # les cases vides vont à la fin
                return False
            if b != b or b is None:
                return True
            return bool(a < b)
        except TypeError:
            return str(a) < str(b)


def _pref(key: str, default: Any) -> Any:
    try:
        return settings().value(key, default, type=type(default))
    except (TypeError, ValueError):
        return default


def hidden_columns() -> set[str]:
    """Les colonnes que l'on a choisi de ne pas montrer (par leur nom, pour tous les tableaux)."""
    try:
        return {str(c) for c in json.loads(_pref("results/hidden_columns", "[]"))}
    except (TypeError, ValueError):
        return set()


class ResultsWidget(QWidget):
    """Un tableau de résultats, son graphique et leurs réglages."""

    def __init__(self, table: Table, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = table
        self.plot: PlotWidget | None = None
        self._shown_plot: Table | None = None
        self._filling = False
        lay = QVBoxLayout(self)
        lay.setSpacing(6)
        self.title = QLabel()
        self.summary = hint_label()
        lay.addWidget(self.title)
        lay.addWidget(self.summary)

        # ---- réglages ----------------------------------------------------------------
        bar = QWidget()
        tools = FlowLayout(bar, spacing=8, line_spacing=6)
        tools.setContentsMargins(0, 0, 0, 0)
        self.columns_btn = QPushButton("Colonnes")
        self.columns_btn.setToolTip(
            "Choisir les colonnes montrées, copiées et exportées (retenu pour les prochains tableaux)"
        )
        self.columns_menu = QMenu(self.columns_btn)
        self.columns_btn.setMenu(self.columns_menu)
        self.decimals = QComboBox()
        for d in DECIMALS:
            self.decimals.addItem("Décimales : auto" if d < 0 else f"Décimales : {d}", d)
        self.decimals.setCurrentIndex(max(0, self.decimals.findData(_pref("results/decimals", -1))))
        self.decimals.setToolTip("Chiffres après la virgule (l'export garde toute la précision)")
        self.decimals.currentIndexChanged.connect(self._format_changed)
        self.comma = QCheckBox("Virgule décimale")
        self.comma.setToolTip(
            "12,5 au lieu de 12.5 : pour coller ou ouvrir dans Excel en français "
            "(le fichier CSV sépare alors les colonnes par des points-virgules)"
        )
        self.comma.setChecked(_pref("results/decimal_comma", default_comma()))
        self.comma.toggled.connect(self._format_changed)
        self.stats_box = QCheckBox("Statistiques")
        self.stats_box.setToolTip("Moyenne, écart type, médiane, min, max, somme de chaque colonne")
        self.stats_box.setChecked(_pref("results/statistics", False))
        self.stats_box.toggled.connect(self._stats_toggled)
        self.graph = QComboBox()
        self.graph.setToolTip("Graphique : celui du tableau, ou la distribution (histogramme) d'une colonne")
        self.graph.currentIndexChanged.connect(self._graph_changed)
        self.bins = QSpinBox()
        self.bins.setRange(0, 200)
        self.bins.setSpecialValueText("classes : auto")
        self.bins.setPrefix("classes : ")
        self.bins.setToolTip("Nombre de barres de la distribution (auto : d'après le nombre de valeurs)")
        self.bins.setKeyboardTracking(False)
        self.bins.valueChanged.connect(self._graph_changed)
        self.copy_btn = QPushButton("Copier (pour Excel)")
        self.copy_btn.clicked.connect(self._copy)
        self.export_btn = QPushButton("Exporter en CSV…")
        self.export_btn.clicked.connect(self._export)
        self.save_plot_btn = QPushButton("Enregistrer le graphique…")
        self.save_plot_btn.clicked.connect(self._save_plot)
        for w in (
            self.columns_btn,
            self.decimals,
            self.comma,
            self.stats_box,
            self.graph,
            self.bins,
            self.copy_btn,
            self.export_btn,
            self.save_plot_btn,
        ):
            tools.addWidget(w)
        lay.addWidget(bar)
        self.note = hint_label()
        lay.addWidget(self.note)

        # ---- graphique | tableau + statistiques ----------------------------------------
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.plot_box = QWidget()
        self.plot_lay = QVBoxLayout(self.plot_box)
        self.plot_lay.setContentsMargins(0, 0, 0, 0)
        self.plot_note = hint_label()
        self.plot_lay.addWidget(self.plot_note)
        right = QSplitter(Qt.Orientation.Vertical)  # tableau au-dessus, statistiques en dessous
        right.setChildrenCollapsible(False)
        self.grid = QTableWidget()
        self.grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.grid.horizontalHeader().setToolTip("Un clic sur un titre trie le tableau d'après cette colonne")
        self.stats_grid = QTableWidget()
        self.stats_grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stats_grid.horizontalHeader().hide()
        self.stats_grid.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.stats_grid.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for g in (self.grid, self.stats_grid):  # lignes serrées : plus de mesures à l'écran
            g.verticalHeader().setMinimumSectionSize(_ROW)
            g.verticalHeader().setDefaultSectionSize(_ROW)
        # les deux tableaux gardent leurs colonnes alignées
        self.grid.horizontalHeader().sectionResized.connect(
            lambda i, _old, new: self.stats_grid.setColumnWidth(i, new)
        )
        self.grid.horizontalScrollBar().valueChanged.connect(self.stats_grid.horizontalScrollBar().setValue)
        self.grid.model().layoutChanged.connect(self._relabel)  # après un tri
        right.addWidget(self.grid)
        right.addWidget(self.stats_grid)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        self.grid.setMinimumHeight(3 * _ROW + 40)
        self.stats_grid.setMinimumHeight(2 * _ROW)
        self.splitter.addWidget(self.plot_box)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        lay.addWidget(self.splitter, 1)
        self.set_table(table)

    # ------------------------------------------------------------------ contenu
    def set_table(self, table: Table) -> None:
        """Montre un autre tableau (ou le même, complété) avec les réglages en cours."""
        self.table = table
        df = table.data
        self.title.setText(f"<b>{table.name}</b> — {len(df)} ligne(s)")
        summary = "  ·  ".join(f"{k} : {_fmt(v)}" for k, v in table.summary.items())
        self.summary.setText(summary)
        self.summary.setVisible(bool(summary))
        shown = min(len(df), MAX_ROWS_SHOWN)
        self.note.setText(
            f"{shown} premières lignes affichées ; l'export les contient toutes." if shown < len(df) else ""
        )
        self.note.setVisible(shown < len(df))
        self._build_columns_menu()
        self._build_graph_choices()
        self._fill()
        self._fill_statistics()
        self._apply_hidden()
        self._graph_changed()

    def columns(self) -> list[str]:
        """Les colonnes montrées (donc copiées et exportées), dans l'ordre du tableau."""
        hidden = hidden_columns()
        return [str(c) for c in self.table.data.columns if str(c) not in hidden]

    def _header(self, column: Any) -> str:
        unit = self.table.units.get(column)
        return f"{column}\n({unit})" if unit else str(column)

    def _text(self, value: Any) -> str:
        return format_value(value, int(self.decimals.currentData()), self.comma.isChecked())

    def _fill(self) -> None:
        df = self.table.data.iloc[:MAX_ROWS_SHOWN]
        grid = self.grid
        self._filling = True
        grid.setSortingEnabled(False)
        grid.clear()
        grid.setRowCount(len(df))
        grid.setColumnCount(len(df.columns))
        grid.setHorizontalHeaderLabels([self._header(c) for c in df.columns])
        grid.setVerticalHeaderLabels([str(i) for i in df.index])
        align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for c, name in enumerate(df.columns):
            for r, v in enumerate(df[name].tolist()):
                item = _Cell(self._text(v), v)
                item.label = str(df.index[r])
                item.setTextAlignment(align)
                grid.setItem(r, c, item)
        grid.resizeColumnsToContents()
        grid.setSortingEnabled(True)
        grid.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self._filling = False

    def _relabel(self, *_: Any) -> None:
        """Après un tri, chaque ligne garde son numéro d'origine (celui de l'objet mesuré)."""
        if self._filling or not self.grid.columnCount():
            return
        cells = [self.grid.item(r, 0) for r in range(self.grid.rowCount())]
        self.grid.setVerticalHeaderLabels([getattr(c, "label", "") for c in cells])

    def _fill_statistics(self) -> None:
        df = self.table.data
        stats = statistics(df)
        grid = self.stats_grid
        grid.clear()
        grid.setRowCount(len(stats.index))
        grid.setColumnCount(len(df.columns))
        grid.setVerticalHeaderLabels([str(i) for i in stats.index])
        align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for c, name in enumerate(df.columns):
            for r, stat in enumerate(stats.index):
                text = self._text(stats.loc[stat, str(name)]) if str(name) in stats.columns else ""
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                grid.setItem(r, c, item)
        for c in range(grid.columnCount()):  # une colonne assez large pour ses statistiques aussi
            self.grid.setColumnWidth(c, max(self.grid.columnWidth(c), grid.sizeHintForColumn(c) + 8))
            grid.setColumnWidth(c, self.grid.columnWidth(c))
        # même largeur pour les titres de lignes : les colonnes des deux tableaux s'alignent
        width = max(self.grid.verticalHeader().sizeHint().width(), grid.verticalHeader().sizeHint().width())
        self.grid.verticalHeader().setFixedWidth(width)
        grid.verticalHeader().setFixedWidth(width)
        rows = sum(grid.rowHeight(r) for r in range(grid.rowCount()))
        grid.setMaximumHeight(rows + 2 * grid.frameWidth())
        grid.setVisible(self.stats_box.isChecked() and bool(len(stats.columns)))

    def statistics(self) -> pd.DataFrame:
        """Les statistiques des colonnes montrées."""
        stats = statistics(self.table.data)
        return stats[[c for c in self.columns() if c in stats.columns]]

    # ------------------------------------------------------------------ colonnes
    def _build_columns_menu(self) -> None:
        menu = self.columns_menu
        menu.clear()
        hidden = hidden_columns()
        self.column_actions = {}
        for c in self.table.data.columns:
            act = menu.addAction(str(c))
            act.setCheckable(True)
            act.setChecked(str(c) not in hidden)
            act.toggled.connect(lambda on, name=str(c): self.set_column_visible(name, on))
            self.column_actions[str(c)] = act
        menu.addSeparator()
        menu.addAction("Tout montrer", self.show_all_columns)

    def set_column_visible(self, name: str, visible: bool) -> None:
        hidden = hidden_columns()
        (hidden.discard if visible else hidden.add)(name)
        settings().setValue("results/hidden_columns", json.dumps(sorted(hidden), ensure_ascii=False))
        act = self.column_actions.get(name)
        if act is not None and act.isChecked() != visible:
            act.blockSignals(True)
            act.setChecked(visible)
            act.blockSignals(False)
        self._apply_hidden()

    def show_all_columns(self) -> None:
        for name in list(self.column_actions):
            self.set_column_visible(name, True)

    def _apply_hidden(self) -> None:
        hidden = hidden_columns()
        count = 0
        for c, name in enumerate(self.table.data.columns):
            hide = str(name) in hidden
            count += hide
            self.grid.setColumnHidden(c, hide)
            self.stats_grid.setColumnHidden(c, hide)
        self.columns_btn.setText(
            f"Colonnes ({count} cachée{'s' if count > 1 else ''})" if count else "Colonnes"
        )

    # ------------------------------------------------------------------ réglages
    def _format_changed(self, *_: Any) -> None:
        s = settings()
        s.setValue("results/decimals", int(self.decimals.currentData()))
        s.setValue("results/decimal_comma", self.comma.isChecked())
        widths = [self.grid.columnWidth(c) for c in range(self.grid.columnCount())]
        self._fill()
        for c, w in enumerate(widths):  # les colonnes élargies à la main le restent
            self.grid.setColumnWidth(c, max(w, self.grid.columnWidth(c)))
        self._fill_statistics()
        self._apply_hidden()

    def _stats_toggled(self, on: bool) -> None:
        settings().setValue("results/statistics", bool(on))
        self.stats_grid.setVisible(on and bool(numeric_columns(self.table.data)))

    # ------------------------------------------------------------------ graphique
    def _build_graph_choices(self) -> None:
        combo = self.graph
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Graphique : aucun", "aucun")
        if self.table.plot is not None and len(self.table.data):
            combo.addItem("Graphique du tableau", "tableau")
        for c in numeric_columns(self.table.data):
            combo.addItem(f"Distribution : {c}", _DISTRIBUTION + c)
        index = combo.findData(previous) if previous is not None else -1
        if index < 0:
            index = 1 if self.table.plot is not None and len(self.table.data) else 0
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def show_distribution(self, column: str, bins: int = 0) -> None:
        """Montre la distribution d'une colonne."""
        index = self.graph.findData(_DISTRIBUTION + column)
        if index < 0:
            raise KeyError(column)
        self.bins.setValue(bins)
        self.graph.setCurrentIndex(index)

    def _graph_changed(self, *_: Any) -> None:
        choice = str(self.graph.currentData() or "aucun")
        kind, _, column = choice.partition(":")
        shown: Table | None = None
        note = ""
        if kind == "tableau":
            shown = self.table
        elif kind == "distribution":
            try:
                shown = distribution(self.table, column, self.bins.value())
                values = _finite(self.table.data[column])
                spread = f" ± {_fmt(float(np.std(values, ddof=1)))}" if values.size > 1 else ""
                note = f"{column} : moyenne {_fmt(float(np.mean(values)))}{spread} (n = {values.size})"
            except ValueError as exc:
                note = str(exc)
        if self.plot is not None:
            self.plot_lay.removeWidget(self.plot)
            self.plot.deleteLater()
            self.plot = None
        self._shown_plot = shown
        if shown is not None:
            self.plot = PlotWidget(shown)
            self.plot_lay.insertWidget(0, self.plot, 1)
        self.plot_note.setText(note)
        self.plot_note.setVisible(bool(note))
        self.plot_box.setVisible(shown is not None or bool(note))
        self.bins.setVisible(kind == "distribution")
        self.save_plot_btn.setVisible(shown is not None)

    def _save_plot(self) -> None:
        if self.plot is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le graphique", "graphique.png", "PNG (*.png)"
        )
        if path:
            self.plot.grab().save(path)

    # ------------------------------------------------------------------ sortie
    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exporter les résultats", "resultats.csv", "CSV (*.csv)")
        if path:
            self.export(path)

    def export(self, path: str) -> None:
        """Écrit les colonnes montrées dans un fichier CSV (toutes les lignes)."""
        self.table.to_csv(path, columns=self.columns(), decimal="," if self.comma.isChecked() else ".")

    def clipboard_text(self) -> str:
        """Les colonnes montrées, séparées par des tabulations (se colle dans un tableur)."""
        frame = self.table.frame(self.columns())
        return frame.to_csv(sep="\t", index_label=" ", decimal="," if self.comma.isChecked() else ".")

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.clipboard_text())


__all__ = [
    "MAX_ROWS_SHOWN",
    "ResultsWidget",
    "accumulate",
    "distribution",
    "format_value",
    "hidden_columns",
    "kind_of",
    "numeric_columns",
    "statistics",
]
