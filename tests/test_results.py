"""Tableau de résultats : colonnes, décimales, tri, statistiques, distribution, cumul."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from fuji.analysis.image import Plot, Table


def _table(name="Particules — lame 1", n=30, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Area": rng.normal(50, 10, n).round(3),
            "Mean": rng.normal(120, 5, n),
            "Type": ["rond"] * n,
            "Count": np.arange(n) * 3 % 11,
        },
        index=pd.RangeIndex(1, n + 1),
    )
    if n >= 3:
        df.loc[3, "Mean"] = np.nan  # une case vide
    return Table(df, name=name, units={"Area": "µm²"}, summary={"Count": n})


# ------------------------------------------------------------------ calculs (sans écran)
def test_statistics_distribution_and_formatting():
    pytest.importorskip("qtpy")
    from fuji.ui.results import distribution, format_value, numeric_columns, statistics

    table = _table()
    assert numeric_columns(table.data) == ["Area", "Mean", "Count"]  # pas la colonne de texte
    stats = statistics(table.data)
    area = table.data["Area"].to_numpy()
    assert stats.loc["Moyenne", "Area"] == pytest.approx(area.mean())
    assert stats.loc["Écart type", "Area"] == pytest.approx(area.std(ddof=1))  # n − 1, comme ImageJ
    assert stats.loc["Somme", "Area"] == pytest.approx(area.sum())
    assert stats.loc["Nombre", "Mean"] == 29 and stats.loc["Nombre", "Area"] == 30  # case vide écartée
    assert stats.loc["Médiane", "Count"] == np.median(table.data["Count"])

    hist = distribution(table, "Area")
    assert hist.plot == Plot("Area", ("Nombre",), "barres", "Area (µm²)", "Nombre")
    assert hist.data["Nombre"].sum() == 30 and len(hist.data) == 6  # √30 → 6 classes
    assert len(distribution(table, "Area", bins=12).data) == 12
    assert distribution(table, "Mean").data["Nombre"].sum() == 29
    with pytest.raises(ValueError, match="aucun nombre"):
        distribution(Table(pd.DataFrame({"A": [np.nan, np.nan]})), "A")

    assert format_value(12.3456) == "12.3456" and format_value(12.3456, 2) == "12.35"
    assert format_value(12.3456, 1, comma=True) == "12,3" and format_value(7, 2) == "7"
    assert format_value(float("nan"), 2) == "" and format_value("rond", 2, True) == "rond"
    assert format_value(np.float64(1234.5), 0) == "1234" or format_value(np.float64(1234.5), 0) == "1235"


def test_accumulate_appends_measurements_of_a_series():
    pytest.importorskip("qtpy")
    from fuji.ui.results import accumulate

    first, second = _table("Particules — lame 1", 4), _table("Particules — lame 2", 3, seed=1)
    merged = accumulate(first, second)
    assert merged.name == "Particules — cumul" and len(merged) == 7
    assert list(merged.data.columns) == ["Image", "Area", "Mean", "Type", "Count"]
    assert merged.data["Image"].tolist() == ["lame 1"] * 4 + ["lame 2"] * 3
    assert merged.data.index.tolist() == list(range(1, 8)) and merged.units == {"Area": "µm²"}
    assert merged.summary == {"Lignes": 7, "Images": 2}
    again = accumulate(merged, _table("Particules — lame 3", 2, seed=2))  # et ainsi de suite
    assert len(again) == 9 and again.data["Image"].nunique() == 3
    # pas le même genre, pas les mêmes colonnes, ou un graphique : chacun son onglet
    assert accumulate(first, _table("Mesure — lame 2", 3)) is None
    other = _table("Particules — lame 4", 3)
    other.data = other.data.drop(columns="Mean")
    assert accumulate(first, other) is None
    plotted = _table("Particules — lame 5", 3)
    plotted.plot = Plot("Area", ("Mean",))
    assert accumulate(first, plotted) is None


def test_csv_export_with_chosen_columns_and_decimal_comma(tmp_path):
    table = _table(n=3)
    plain = table.to_csv(tmp_path / "a.csv").read_text(encoding="utf-8-sig").splitlines()
    assert plain[0] == " ,Area (µm²),Mean,Type,Count"
    french = table.to_csv(tmp_path / "b.csv", columns=["Count", "Area"], decimal=",")
    lines = french.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == " ;Area (µm²);Count"  # l'ordre du tableau, des points-virgules
    area = f"{table.data['Area'].iloc[0]}".replace(".", ",")
    assert lines[1] == f"1;{area};0"


# ------------------------------------------------------------------ interface
gui = pytest.mark.gui


@pytest.fixture
def widget(qtbot, tmp_path):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("pas d'affichage (lancer avec xvfb-run)")
    from fuji.ui.app import isolate_settings
    from fuji.ui.prefs import settings
    from fuji.ui.results import ResultsWidget

    isolate_settings(str(tmp_path))
    settings().setValue("results/decimal_comma", False)  # d'origine : d'après la langue du système
    w = ResultsWidget(_table())
    qtbot.addWidget(w)
    return w


def _column(widget, name):
    index = list(widget.table.data.columns).index(name)
    return [widget.grid.item(r, index).text() for r in range(widget.grid.rowCount())]


@gui
def test_decimals_comma_and_sorting(widget):

    from fuji.ui.prefs import settings

    assert "30 lignes" in widget.title.text() and widget.summary.text() == "Count : 30"
    first = widget.table.data["Area"].iloc[0]
    widget.decimals.setCurrentIndex(widget.decimals.findData(1))
    assert _column(widget, "Area")[0] == f"{first:.1f}" and _column(widget, "Type")[0] == "rond"
    widget.comma.setChecked(True)
    assert _column(widget, "Area")[0] == f"{first:.1f}".replace(".", ",")
    assert _column(widget, "Mean")[2] == ""  # la case vide le reste
    assert settings().value("results/decimals", type=int) == 1
    assert settings().value("results/decimal_comma", type=bool) is True

    # tri d'après les valeurs (et non le texte) ; les numéros de ligne suivent
    widget.sort_by("Count", descending=True)
    values = [int(v) for v in _column(widget, "Count")]
    assert values == sorted(values, reverse=True) and values[0] == 10
    # chaque ligne garde son numéro d'origine : celui de l'objet mesuré
    best = widget.table.data["Count"].idxmax()
    assert widget.grid.verticalHeaderItem(0).text() == str(best)
    row = widget.table.data.loc[int(widget.grid.verticalHeaderItem(5).text())]
    assert _column(widget, "Count")[5] == str(row["Count"])
    # … y compris après un changement de décimales (les numéros repartaient de 1, 2, 3…)
    widget.decimals.setCurrentIndex(widget.decimals.findData(2))
    assert widget.grid.verticalHeaderItem(0).text() == str(best)
    row = widget.table.data.loc[int(widget.grid.verticalHeaderItem(5).text())]
    assert _column(widget, "Count")[5] == str(row["Count"])
    # un clic sur le titre : croissant, puis décroissant, puis l'ordre d'origine
    mean = list(widget.table.data.columns).index("Mean")
    widget._header_clicked(mean)
    assert _column(widget, "Mean")[-1] == ""  # les cases vides à la fin
    shown = [float(v.replace(",", ".")) for v in _column(widget, "Mean") if v]
    assert shown == sorted(shown)
    widget._header_clicked(mean)
    shown = [float(v.replace(",", ".")) for v in _column(widget, "Mean") if v]
    assert shown == sorted(shown, reverse=True) and _column(widget, "Mean")[-1] == ""
    widget._header_clicked(mean)
    assert widget.grid.verticalHeaderItem(0).text() == str(widget.table.data.index[0])


@gui
def test_sort_covers_the_whole_table_and_export_follows(qtbot, tmp_path):
    """Au-delà de 2000 lignes, le tri ne portait que sur les lignes affichées."""
    from fuji.ui.results import MAX_ROWS_SHOWN, ResultsWidget, sorted_frame

    n = MAX_ROWS_SHOWN + 500
    df = pd.DataFrame({"Area": np.arange(n, dtype=float), "Nom": [f"objet {i}" for i in range(n)]})
    df.index = pd.RangeIndex(1, n + 1)
    widget = ResultsWidget(Table(df, name="Particules — test", units={"Area": "µm²"}))
    qtbot.addWidget(widget)
    widget.sort_by("Area", descending=True)
    # la plus grande valeur est dans les lignes cachées d'origine : elle arrive en tête
    assert widget.grid.verticalHeaderItem(0).text() == str(n)
    assert widget.grid.rowCount() == MAX_ROWS_SHOWN
    path = tmp_path / "tri.csv"
    widget.export(str(path))
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert lines[1].startswith(f"{n};") or lines[1].startswith(f"{n},")
    assert widget.clipboard_text().splitlines()[1].startswith(f"{n}\t")
    # colonne qui mêle nombres et textes : triée quand même, sans erreur
    mixed = pd.DataFrame({"x": [3, "a", 1.5, None]})
    assert len(sorted_frame(mixed, "x")) == 4 and sorted_frame(mixed, "absente") is mixed


def test_accumulate_refuses_mixed_units():
    """Des µm² et des px² ne vont pas dans la même colonne."""
    from fuji.ui.results import accumulate

    a = Table(pd.DataFrame({"Area": [1.0]}), name="Mesure — a", units={"Area": "µm²"})
    b = Table(pd.DataFrame({"Area": [2.0]}), name="Mesure — b", units={"Area": "px²"})
    c = Table(pd.DataFrame({"Area": [3.0]}), name="Mesure — c", units={"Area": "µm²"})
    assert accumulate(a, b) is None
    assert len(accumulate(a, c)) == 2


@gui
def test_columns_statistics_and_export(widget, tmp_path, qtbot):
    from fuji.ui.results import ResultsWidget, hidden_columns

    widget.show()
    assert widget.stats_grid.isHidden()
    widget.stats_box.setChecked(True)
    assert not widget.stats_grid.isHidden() and widget.stats_grid.rowCount() == 7
    names = [widget.stats_grid.verticalHeaderItem(r).text() for r in range(7)]
    assert names[:2] == ["Moyenne", "Écart type"] and names[-1] == "Nombre"
    area = widget.table.data["Area"].to_numpy()
    widget.decimals.setCurrentIndex(widget.decimals.findData(2))
    assert widget.stats_grid.item(0, 0).text() == f"{area.mean():.2f}"
    assert widget.stats_grid.item(6, 1).text() == "29" and widget.stats_grid.item(0, 2).text() == ""
    # les colonnes des deux tableaux restent alignées
    widget.grid.setColumnWidth(0, 143)
    assert widget.stats_grid.columnWidth(0) == 143

    widget.column_actions["Mean"].trigger()
    widget.column_actions["Type"].trigger()
    assert widget.columns() == ["Area", "Count"] and hidden_columns() == {"Mean", "Type"}
    assert widget.grid.isColumnHidden(1) and widget.stats_grid.isColumnHidden(1)
    assert widget.columns_btn.text() == "Colonnes (2 cachées)"
    assert list(widget.statistics().columns) == ["Area", "Count"]
    # copié et exporté : les colonnes montrées, toutes les lignes, la précision entière
    widget.comma.setChecked(True)
    text = widget.clipboard_text().splitlines()
    assert text[0] == " \tArea (µm²)\tCount" and len(text) == 31
    assert text[1] == f"1\t{area[0]}".replace(".", ",") + "\t0"
    widget.export(str(tmp_path / "r.csv"))
    assert (tmp_path / "r.csv").read_text(encoding="utf-8-sig").splitlines()[0] == " ;Area (µm²);Count"
    # retenu pour les tableaux suivants
    other = ResultsWidget(_table("Particules — lame 2", 5, seed=3))
    qtbot.addWidget(other)
    assert other.columns() == ["Area", "Count"] and other.stats_box.isChecked() and other.comma.isChecked()
    other.show_all_columns()
    assert other.columns() == ["Area", "Mean", "Type", "Count"] and other.columns_btn.text() == "Colonnes"


@gui
def test_distribution_of_a_column(widget):
    widget.show()
    assert widget.plot is None and widget.plot_box.isHidden() and widget.bins.isHidden()
    choices = [widget.graph.itemText(i) for i in range(widget.graph.count())]
    assert choices == [
        "Graphique : aucun",
        "Distribution : Area",
        "Distribution : Mean",
        "Distribution : Count",
    ]
    widget.show_distribution("Area")
    assert widget.plot is not None and widget.plot.spec.kind == "barres"
    assert len(widget.plot.x) == 6 and not widget.bins.isHidden() and not widget.save_plot_btn.isHidden()
    assert widget.plot_note.text().startswith("Area : moyenne ") and "(n = 30)" in widget.plot_note.text()
    widget.bins.setValue(15)
    assert len(widget.plot.x) == 15
    widget.plot.grab()  # se dessine sans erreur
    widget.graph.setCurrentIndex(0)
    assert widget.plot is None and widget.plot_box.isHidden()
    with pytest.raises(KeyError):
        widget.show_distribution("Type")


@gui
def test_a_table_with_its_own_plot_shows_it(qtbot, tmp_path):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("pas d'affichage (lancer avec xvfb-run)")
    from fuji.ui.app import isolate_settings
    from fuji.ui.results import ResultsWidget

    isolate_settings(str(tmp_path))
    x = np.arange(20.0)
    table = Table(
        pd.DataFrame({"Distance": x, "Valeur": x**2}),
        name="Profil — lame",
        plot=Plot("Distance", ("Valeur",), "ligne", "Distance (µm)", "Intensité"),
    )
    w = ResultsWidget(table)
    qtbot.addWidget(w)
    assert w.graph.currentText() == "Graphique du tableau" and w.plot.spec.kind == "ligne"
    w.show_distribution("Valeur")
    assert w.plot.spec.kind == "barres"
    w.set_table(table)  # le même tableau, complété : le choix du graphique est gardé
    assert w.graph.currentText() == "Distribution : Valeur"


@gui
def test_accumulate_box_gathers_a_series_in_one_tab(qtbot, tmp_path):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("pas d'affichage (lancer avec xvfb-run)")
    pytest.importorskip("napari")
    from fuji.ui.app import create_window, isolate_settings
    from fuji.ui.prefs import settings

    isolate_settings(str(tmp_path))
    _, win = create_window("demo")
    win.m.camera._timing = False
    qtbot.addWidget(win)
    tabs = win.results_tabs
    win.show_table(_table("Mesure — lame 1", 3))
    win.show_table(_table("Mesure — lame 2", 2, seed=1))
    assert tabs.count() == 2 and not win.accumulate_box.isChecked()  # d'origine : un onglet chacun
    win.accumulate_box.setChecked(True)
    assert settings().value("results/accumulate", type=bool) is True
    win.show_table(_table("Mesure — lame 3", 4, seed=2))  # à la suite du dernier du même genre
    assert tabs.count() == 2 and tabs.currentIndex() == 1 and tabs.tabText(1) == "Mesure — cumul"
    merged = tabs.widget(1)
    assert len(merged.table) == 6 and merged.grid.rowCount() == 6
    assert merged.table.data["Image"].tolist() == ["lame 2"] * 2 + ["lame 3"] * 4
    assert "6 lignes" in merged.title.text() and merged.summary.text() == "Lignes : 6  ·  Images : 2"
    win.show_table(_table("Particules — lame 3", 4))  # un autre genre de mesure : son onglet
    assert tabs.count() == 3
    win.show_table(_table("Mesure — lame 4", 1, seed=3))
    assert tabs.count() == 3 and len(tabs.widget(1).table) == 7 and tabs.currentIndex() == 1
