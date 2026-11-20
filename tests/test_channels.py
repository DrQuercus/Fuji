"""Canaux superposés : couleurs, captures couleur ramenées à un canal, image couleur finale,
et dans l'interface : superposer des images, régler chaque canal, ouvrir une acquisition."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from fuji.analysis.channels import (
    Shown,
    best_component,
    channel_color,
    channel_plane,
    color_from_name,
    composite_rgb,
    order_channels,
)


def test_channel_colors():
    assert channel_color("DAPI", "fluorescence", "blue") == "blue"  # celle de la configuration
    assert channel_color("GFP", "fluorescence", "gray") == "green"  # gris par défaut : ignoré
    assert channel_color("Canal 1", "fluorescence", emission_nm=610) == "magenta"
    assert channel_color("Fond clair", "transmission") == "gray"
    assert channel_color("Fond noir", "fond noir", "green") == "green"  # choix explicite gardé
    assert channel_color("C2", None, taken=["green"]) == "magenta"  # une couleur encore libre
    names = ["DAPI", "Hoechst", "FITC", "GFP", "TRITC", "mCherry", "Cy5", "Alexa 647 (far red)", "PI"]
    assert [color_from_name(n) for n in names] == [
        "blue",
        "blue",
        "green",
        "green",
        "red",
        "red",
        "magenta",
        "magenta",
        "red",
    ]
    assert color_from_name("pile 3") is None and color_from_name("Fond clair") is None


def _fluo_capture(component: int, shape=(40, 50)) -> np.ndarray:
    """Une capture couleur de fluorescence : du signal dans une seule couleur de la caméra."""
    rgb = np.full((*shape, 3), 60, np.uint16)
    rgb[10:20, 10:20, component] = 3000
    return rgb


def test_colour_captures_become_one_channel():
    gfp = _fluo_capture(1)
    assert best_component(gfp).max() == 3000  # le vert porte le signal
    assert best_component(gfp, "blue").max() == 60  # couleur imposée : le bleu
    ch = channel_plane(gfp, "GFP", kind="fluorescence")
    assert ch.color == "green" and not ch.rgb and ch.data.shape == (40, 50) and ch.data.max() == 3000
    magenta = _fluo_capture(0)
    magenta[10:20, 10:20, 2] = 2500
    assert best_component(magenta, "magenta").max() == 3000  # rouge et bleu
    bright = np.full((40, 50, 3), 3000, np.uint16)
    bright[10:20, 10:20] = 800  # une cellule sombre sur fond clair
    ch = channel_plane(bright, "capture_001")  # type inconnu : reconnu au fond clair
    assert ch.kind == "transmission" and ch.rgb and ch.color == "gray"
    dark = channel_plane(bright, "Fond noir", kind="fond noir")
    assert dark.rgb  # un fond noir en couleur reste en couleur
    first = order_channels([channel_plane(gfp, "GFP", kind="fluorescence"), ch])
    assert first[0].kind == "transmission"  # le fond clair en dessous


def test_composite_rgb_is_what_is_shown():
    a = np.zeros((4, 5), np.uint16)
    a[0, 0] = 1000
    b = np.zeros((4, 5), np.uint16)
    b[0, 0] = 500
    b[1, 1] = 1000
    rgb = composite_rgb(
        [
            Shown(a, "blue", (0, 1000)),
            Shown(b, "green", (0, 1000)),
            Shown(b, "red", (0, 1000), visible=False),  # caché : absent de l'image
        ]
    )
    assert rgb.dtype == np.uint8 and rgb.shape == (4, 5, 3)
    assert tuple(rgb[0, 0]) == (0, 128, 255) and tuple(rgb[1, 1]) == (0, 255, 0)
    assert tuple(composite_rgb([Shown(a, tint=(1.0, 0.5, 0.0), limits=(0, 1000))])[0, 0]) == (255, 128, 0)
    base = np.full((4, 5, 3), 200, np.uint8)
    over = composite_rgb(
        [Shown(base, rgb=True, limits=(0, 255), additive=False), Shown(a, "blue", (0, 1000))]
    )
    assert tuple(over[0, 0]) == (200, 200, 255) and tuple(over[2, 2]) == (200, 200, 200)
    with pytest.raises(ValueError, match="visible"):
        composite_rgb([Shown(a, visible=False)])


# ------------------------------------------------------------------ interface
gui = pytest.mark.gui


@pytest.fixture
def window(qtbot, tmp_path):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("pas d'affichage (lancer avec xvfb-run)")
    pytest.importorskip("napari")
    from fuji.ui.app import create_window, isolate_settings

    isolate_settings(str(tmp_path))
    _, win = create_window("demo")
    win.m.camera._timing = False
    qtbot.addWidget(win)
    return win


def _capture(window, qtbot, channel):
    window.m.apply_channel(channel)
    window._on_snap(window.m.snap())
    qtbot.waitUntil(lambda: not window.gallery._reserved, timeout=10000)


@gui
def test_superpose_captures_and_tune_each_channel(window, qtbot):
    for channel in ("Fond clair", "DAPI", "GFP"):
        _capture(window, qtbot, channel)
    gallery = window.gallery
    assert not gallery.superpose_btn.isEnabled()
    gallery.list.selectAll()
    assert gallery.superpose_btn.isEnabled()
    assert [i.metadata["channel"] for i in gallery.selected()] == ["Fond clair", "DAPI", "GFP"]
    gallery.superpose_btn.click()
    qtbot.wait(50)
    doc = "Fond clair + DAPI + GFP"
    layers = {ly.metadata["fuji_channel"]: ly for ly in window.viewer.layers}
    assert set(layers) == {"Fond clair", "DAPI", "GFP"} and window.mode == "analyse"
    assert layers["Fond clair"].rgb and str(layers["Fond clair"].blending) == "translucent_no_depth"
    assert layers["DAPI"].colormap.name == "blue" and layers["GFP"].colormap.name == "green"
    assert str(layers["GFP"].blending) == "additive" and layers["GFP"].data.ndim == 2
    tabs = window.doc_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [doc]

    bar = window.display_bar
    chips = {chip.name.text(): chip for chip in bar._chips}
    assert list(chips) == ["Fond clair", "DAPI", "GFP"] and not bar.channels_box.isHidden()
    chips["GFP"].name.click()
    assert bar.layer is layers["GFP"] and bar.name_label.text() == "Canal : GFP"
    bar.max_box.setValue(1500)  # la barre règle le canal choisi
    assert layers["GFP"].contrast_limits[1] == 1500 and layers["DAPI"].contrast_limits[1] != 1500
    chips["GFP"].set_color("magenta")
    assert layers["GFP"].colormap.name == "magenta"
    chips["Fond clair"].eye.setChecked(False)
    assert not layers["Fond clair"].visible
    bar.solo_channel(layers["DAPI"])
    assert [ly.visible for ly in layers.values()] == [False, True, False]
    for ly in layers.values():
        ly.visible = True
    bar.auto_all_channels()

    # l'image couleur, telle qu'on la voit : une nouvelle image, à côté
    name = window.color_image()
    qtbot.wait(20)  # les onglets suivent juste après
    rgb = window.viewer.layers[name]
    assert name == f"{doc} (couleur)" and rgb.rgb and rgb.data.dtype == np.uint8
    assert [tabs.tabText(i) for i in range(tabs.count())] == [doc, name]
    # revenir à la superposition : les canaux retrouvent leur réglage
    tabs.setCurrentIndex(0)
    qtbot.wait(20)
    assert all(ly.visible for ly in layers.values()) and not rgb.visible
    assert layers["GFP"].colormap.name == "magenta"


@gui
def test_multichannel_acquisition_opens_superposed(window, qtbot, tmp_path):
    from fuji.io.ome_tiff import save_image

    frames = []
    for channel in ("DAPI", "GFP"):
        window.m.apply_channel(channel)
        frames.append(window.m.snap().data)
    path = tmp_path / "acquisition_p000.ome.tif"
    save_image(
        path,
        np.stack(frames),
        pixel_size_um=0.29,
        axes="CYXS",
        metadata={"Channel": {"Name": ["DAPI", "GFP"]}},
    )
    window.open_paths([str(path)])
    qtbot.wait(50)
    layers = list(window.viewer.layers)
    assert [ly.metadata["fuji_channel"] for ly in layers] == ["DAPI", "GFP"]
    assert [ly.colormap.name for ly in layers] == ["blue", "green"]  # couleurs de la configuration
    assert all(ly.data.shape == (768, 1024) for ly in layers)  # la couleur de la caméra qui porte le signal
    assert layers[0].scale[-1] == pytest.approx(0.29)
    assert [c.name.text() for c in window.display_bar._chips] == ["DAPI", "GFP"]
    # les outils travaillent sur un canal à la fois
    panel = window.analysis
    panel.select_operation("Mesurer")
    assert panel._inputs["image"].findText(layers[1].name) >= 0


@gui
def test_superposition_refuses_images_of_different_sizes(window, qtbot, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    from fuji.io.ome_tiff import save_image

    told = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: told.append(a[2]))
    a, b = tmp_path / "a.ome.tif", tmp_path / "b.ome.tif"
    save_image(a, np.zeros((40, 50), np.uint16))
    save_image(b, np.zeros((30, 50), np.uint16))
    assert window.superpose_paths([str(a), str(b)]) is None
    assert "pas la même taille" in told[0] and len(window.viewer.layers) == 0
    assert window.superpose_paths([str(a)]) is None  # une seule image : rien à superposer


@gui
def test_images_of_other_days(window, qtbot):
    from fuji.io.ome_tiff import save_image

    gallery = window.gallery
    root = gallery.store.root
    old = root / "2026-01-15"
    old.mkdir(parents=True)
    save_image(old / "capture_001.ome.tif", np.zeros((20, 30), np.uint16))
    save_image(old / "capture_002.ome.tif", np.zeros((20, 30), np.uint16))
    gallery._fill_days()
    assert not gallery.day.isHidden()
    assert gallery.day.itemText(0) == "Aujourd'hui" and gallery.day.itemText(1) == "15/01/2026 — 2 images"
    gallery.day.setCurrentIndex(1)
    gallery.day.activated.emit(1)
    assert [i.name for i in gallery.entries()] == ["capture_002", "capture_001"]
    gallery.list.setCurrentRow(1)
    gallery.open_selected()
    assert window.viewer.layers[0].metadata["source"].endswith("2026-01-15/capture_001.ome.tif")
    # une capture faite pendant que l'on regarde un autre jour : retour à aujourd'hui
    _capture(window, qtbot, "Fond clair")
    assert gallery.day.currentIndex() == 0 and [i.name for i in gallery.entries()] == ["capture_001"]
    assert gallery.entries()[0].path.parent == gallery.store.day_folder()
