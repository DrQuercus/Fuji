"""Essai automatique de l'installation : ``fuji selftest``.

Démarre Fuji sur le microscope simulé, vérifie l'essentiel (fenêtre, live, capture, analyse,
enregistrement, moteurs simulés, serveur pour IA, affichage), puis ferme. Aucun réglage de
l'utilisateur n'est touché : tout se passe dans un dossier temporaire.

Sert à valider une installation sur un nouvel ordinateur, et l'application empaquetée
(``Fuji.app``) avant de la distribuer.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _versions() -> dict[str, str]:
    """Versions des bibliothèques dont dépend l'interface (une erreur = installation incomplète)."""
    import importlib

    out: dict[str, str] = {"python": platform.python_version(), "systeme": platform.platform()}
    for module in ("numpy", "scipy", "skimage", "pandas", "tifffile", "qtpy", "napari", "vispy", "superqt"):
        out[module] = str(getattr(importlib.import_module(module), "__version__", "?"))
    import qtpy

    out["qt"] = f"{qtpy.API_NAME} {qtpy.QT_VERSION}"
    for module in ("pymmcore_plus", "serial", "mcp", "imagej"):  # facultatifs
        try:
            out[module] = str(getattr(importlib.import_module(module), "__version__", "présent"))
        except Exception as exc:
            out[module] = f"absent ({type(exc).__name__})"
    return out


def run(report: str | None = None, screenshot: str | None = None) -> int:
    """Lance l'essai. Renvoie 0 si tout va bien, 1 sinon ; ``report`` : fichier JSON détaillé,
    ``screenshot`` : image de la fenêtre (pour juger de l'affichage)."""
    steps: list[dict[str, Any]] = []
    scratch = Path(tempfile.mkdtemp(prefix="fuji-essai-"))
    os.environ["FUJI_HOME"] = str(scratch / "fuji")  # ni profils ni calibrations de l'utilisateur
    os.environ["FUJI_CAPTURES"] = str(scratch / "images")  # ni ses images

    def step(name: str, fn: Callable[[], Any]) -> Any:
        start = time.monotonic()
        try:
            detail = fn()
            steps.append(
                {"etape": name, "ok": True, "detail": detail, "s": round(time.monotonic() - start, 2)}
            )
            print(f"  ✔ {name}", flush=True)
            return detail
        except Exception as exc:
            steps.append(
                {
                    "etape": name,
                    "ok": False,
                    "detail": f"{type(exc).__name__} : {exc}",
                    "trace": traceback.format_exc(),
                    "s": round(time.monotonic() - start, 2),
                }
            )
            print(f"  ✘ {name} — {type(exc).__name__} : {exc}", flush=True)
            return None

    print("Essai de Fuji…", flush=True)
    versions = step("bibliothèques", _versions) or {}

    state: dict[str, Any] = {}

    def wait(condition: Callable[[], bool], timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError(f"rien après {timeout:.0f} s")
            state["app"].processEvents()
            time.sleep(0.01)

    def window() -> str:
        from fuji.ui.app import create_window, isolate_settings

        isolate_settings(str(scratch / "reglages"))  # les préférences de l'utilisateur ne bougent pas
        app, win = create_window("demo")
        state["app"], state["win"] = app, win
        win.resize(1400, 900)
        win.show()
        wait(win.isVisible, 10)
        return f"{win.m.profile.name} — {app.platformName()}"

    def live() -> str:
        from fuji.ui.live import LIVE_LAYER

        win = state["win"]
        win.live.start()
        win.set_mode("microscope")
        wait(lambda: LIVE_LAYER in win.scope_viewer.layers)
        # le live tourne-t-il vraiment ? on attend trois images différentes (un ordinateur
        # lent en met plus de temps, ce n'est pas une panne)
        start, seen, last = time.monotonic(), 0, None
        while seen < 3:
            if time.monotonic() - start > 45.0:
                win.live.stop()
                raise RuntimeError(f"le live n'a affiché que {seen} image(s) en 45 s")
            state["app"].processEvents()
            data = win.scope_viewer.layers[LIVE_LAYER].data
            if data is not last:
                last, seen = data, seen + 1
            time.sleep(0.005)
        elapsed = time.monotonic() - start
        win.live.stop()
        return f"{seen} images en {elapsed:.1f} s, {tuple(last.shape)}"

    def capture() -> str:
        win = state["win"]
        frame = win.m.snap()
        win._on_snap(frame)
        state["frame"] = frame
        wait(lambda: not win.gallery._reserved, 30)  # enregistrement en arrière-plan
        info = win.gallery.entries()[0]
        if not info.path.exists() or not info.thumbnail_path.exists():
            raise RuntimeError(f"capture non enregistrée : {info.path}")
        state["capture"] = info
        size = info.path.stat().st_size // 1024
        return (
            f"{info.name} : {frame.data.shape} {frame.data.dtype}, {size} Ko, pixel {info.pixel_size_um} µm"
        )

    def open_in_analysis() -> str:
        win = state["win"]
        info = state["capture"]
        win.open_capture(info, None)  # relue depuis le disque, comme le lendemain
        if win.mode != "analyse" or info.name not in win.viewer.layers:
            raise RuntimeError(f"image non ouverte dans Analyse ({win.mode})")
        layer = win.viewer.layers[info.name]
        if layer.metadata.get("pixel_size_um") != info.pixel_size_um:
            raise RuntimeError("calibration perdue à la réouverture")
        if win.display_bar.layer is not layer:
            raise RuntimeError("la barre d'affichage ne suit pas l'image ouverte")
        win.display_bar.auto_contrast()
        lo, hi = (round(float(v)) for v in layer.contrast_limits)
        return f"espace {win.mode}, calque {info.name}, {layer.data.shape}, contraste auto {lo}–{hi}"

    def analysis() -> str:
        from fuji.analysis import Image
        from fuji.analysis import run as run_operation

        image = Image.from_frame(state["frame"])
        gray = run_operation("Niveaux de gris", image)
        mask = run_operation("Seuillage automatique", gray, method="Otsu", objects="sombres")
        table, _ = run_operation("Analyse de particules", mask, intensity=gray, min_area_um2=20)
        count = len(table.data)
        if count < 5:
            raise RuntimeError(f"seulement {count} objets trouvés sur la lame simulée")
        state["table"] = table
        return f"{count} objets mesurés"

    def analysis_settings() -> str:
        """Les réglages de l'espace Analyse : affichage, dessin mesuré, tableau de résultats."""
        import numpy as np

        from fuji.analysis import Image
        from fuji.analysis import run as run_operation

        win = state["win"]
        bar, panel = win.display_bar, win.analysis
        layer = win.viewer.layers[state["capture"].name]
        # affichage : zoom, luminosité / contraste, barre d'échelle
        bar.set_zoom(100)
        bar.full_contrast()
        bar.contrast.setValue(30)
        lo, hi = (float(v) for v in layer.contrast_limits)
        bar.contrast_label.click()
        if not (hi - lo) < float(layer.contrast_limits[1]) - float(layer.contrast_limits[0]):
            raise RuntimeError("le curseur de contraste ne resserre pas les bornes")
        bar.set_scale_bar(color="white")
        bar.set_scale_bar(color="auto")
        bar.zoom_fit()
        # dessin : une région et une ligne, avec leur mesure à côté
        panel.set_tool("shapes", "add_rectangle")
        roi = win.viewer.layers["ROI"]
        roi.add_rectangles(np.array([[20, 20], [20, 120], [90, 120], [90, 20]], float))
        roi.add_lines(np.array([[10, 10], [70, 90]], float))
        panel.set_tool("", "pan_zoom")
        panel.drawing.refresh()
        labels = [str(t) for t in roi.text.values]
        if len(labels) != 2 or "µm²" not in labels[0] or not labels[1].endswith("µm"):
            raise RuntimeError(f"mesures non affichées sur les formes : {labels}")
        for _ in range(10):  # le texte se dessine sur l'image
            state["app"].processEvents()
            time.sleep(0.02)
        panel.clear_rois()
        # un outil de retouche, puis le tableau de résultats et ses réglages
        image = Image.from_frame(state["frame"])
        tuned = run_operation("Réglages de l'image", image, contrast=20.0, saturation=15.0, sharpness=20.0)
        if tuned.data.shape != image.data.shape:
            raise RuntimeError("« Réglages de l'image » change la taille de l'image")
        tabs = win.results_tabs
        win.show_table(state["table"])
        widget = tabs.widget(tabs.count() - 1)
        widget.stats_box.setChecked(True)
        widget.show_distribution("Area")
        widget.plot.grab()
        mean = float(widget.statistics().loc["Moyenne", "Area"])
        widget.stats_box.setChecked(False)
        win._close_results_tab(tabs.count() - 1)
        win.results_dock.hide()
        return (
            f"zoom, contraste, barre d'échelle ; formes : {' | '.join(labels)} ; surface moyenne {mean:.1f}"
        )

    def files() -> str:
        import numpy as np
        import tifffile

        from fuji.io import save_image

        frame = state["frame"]
        path = scratch / "essai.ome.tif"
        save_image(path, frame.data, pixel_size_um=frame.metadata.pixel_size_um)
        back = tifffile.imread(path)
        if not np.array_equal(back, frame.data):
            raise RuntimeError("l'image relue diffère de l'image enregistrée")
        return f"{path.stat().st_size // 1024} Ko relus à l'identique"

    def compositions() -> str:
        """Profondeur de champ étendue et panorama, sur des images du microscope simulé
        (vérifie aussi que les outils de calcul sont bien présents dans l'application)."""
        from fuji.core import Microscope
        from fuji.core.compose import FocusStacker, LiveMosaic

        with Microscope.from_file("motorise-simulation") as scope:
            scope.device("moteurs").link._io.speed = 3000
            scope.apply_channel("Fond clair")
            scope.camera.set_property("Épaisseur de l'échantillon", 40.0)
            stack = FocusStacker()
            for z in (-20.0, 0.0, 20.0):
                scope.focus.move_to(z)
                stack.add(scope.snap().data)
            scope.camera.set_property("Épaisseur de l'échantillon", 0.0)
            scope.focus.move_to(0.0)
            mosaic = LiveMosaic()
            try:
                for x in (0.0, 0.0, 0.0, 40.0, 80.0, 120.0, 120.0, 120.0):
                    scope.xy_stage.move_to(x, 0.0)
                    mosaic.add(scope.snap().data)
                image, _ = mosaic.render()
            finally:
                mosaic.close()
        if stack.status.used < 2 or len(mosaic.tiles) != 2:
            raise RuntimeError(f"{stack.status.used} images fusionnées, {len(mosaic.tiles)} champs")
        return (
            f"{stack.status.used} images fusionnées ; panorama de 2 champs {image.shape[1]}×{image.shape[0]}"
        )

    def motors() -> str:
        from fuji.core import Microscope

        with Microscope.from_file("motorise-simulation") as scope:
            scope.device("moteurs").link._io.speed = 3000  # contrôleur simulé accéléré
            scope.xy_stage.move_to(120.0, -80.0)
            scope.focus.move_to(7.5)
            x, y = scope.xy_stage.get_position()
            z = scope.focus.get_position()
            if (round(x, 3), round(y, 3), round(z, 3)) != (120.0, -80.0, 7.5):
                raise RuntimeError(f"position inattendue : {x}, {y}, {z}")
        return "platine et mise au point simulées : positions exactes"

    def ai_server() -> str:
        import asyncio
        import socket
        import threading

        from mcp import Client

        win = state["win"]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        win.set_ai_server(True, port, True)
        if win._ai_server is None:
            raise RuntimeError(win.settings_panel.ai_status.text())
        out: dict[str, Any] = {}

        async def session() -> None:
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                out["outils"] = len((await client.list_tools()).tools)
                result = await client.call_tool("snap", {"max_side": 128})
                out["contenu"] = [c.type for c in result.content]

        def work() -> None:
            try:
                asyncio.run(session())
            except Exception as exc:
                out["erreur"] = f"{type(exc).__name__} : {exc}"
            out["fini"] = True

        threading.Thread(target=work, daemon=True).start()
        try:
            wait(lambda: "fini" in out, 60)
        finally:
            win.set_ai_server(False)
        if "erreur" in out:
            raise RuntimeError(out["erreur"])
        if out["contenu"] != ["text", "image"]:
            raise RuntimeError(f"réponse inattendue : {out['contenu']}")
        return f"{out['outils']} outils, image reçue par le client"

    def display() -> str:
        import numpy as np

        win = state["win"]
        win.viewer.reset_view()
        for _ in range(30):
            state["app"].processEvents()
            time.sleep(0.02)
        shot = np.asarray(win.qt_viewer.screenshot(flash=False))  # rendu OpenGL de l'image
        if screenshot:
            win.grab().save(screenshot)
        spread = float(shot[..., :3].std())
        if spread < 2.0:
            raise RuntimeError("le rendu de l'image est uniforme (OpenGL ne dessine rien ?)")
        return f"rendu {shot.shape[1]}×{shot.shape[0]}, contraste {spread:.0f}"

    if step("fenêtre", window) is not None:
        step("live", live)
        if step("capture", capture) is not None:
            step("ouverture dans Analyse", open_in_analysis)
            if step("analyse", analysis) is not None:
                step("réglages de l'analyse", analysis_settings)
            step("enregistrement", files)
        step("affichage", display)
        if "absent" not in versions.get("mcp", "absent"):
            step("serveur pour IA", ai_server)
    step("compositions en direct", compositions)
    step("moteurs simulés", motors)

    win = state.get("win")
    if win is not None:
        try:
            win.close()
            state["app"].processEvents()
        except Exception:  # la fermeture ne doit pas masquer le résultat
            pass

    ok = all(s["ok"] for s in steps)
    if report:
        Path(report).write_text(
            json.dumps({"ok": ok, "etapes": steps}, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
    failed = [s["etape"] for s in steps if not s["ok"]]
    print("Tout fonctionne." if ok else f"Échec : {', '.join(failed)}", flush=True)
    for s in steps:
        if not s["ok"]:
            print(s["trace"], file=sys.stderr, flush=True)
    return 0 if ok else 1
