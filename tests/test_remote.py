"""Pilotage par un programme ou une IA externe : contrôleur, garde-fous, serveur MCP."""

from __future__ import annotations

import asyncio
import json

import pytest

from fuji.core import Microscope
from fuji.remote import ControlError, ManualActionRequired, MicroscopeController, NotAllowed, SafetyPolicy


@pytest.fixture
def scope():
    """Microscope motorisé simulé (contrôleur GRBL simulé, caméra qui suit la platine)."""
    m = Microscope.from_file("motorise-simulation")
    m.connect()
    m.camera._timing = False
    m.device("moteurs").link._io.speed = 3000.0  # moteurs simulés accélérés
    yield m
    m.disconnect()


@pytest.fixture
def controller(scope, tmp_path):
    return MicroscopeController(scope, SafetyPolicy(workspace=tmp_path))


# ---------------------------------------------------------------------- contrôleur
def test_status_describes_hardware_and_safeguards(controller):
    status = controller.status()
    json.dumps(status)  # entièrement sérialisable
    assert status["microscope"].startswith("Microscope motorisé")
    assert status["objectif"] == "10x/0.25" and status["changement_objectif"] == "manuel"
    assert status["taille_pixel_um"] == pytest.approx(0.29)
    assert status["platine"]["motorisee"] and status["mise_au_point"]["limites_um"] == [-500, 500]
    assert status["camera"]["largeur_px"] == 1024 and status["camera"]["champ_um"][0] == pytest.approx(
        297, abs=1
    )
    assert {k: status["lampes"][0][k] for k in ("nom", "allumee")} == {
        "nom": "moteurs_lampe",
        "allumee": False,
    }
    assert status["garde_fous"]["moteurs_autorises"] and not status["garde_fous"]["lecture_seule"]


def test_snapshot_flags_bad_exposure_and_gives_a_preview(controller):
    dark = controller.snap()
    assert dark["nom"] == "capture_1" and "très sombre" in dark["diagnostic"]  # lampe éteinte
    controller.set_channel("Fond clair")
    controller.auto_exposure()
    image = controller.snap()
    assert image["diagnostic"] is None and image["forme"] == [768, 1024, 3] and image["sature_pourcent"] < 1
    assert image["metadonnees"]["objective"] == "10x/0.25" and image["taille_pixel_um"] == pytest.approx(0.29)
    data, fmt = controller.preview(image["nom"], max_side=256)
    assert fmt == "jpeg" and data[:2] == b"\xff\xd8" and 2000 < len(data) < 60000
    import io

    from PIL import Image as PILImage

    assert max(PILImage.open(io.BytesIO(data)).size) == 256
    controller.set_exposure(4000)
    assert "surexposée" in controller.snap()["diagnostic"]
    with pytest.raises(ControlError, match="hors limites"):
        controller.set_exposure(99999)


def test_motion_is_bounded_by_policy_and_hardware_limits(controller, scope):
    assert controller.move_stage(100, -50) == {"x_um": 100.0, "y_um": -50.0}
    assert controller.move_stage(0, 0, relative=False) == {"x_um": 0.0, "y_um": 0.0}
    with pytest.raises(NotAllowed, match="trop grand"):
        controller.move_stage(6000, 0)  # garde-fou : 5 mm par commande
    assert controller.move_focus(40)["z_um"] == 40.0
    with pytest.raises(NotAllowed, match="protéger l'objectif"):
        controller.move_focus(150)
    for _ in range(4):
        controller.move_focus(95, relative=True)
    with pytest.raises(ControlError, match="hors de la course"):  # limite du matériel : ±500 µm
        controller.move_focus(95, relative=True)
    assert scope.focus.get_position() == pytest.approx(40 + 4 * 95)  # le pas refusé n'a rien bougé
    assert controller.set_origin("z") == {"origine": "z"} and scope.focus.get_position() == pytest.approx(0)
    assert controller.stop() == {"arrete": True}
    with pytest.raises(ControlError, match="prise d'origine"):
        controller.home()


def test_read_only_and_no_motion_policies(scope, tmp_path):
    observer = MicroscopeController(scope, SafetyPolicy(read_only=True, workspace=tmp_path))
    assert observer.snap()["nom"] == "capture_1"  # regarder reste permis
    assert observer.run_operation("Flou gaussien", {"image": "capture_1"})["resultats"]  # analyser aussi
    for call in (
        lambda: observer.set_exposure(20),
        lambda: observer.move_stage(10, 0),
        lambda: observer.set_light(on=True),
        lambda: observer.set_channel("Fond clair"),
        lambda: observer.acquire("x"),
    ):
        with pytest.raises(NotAllowed):
            call()
    assert observer.stop() == {"arrete": True}  # l'arrêt d'urgence n'est jamais refusé
    still = MicroscopeController(scope, SafetyPolicy(allow_motion=False, workspace=tmp_path))
    still.set_exposure(20)
    with pytest.raises(NotAllowed, match="moteurs"):
        still.move_focus(1)
    with pytest.raises(NotAllowed, match="moteurs"):
        still.autofocus()


def test_manual_actions_are_relayed_then_confirmed(controller, scope):
    with pytest.raises(ManualActionRequired, match="Mettez en place « 40x/0\\.65 »") as info:
        controller.set_objective("40x/0.65")
    assert "manual_done=true" in str(info.value)
    assert scope.objective.label == "10x/0.25"  # rien n'est supposé fait
    assert controller.set_objective("40x/0.65", manual_done=True) == {
        "objectif": "40x/0.65",
        "taille_pixel_um": pytest.approx(0.0725),
    }
    with pytest.raises(ControlError, match="Objectif inconnu"):
        controller.set_objective("63x")


def test_files_stay_inside_the_workspace(controller, tmp_path):
    controller.set_channel("Fond clair")
    controller.snap("vue")
    saved = controller.save_item("vue")
    assert saved["fichier"] == str(tmp_path / "vue.ome.tif") and (tmp_path / "vue.ome.tif").exists()
    reopened = controller.open_image("vue.ome.tif", "relue")
    assert reopened["taille_pixel_um"] == pytest.approx(0.29) and reopened["forme"] == [768, 1024, 3]
    for bad in ("../ailleurs.tif", "/etc/passwd", str(tmp_path.parent / "x.tif")):
        with pytest.raises(NotAllowed, match="dossier de travail"):
            controller.save_item("vue", bad)
        with pytest.raises(NotAllowed, match="dossier de travail"):
            controller.open_image(bad)
    with pytest.raises(ControlError, match="introuvable"):
        controller.open_image("absent.tif")


def test_analysis_pipeline_through_the_controller(controller):
    controller.set_channel("Fond clair")
    controller.auto_exposure()
    controller.snap("cellules")
    found = controller.operations(search="threshold")
    assert {"Seuillage automatique", "Seuillage local"} <= {o["nom"] for o in found}
    detail = controller.describe_operation("Seuillage automatique")
    assert [e["nom"] for e in detail["entrees"]] == ["image"] and "Otsu" in detail["reglages"][0]["choix"]
    mask = controller.run_operation(
        "Seuillage automatique", {"image": "cellules"}, {"method": "Otsu", "objects": "sombres"}
    )["resultats"][0]
    assert mask["type"] == "masque" and 0.01 < mask["fraction_couverte"] < 0.5
    out = controller.run_operation(
        "Analyse de particules", {"mask": mask["nom"], "intensity": "cellules"}, {"min_area_um2": 20}
    )["resultats"]
    table = controller.table(next(o["nom"] for o in out if o["type"] == "tableau"), max_rows=3)
    assert table["lignes"] > 5 and len(table["donnees"]) == 3 and table["tronque"]
    assert table["unites"]["Area"] == "µm²" and table["statistiques"]["Area"]["min"] >= 20
    roi = controller.create_roi("rectangle", [100, 100, 200, 150])
    measured = controller.run_operation("Mesurer les ROI", {"image": "cellules", "roi": roi["nom"]})
    area = controller.table(measured["resultats"][0]["nom"])["donnees"][0]["Area"]
    assert area == pytest.approx(200 * 150 * 0.29**2, rel=1e-3)
    data, fmt = controller.preview(mask["nom"], 128)
    assert fmt == "png" and data[:4] == b"\x89PNG"
    csv = controller.save_item(table["nom"])["fichier"]
    assert csv.endswith(".csv")
    assert controller.delete_item(mask["nom"])["supprime"] == mask["nom"]
    # erreurs explicites, exploitables par une IA
    with pytest.raises(ControlError, match="Vouliez-vous dire"):
        controller.run_operation("seuillage")
    with pytest.raises(ControlError, match="il manque l'entrée 'image'"):
        controller.run_operation("Flou gaussien")
    with pytest.raises(ControlError, match="réglages inconnus"):
        controller.run_operation("Flou gaussien", {"image": "cellules"}, {"rayon": 2})
    with pytest.raises(ControlError, match="choix possibles"):
        controller.run_operation("Seuillage automatique", {"image": "cellules"}, {"method": "Magique"})
    with pytest.raises(ControlError, match="Aucun élément"):
        controller.run_operation("Flou gaussien", {"image": "inconnue"})


def test_acquisition_z_stack_and_mosaic(controller, tmp_path):
    controller.set_channel("Fond clair")
    done = controller.acquire("pile", z_range_um=4, z_step_um=2, wait_s=30)
    assert done["etat"] == "terminée" and done["faites"] == 3 and done["erreur"] is None
    assert done["fichiers"] == [str(tmp_path / "acquisitions" / "pile.ome.tif")]
    calibration = controller.calibrate_stage()
    assert abs(calibration["rotation_degres"]) < 3 and not calibration["enregistre"]
    with pytest.raises(NotAllowed, match="profil"):
        controller.calibrate_stage(save=True)  # écrire dans le profil demande une autorisation
    mosaic = controller.acquire("zone", mosaic=[2, 2], mosaic_overlap=0.2, wait_s=60)
    assert mosaic["etat"] == "terminée" and mosaic["faites"] == 4 and mosaic["mosaiques"] == ["mosaique_1"]
    shape = controller.describe_item("mosaique_1")["forme"]
    assert shape[0] > 1.6 * 768 and shape[1] > 1.6 * 1024 and shape[2] == 3
    assert controller.acquisition_status()["etat"] == "terminée"
    with pytest.raises(NotAllowed, match="au plus"):
        controller.acquire("trop", time_points=5000)
    with pytest.raises(ControlError, match="Canaux inconnus"):
        controller.acquire("x", channels=["Inconnu"])


def test_long_acquisition_can_be_followed_and_cancelled(controller):
    started = controller.acquire("longue", time_points=50, interval_s=0.2, wait_s=0.3)
    assert started["etat"] == "en cours" and started["total"] == 50
    with pytest.raises(ControlError, match="déjà en cours"):
        controller.acquire("autre")
    cancelled = controller.cancel_acquisition()
    assert cancelled["etat"] == "annulée" and 0 < cancelled["faites"] < 50


def test_assistants_through_the_controller(controller, scope):
    controller.set_channel("Fond clair")
    assert controller.focus_score()["nettete"] > 0
    assert "diagnostic" in controller.check_illumination()
    assert "conseil" in controller.check_field_diaphragm()
    controller.move_focus(6)
    assert abs(controller.autofocus(range_um=20, step_um=2)["z_um"]) < 1.5
    props = controller.device_properties("camera")
    assert any(p["nom"] == "Échantillon" for p in props)
    assert controller.set_device_property("camera", "Échantillon", "micromètre")["valeur"] == "micromètre"
    cal = controller.calibrate_pixel_size(10.0)
    assert cal["fiable"] and cal["taille_pixel_um"] == pytest.approx(0.29, rel=5e-3)
    with pytest.raises(ControlError, match="choix possibles"):
        controller.set_device_property("camera", "Échantillon", "lune")
    with pytest.raises(ControlError, match="Périphérique inconnu"):
        controller.device_properties("fantôme")


# ---------------------------------------------------------------------- serveur MCP
mcp = pytest.importorskip("mcp", reason='pip install "fuji-microscopy[ia]"')


def _text(result) -> str:
    return next(c.text for c in result.content if hasattr(c, "text"))


def _is_error(result) -> bool:
    return bool(getattr(result, "is_error", None) or getattr(result, "isError", None))


def _call(server, steps):
    """Appelle des outils dans une session MCP en mémoire ; renvoie les résultats."""
    from mcp import Client

    async def main():
        async with Client(server) as client:
            return await steps(client)

    return asyncio.run(main())


def test_mcp_server_exposes_documented_tools(controller):
    from fuji.remote.mcp_server import build_server

    async def steps(client):
        return (
            (await client.list_tools()).tools,
            (await client.list_resources()).resources,
            client.instructions,
        )

    tools, resources, instructions = _call(build_server(controller), steps)
    names = {t.name for t in tools}
    assert {
        "microscope_status",
        "snap",
        "set_exposure",
        "set_objective",
        "move_stage",
        "move_focus",
        "center_on_point",
        "autofocus",
        "stop_motion",
        "acquire",
        "list_operations",
        "run_operation",
        "get_table",
        "get_image",
    } <= names
    assert all(t.description and len(t.description) > 30 for t in tools)  # chaque outil explique son usage
    snap = next(t for t in tools if t.name == "snap")
    schema = getattr(snap, "input_schema", None) or getattr(snap, "inputSchema", None)
    assert schema["properties"]["max_side"]["description"]
    assert {str(r.uri) for r in resources} == {"fuji://etat", "fuji://operations", "fuji://profil"}
    assert "manual_done=true" in instructions and "micromètres" in instructions


def test_mcp_session_snap_analyse_and_report_errors(controller):
    from fuji.remote.mcp_server import build_server

    async def steps(client):
        out = {}
        out["status"] = await client.call_tool("microscope_status", {})
        await client.call_tool("set_channel", {"name": "Fond clair"})
        await client.call_tool("auto_exposure", {})
        out["snap"] = await client.call_tool("snap", {"max_side": 200})
        out["mask"] = await client.call_tool(
            "run_operation",
            {
                "name": "Seuillage automatique",
                "inputs": {"image": "capture_1"},
                "params": {"objects": "sombres"},
            },
        )
        out["view"] = await client.call_tool(
            "get_image", {"name": "seuillage_automatique_1", "max_side": 100}
        )
        out["too_far"] = await client.call_tool("move_stage", {"x_um": 1e6, "y_um": 0})
        out["manual"] = await client.call_tool("set_objective", {"label": "40x/0.65"})
        out["confirmed"] = await client.call_tool("set_objective", {"label": "40x/0.65", "manual_done": True})
        out["moved"] = await client.call_tool("move_stage", {"x_um": 50, "y_um": 0})
        out["state"] = await client.read_resource("fuji://etat")
        out["prompt"] = await client.get_prompt("compter_des_objets", {"objets": "noyaux"})
        return out

    out = _call(build_server(controller), steps)
    assert json.loads(_text(out["status"]))["objectif"] == "10x/0.25"
    kinds = [type(c).__name__ for c in out["snap"].content]
    assert kinds == ["TextContent", "ImageContent"]  # l'IA reçoit la description et l'image
    image = out["snap"].content[1]
    assert getattr(image, "mime_type", None) or getattr(image, "mimeType", None) == "image/jpeg"
    assert json.loads(_text(out["snap"]))["nom"] == "capture_1"
    assert json.loads(_text(out["mask"]))["resultats"][0]["type"] == "masque"
    assert [type(c).__name__ for c in out["view"].content] == ["TextContent", "ImageContent"]
    assert _is_error(out["too_far"]) and "au plus 5000 µm" in _text(out["too_far"])
    assert _is_error(out["manual"]) and "Mettez en place « 40x/0.65 »" in _text(out["manual"])
    assert not _is_error(out["confirmed"]) and json.loads(_text(out["confirmed"]))["objectif"] == "40x/0.65"
    assert json.loads(_text(out["moved"])) == {"x_um": 50.0, "y_um": 0.0}
    assert json.loads(out["state"].contents[0].text)["platine"]["x_um"] == 50.0
    assert "noyaux" in out["prompt"].messages[0].content.text


def test_mcp_emergency_stop_passes_during_a_long_move(scope, tmp_path):
    from fuji.remote.mcp_server import build_server

    scope.device("moteurs").link._io.speed = 1.0  # temps réel : 5 mm à 600 mm/min = 0,5 s... × 10
    controller = MicroscopeController(scope, SafetyPolicy(workspace=tmp_path, max_xy_step_um=1e6))

    async def steps(client):
        move = asyncio.create_task(client.call_tool("move_stage", {"x_um": 15000, "y_um": 0}))
        await asyncio.sleep(0.3)
        stop = await client.call_tool("stop_motion", {})  # traité pendant que la platine avance
        return await move, stop

    import time

    t0 = time.monotonic()
    move, stop = _call(build_server(controller), steps)
    assert time.monotonic() - t0 < 1.4  # sans arrêt, le déplacement durerait 1,5 s
    assert json.loads(_text(stop)) == {"arrete": True}
    assert _is_error(move) and "interrompu" in _text(move)
    assert 0 < scope.xy_stage.get_position()[0] < 14000


def test_mcp_http_transport_requires_the_token(controller):
    import urllib.error
    import urllib.request

    from mcp import Client

    from fuji.remote.mcp_server import HttpServer, http_app

    with pytest.raises(ControlError, match="jeton"):
        http_app(controller, host="0.0.0.0")  # ouvrir au réseau sans jeton est refusé
    server = HttpServer(controller, port=8793, token="sésame-ouvre-toi".encode().hex())
    server.start()
    try:
        assert server.running and server.url == "http://127.0.0.1:8793/mcp"
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(urllib.request.Request(server.url, data=b"{}", method="POST"))
        assert info.value.code == 401
        busy = HttpServer(controller, port=8793)
        with pytest.raises(ControlError, match="port"):
            busy.start(timeout=3)
    finally:
        server.stop()
    assert not server.running
    open_server = HttpServer(controller, port=8794)  # sur cette machine seulement : pas de jeton
    open_server.start()
    try:

        async def main():
            async with Client(open_server.url) as client:
                return await client.call_tool("microscope_status", {})

        assert json.loads(_text(asyncio.run(main())))["microscope"].startswith("Microscope motorisé")
    finally:
        open_server.stop()


def test_cli_lists_ports_and_declares_the_mcp_command(capsys):
    from fuji.cli import main

    assert main(["moteurs", "ports"]) in (0, 1)
    with pytest.raises(SystemExit):
        main(["mcp", "--help"])
    out = capsys.readouterr().out
    assert "--read-only" in out and "--no-motion" in out and "--token" in out
