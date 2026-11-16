"""Ligne de commande : ``fuji`` (interface), ``fuji check``, ``fuji touptek``..."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from fuji.ui.app import run_app
    except ImportError as exc:
        print(f'Interface indisponible ({exc}).\nInstalle-la avec : pip install "fuji-microscopy[gui]"')
        return 1
    return run_app(args.profile)


def _cmd_profiles(_: argparse.Namespace) -> int:
    from fuji.core.profile import MicroscopeProfile, available_profiles

    for name, path in available_profiles().items():
        try:
            title = MicroscopeProfile.load(path).name
        except Exception as exc:  # profil invalide : on le signale sans s'arrêter
            title = f"(invalide : {exc})"
        print(f"{name:28s} {title}\n{'':28s} {path}")
    return 0


def _cmd_drivers(_: argparse.Namespace) -> int:
    from fuji.drivers import available_drivers

    print("\n".join(available_drivers()))
    return 0


def _cmd_operations(_: argparse.Namespace) -> int:
    from fuji.analysis import list_operations

    cat = None
    for op in list_operations():
        if op.category != cat:
            cat = op.category
            print(f"\n[{cat}]")
        extra = f"   ← ImageJ : {op.imagej}" if op.imagej else ""
        print(f"  {op.name}{extra}")
    return 0


def _ok(flag: bool) -> str:
    return "✔" if flag else "✘"


def _cmd_check(_: argparse.Namespace) -> int:
    """Diagnostic de l'installation."""
    print(f"Fuji sur Python {platform.python_version()} ({platform.system()} {platform.machine()})")
    for mod, why in (
        ("napari", "interface"),
        ("qtpy", "interface"),
        ("pymmcore_plus", "Micro-Manager"),
        ("imagej", "pont Fiji (PyImageJ)"),
    ):
        print(f"  {_ok(importlib.util.find_spec(mod) is not None)} {mod:14s} {why}")
    print(
        f"  {_ok(shutil.which('java') is not None)} {'java':14s} requis par le pont Fiji "
        "(PyImageJ peut aussi le télécharger)"
    )
    from fuji.analysis.imagej import find_fiji

    fiji = find_fiji()
    print(f"  {_ok(fiji is not None)} {'Fiji':14s} {fiji or 'non trouvé (FUJI_FIJI_PATH)'}")
    from fuji.drivers.touptek import ToupTekSDKNotFound, default_sdk_dir, list_cameras

    try:
        cams = list_cameras()
        print(f"  ✔ {'SDK ToupTek':14s} {default_sdk_dir()} — {len(cams)} caméra(s)")
        for c in cams:
            print(f"      • {c['nom']} ({', '.join(c['résolutions'])})")
    except ToupTekSDKNotFound:
        print(f"  ✘ {'SDK ToupTek':14s} non installé (fuji touptek install …)")
    except Exception as exc:
        print(f"  ✘ {'SDK ToupTek':14s} erreur : {exc}")
    return 0


def _cmd_touptek(args: argparse.Namespace) -> int:
    from fuji.drivers.touptek import install_sdk, list_cameras

    if args.action == "install":
        if not args.source:
            print("Indique le SDK : fuji touptek install /chemin/toupcamsdk.zip")
            return 2
        dest = install_sdk(args.source)
        print(f"SDK ToupTek installé dans {dest}")
        args.action = "info"
    cams = list_cameras()
    if not cams:
        print("SDK chargé, mais aucune caméra détectée (câble USB 3 ? ToupView fermé ?)")
        return 1
    print(json.dumps(cams, indent=2, ensure_ascii=False))
    return 0


def _cmd_snap(args: argparse.Namespace) -> int:
    from fuji.core import Microscope
    from fuji.io import save_image

    with Microscope.from_file(args.profile) as scope:
        if args.exposure:
            scope.require_camera().set_exposure(args.exposure)
        frame = scope.snap()
    out = Path(args.output)
    save_image(out, frame.data, pixel_size_um=frame.metadata.pixel_size_um)
    print(f"{out} : {frame.data.shape} {frame.data.dtype}, pixel {frame.metadata.pixel_size_um or '?'} µm")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fuji", description="Fuji : microscopie et analyse")
    parser.add_argument("--profile", "-p", help="profil de microscope (nom ou fichier .yaml)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="interface graphique (par défaut)")
    sub.add_parser("profiles", help="liste les profils disponibles")
    sub.add_parser("drivers", help="liste les pilotes de périphériques")
    sub.add_parser("operations", help="liste les opérations d'analyse")
    sub.add_parser("check", help="diagnostic de l'installation")
    tp = sub.add_parser("touptek", help="SDK et caméras ToupTek")
    tp.add_argument("action", choices=["install", "info"])
    tp.add_argument("source", nargs="?", help="toupcamsdk.zip ou dossier du SDK")
    sp = sub.add_parser("snap", help="capture une image sans interface")
    sp.add_argument("-o", "--output", default="capture.ome.tif")
    sp.add_argument("-e", "--exposure", type=float, help="exposition (ms)")
    args = parser.parse_args(argv)
    if args.command == "snap" and not args.profile:
        args.profile = "demo"
    commands = {
        None: _cmd_gui,
        "gui": _cmd_gui,
        "profiles": _cmd_profiles,
        "drivers": _cmd_drivers,
        "operations": _cmd_operations,
        "check": _cmd_check,
        "touptek": _cmd_touptek,
        "snap": _cmd_snap,
    }
    return commands[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
