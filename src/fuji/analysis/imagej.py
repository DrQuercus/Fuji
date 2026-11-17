"""Pont vers Fiji / ImageJ.

Deux façons d'utiliser Fiji depuis Fuji :

1. **Dans le même programme** (PyImageJ) : exécuter une macro ImageJ, une commande
   de menu ou un plugin sur une image Fuji et récupérer l'image et le tableau
   « Results ». C'est ce qui garantit d'avoir *toute* la puissance de Fiji.
   On utilise de préférence ton installation Fiji (avec tes plugins) :
   variable ``FUJI_FIJI_PATH`` ou emplacement standard (``/Applications/Fiji.app``).

2. **Ouvrir dans Fiji** : l'image est enregistrée en TIFF ImageJ calibré et ouverte
   dans l'application Fiji, pour travailler à la main.

Le démarrage de Java prend quelques secondes : il n'a lieu qu'au premier usage.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .image import Image, Table

log = logging.getLogger(__name__)

_FIJI_CANDIDATES = (
    "/Applications/Fiji.app",
    "~/Applications/Fiji.app",
    "~/Fiji.app",
    "~/fiji",
    "C:/Fiji.app",
    "~/Desktop/Fiji.app",
)


def find_fiji() -> Path | None:
    """Emplacement d'une installation Fiji locale, si elle existe."""
    env = os.environ.get("FUJI_FIJI_PATH")
    candidates = [env] if env else []
    candidates += list(_FIJI_CANDIDATES)
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_dir() and ((p / "jars").is_dir() or (p / "Contents").is_dir()):
            return p
    return None


@dataclass
class MacroResult:
    image: Image | None
    table: Table | None
    log: str


class ImageJBridge:
    """Une seule instance par programme (Java ne démarre qu'une fois)."""

    _instance: ImageJBridge | None = None
    _lock = threading.Lock()

    def __init__(self, target: str | Path | None = None, mode: str = "headless") -> None:
        self.target = str(target) if target else None
        self.mode = mode
        self.ij: Any = None

    @classmethod
    def get(cls) -> ImageJBridge:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # --------------------------------------------------------------- démarrage
    @property
    def ready(self) -> bool:
        return self.ij is not None

    def start(self) -> Any:
        if self.ij is not None:
            return self.ij
        try:
            import imagej
        except ImportError as exc:
            raise ImportError(
                'Le pont Fiji nécessite PyImageJ : pip install "fuji-microscopy[imagej]"'
            ) from exc
        target = self.target or (str(find_fiji()) if find_fiji() else "sc.fiji:fiji")
        log.info("Démarrage d'ImageJ (%s, mode %s)...", target, self.mode)
        self.ij = imagej.init(target, mode=self.mode, add_legacy=True)
        return self.ij

    def version(self) -> str:
        return str(self.start().getVersion())

    # --------------------------------------------------------------- conversions
    def to_imageplus(self, image: Image) -> Any:
        ij = self.start()
        data = np.ascontiguousarray(image.data)
        rgb8 = image.is_rgb and data.dtype == np.uint8
        if rgb8:
            import jpype
            from scyjava import jimport

            ColorProcessor = jimport("ij.process.ColorProcessor")
            ImagePlus = jimport("ij.ImagePlus")
            h, w = data.shape[:2]
            packed = (
                (data[..., 0].astype(np.int32) << 16)
                | (data[..., 1].astype(np.int32) << 8)
                | data[..., 2].astype(np.int32)
            ) | np.int32(-16777216)
            cp = ColorProcessor(w, h, jpype.JArray(jpype.JInt)(packed.ravel()))
            imp = ImagePlus(image.name, cp)
        else:
            if image.is_rgb:  # RGB 16 bits : pile de 3 canaux
                data = np.moveaxis(data, -1, 0)
            imp = ij.py.to_imageplus(data)
            imp.setTitle(image.name)
        if image.pixel_size_um:
            cal = imp.getCalibration()
            cal.pixelWidth = cal.pixelHeight = float(image.pixel_size_um)
            cal.setUnit("micron")
        return imp

    def from_imageplus(self, imp: Any, name: str | None = None) -> Image:
        ij = self.start()
        if int(imp.getType()) == 4:  # COLOR_RGB
            pixels = np.asarray(imp.getProcessor().getPixels(), dtype=np.int64)
            h, w = int(imp.getHeight()), int(imp.getWidth())
            pixels = pixels.reshape(h, w)
            data = np.stack([(pixels >> 16) & 255, (pixels >> 8) & 255, pixels & 255], -1)
            data = data.astype(np.uint8)
        else:
            data = np.asarray(ij.py.from_java(imp))
        cal = imp.getCalibration()
        px = float(cal.pixelWidth) if cal.scaled() else None
        return Image(data, px, name or str(imp.getTitle()))

    @staticmethod
    def results_to_table(rt: Any, name: str = "Résultats ImageJ") -> Table | None:
        if rt is None or int(rt.size()) == 0:
            return None
        cols: dict[str, Any] = {}
        for heading in rt.getHeadings():
            heading = str(heading)
            if heading == " ":
                continue
            idx = rt.getColumnIndex(heading)
            if idx >= 0 and rt.columnExists(idx):
                cols[heading] = np.asarray(rt.getColumn(idx), dtype=np.float64)
            else:
                cols[heading] = [str(rt.getStringValue(heading, i)) for i in range(rt.size())]
        df = pd.DataFrame(cols)
        df.index = pd.RangeIndex(1, len(df) + 1)
        return Table(df, name=name)

    # ---------------------------------------------------------------- exécution
    def run_macro(self, macro: str, image: Image | None = None, args: str = "") -> MacroResult:
        """Exécute une macro ImageJ (langage IJM) sur ``image``.

        L'image devient l'image courante de la macro ; l'image courante à la fin et
        le tableau « Results » sont renvoyés.
        """
        self.start()
        from scyjava import jimport

        IJ = jimport("ij.IJ")
        WindowManager = jimport("ij.WindowManager")
        ResultsTable = jimport("ij.measure.ResultsTable")
        Interpreter = jimport("ij.macro.Interpreter")
        rt = ResultsTable.getResultsTable()
        if rt is not None:
            rt.reset()
        IJ.log("\\Clear")
        imp = None
        if image is not None:
            imp = self.to_imageplus(image)
            WindowManager.setTempCurrentImage(imp)
        Interpreter.batchMode = True
        try:
            IJ.runMacro("setBatchMode(true);\n" + macro, args)
            out_imp = WindowManager.getCurrentImage() or imp
        finally:
            Interpreter.batchMode = False
        result_image = self.from_imageplus(out_imp) if out_imp is not None else None
        table = self.results_to_table(ResultsTable.getResultsTable())
        return MacroResult(result_image, table, str(IJ.getLog() or ""))

    def run_command(self, command: str, options: str = "", image: Image | None = None) -> MacroResult:
        """Exécute une commande de menu ImageJ, ex. ``("Gaussian Blur...", "sigma=2")``."""
        esc = command.replace('"', '\\"')
        opts = options.replace('"', '\\"')
        return self.run_macro(f'run("{esc}", "{opts}");', image)


# ------------------------------------------------------------ ouvrir dans Fiji
def save_imagej_tiff(image: Image, path: str | Path) -> Path:
    """TIFF au format ImageJ avec calibration (ouvert tel quel par Fiji)."""
    import tifffile

    path = Path(path)
    data = image.data
    kwargs: dict[str, Any] = {"imagej": True}
    if image.pixel_size_um:
        res = 1.0 / float(image.pixel_size_um)
        kwargs.update(resolution=(res, res), metadata={"unit": "um"})
    if image.is_rgb and data.dtype == np.uint8:
        kwargs["photometric"] = "rgb"
    elif image.is_rgb:
        data = np.moveaxis(data, -1, 0)
        kwargs.setdefault("metadata", {})["axes"] = "CYX"
    if data.dtype not in (np.uint8, np.uint16, np.float32):
        data = data.astype(np.float32)
    tifffile.imwrite(path, data, **kwargs)
    return path


def open_in_fiji(image: Image, fiji: str | Path | None = None) -> Path:
    """Enregistre l'image dans un fichier temporaire et l'ouvre dans Fiji."""
    folder = Path(tempfile.gettempdir()) / "fuji-vers-fiji"
    folder.mkdir(exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in image.name) or "image"
    path = save_imagej_tiff(image, folder / f"{safe}.tif")
    app = Path(fiji).expanduser() if fiji else find_fiji()
    if sys.platform == "darwin":
        cmd = ["open", "-a", str(app) if app else "Fiji", str(path)]
    elif app is None:
        raise FileNotFoundError(
            "Fiji introuvable : installez-le, puis indiquez son dossier dans Préférences → Fiji / ImageJ"
        )
    else:
        exe = _fiji_executable(app)
        cmd = [str(exe), str(path)]
    subprocess.Popen(cmd)
    return path


def _fiji_executable(app: Path) -> Path:
    machine = platform.machine().lower()
    names = (
        ["fiji-windows-x64.exe", "ImageJ-win64.exe"]
        if sys.platform.startswith("win")
        else [
            "fiji-linux-arm64" if "arm" in machine or "aarch" in machine else "fiji-linux-x64",
            "ImageJ-linux64",
        ]
    )
    for n in names:
        if (app / n).exists():
            return app / n
    raise FileNotFoundError(f"Exécutable Fiji introuvable dans {app}")
