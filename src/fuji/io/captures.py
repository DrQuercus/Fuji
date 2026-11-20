"""Images prises au microscope : enregistrement automatique et liste du jour.

Chaque capture est écrite tout de suite dans un dossier daté (``~/Fuji/Captures/2026-11-17``),
en OME-TIFF avec sa calibration et ses métadonnées : rien ne se perd, même si Fuji se ferme
brutalement. Une vignette et un petit résumé (``.vignettes/<nom>.png`` et ``.json``)
permettent d'afficher la liste sans relire les images entières.

Indépendant de l'interface : utilisable depuis un script.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fuji.core.frame import is_rgb
from fuji.text import number

from .ome_tiff import save_image

#: dossier des vignettes et des résumés, à côté des images
THUMBS = ".vignettes"
THUMB_SIDE = 192
SUFFIX = ".ome.tif"
_NAME = re.compile(r"^[^/\\:*?\"<>|]+$")

#: types d'images et préfixe de leur nom de fichier
KINDS = {
    "capture": "capture",
    "edf": "profondeur",
    "panorama": "panorama",
    "mosaique": "mosaique",
    "acquisition": "acquisition",
}


def default_root() -> Path:
    """``~/Fuji/Captures``, ou le dossier de la variable ``FUJI_CAPTURES`` (tests, essais)."""
    override = os.environ.get("FUJI_CAPTURES")
    return Path(override).expanduser() if override else Path.home() / "Fuji" / "Captures"


def thumbnail(data: np.ndarray, side: int = THUMB_SIDE) -> np.ndarray:
    """Vignette 8 bits (contraste étalé), plus grand côté ≤ ``side`` : quelques millisecondes,
    même pour une image 4K (sous-échantillonnage, pas de calcul sur l'image entière)."""
    a = np.asarray(data)
    rgb = is_rgb(a)
    while a.ndim > (3 if rgb else 2):  # pile : plan du milieu
        a = a[a.shape[0] // 2]
    if rgb:
        a = a[..., :3]
    step = max(1, int(np.ceil(max(a.shape[:2]) / side)))
    small = a[::step, ::step].astype(np.float32)
    lo, hi = np.percentile(small, (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((small - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def describe(metadata: dict[str, Any] | None) -> str:
    """Résumé d'une ligne : objectif · canal · exposition."""
    md = metadata or {}
    parts = [str(md[k]) for k in ("objective", "channel") if md.get(k)]
    if md.get("exposure_ms"):
        exposure = float(md["exposure_ms"])
        parts.append(f"{number(exposure, 0 if exposure >= 10 else 1)} ms")
    if md.get("note"):
        parts.append(str(md["note"]))
    return " · ".join(parts)


@dataclass
class CaptureInfo:
    """Une image de la liste : où elle est et ce qu'elle contient."""

    path: Path
    kind: str = "capture"
    created: float = field(default_factory=time.time)
    summary: str = ""
    shape: tuple[int, ...] = ()
    pixel_size_um: float | None = None
    #: métadonnées de la prise de vue (objectif, canal, exposition, positions...)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name[: -len(SUFFIX)] if self.path.name.endswith(SUFFIX) else self.path.stem

    @property
    def thumbnail_path(self) -> Path:
        return self.path.parent / THUMBS / f"{self.name}.png"

    @property
    def info_path(self) -> Path:
        return self.path.parent / THUMBS / f"{self.name}.json"

    @property
    def time_text(self) -> str:
        return datetime.fromtimestamp(self.created).strftime("%H:%M:%S")

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = self.path.name
        out["shape"] = list(self.shape)
        return out


class CaptureStore:
    """Dossier des images prises, rangées par jour."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root else default_root()

    def day_folder(self, when: datetime | None = None) -> Path:
        return self.root / (when or datetime.now()).strftime("%Y-%m-%d")

    def next_name(self, prefix: str = "capture", folder: Path | None = None) -> str:
        """``capture_001``, ``capture_002``... sans jamais écraser un fichier existant."""
        folder = folder or self.day_folder()
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)")
        used = [int(m.group(1)) for p in folder.glob(f"{prefix}_*") if (m := pattern.match(p.name))]
        return f"{prefix}_{max(used, default=0) + 1:03d}"

    def save(
        self,
        data: np.ndarray,
        *,
        kind: str = "capture",
        metadata: dict[str, Any] | None = None,
        pixel_size_um: float | None = None,
        name: str | None = None,
        summary: str | None = None,
        thumb: np.ndarray | None = None,
    ) -> CaptureInfo:
        """Écrit l'image (OME-TIFF), sa vignette et son résumé dans le dossier du jour."""
        folder = self.day_folder()
        (folder / THUMBS).mkdir(parents=True, exist_ok=True)
        if name is None:
            name = self.next_name(KINDS.get(kind, kind), folder)
        path = folder / f"{name}{SUFFIX}"
        if path.exists():
            raise FileExistsError(f"{path.name} existe déjà")
        save_image(path, data, pixel_size_um=pixel_size_um, metadata=_ome_safe(metadata))
        info = CaptureInfo(
            path=path,
            kind=kind,
            summary=describe(metadata) if summary is None else summary,
            shape=tuple(int(s) for s in np.shape(data)),
            pixel_size_um=pixel_size_um,
            metadata=json.loads(json.dumps(metadata or {}, default=str)),
        )
        self.write_sidecars(info, data if thumb is None else None, thumb)
        return info

    def write_sidecars(
        self, info: CaptureInfo, data: np.ndarray | None, thumb: np.ndarray | None = None
    ) -> None:
        from PIL import Image as PILImage

        info.thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        if thumb is None and data is not None:
            thumb = thumbnail(data)
        if thumb is not None:
            PILImage.fromarray(thumb).save(info.thumbnail_path)
        info.info_path.write_text(json.dumps(info.to_json(), ensure_ascii=False), encoding="utf-8")

    def add_existing(self, path: str | Path, kind: str = "acquisition", summary: str = "") -> CaptureInfo:
        """Ajoute à la liste une image déjà écrite ailleurs (acquisition), sans la recopier."""
        path = Path(path)
        return CaptureInfo(path=path, kind=kind, created=path.stat().st_mtime, summary=summary)

    def list(self, folder: Path | None = None) -> list[CaptureInfo]:
        """Images d'un dossier (celui du jour par défaut), de la plus ancienne à la plus récente."""
        folder = folder or self.day_folder()
        if not folder.is_dir():
            return []
        out = []
        for path in folder.iterdir():
            if not path.is_file() or not path.name.lower().endswith((".tif", ".tiff")):
                continue
            info = CaptureInfo(path=path, created=path.stat().st_mtime)
            try:
                saved = json.loads(info.info_path.read_text(encoding="utf-8"))
                info.kind = saved.get("kind", "capture")
                info.summary = saved.get("summary", "")
                info.shape = tuple(saved.get("shape", ()))
                info.pixel_size_um = saved.get("pixel_size_um")
                info.created = float(saved.get("created", info.created))
                info.metadata = dict(saved.get("metadata") or {})
            except (OSError, ValueError):
                pass  # image déposée à la main : pas de résumé
            out.append(info)
        return sorted(out, key=lambda i: i.created)

    def rename(self, info: CaptureInfo, new_name: str) -> CaptureInfo:
        """Renomme l'image et ses fichiers associés (refuse un nom déjà pris ou invalide)."""
        new_name = new_name.strip()
        if new_name.endswith(SUFFIX):
            new_name = new_name[: -len(SUFFIX)]
        if not new_name or not _NAME.match(new_name) or new_name.startswith("."):
            raise ValueError(f"Nom invalide : « {new_name} »")
        target = info.path.with_name(f"{new_name}{SUFFIX}")
        if target.exists():
            raise FileExistsError(f"« {new_name} » existe déjà")
        old_thumb, old_info = info.thumbnail_path, info.info_path
        info.path.rename(target)
        renamed = CaptureInfo(
            target, info.kind, info.created, info.summary, info.shape, info.pixel_size_um, info.metadata
        )
        for old, new in ((old_thumb, renamed.thumbnail_path), (old_info, renamed.info_path)):
            if old.exists():
                old.rename(new)
        if renamed.info_path.exists():
            renamed.info_path.write_text(json.dumps(renamed.to_json(), ensure_ascii=False), encoding="utf-8")
        return renamed

    def forget(self, info: CaptureInfo) -> None:
        """Retire la vignette et le résumé (l'image elle-même est supprimée par l'appelant)."""
        for path in (info.thumbnail_path, info.info_path):
            try:
                path.unlink()
            except OSError:
                pass


def _ome_safe(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Métadonnées enregistrables dans l'OME-TIFF : valeurs simples seulement."""
    if not metadata:
        return None
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) and value is not None:
            out[key] = value
        elif isinstance(value, dict) and value:
            out[key] = json.dumps(value, ensure_ascii=False, default=str)
    return out
