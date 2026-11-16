"""Écriture OME-TIFF au fil de l'acquisition.

- un fichier par position (``nom_p000.ome.tif``), dimensions T, C, Z, Y, X (+ RGB) ;
- écriture directe sur disque (mémoire mappée) : pas de limite de RAM, et une
  acquisition interrompue reste lisible ;
- métadonnées OME complètes : taille du pixel, pas en Z, noms des canaux, et pour
  chaque plan l'heure (DeltaT), l'exposition et la position de la platine ;
- un fichier ``.json`` à côté conserve toutes les métadonnées Fuji (objectif,
  état de chaque périphérique...).

Les fichiers s'ouvrent directement dans Fiji (Bio-Formats), napari, QuPath, Python.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import useq

from fuji.core.frame import Frame

log = logging.getLogger(__name__)

_FILE_AXES = "tcz"


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name).strip("_") or "acquisition"


@dataclass
class _PositionFile:
    path: Path
    mm: np.memmap
    axes: str  # ex. "TCZYX" ou "TZYXS"
    loop_shape: tuple[int, ...]
    planes: dict[tuple[int, ...], dict[str, Any]] = field(default_factory=dict)


class OMETiffWriter:
    """Écrit les images d'une séquence ``useq`` dans des fichiers OME-TIFF."""

    def __init__(self, directory: str | Path, name: str = "acquisition", *, overwrite: bool = False) -> None:
        self.directory = Path(directory).expanduser()
        self.name = _safe_name(name)
        self.overwrite = overwrite
        self._sequence: useq.MDASequence | None = None
        self._files: dict[tuple[int, int], _PositionFile] = {}
        self._frames_md: list[dict[str, Any]] = []
        self._channel_names: list[str] = []
        self._z_step: float | None = None
        self._interval: float | None = None

    @property
    def paths(self) -> list[Path]:
        return [f.path for f in self._files.values()]

    # -------------------------------------------------------------- séquence
    def open(self, sequence: useq.MDASequence) -> None:
        self._sequence = sequence
        self.directory.mkdir(parents=True, exist_ok=True)
        self._channel_names = [c.config for c in sequence.channels]
        step = getattr(sequence.z_plan, "step", None) if sequence.z_plan is not None else None
        self._z_step = float(step) if step else None
        t = sequence.time_plan
        interval = getattr(t, "interval", None) if t is not None else None
        self._interval = interval.total_seconds() if hasattr(interval, "total_seconds") else None

    def _loop_axes(self) -> list[str]:
        assert self._sequence is not None
        sizes = self._sequence.sizes
        return [a for a in _FILE_AXES if sizes.get(a, 0) > 0]

    def _create(self, key: tuple[int, int], frame: Frame) -> _PositionFile:
        assert self._sequence is not None
        sizes = self._sequence.sizes
        loop_axes = self._loop_axes()
        loop_shape = tuple(sizes[a] for a in loop_axes)
        data = frame.data
        rgb = frame.is_rgb
        plane_shape = data.shape
        axes = "".join(a.upper() for a in loop_axes) + ("YXS" if rgb else "YX")
        shape = (*loop_shape, *plane_shape)
        p, g = key
        suffix = f"_p{p:03d}" if sizes.get("p", 0) > 1 or sizes.get("g", 0) > 1 else ""
        if sizes.get("g", 0) > 1:
            suffix += f"_g{g:03d}"
        path = self.directory / f"{self.name}{suffix}.ome.tif"
        if path.exists() and not self.overwrite:
            i = 1
            while (alt := path.with_name(f"{self.name}{suffix}_{i}.ome.tif")).exists():
                i += 1
            path = alt
        nbytes = int(np.prod(shape)) * data.dtype.itemsize
        metadata = self._ome_metadata(axes, frame)
        mm = tifffile.memmap(
            path,
            shape=shape,
            dtype=data.dtype,
            ome=True,
            bigtiff=nbytes > 2**31,
            photometric="rgb" if rgb else "minisblack",
            metadata=metadata,
        )
        pf = _PositionFile(path, mm, axes, loop_shape)
        self._files[key] = pf
        return pf

    def _ome_metadata(self, axes: str, frame: Frame) -> dict[str, Any]:
        md = frame.metadata
        out: dict[str, Any] = {"axes": axes, "Name": self.name}
        if md.pixel_size_um:
            out.update(
                PhysicalSizeX=md.pixel_size_um,
                PhysicalSizeXUnit="µm",
                PhysicalSizeY=md.pixel_size_um,
                PhysicalSizeYUnit="µm",
            )
        if "Z" in axes and self._z_step:
            out.update(PhysicalSizeZ=self._z_step, PhysicalSizeZUnit="µm")
        if "T" in axes and self._interval:
            out.update(TimeIncrement=self._interval, TimeIncrementUnit="s")
        if "C" in axes and self._channel_names:
            out["Channel"] = {"Name": self._channel_names}
        return out

    # ----------------------------------------------------------------- écriture
    def write(self, event: useq.MDAEvent, frame: Frame) -> None:
        index = dict(event.index)
        key = (index.get("p", 0), index.get("g", 0))
        pf = self._files.get(key) or self._create(key, frame)
        loop_idx = tuple(index.get(a, 0) for a in self._loop_axes())
        pf.mm[loop_idx] = frame.data
        md = frame.metadata
        pf.planes[loop_idx] = {
            "DeltaT": md.elapsed_s,
            "ExposureTime": md.exposure_ms,
            "PositionX": md.x_um,
            "PositionY": md.y_um,
            "PositionZ": md.z_um,
        }
        self._frames_md.append({"file": pf.path.name, "index": index, **md.to_dict()})

    def close(self) -> None:
        for pf in self._files.values():
            pf.mm.flush()
            dtype, shape = pf.mm.dtype, pf.mm.shape
            del pf.mm
            try:
                self._write_planes(pf, dtype, shape)
            except Exception:  # pragma: no cover - les images restent valides
                log.exception("Métadonnées par plan non écrites pour %s", pf.path)
        if self._frames_md:
            sidecar = self.directory / f"{self.name}_metadata.json"
            payload = {
                "sequence": json.loads(self._sequence.model_dump_json()) if self._sequence else None,
                "files": [p.name for p in self.paths],
                "frames": self._frames_md,
            }
            sidecar.write_text(
                json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
            )

    def _write_planes(self, pf: _PositionFile, dtype: np.dtype, shape: tuple[int, ...]) -> None:
        """Réécrit l'OME-XML avec une entrée par plan (temps, exposition, position)."""
        n_planes = int(np.prod(pf.loop_shape)) if pf.loop_shape else 1
        order = list(np.ndindex(*pf.loop_shape)) if pf.loop_shape else [()]
        keys = ("DeltaT", "ExposureTime", "PositionX", "PositionY", "PositionZ")
        plane_md: dict[str, list[Any]] = {}
        for k in keys:
            values = [pf.planes.get(idx, {}).get(k) for idx in order]
            if all(v is not None for v in values):
                plane_md[k] = [float(v) for v in values]
        if not plane_md:
            return
        units = {"DeltaT": "s", "ExposureTime": "ms", "PositionX": "µm", "PositionY": "µm", "PositionZ": "µm"}
        for k in list(plane_md):
            plane_md[f"{k}Unit"] = [units[k]] * n_planes
        rgb = pf.axes.endswith("S")
        h, w = (shape[-3], shape[-2]) if rgb else (shape[-2], shape[-1])
        stored = (n_planes, 1, 1, h, w, shape[-1] if rgb else 1)
        with tifffile.TiffFile(pf.path) as tif:
            first = tif.ome_metadata or ""
        base = {k: v for k, v in _parse_basic(first).items()}
        base["axes"] = pf.axes
        base["Plane"] = plane_md
        ome = tifffile.OmeXml()
        ome.addimage(dtype, shape, stored, **base)
        tifffile.tiffcomment(pf.path, ome.tostring().encode())

    def __enter__(self) -> OMETiffWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _parse_basic(xml: str) -> dict[str, Any]:
    """Récupère les attributs principaux de l'OME-XML initial (pour le réécrire)."""
    out: dict[str, Any] = {}
    m = re.search(r"<Image [^>]*Name=\"([^\"]*)\"", xml)
    if m:
        out["Name"] = m.group(1)
    for key in ("PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ", "TimeIncrement"):
        m = re.search(rf'{key}="([^"]+)"', xml)
        if m:
            out[key] = float(m.group(1))
            u = re.search(rf'{key}Unit="([^"]+)"', xml)
            if u:
                out[f"{key}Unit"] = u.group(1)
    names = re.findall(r"<Channel [^>]*Name=\"([^\"]*)\"", xml)
    if names and any(n for n in names):
        out["Channel"] = {"Name": names}
    return out


def save_image(
    path: str | Path,
    data: np.ndarray,
    *,
    pixel_size_um: float | None = None,
    axes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Enregistre une image (ou une pile) en OME-TIFF avec sa calibration."""
    path = Path(path)
    rgb = data.ndim >= 3 and data.shape[-1] in (3, 4) and data.dtype in (np.uint8, np.uint16)
    if axes is None:
        base = "YXS" if rgb else "YX"
        extra = data.ndim - len(base)
        axes = "TCZ"[3 - extra :] + base if extra > 0 else base
    md: dict[str, Any] = {"axes": axes}
    if pixel_size_um:
        md.update(
            PhysicalSizeX=pixel_size_um,
            PhysicalSizeXUnit="µm",
            PhysicalSizeY=pixel_size_um,
            PhysicalSizeYUnit="µm",
        )
    if metadata:
        md.update(metadata)
    tifffile.imwrite(path, data, ome=True, photometric="rgb" if rgb else "minisblack", metadata=md)
    return path
