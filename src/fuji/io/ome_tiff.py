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

import glob
import json
import logging
import re
import time
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
    mm: np.memmap | None
    axes: str  # ex. "TCZYX" ou "TZYXS"
    loop_shape: tuple[int, ...]
    metadata: dict[str, Any]
    dtype: np.dtype
    shape: tuple[int, ...]
    planes: dict[tuple[int, ...], dict[str, Any]] = field(default_factory=dict)


class OMETiffWriter:
    """Écrit les images d'une séquence ``useq`` dans des fichiers OME-TIFF.

    Une acquisition n'écrase jamais la précédente : si le nom est déjà pris dans le
    dossier, un suffixe ``_1``, ``_2``... est ajouté à tous ses fichiers (images et
    métadonnées), sauf si ``overwrite=True``.
    """

    def __init__(
        self,
        directory: str | Path,
        name: str = "acquisition",
        *,
        overwrite: bool = False,
        flush_interval_s: float = 10.0,
    ) -> None:
        self.directory = Path(directory).expanduser()
        self.name = _safe_name(name)
        self.overwrite = overwrite
        self.flush_interval_s = flush_interval_s
        self._base = self.name
        self._sequence: useq.MDASequence | None = None
        self._files: dict[tuple[int, int], _PositionFile] = {}
        self._frames_md: list[dict[str, Any]] = []
        self._channel_names: list[str] = []
        self._z_step: float | None = None
        self._interval: float | None = None
        self._last_flush = time.monotonic()
        self._closed = False

    @property
    def paths(self) -> list[Path]:
        return [f.path for f in self._files.values()]

    @property
    def metadata_path(self) -> Path:
        return self.directory / f"{self._base}_metadata.json"

    # -------------------------------------------------------------- séquence
    def open(self, sequence: useq.MDASequence) -> None:
        self._sequence = sequence
        self.directory.mkdir(parents=True, exist_ok=True)
        self._base = self.name if self.overwrite else self._free_base()
        self._channel_names = [c.config for c in sequence.channels]
        step = getattr(sequence.z_plan, "step", None) if sequence.z_plan is not None else None
        self._z_step = float(step) if step else None
        t = sequence.time_plan
        interval = getattr(t, "interval", None) if t is not None else None
        self._interval = interval.total_seconds() if hasattr(interval, "total_seconds") else None

    def _free_base(self) -> str:
        """Premier nom (``nom``, ``nom_1``...) dont aucun fichier n'existe encore."""

        def taken(base: str) -> bool:
            if (self.directory / f"{base}.ome.tif").exists():
                return True
            if (self.directory / f"{base}_metadata.json").exists():
                return True
            return any(self.directory.glob(f"{glob.escape(base)}_p[0-9][0-9][0-9]*.ome.tif"))

        base, i = self.name, 1
        while taken(base):
            base = f"{self.name}_{i}"
            i += 1
        return base

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
        axes = "".join(a.upper() for a in loop_axes) + ("YXS" if rgb else "YX")
        shape = (*loop_shape, *data.shape)
        p, g = key
        multi_p, multi_g = sizes.get("p", 0) > 1, sizes.get("g", 0) > 1
        suffix = f"_p{p:03d}" if multi_p or multi_g else ""
        if multi_g:
            suffix += f"_g{g:03d}"
        path = self.directory / f"{self._base}{suffix}.ome.tif"
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
        pf = _PositionFile(path, mm, axes, loop_shape, metadata, data.dtype, shape)
        self._files[key] = pf
        return pf

    def _ome_metadata(self, axes: str, frame: Frame) -> dict[str, Any]:
        md = frame.metadata
        out: dict[str, Any] = {"axes": axes, "Name": self._base}
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
        if self._closed:
            raise RuntimeError("Écriture après la fermeture du fichier")
        index = {str(getattr(k, "value", k)): int(v) for k, v in event.index.items()}
        key = (index.get("p", 0), index.get("g", 0))
        pf = self._files.get(key) or self._create(key, frame)
        assert pf.mm is not None
        if frame.data.shape != pf.shape[len(pf.loop_shape) :]:
            raise ValueError(
                f"La taille de l'image a changé pendant l'acquisition : {frame.data.shape} au lieu de "
                f"{pf.shape[len(pf.loop_shape) :]} (résolution ou profondeur modifiée ?)"
            )
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
        self._frames_md.append({"file": pf.path.name, **md.to_dict(), "index": index})
        now = time.monotonic()
        if now - self._last_flush > self.flush_interval_s:  # longues acquisitions : sécuriser
            self._last_flush = now
            for f in self._files.values():
                if f.mm is not None:
                    f.mm.flush()

    def close(self) -> None:
        """Termine les fichiers. Peut être appelée plusieurs fois sans risque."""
        if self._closed:
            return
        self._closed = True
        for pf in self._files.values():
            if pf.mm is not None:
                pf.mm.flush()
                pf.mm = None  # libère le fichier avant de réécrire ses métadonnées
            try:
                self._write_planes(pf)
            except Exception:  # pragma: no cover - les images restent valides
                log.exception("Métadonnées par plan non écrites pour %s", pf.path)
        if self._frames_md:
            payload = {
                "sequence": json.loads(self._sequence.model_dump_json()) if self._sequence else None,
                "files": [p.name for p in self.paths],
                "frames": self._frames_md,
            }
            self.metadata_path.write_text(
                json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
            )

    def _write_planes(self, pf: _PositionFile) -> None:
        """Réécrit l'OME-XML avec une entrée par plan (temps, exposition, position).

        Seulement si tous les plans ont été acquis ; sinon (acquisition interrompue)
        l'en-tête d'origine reste valide et le fichier JSON contient le détail.
        """
        n_planes = int(np.prod(pf.loop_shape)) if pf.loop_shape else 1
        order = list(np.ndindex(*pf.loop_shape)) if pf.loop_shape else [()]
        units = {"DeltaT": "s", "ExposureTime": "ms", "PositionX": "µm", "PositionY": "µm", "PositionZ": "µm"}
        plane_md: dict[str, list[Any]] = {}
        for k, unit in units.items():
            values = [pf.planes.get(idx, {}).get(k) for idx in order]
            if all(v is not None for v in values):
                plane_md[k] = [float(v) for v in values]
                plane_md[f"{k}Unit"] = [unit] * n_planes
        if not plane_md:
            return
        rgb = pf.axes.endswith("S")
        shape = pf.shape
        h, w = (shape[-3], shape[-2]) if rgb else (shape[-2], shape[-1])
        stored = (n_planes, 1, 1, h, w, shape[-1] if rgb else 1)
        ome = tifffile.OmeXml()
        ome.addimage(pf.dtype, shape, stored, **{**pf.metadata, "Plane": plane_md})
        tifffile.tiffcomment(pf.path, ome.tostring().encode())

    def __enter__(self) -> OMETiffWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _is_rgb(data: np.ndarray) -> bool:
    return data.ndim >= 3 and data.shape[-1] in (3, 4) and data.shape[-2] > 4


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
    rgb = _is_rgb(data)
    if axes is None:
        base = "YXS" if rgb else "YX"
        extra = data.ndim - len(base)
        if not 0 <= extra <= 3:
            raise ValueError(f"Image à {data.ndim} dimensions : précisez l'argument axes")
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
