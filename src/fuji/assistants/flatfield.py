"""Correction de champ plat : acquisition et stockage des références par objectif.

Les références (lame vide = « plat », obturateur fermé = « noir ») sont stockées
dans ``~/.fuji/flatfield/<microscope>/``. La correction elle-même est
:func:`fuji.analysis.ops.basic.flat_field_correct`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fuji.analysis.ops.basic import flat_field_correct

if TYPE_CHECKING:
    from fuji.core.microscope import Microscope


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "x"


def average_frames(microscope: Microscope, n: int = 8) -> np.ndarray:
    """Moyenne de ``n`` images (réduit le bruit des références)."""
    acc = None
    for _ in range(n):
        img = microscope.snap().data.astype(np.float64)
        acc = img if acc is None else acc + img
    assert acc is not None
    return (acc / n).astype(np.float32)


class FlatFieldStore:
    def __init__(self, microscope_name: str, root: str | Path | None = None) -> None:
        home = Path(os.environ.get("FUJI_HOME", Path.home() / ".fuji")).expanduser()
        self.folder = Path(root) if root else home / "flatfield" / _slug(microscope_name)

    def _path(self, objective: str | None, channel: str | None) -> Path:
        return self.folder / f"{_slug(objective or 'objectif')}__{_slug(channel or 'canal')}.npz"

    def save(
        self, objective: str | None, channel: str | None, flat: np.ndarray, dark: np.ndarray | None = None
    ) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._path(objective, channel)
        arrays = {"flat": flat.astype(np.float32)}
        if dark is not None:
            arrays["dark"] = dark.astype(np.float32)
        np.savez_compressed(path, **arrays)
        return path

    def load(self, objective: str | None, channel: str | None) -> tuple[np.ndarray, np.ndarray | None] | None:
        path = self._path(objective, channel)
        if not path.exists():
            return None
        with np.load(path) as z:
            return z["flat"], (z["dark"] if "dark" in z else None)

    def correct(self, image: np.ndarray, objective: str | None, channel: str | None) -> np.ndarray | None:
        ref = self.load(objective, channel)
        if ref is None or ref[0].shape != image.shape:
            return None
        out = flat_field_correct(image, *ref)
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            return np.clip(np.round(out), info.min, info.max).astype(image.dtype)
        return out
