"""Images prises : enregistrement automatique, liste du jour, renommage."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from fuji.io.captures import CaptureStore, default_root, describe, thumbnail


def _image(shape=(120, 160, 3)) -> np.ndarray:
    return np.random.default_rng(0).integers(100, 4000, shape, dtype=np.uint16)


def test_default_folder_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FUJI_CAPTURES", str(tmp_path / "ici"))
    assert default_root() == tmp_path / "ici"
    monkeypatch.delenv("FUJI_CAPTURES")
    assert default_root().parts[-2:] == ("Fuji", "Captures")


def test_captures_are_saved_with_calibration_and_listed(tmp_path):
    store = CaptureStore(tmp_path)
    md = {"objective": "40x/0.65", "channel": "Fond clair", "exposure_ms": 12.0, "devices": {"lampe": "on"}}
    first = store.save(_image(), metadata=md, pixel_size_um=0.0725)
    second = store.save(_image((80, 90)), kind="edf", metadata={"objective": "10x/0.25", "note": "12 images"})
    assert first.path.parent == store.day_folder() and first.name == "capture_001"
    assert second.name == "profondeur_001" and store.next_name() == "capture_002"
    assert first.summary == "40x/0.65 · Fond clair · 12 ms" and second.summary == "10x/0.25 · 12 images"
    with tifffile.TiffFile(first.path) as tif:
        assert tif.series[0].axes == "YXS" and tif.series[0].shape == (120, 160, 3)
        assert "0.0725" in tif.ome_metadata  # taille du pixel enregistrée
    listed = store.list()
    assert [i.name for i in listed] == ["capture_001", "profondeur_001"]
    assert listed[0].metadata["objective"] == "40x/0.65" and listed[0].pixel_size_um == 0.0725
    assert listed[0].thumbnail_path.exists() and listed[1].kind == "edf"


def test_names_are_never_reused_and_renaming_is_checked(tmp_path):
    store = CaptureStore(tmp_path)
    info = store.save(_image())
    with pytest.raises(FileExistsError):
        store.save(_image(), name="capture_001")
    renamed = store.rename(info, "lame 3 — zone B")
    assert renamed.path.name == "lame 3 — zone B.ome.tif" and renamed.thumbnail_path.exists()
    assert not info.path.exists() and store.list()[0].name == "lame 3 — zone B"
    other = store.save(_image())
    assert other.name == "capture_001"  # le nom est libre à nouveau
    for bad in ("", "a/b", ".cache", "x:y"):
        with pytest.raises(ValueError):
            store.rename(other, bad)
    with pytest.raises(FileExistsError):
        store.rename(other, "lame 3 — zone B")


def test_thumbnails_and_summaries():
    small = thumbnail(np.zeros((2160, 3840, 3), np.uint16) + np.arange(3840, dtype=np.uint16)[:, None])
    assert small.dtype == np.uint8 and max(small.shape[:2]) <= 192 and small.shape[2] == 3
    assert thumbnail(np.ones((5, 64, 64), np.uint16)).shape == (64, 64)  # pile : plan du milieu
    assert describe({"exposure_ms": 0.5, "note": "panorama de 4 champs"}) == "0.5 ms · panorama de 4 champs"
    assert describe(None) == ""
