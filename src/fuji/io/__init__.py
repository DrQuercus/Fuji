"""Entrées/sorties : OME-TIFF (lisible par Fiji, napari, QuPath...)."""

from .ome_tiff import OMETiffWriter, save_image

__all__ = ["OMETiffWriter", "save_image"]
