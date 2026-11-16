"""Analyse d'images calibrée (définitions d'ImageJ) + pont vers Fiji.

Exemple ::

    from fuji.analysis import Image, run, list_operations

    img = Image.from_file("cellules.ome.tif")
    mask = run("Seuillage automatique", img, method="Otsu", objects="sombres")
    table, objets = run("Analyse de particules", mask, intensity=img, min_area_um2=20)
    table.to_csv("resultats.csv")
"""

from .image import Image, Labels, Table
from .registry import Operation, Param, categories, get_operation, list_operations, operation, run

__all__ = [
    "Image",
    "Labels",
    "Operation",
    "Param",
    "Table",
    "categories",
    "get_operation",
    "list_operations",
    "operation",
    "run",
]
