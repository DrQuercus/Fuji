"""Fuji : acquisition multi-microscopes, assistants d'alignement et analyse d'images.

Organisation du paquet :

- :mod:`fuji.core`      — abstraction du matériel (caméras, platines, tourelles...),
                          profils de microscopes, moteur d'acquisition.
- :mod:`fuji.drivers`   — pilotes : démo simulée, ToupTek, Micro-Manager, manuels.
- :mod:`fuji.assistants`— Köhler, mise au point, calibration, champ plat.
- :mod:`fuji.analysis`  — opérations d'analyse calibrées + pont vers Fiji/ImageJ.
- :mod:`fuji.io`        — écriture OME-TIFF.
- :mod:`fuji.ui`        — interface Qt/napari (optionnelle).

Le cœur ne dépend jamais de Qt : il peut tourner sans interface (scripts, tests,
ou plus tard sur un Raspberry Pi piloté à distance).
"""

__version__ = "1.0.0"
