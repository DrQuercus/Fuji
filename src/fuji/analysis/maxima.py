"""Recherche de maxima locaux avec la règle de ``MaximumFinder`` d'ImageJ.

Un maximum est retenu s'il domine son voisinage de plus que la *proéminence* (appelée
« noise tolerance » dans les anciennes versions d'ImageJ) : en partant de lui et en ne
traversant que des pixels situés à moins de cette valeur sous lui, on ne doit atteindre
aucun pixel plus haut. Plusieurs pixels de même hauteur reliés de cette façon ne comptent
que pour un maximum, placé sur celui qui est le plus proche de leur centre.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_EIGHT = np.ones((3, 3), bool)


def find_maxima(
    plane: np.ndarray,
    prominence: float = 10.0,
    *,
    strict: bool = False,
    exclude_edges: bool = False,
    light_background: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maxima de ``plane``.

    Renvoie ``(positions, valeurs, zones)`` : les positions (ligne, colonne) triées par
    valeur décroissante, la valeur de l'image en chacune, et l'image d'étiquettes des zones
    « dans la tolérance » de chaque maximum (numéro = rang du maximum, à partir de 1).

    ``light_background`` : cherche les minima (objets sombres sur fond clair).
    ``strict`` : aucun maximum si toute l'image tient dans la tolérance ; sinon le maximum
    global est toujours retenu. ``exclude_edges`` : ignore les maxima qui touchent le bord.
    """
    from skimage.morphology import local_maxima, reconstruction

    img = np.asarray(plane, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("La recherche de maxima demande une image 2D en niveaux de gris")
    if light_background:
        img = -img
    img = np.where(np.isfinite(img), img, -np.inf if not np.isfinite(img).all() else img)
    finite = img[np.isfinite(img)]
    empty = (np.zeros((0, 2), np.int64), np.zeros(0), np.zeros(img.shape, np.int32))
    if finite.size == 0:
        return empty
    lowest, highest = float(finite.min()), float(finite.max())
    img = np.where(np.isfinite(img), img, lowest)
    if highest - lowest <= prominence:
        if strict or highest == lowest:
            return empty
        zones = np.ones(img.shape, np.int32)  # toute l'image tient dans la tolérance
    else:
        flooded = reconstruction(img - prominence, img, method="dilation", footprint=_EIGHT)
        plateaus = local_maxima(flooded, connectivity=2, allow_borders=True)
        zones, _ = ndi.label(plateaus, structure=_EIGHT)
    n = int(zones.max())
    if n == 0:
        return empty
    index = np.arange(1, n + 1)
    peak = np.asarray(ndi.maximum(img, zones, index))
    top = (zones > 0) & (img == peak[np.maximum(zones, 1) - 1])
    rr, cc = np.indices(img.shape)
    mean_r = np.asarray(ndi.mean(rr, np.where(top, zones, 0), index))
    mean_c = np.asarray(ndi.mean(cc, np.where(top, zones, 0), index))
    safe = np.maximum(zones, 1) - 1
    dist = np.where(top, (rr - mean_r[safe]) ** 2 + (cc - mean_c[safe]) ** 2, np.inf)
    positions = np.asarray(ndi.minimum_position(dist, zones, index), dtype=np.int64).reshape(-1, 2)
    keep = np.ones(n, bool)
    if exclude_edges:
        touching = top if not strict else zones > 0
        border = np.zeros(img.shape, bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        keep[np.unique(zones[touching & border]) - 1] = False
    order = np.argsort(-peak, kind="stable")
    order = order[keep[order]]
    relabel = np.zeros(n + 1, np.int32)
    relabel[order + 1] = np.arange(1, len(order) + 1)
    values = plane[positions[order, 0], positions[order, 1]] if len(order) else np.zeros(0)
    return positions[order], np.asarray(values, dtype=np.float64), relabel[zones]
