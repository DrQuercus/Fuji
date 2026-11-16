"""Descripteurs de forme calculés **exactement comme ImageJ**.

Pourquoi ne pas utiliser directement ``skimage.measure.regionprops`` ? Parce que
ses définitions (périmètre, Feret, ellipse) diffèrent légèrement de celles d'ImageJ,
et qu'un résultat doit être identique d'un logiciel à l'autre pour être comparable
avec la littérature. Ici :

- **contour** : tracé le long des bords des pixels (comme la baguette d'ImageJ),
  objets en 8-connexité ;
- **périmètre** : ``PolygonRoi.getTracedPerimeter`` (bords droits = 1, coins
  non adjacents corrigés de 2 − √2) ;
- **ellipse** : mêmes moments qu'``EllipseFitter`` (pixel = carré, +1/12), puis mise
  à l'échelle pour avoir la même surface que l'objet ;
- **Feret** : sur les sommets du contour tracé (coins des pixels) ;
- **centroïde** : centre des pixels (+0.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# directions : droite, bas, gauche, haut (y vers le bas)
_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def trace_outline(mask: np.ndarray) -> np.ndarray:
    """Sommets du contour extérieur (coordonnées des coins de pixels, x puis y).

    ``mask`` : objet unique (booléen). Le contour suit les bords des pixels, objet à
    droite, en considérant les pixels qui se touchent par un coin comme connectés.
    """
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    rows, cols = np.nonzero(m)
    if rows.size == 0:
        return np.zeros((0, 2), np.int64)
    r0 = rows.min()
    c0 = cols[rows == r0].min()

    def px(r: int, c: int) -> bool:
        return bool(m[r, c])

    start = (int(c0), int(r0))  # coin haut-gauche du pixel le plus haut à gauche
    x, y = start[0] + 1, start[1]
    d = 0  # vers la droite
    verts = [start]
    max_steps = 4 * m.size + 8
    for _ in range(max_steps):
        if (x, y) == start:
            break
        ul, ur, ll, lr = px(y - 1, x - 1), px(y - 1, x), px(y, x - 1), px(y, x)
        # pixels devant (gauche / droite) selon la direction actuelle
        fl, fr = ((ur, lr), (lr, ll), (ll, ul), (ul, ur))[d]
        if fl:
            nd = (d - 1) % 4  # tourner à gauche (connexité 8)
        elif fr:
            nd = d  # tout droit
        else:
            nd = (d + 1) % 4  # tourner à droite
        if nd != d:
            verts.append((x, y))
        d = nd
        x += _DIRS[d][0]
        y += _DIRS[d][1]
    else:  # pragma: no cover - garde-fou
        raise RuntimeError("Contour non refermé")
    out = np.asarray(verts, dtype=np.int64) - 1  # retirer le bord ajouté
    return out


def traced_perimeter(verts: np.ndarray, px: float = 1.0) -> float:
    """``PolygonRoi.getTracedPerimeter`` d'ImageJ."""
    n = len(verts)
    if n == 0:
        return 0.0
    xp, yp = verts[:, 0], verts[:, 1]
    sumdx = sumdy = n_corners = 0
    dx1, dy1 = int(xp[0] - xp[-1]), int(yp[0] - yp[-1])
    side1 = abs(dx1) + abs(dy1)
    corner = False
    for i in range(n):
        j = (i + 1) % n
        dx2, dy2 = int(xp[j] - xp[i]), int(yp[j] - yp[i])
        sumdx += abs(dx1)
        sumdy += abs(dy1)
        side2 = abs(dx2) + abs(dy2)
        if side1 > 1 or not corner:
            corner = True
            n_corners += 1
        else:
            corner = False
        dx1, dy1, side1 = dx2, dy2, side2
    return (sumdx + sumdy) * px - n_corners * (2.0 - math.sqrt(2.0)) * px


def _hull(points: np.ndarray) -> np.ndarray:
    """Enveloppe convexe (chaîne monotone d'Andrew), sommets dans l'ordre."""
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def polygon_area(verts: np.ndarray) -> float:
    if len(verts) < 3:
        return 0.0
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass
class Feret:
    max: float
    min: float
    angle: float  # degrés, 0–180, sens trigonométrique (y vers le haut)
    x: float
    y: float


def feret(verts: np.ndarray) -> Feret:
    hull = _hull(verts)
    if len(hull) == 1:
        return Feret(0.0, 0.0, 0.0, float(hull[0, 0]), float(hull[0, 1]))
    diff = hull[:, None, :] - hull[None, :, :]
    d2 = (diff**2).sum(-1)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    dmax = math.sqrt(float(d2[i, j]))
    p1, p2 = hull[i], hull[j]
    if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] < p2[1]):
        p1, p2 = p2, p1
    angle = math.degrees(math.atan2(-(p2[1] - p1[1]), p2[0] - p1[0])) % 180.0
    # largeur minimale : pieds à coulisse tournants sur les arêtes de l'enveloppe
    dmin = dmax
    if len(hull) >= 3:
        for k in range(len(hull)):
            a, b = hull[k], hull[(k + 1) % len(hull)]
            edge = b - a
            length = math.hypot(*edge)
            if length == 0:
                continue
            rel = hull - a
            dist = np.abs(edge[0] * rel[:, 1] - edge[1] * rel[:, 0]) / length
            dmin = min(dmin, float(dist.max()))
    else:
        dmin = 0.0
    return Feret(dmax, dmin, angle, float(p1[0]), float(p1[1]))


@dataclass
class Ellipse:
    major: float
    minor: float
    angle: float  # degrés, 0–180


def fit_ellipse(rows: np.ndarray, cols: np.ndarray) -> Ellipse:
    """Ellipse de mêmes moments et de même surface (``EllipseFitter`` d'ImageJ)."""
    n = rows.size
    if n == 0:
        return Ellipse(0.0, 0.0, 0.0)
    x = cols.astype(np.float64)
    y = rows.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    cxx = ((x - xm) ** 2).mean() + 1 / 12
    cyy = ((y - ym) ** 2).mean() + 1 / 12
    cxy = ((x - xm) * (y - ym)).mean()
    tr, det = cxx + cyy, cxx * cyy - cxy**2
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-12)
    ratio = math.sqrt(l1 / l2)
    b = math.sqrt(n / (math.pi * ratio))
    a = ratio * b
    theta_down = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    angle = (-math.degrees(theta_down)) % 180.0
    if abs(cxx - cyy) < 1e-12 and abs(cxy) < 1e-12:
        angle = 0.0
    return Ellipse(2 * a, 2 * b, angle)
