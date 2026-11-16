"""Descripteurs de forme calculés **exactement comme ImageJ**.

Pourquoi ne pas utiliser directement ``skimage.measure.regionprops`` ? Parce que
ses définitions (périmètre, Feret, ellipse) diffèrent légèrement de celles d'ImageJ,
et qu'un résultat doit être identique d'un logiciel à l'autre pour être comparable
avec la littérature. Ici :

- **contour** : tracé le long des bords des pixels, objets en 8-connexité, dans le
  même sens et à partir du même sommet que la baguette d'ImageJ (``Wand.traceEdge``).
  Le sommet de départ compte : le périmètre en dépend dans les « escaliers » ;
- **périmètre** : ``PolygonRoi.getTracedPerimeter`` (bords droits = 1, coins
  non adjacents corrigés de 2 − √2) ;
- **ellipse** : transcription d'``EllipseFitter`` (pixel = carré, +1/12, même surface
  que l'objet, mêmes conventions d'angle) ;
- **Feret** : sur l'enveloppe convexe du contour, parcourue dans l'ordre d'ImageJ pour
  que l'angle soit le même quand plusieurs diamètres sont égaux (formes symétriques) ;
- **centroïde** : centre des pixels (+0.5).

Ces fonctions transcrivent le code source d'ImageJ ; le test
``tests/test_imagej_parity.py`` les compare au vrai ImageJ quand il est disponible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# directions : droite, bas, gauche, haut (y vers le bas)
_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def trace_outline(mask: np.ndarray) -> np.ndarray:
    """Sommets du contour extérieur (coordonnées des coins de pixels, x puis y).

    ``mask`` : objet unique (booléen). Le contour suit les bords des pixels dans le sens
    des aiguilles d'une montre (objet à droite), en considérant les pixels qui se touchent
    par un coin comme connectés. Seuls les changements de direction sont des sommets.
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
    # ImageJ part du bord droit du premier segment de la ligne du haut et descend : son
    # premier sommet est donc le troisième du nôtre (après les deux coins du haut).
    return np.roll(out, -2, axis=0)


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
    """Enveloppe convexe stricte (sans points alignés), dans l'ordre d'ImageJ : départ au
    sommet le plus haut (puis le plus à gauche), sens inverse des aiguilles d'une montre à
    l'écran (``FloatPolygon.getConvexHull``, algorithme du papier cadeau)."""
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) <= 2:
        return pts[np.lexsort((pts[:, 0], pts[:, 1]))]
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
    hull = np.asarray(lower[:-1] + upper[:-1])[::-1]  # chaîne monotone : sens horaire → inversé
    if len(hull) < 3:  # points alignés
        return hull[np.lexsort((hull[:, 0], hull[:, 1]))]
    start = int(np.lexsort((hull[:, 0], hull[:, 1]))[0])
    return np.roll(hull, -start, axis=0)


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
    """Diamètres de Feret (``Roi.getFeretValues``)."""
    hull = _hull(verts)
    if len(hull) == 0:
        return Feret(0.0, 0.0, 0.0, 0.0, 0.0)
    if len(hull) == 1:
        return Feret(0.0, 0.0, 0.0, float(hull[0, 0]), float(hull[0, 1]))
    diff = hull[:, None, :] - hull[None, :, :]
    d2 = (diff**2).sum(-1)
    # première paire (i ≤ j, dans l'ordre de l'enveloppe) atteignant le maximum, comme ImageJ
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    dmax = math.sqrt(float(d2[i, j]))
    (x1, y1), (x2, y2) = hull[i], hull[j]
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    angle = math.degrees(math.atan2(y1 - y2, x2 - x1))  # y vers le haut
    if angle < 0:
        angle += 180.0
    # largeur minimale : pieds à coulisse tournants sur les arêtes de l'enveloppe
    dmin = 0.0
    if len(hull) >= 3:
        dmin = dmax
        for k in range(len(hull)):
            a, b = hull[k], hull[(k + 1) % len(hull)]
            edge = b - a
            length = math.hypot(*edge)
            if length == 0:
                continue
            rel = hull - a
            dist = np.abs(edge[0] * rel[:, 1] - edge[1] * rel[:, 0]) / length
            dmin = min(dmin, float(dist.max()))
    return Feret(dmax, dmin, angle, float(x1), float(y1))


@dataclass
class Ellipse:
    major: float
    minor: float
    angle: float  # degrés, 0–180


def fit_ellipse(rows: np.ndarray, cols: np.ndarray) -> Ellipse:
    """Ellipse de mêmes moments d'ordre 2 et de même surface que l'objet.

    Transcription d'``ij.process.EllipseFitter`` (``computeSums``, ``getMoments``,
    ``getEllipseParam``), constantes comprises, pour obtenir les mêmes chiffres.
    """
    n = int(rows.size)
    if n == 0:
        return Ellipse(0.0, 0.0, 0.0)
    x = cols.astype(np.float64)
    y = rows.astype(np.float64)
    x1, y1 = float(x.sum()) / n, float(y.sum()) / n
    x2 = (float((x * x).sum()) + 0.08333333 * n) / n
    y2 = (float((y * y).sum()) + 0.08333333 * n) / n
    xy = float((x * y).sum()) / n
    u20 = x2 - x1 * x1  # variance en x
    u02 = y2 - y1 * y1  # variance en y
    u11 = xy - x1 * y1
    m4 = 4.0 * abs(u02 * u20 - u11 * u11)
    if m4 < 0.000001:
        m4 = 0.000001
    a11, a12, a22 = u02 / m4, u11 / m4, u20 / m4
    tmp = a11 - a22
    if tmp == 0.0:
        tmp = 0.000001
    theta = 0.5 * math.atan(2.0 * a12 / tmp)
    if theta < 0.0:
        theta += math.pi / 2
    if a12 > 0.0:
        theta += math.pi / 2
    elif a12 == 0.0:
        if a22 > a11:
            theta = 0.0
            a22, a11 = a11, a22
        elif a11 != a22:
            theta = math.pi / 2
    tmp = math.sin(theta)
    if tmp == 0.0:
        tmp = 0.000001
    z = a12 * math.cos(theta) / tmp
    major = math.sqrt(1.0 / abs(a22 + z))
    minor = math.sqrt(1.0 / abs(a11 - z))
    scale = math.sqrt(n / (math.pi * major * minor))  # même surface que l'objet
    major, minor = major * scale * 2.0, minor * scale * 2.0
    angle = 180.0 * theta / math.pi
    if angle == 180.0:
        angle = 0.0
    if major < minor:
        major, minor = minor, major
    return Ellipse(major, minor, angle)
