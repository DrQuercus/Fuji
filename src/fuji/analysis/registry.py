"""Registre des opérations d'analyse.

Une opération est une fonction Python ordinaire, décorée avec :func:`operation`.
Ses paramètres (type, valeur par défaut, bornes via :class:`Param`) suffisent à
l'interface pour générer automatiquement le formulaire de réglages, et à un
protocole pour l'enregistrer et la rejouer.

Exemple ::

    @operation("Flou gaussien", category="Filtres", imagej="Process > Filters > Gaussian Blur...")
    def gaussian(image: Image, sigma: Annotated[float, Param(min=0.1, max=100, unit="px")] = 2.0):
        ...

Les paramètres annotés ``Image`` / ``Labels`` sont des *entrées* (calques) ; les autres
sont des réglages. Un paquet tiers peut ajouter des opérations via le point d'entrée
``fuji.operations``.
"""

from __future__ import annotations

import inspect
import logging
import types
import typing
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from .image import Image, Labels, Points, Rois, Table

#: Types d'entrée d'une opération : ce sont des calques (images, masques, ROI, points).
INPUT_KINDS = (Image, Labels, Rois, Points)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Param:
    """Métadonnées d'un réglage (bornes, unité, libellé, mode expert)."""

    label: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    choices: tuple[str, ...] | None = None
    description: str = ""
    advanced: bool = False
    multiline: bool = False


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: type
    default: Any
    meta: Param

    @property
    def label(self) -> str:
        return self.meta.label or self.name.replace("_", " ")


@dataclass(frozen=True)
class InputSpec:
    name: str
    kind: type  # Image, Labels, Rois ou Points
    optional: bool


@dataclass
class Operation:
    name: str
    func: Callable[..., Any]
    category: str
    description: str = ""
    #: Commande ImageJ équivalente (aide à la migration depuis Fiji).
    imagej: str | None = None
    inputs: list[InputSpec] = field(default_factory=list)
    params: list[ParamSpec] = field(default_factory=list)
    #: Vrai si l'opération travaille sur la pile entière (projection, montage...). Les
    #: autres opérations reçoivent un plan 2D ; pour une pile, on les applique plan par plan.
    stack: bool = False

    def __call__(self, *inputs: Any, **params: Any) -> Any:
        return self.func(*inputs, **params)


_REGISTRY: dict[str, Operation] = {}
_loaded = False


def _unwrap(annotation: Any) -> tuple[Any, Param | None, bool]:
    """Renvoie (type de base, Param, optionnel)."""
    meta = None
    if get_origin(annotation) is Annotated:
        base, *extras = get_args(annotation)
        meta = next((e for e in extras if isinstance(e, Param)), None)
        annotation = base
    optional = False
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        optional = len(args) < len(get_args(annotation))
        annotation = args[0] if len(args) == 1 else annotation
    if get_origin(annotation) is typing.Literal:
        choices = tuple(str(a) for a in get_args(annotation))
        meta = Param(**{**(meta.__dict__ if meta else {}), "choices": choices})
        annotation = str
    return annotation, meta, optional


def operation(
    name: str, *, category: str, description: str = "", imagej: str | None = None, stack: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        hints = get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)
        inputs: list[InputSpec] = []
        params: list[ParamSpec] = []
        for pname, p in sig.parameters.items():
            base, meta, optional = _unwrap(hints.get(pname, Any))
            if base in INPUT_KINDS:
                inputs.append(InputSpec(pname, base, optional or p.default is None))
            else:
                if p.default is inspect.Parameter.empty:
                    raise TypeError(f"{name} : le réglage {pname!r} doit avoir une valeur par défaut")
                params.append(ParamSpec(pname, base, p.default, meta or Param()))
        doc = description or inspect.getdoc(func) or ""
        _REGISTRY[name] = Operation(name, func, category, doc, imagej, inputs, params, stack)
        return func

    return deco


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import ops  # noqa: F401 - enregistre les opérations fournies

    for ep in entry_points(group="fuji.operations"):
        try:
            ep.load()
        except Exception:  # pragma: no cover
            log.exception("Impossible de charger les opérations externes %s", ep.name)


def _sort_key(text: str) -> str:
    """Ordre alphabétique français : « Érosion » se range à E, pas après Z."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def list_operations(category: str | None = None) -> list[Operation]:
    _load_all()
    ops = sorted(_REGISTRY.values(), key=lambda o: (_sort_key(o.category), _sort_key(o.name)))
    return [o for o in ops if category is None or o.category == category]


def categories() -> list[str]:
    return sorted({o.category for o in list_operations()})


def get_operation(name: str) -> Operation:
    _load_all()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Opération inconnue {name!r}. Disponibles : {sorted(_REGISTRY)}") from None


def combine_planes(results: list[Any], indices: list[tuple[int, ...]], lead_shape: tuple[int, ...]) -> Any:
    """Rassemble les résultats d'une opération appliquée plan par plan à une pile.

    Images et masques sont réempilés, les tableaux mis bout à bout avec une colonne
    « Plan », les points regroupés avec leurs coordonnées dans la pile.
    """
    import numpy as np
    import pandas as pd

    if not results:
        raise ValueError("Pile vide")
    first = results[0]
    if isinstance(first, tuple):
        return tuple(combine_planes([r[i] for r in results], indices, lead_shape) for i in range(len(first)))
    if first is None:
        return None

    def label(index: tuple[int, ...]) -> str:
        return ", ".join(str(i + 1) for i in index)

    if isinstance(first, (Image, Labels)):
        shapes = {r.data.shape for r in results}
        if len(shapes) != 1:
            raise ValueError("Les plans n'ont pas tous la même taille après l'opération")
        data = np.stack([r.data for r in results]).reshape(*lead_shape, *first.data.shape)
        return type(first)(data, first.pixel_size_um, first.name, dict(first.metadata))
    if isinstance(first, Table):
        frames = []
        for index, table in zip(indices, results, strict=True):
            df = table.data.copy()
            df.insert(0, "Plan", label(index))
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)
        data.index = pd.RangeIndex(1, len(data) + 1)
        return Table(data, first.name, dict(first.units), {}, first.plot)
    if isinstance(first, Points):
        rows = [
            np.column_stack([np.tile(index, (len(p.data), 1)), p.data])
            for index, p in zip(indices, results, strict=True)
            if len(p.data)
        ]
        data = np.vstack(rows) if rows else np.zeros((0, len(lead_shape) + 2))
        return Points(data, first.pixel_size_um, first.name)
    return first


def run(name: str, *inputs: Any, **params: Any) -> Image | Labels | Table | tuple[Any, ...]:
    """Exécute une opération par son nom : ``run("Flou gaussien", img, sigma=2)``."""
    op = get_operation(name)
    known = {p.name for p in op.params} | {i.name for i in op.inputs}
    unknown = set(params) - known
    if unknown:
        raise TypeError(f"{name} : réglages inconnus {sorted(unknown)} ; attendus {sorted(known)}")
    return op(*inputs, **params)
