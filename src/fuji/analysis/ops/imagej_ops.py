"""Opérations qui délèguent à Fiji/ImageJ (macros, commandes de menu, plugins)."""

from __future__ import annotations

from typing import Annotated

from ..image import Image, Table
from ..imagej import ImageJBridge
from ..registry import Param, operation


@operation("Macro ImageJ", category="Fiji / ImageJ", imagej="Plugins > Macros > Run...")
def imagej_macro(
    image: Image | None = None,
    macro: Annotated[str, Param(multiline=True, description="langage macro ImageJ (IJM)")] = (
        'run("Gaussian Blur...", "sigma=2");'
    ),
) -> tuple[Image | None, Table | None]:
    """Exécute une macro ImageJ dans Fiji (Java) et récupère image + résultats."""
    res = ImageJBridge.get().run_macro(macro, image)
    return res.image, res.table


@operation("Commande ImageJ", category="Fiji / ImageJ")
def imagej_command(
    image: Image | None = None,
    command: Annotated[str, Param(description="nom exact du menu, ex. « Find Edges »")] = "Find Edges",
    options: Annotated[str, Param(description="options de l'enregistreur de macros")] = "",
) -> tuple[Image | None, Table | None]:
    """Exécute n'importe quelle commande de menu Fiji (y compris les plugins)."""
    res = ImageJBridge.get().run_command(command, options, image)
    return res.image, res.table
