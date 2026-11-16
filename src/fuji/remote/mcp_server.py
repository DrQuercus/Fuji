"""Serveur MCP de Fuji : le microscope et l'analyse d'images, offerts comme outils à une IA.

MCP (Model Context Protocol) est le standard par lequel un assistant IA — Claude Desktop,
Claude Code, ou tout autre client compatible — appelle des outils externes. Ce module
expose le :class:`~fuji.remote.controller.MicroscopeController` sous cette forme :

- en **stdio** : le client lance lui-même ``fuji mcp`` (cas d'un assistant installé sur
  l'ordinateur du microscope) ;
- en **HTTP** : le serveur tourne à côté du microscope (ou dans l'interface de Fuji) et le
  client s'y connecte par une adresse, éventuellement depuis une autre machine.

Chaque outil est une fonction courte qui délègue au contrôleur ; les garde-fous
(lecture seule, moteurs, limites de déplacement, dossier de travail) y sont appliqués.
Les erreurs reviennent à l'IA sous forme de messages clairs qu'elle peut exploiter.

Installation : ``pip install "fuji-microscopy[ia]"``.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from functools import partial
from typing import Annotated, Any

from pydantic import Field

from .controller import ControlError, MicroscopeController

log = logging.getLogger(__name__)

# types des paramètres communs (au niveau du module : les annotations des outils y sont résolues)
Manual = Annotated[
    bool,
    Field(
        description="true seulement après que l'utilisateur a confirmé avoir fait la manipulation demandée"
    ),
]
Name = Annotated[str, Field(description="nom d'un élément en mémoire (voir list_items)")]

INSTRUCTIONS = """\
Fuji pilote un microscope optique (caméra, objectifs, éclairage, platine, mise au point) et \
analyse les images avec les définitions d'ImageJ.

Pour commencer : microscope_status donne le matériel disponible (manuel ou motorisé), \
l'objectif en place, la taille du pixel et les garde-fous actifs. snap prend une image et la \
montre ; elle reste en mémoire sous un nom (ex. capture_1) utilisable par les autres outils.

Unités : micromètres pour les positions et les tailles, millisecondes pour l'exposition, \
pourcents pour l'éclairage. Dans une image, x va vers la droite et y vers le bas, en pixels.

Éléments manuels : sur un microscope sans moteur, changer d'objectif ou de filtre demande la \
main de l'utilisateur. L'outil répond alors « action manuelle nécessaire » : transmettez la \
consigne à l'utilisateur, attendez qu'il confirme, puis rappelez le même outil avec \
manual_done=true. Ne supposez jamais qu'une manipulation a été faite.

Avant de mesurer : vérifiez que l'image n'est ni saturée ni sombre (snap l'indique), que le \
bon objectif est déclaré (la taille du pixel en dépend), et que la mise au point est faite.

Analyse : list_operations pour trouver une opération, describe_operation pour ses réglages, \
run_operation pour l'exécuter sur des éléments en mémoire. Enchaînement typique pour compter \
des objets : snap → « Seuillage automatique » → « Analyse de particules » → get_table.

Moteurs : les déplacements sont bornés par commande et par la course du matériel ; avancez \
par petites étapes et regardez l'image entre deux. stop_motion arrête tout, à tout moment.\
"""


def _require_mcp() -> Any:
    try:
        from mcp.server import mcpserver
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise ImportError(
            'Le serveur pour IA a besoin du paquet « mcp » (version 2 ou plus) : pip install "fuji-microscopy[ia]"'
        ) from exc
    return mcpserver


def build_server(controller: MicroscopeController) -> Any:
    """Construit le serveur MCP (``mcp.server.mcpserver.MCPServer``) autour d'un contrôleur."""
    import anyio
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    from fuji import __version__

    mcpserver = _require_mcp()
    server = mcpserver.MCPServer(
        "fuji",
        title="Fuji — microscope et analyse d'images",
        instructions=INSTRUCTIONS,
        version=__version__,
        log_level="WARNING",
    )
    read = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    hardware = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
    compute = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    async def run(fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Exécute une commande du contrôleur hors de la boucle d'événements : un déplacement
        long ne doit pas empêcher de recevoir un arrêt d'urgence.

        Toute erreur revient à l'IA sous forme de message (refus d'un garde-fou, consigne
        manuelle, panne du matériel) : c'est une information dont elle a besoin pour agir."""
        try:
            return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))
        except ControlError as exc:
            raise ToolError(str(exc)) from None
        except Exception as exc:
            log.warning("Outil en erreur : %s", exc, exc_info=log.isEnabledFor(logging.DEBUG))
            raise ToolError(f"{type(exc).__name__} : {exc}") from None

    def text(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=1, default=str)

    async def with_preview(info: dict[str, Any], name: str, max_side: int) -> list[Any]:
        data, fmt = await run(controller.preview, name, max_side)
        return [text(info), mcpserver.Image(data=data, format=fmt)]

    # ------------------------------------------------------------------- état
    @server.tool(title="État du microscope", annotations=read)
    async def microscope_status() -> dict[str, Any]:
        """État complet : matériel (manuel ou motorisé), objectif en place, canal, taille du
        pixel, exposition, positions de la platine et de la mise au point, lampes, garde-fous."""
        return await run(controller.status)

    @server.tool(title="Réglages d'un périphérique", annotations=read)
    async def get_device_properties(
        device: Annotated[str, Field(description="nom du périphérique (voir microscope_status)")],
    ) -> list[dict[str, Any]]:
        """Tous les réglages détaillés d'un périphérique (balance des blancs, gamma, vitesse...)."""
        return await run(controller.device_properties, device)

    @server.tool(title="Modifier un réglage", annotations=hardware)
    async def set_device_property(device: str, name: str, value: Any) -> dict[str, Any]:
        """Modifie un réglage détaillé d'un périphérique (nom exact donné par get_device_properties)."""
        return await run(controller.set_device_property, device, name, value)

    # ---------------------------------------------------------------- réglages
    @server.tool(title="Exposition", annotations=hardware)
    async def set_exposure(
        exposure_ms: Annotated[float, Field(description="temps d'exposition en millisecondes", gt=0)],
    ) -> dict[str, Any]:
        """Règle le temps d'exposition de la caméra."""
        return await run(controller.set_exposure, exposure_ms)

    @server.tool(title="Exposition automatique", annotations=hardware)
    async def auto_exposure(
        target: Annotated[
            float, Field(description="niveau visé pour les hautes lumières (0,7 = 70 % de la dynamique)")
        ] = 0.7,
    ) -> dict[str, Any]:
        """Trouve l'exposition qui remplit la dynamique sans saturer (prend quelques images)."""
        return await run(controller.auto_exposure, target)

    @server.tool(title="Gain", annotations=hardware)
    async def set_gain(gain: float) -> dict[str, Any]:
        """Règle le gain de la caméra (à garder bas pour mesurer : il amplifie aussi le bruit)."""
        return await run(controller.set_gain, gain)

    @server.tool(title="Objectif", annotations=hardware)
    async def set_objective(
        label: Annotated[str, Field(description="nom exact de l'objectif (voir microscope_status)")],
        manual_done: Manual = False,
    ) -> dict[str, Any]:
        """Change d'objectif. La taille du pixel suit. Sur une tourelle manuelle, l'utilisateur
        doit tourner la tourelle lui-même : relayez la consigne, puis rappelez avec manual_done=true."""
        return await run(controller.set_objective, label, manual_done)

    @server.tool(title="Canal", annotations=hardware)
    async def set_channel(
        name: Annotated[str, Field(description="nom du canal (voir microscope_status)")],
        manual_done: Manual = False,
    ) -> dict[str, Any]:
        """Applique un canal d'imagerie : cube de filtres, lampes, exposition (ex. « Fond clair », « DAPI »)."""
        return await run(controller.set_channel, name, manual_done)

    @server.tool(title="Éclairage", annotations=hardware)
    async def set_light(
        on: Annotated[
            bool | None, Field(description="allumer (true) ou éteindre (false) ; rien = inchangé")
        ] = None,
        intensity: Annotated[
            float | None, Field(description="intensité en pourcents ; rien = inchangée")
        ] = None,
        name: Annotated[str | None, Field(description="nom de la lampe ; rien = lumière transmise")] = None,
        manual_done: Manual = False,
    ) -> dict[str, Any]:
        """Allume, éteint ou règle une lampe."""
        return await run(controller.set_light, on, intensity, name, manual_done)

    # ------------------------------------------------------------------ images
    @server.tool(title="Capturer une image", annotations=hardware, structured_output=False)
    async def snap(
        name: Annotated[
            str | None, Field(description="nom à donner à l'image ; rien = capture_1, capture_2...")
        ] = None,
        show: Annotated[bool, Field(description="joindre l'image à la réponse pour la regarder")] = True,
        max_side: Annotated[
            int, Field(description="taille de l'aperçu joint (plus grand côté, en pixels)", ge=64, le=4096)
        ] = 1024,
    ) -> list[Any]:
        """Capture une image avec les réglages actuels. Elle est gardée en mémoire sous le nom
        renvoyé. L'aperçu joint est réduit et son contraste étalé pour être lisible : les
        mesures, elles, se font sur l'image d'origine. La réponse signale une image saturée ou sombre."""
        info = await run(controller.snap, name)
        if not show:
            return [text(info)]
        return await with_preview(info, info["nom"], max_side)

    @server.tool(title="Regarder un élément", annotations=read, structured_output=False)
    async def get_image(
        name: Name,
        max_side: Annotated[
            int, Field(description="taille de l'aperçu (plus grand côté, en pixels)", ge=64, le=4096)
        ] = 1024,
    ) -> list[Any]:
        """Montre une image ou un masque en mémoire (résultat d'une opération, mosaïque...)."""
        info = await run(controller.describe_item, name)
        return await with_preview(info, name, max_side)

    @server.tool(title="Éléments en mémoire", annotations=read)
    async def list_items() -> list[dict[str, Any]]:
        """Images, masques, tableaux, points et ROI gardés en mémoire, avec leur résumé."""
        return await run(controller.list_items)

    @server.tool(title="Lire un tableau", annotations=read)
    async def get_table(
        name: Name,
        max_rows: Annotated[int, Field(description="nombre de lignes renvoyées", ge=1, le=1000)] = 50,
    ) -> dict[str, Any]:
        """Contenu d'un tableau de résultats : premières lignes, unités, statistiques par colonne."""
        return await run(controller.table, name, max_rows)

    @server.tool(title="Ouvrir un fichier image", annotations=compute)
    async def open_image(
        filename: Annotated[
            str, Field(description="fichier du dossier de travail (TIFF, OME-TIFF, PNG, JPEG)")
        ],
        name: str | None = None,
    ) -> dict[str, Any]:
        """Charge en mémoire une image du dossier de travail."""
        return await run(controller.open_image, filename, name)

    @server.tool(title="Enregistrer", annotations=compute)
    async def save_item(
        name: Name,
        filename: Annotated[
            str | None, Field(description="nom du fichier dans le dossier de travail ; rien = automatique")
        ] = None,
    ) -> dict[str, Any]:
        """Enregistre une image (OME-TIFF, avec sa calibration) ou un tableau (CSV) dans le dossier de travail."""
        return await run(controller.save_item, name, filename)

    @server.tool(title="Oublier un élément", annotations=compute)
    async def delete_item(name: Name) -> dict[str, Any]:
        """Retire un élément de la mémoire (les fichiers enregistrés ne sont pas touchés)."""
        return await run(controller.delete_item, name)

    @server.tool(title="Créer une ROI", annotations=compute)
    async def create_roi(
        kind: Annotated[str, Field(description="rectangle, ellipse, line ou polygon")],
        coordinates: Annotated[
            list[float],
            Field(
                description="rectangle/ellipse : [x, y, largeur, hauteur] ; line : [x1, y1, x2, y2] ; polygon : [x1, y1, x2, y2, ...] — en pixels"
            ),
        ],
        name: str | None = None,
    ) -> dict[str, Any]:
        """Crée une région d'intérêt, à utiliser comme entrée « roi » d'une opération
        (« Mesurer les ROI », « Profil d'intensité », « Rogner », « Histogramme »...)."""
        return await run(controller.create_roi, kind, coordinates, name)

    @server.tool(title="Créer des points", annotations=compute)
    async def create_points(
        coordinates: Annotated[list[list[float]], Field(description="[[x, y], ...] en pixels")],
        name: str | None = None,
    ) -> dict[str, Any]:
        """Crée un ensemble de points (entrée « points » de « Compter les points »)."""
        return await run(controller.create_points, coordinates, name)

    # ----------------------------------------------------------------- moteurs
    @server.tool(title="Déplacer la platine", annotations=hardware)
    async def move_stage(
        x_um: Annotated[float, Field(description="position ou déplacement en X (µm)")],
        y_um: Annotated[float, Field(description="position ou déplacement en Y (µm)")],
        relative: Annotated[
            bool,
            Field(description="true : déplacement depuis la position actuelle ; false : position absolue"),
        ] = True,
    ) -> dict[str, Any]:
        """Déplace la platine motorisée. Les axes de la platine ne sont pas forcément ceux de
        l'image : pour amener un point précis de l'image au centre, utilisez center_on_point."""
        return await run(controller.move_stage, x_um, y_um, relative)

    @server.tool(title="Centrer un point de l'image", annotations=hardware)
    async def center_on_point(
        x_px: Annotated[float, Field(description="colonne du point dans l'image (pixels, depuis la gauche)")],
        y_px: Annotated[float, Field(description="ligne du point dans l'image (pixels, depuis le haut)")],
        image: Annotated[
            str | None, Field(description="image en mémoire où le point a été repéré (en pleine résolution)")
        ] = None,
    ) -> dict[str, Any]:
        """Déplace la platine pour amener au centre du champ un point repéré sur la dernière
        image. Attention : donnez les coordonnées dans l'image d'origine, pas dans l'aperçu réduit
        (multipliez par largeur d'origine / largeur de l'aperçu). L'orientation de la platine doit
        avoir été calibrée (calibrate_stage)."""
        return await run(controller.center_on, x_px, y_px, image)

    @server.tool(title="Déplacer la mise au point", annotations=hardware)
    async def move_focus(
        z_um: Annotated[float, Field(description="position ou déplacement en Z (µm)")],
        relative: Annotated[bool, Field(description="true : déplacement depuis la position actuelle")] = True,
    ) -> dict[str, Any]:
        """Déplace la mise au point motorisée. Avancez par petits pas en vérifiant la netteté
        (focus_score) : un grand déplacement peut amener l'objectif contre la lame."""
        return await run(controller.move_focus, z_um, relative)

    @server.tool(title="Autofocus", annotations=hardware)
    async def autofocus(
        range_um: Annotated[
            float, Field(description="plage balayée autour de la position actuelle (µm)", gt=0)
        ] = 20.0,
        step_um: Annotated[float, Field(description="pas du balayage (µm)", gt=0)] = 2.0,
    ) -> dict[str, Any]:
        """Cherche la mise au point la plus nette autour de la position actuelle (mise au point
        motorisée). Échoue proprement, sans bouger, s'il n'y a rien de net dans le champ."""
        return await run(controller.autofocus, range_um, step_um)

    @server.tool(
        title="Arrêt d'urgence", annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True)
    )
    async def stop_motion() -> dict[str, Any]:
        """Arrête immédiatement tous les moteurs et l'acquisition en cours. Toujours disponible."""
        return await run(controller.stop)

    @server.tool(title="Prise d'origine", annotations=hardware)
    async def home_stage() -> dict[str, Any]:
        """Envoie la platine sur ses fins de course pour retrouver son origine (si elle en a)."""
        return await run(controller.home)

    @server.tool(title="Définir l'origine", annotations=hardware)
    async def set_origin(
        axis: Annotated[str, Field(description="« xy » pour la platine, « z » pour la mise au point")] = "xy",
    ) -> dict[str, Any]:
        """Déclare la position actuelle comme zéro."""
        return await run(controller.set_origin, axis)

    # -------------------------------------------------------------- assistants
    @server.tool(title="Netteté", annotations=hardware)
    async def focus_score() -> dict[str, Any]:
        """Mesure la netteté de l'image actuelle (plus grand = plus net) : utile pour guider une
        mise au point manuelle (« tournez encore », « revenez en arrière »)."""
        return await run(controller.focus_score)

    @server.tool(title="Uniformité de l'éclairage", annotations=hardware)
    async def check_illumination() -> dict[str, Any]:
        """Vérifie l'uniformité de l'éclairage sur une zone vide de la lame, et dit quoi régler."""
        return await run(controller.check_illumination)

    @server.tool(title="Diaphragme de champ (Köhler)", annotations=hardware)
    async def check_field_diaphragm() -> dict[str, Any]:
        """Aide au réglage de Köhler : avec le diaphragme de champ partiellement fermé, indique
        si son bord est net (hauteur du condenseur) et centré (vis de centrage)."""
        return await run(controller.check_field_diaphragm)

    @server.tool(title="Calibrer le pixel", annotations=hardware)
    async def calibrate_pixel_size(
        spacing_um: Annotated[
            float, Field(description="écart entre deux graduations du micromètre objet (µm)", gt=0)
        ] = 10.0,
        save: Annotated[
            bool, Field(description="enregistrer pour l'objectif en place (si autorisé)")
        ] = False,
    ) -> dict[str, Any]:
        """Mesure la taille du pixel sur un micromètre objet placé sous l'objectif."""
        return await run(controller.calibrate_pixel_size, spacing_um, save)

    @server.tool(title="Calibrer la platine", annotations=hardware)
    async def calibrate_stage(
        save: Annotated[bool, Field(description="enregistrer dans le profil (si autorisé)")] = False,
    ) -> dict[str, Any]:
        """Mesure l'orientation de la platine motorisée par rapport à la caméra (deux petits pas,
        sur une lame avec des détails). Nécessaire pour center_on_point et les mosaïques."""
        return await run(controller.calibrate_stage, save)

    @server.tool(title="Jeu de la platine", annotations=hardware)
    async def measure_backlash(
        axis: Annotated[str, Field(description="« x » ou « y »")] = "x",
    ) -> dict[str, Any]:
        """Mesure le jeu mécanique d'un axe de la platine (µm)."""
        return await run(controller.measure_backlash, axis)

    # ------------------------------------------------------------- acquisition
    @server.tool(title="Acquisition", annotations=hardware)
    async def acquire(
        name: Annotated[str, Field(description="nom de l'acquisition (fichiers OME-TIFF)")] = "acquisition",
        time_points: Annotated[int, Field(description="nombre de points de temps", ge=1)] = 1,
        interval_s: Annotated[
            float, Field(description="intervalle entre deux points de temps (s)", ge=0)
        ] = 0.0,
        z_range_um: Annotated[
            float,
            Field(
                description="étendue de la pile Z, centrée sur la mise au point (µm) ; 0 = pas de pile", ge=0
            ),
        ] = 0.0,
        z_step_um: Annotated[float, Field(description="pas de la pile Z (µm)", gt=0)] = 1.0,
        channels: Annotated[
            list[str] | None, Field(description="canaux à acquérir ; rien = réglages actuels")
        ] = None,
        positions: Annotated[
            list[list[float]] | None, Field(description="positions de platine [[x, y], ...] en µm")
        ] = None,
        mosaic: Annotated[
            list[int] | None,
            Field(
                description="[colonnes, rangées] : mosaïque autour de la position actuelle, assemblée à la fin"
            ),
        ] = None,
        mosaic_overlap: Annotated[
            float, Field(description="recouvrement entre tuiles (0,15 = 15 %)", ge=0, lt=0.9)
        ] = 0.15,
        autofocus: Annotated[bool, Field(description="refaire la mise au point à chaque position")] = False,
        wait_s: Annotated[
            float,
            Field(
                description="durée maximale d'attente de la fin (s) ; au-delà, suivre avec acquisition_status",
                ge=0,
                le=300,
            ),
        ] = 30.0,
        manual_done: Manual = False,
    ) -> dict[str, Any]:
        """Lance une acquisition multidimensionnelle (temps, pile Z, canaux, positions ou
        mosaïque), enregistrée en OME-TIFF dans le dossier de travail. Renvoie l'état : s'il est
        « en cours », rappelez acquisition_status. Une mosaïque terminée est gardée en mémoire
        (nom renvoyé dans « mosaiques »)."""
        return await run(
            controller.acquire,
            name,
            time_points,
            interval_s,
            z_range_um,
            z_step_um,
            channels,
            positions,
            mosaic,
            mosaic_overlap,
            autofocus,
            wait_s,
            manual_done,
        )

    @server.tool(title="Suivre l'acquisition", annotations=read)
    async def acquisition_status(
        wait_s: Annotated[
            float, Field(description="attendre la fin jusqu'à ce nombre de secondes", ge=0, le=300)
        ] = 0.0,
    ) -> dict[str, Any]:
        """État de l'acquisition en cours ou de la dernière : images faites, fichiers, erreur."""
        return await run(controller.acquisition_status, wait_s)

    @server.tool(title="Annuler l'acquisition", annotations=hardware)
    async def cancel_acquisition() -> dict[str, Any]:
        """Interrompt l'acquisition en cours (les images déjà prises sont conservées)."""
        return await run(controller.cancel_acquisition)

    # ----------------------------------------------------------------- analyse
    @server.tool(title="Opérations d'analyse", annotations=read)
    async def list_operations(
        search: Annotated[
            str,
            Field(
                description="mot-clé, en français ou nom de commande ImageJ (ex. « seuil », « threshold », « particules »)"
            ),
        ] = "",
        category: Annotated[
            str,
            Field(
                description="catégorie exacte (ex. Filtres, Mesures, Segmentation, Binaire, Piles, Couleur)"
            ),
        ] = "",
    ) -> list[dict[str, Any]]:
        """Catalogue des opérations d'analyse d'images (les fonctions essentielles d'ImageJ) :
        filtres, seuillages, binaire, mesures, particules, profils, colocalisation, piles..."""
        return await run(controller.operations, search, category)

    @server.tool(title="Détail d'une opération", annotations=read)
    async def describe_operation(
        name: Annotated[str, Field(description="nom exact de l'opération (voir list_operations)")],
    ) -> dict[str, Any]:
        """Entrées attendues et réglages d'une opération, avec leurs valeurs par défaut et leurs choix."""
        return await run(controller.describe_operation, name)

    @server.tool(title="Exécuter une opération", annotations=compute)
    async def run_operation(
        name: Annotated[str, Field(description="nom exact de l'opération")],
        inputs: Annotated[
            dict[str, str] | None,
            Field(
                description='entrée → nom d\'un élément en mémoire, ex. {"image": "capture_1"} ou {"mask": "seuillage_automatique_1", "intensity": "capture_1"}'
            ),
        ] = None,
        params: Annotated[
            dict[str, Any] | None, Field(description='réglages, ex. {"sigma": 2} ; rien = valeurs par défaut')
        ] = None,
    ) -> dict[str, Any]:
        """Exécute une opération d'analyse. Les résultats (images, masques, tableaux, points) sont
        gardés en mémoire ; leurs noms sont renvoyés : regardez-les avec get_image ou get_table."""
        return await run(controller.run_operation, name, inputs, params)

    # --------------------------------------------------------------- ressources
    @server.resource("fuji://etat", title="État du microscope", mime_type="application/json")
    async def status_resource() -> str:
        return text(await run(controller.status))

    @server.resource("fuji://operations", title="Catalogue des opérations", mime_type="application/json")
    async def operations_resource() -> str:
        return text(await run(controller.operations))

    @server.resource("fuji://profil", title="Profil du microscope (YAML)", mime_type="text/yaml")
    async def profile_resource() -> str:
        return await run(lambda: controller.microscope.profile.to_yaml())

    # ------------------------------------------------------------------ modèles
    @server.prompt(title="Première image")
    def premiere_image() -> str:
        """Prendre en main le microscope et obtenir une première image correcte."""
        return (
            "Prends en main le microscope avec Fuji. Commence par microscope_status et dis-moi en une phrase "
            "ce qui est motorisé et ce qui est manuel. Allume l'éclairage si besoin (set_channel), règle "
            "l'exposition (auto_exposure), prends une image (snap) et décris ce que tu vois : netteté, "
            "éclairage, densité des objets. Propose ensuite ce qu'il faudrait régler, sans rien déplacer "
            "sans me le dire."
        )

    @server.prompt(title="Compter et mesurer des objets")
    def compter_des_objets(
        objets: Annotated[
            str, Field(description="ce qu'il faut compter (cellules, noyaux, grains...)")
        ] = "cellules",
    ) -> str:
        """Compter et mesurer des objets sur l'image actuelle, avec les mesures d'ImageJ."""
        return (
            f"Compte et mesure les {objets} sur l'image actuelle. Vérifie d'abord l'objectif en place et la "
            "taille du pixel (microscope_status), prends une image correctement exposée (snap), segmente "
            "(« Seuillage automatique », en choisissant objets clairs ou sombres ; « Séparer les objets "
            "(watershed) » s'ils se touchent), puis « Analyse de particules » avec une surface minimale "
            "raisonnable. Montre-moi le masque (get_image) pour que je valide, puis donne le nombre "
            "d'objets, la surface moyenne et sa dispersion, en µm²."
        )

    return server


# ------------------------------------------------------------------------- transports
def run_stdio(controller: MicroscopeController) -> None:
    """Sert le contrôleur sur l'entrée/sortie standard (le client MCP lance ce processus)."""
    build_server(controller).run("stdio")


_LOOPBACK = ("127.0.0.1", "localhost", "::1")


class _TokenAuth:
    """Exige ``Authorization: Bearer <jeton>`` sur chaque requête HTTP."""

    def __init__(self, app: Any, token: str) -> None:
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            given = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"text/plain; charset=utf-8"),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send(
                    {"type": "http.response.body", "body": "Jeton d'accès manquant ou incorrect".encode()}
                )
                return
        await self.app(scope, receive, send)


def http_app(controller: MicroscopeController, host: str = "127.0.0.1", token: str | None = None) -> Any:
    """Application web (ASGI) du serveur MCP, au chemin ``/mcp``.

    Hors de la machine locale, un jeton est obligatoire : ce serveur commande du matériel.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    local = host in _LOOPBACK
    if not local and not token:
        raise ControlError(
            "Un jeton d'accès est obligatoire pour ouvrir le serveur au réseau (option --token) : "
            "n'importe qui sur le réseau pourrait sinon commander le microscope."
        )
    security = None if local else TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = build_server(controller).streamable_http_app(host=host, transport_security=security)
    return _TokenAuth(app, token) if token else app


class HttpServer:
    """Serveur MCP en HTTP dans un thread (interface graphique, ou ``fuji mcp --http``)."""

    def __init__(
        self,
        controller: MicroscopeController,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
    ) -> None:
        self.controller, self.host, self.port, self.token = controller, host, int(port), token
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        shown = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{shown}:{self.port}/mcp"

    @property
    def running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and bool(getattr(self._server, "started", False))
        )

    def start(self, timeout: float = 10.0) -> None:
        import uvicorn

        if self.running:
            return
        config = uvicorn.Config(
            http_app(self.controller, self.host, self.token),
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="fuji-mcp", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._server.started:
            self.stop()
            raise ControlError(
                f"Le serveur n'a pas pu démarrer sur {self.host}:{self.port} (port déjà utilisé ?)"
            )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = self._thread = None
