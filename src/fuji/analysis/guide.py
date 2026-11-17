"""Mode d'emploi des outils d'analyse, en langage courant.

Les opérations sont rangées par **ce que l'on veut faire** (améliorer l'image, isoler les
objets, compter…) plutôt que par famille technique. Chacune a :

- un **résumé** d'une ligne (ce que ça fait) ;
- un conseil **quand l'utiliser** ;

et des **démarches guidées** enchaînent les outils pour les besoins courants (compter des
cellules, mesurer une surface…). Indépendant de l'interface (testable, réutilisable).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Help:
    summary: str
    when: str


@dataclass(frozen=True)
class Menu:
    key: str
    title: str
    icon: str
    intro: str
    operations: tuple[str, ...]


@dataclass(frozen=True)
class Step:
    """Une étape de démarche : une opération à ouvrir, un outil de dessin à prendre (« tool »),
    ou une simple consigne."""

    text: str
    operation: str | None = None
    tool: tuple[str, str] | None = None  # (type de calque, mode de dessin)
    optional: bool = False
    #: réglages proposés pour cette étape (nom du paramètre → valeur, ou :data:`GUESS`)
    params: tuple[tuple[str, object], ...] = ()


#: valeur de réglage à deviner d'après l'image (objets clairs ou sombres selon le fond)
GUESS = "deviner"


@dataclass(frozen=True)
class Recipe:
    key: str
    title: str
    intro: str
    steps: tuple[Step, ...] = field(default_factory=tuple)


MENUS: tuple[Menu, ...] = (
    Menu(
        "ameliorer",
        "Améliorer l'image",
        "wand",
        "Contraste, bruit, netteté, éclairage inégal",
        (
            "Rehausser le contraste",
            "Contraste local (CLAHE)",
            "Soustraction de fond",
            "Correction de champ plat",
            "Balance des blancs",
            "Flou gaussien",
            "Filtre médian",
            "Débruiter (despeckle)",
            "Retirer les valeurs aberrantes",
            "Lisser",
            "Filtre moyenne",
            "Accentuer",
            "Masque flou (netteté)",
            "Filtre passe-bande (FFT)",
            "Inverser",
        ),
    ),
    Menu(
        "isoler",
        "Isoler les objets",
        "target",
        "Séparer les objets du fond : créer un masque",
        ("Seuillage automatique", "Seuil manuel", "Seuillage local", "Seuillage couleur"),
    ),
    Menu(
        "masque",
        "Retoucher un masque",
        "roi_polygon",
        "Remplir, nettoyer, séparer les objets collés",
        (
            "Remplir les trous",
            "Retirer les petits objets",
            "Séparer les objets (watershed)",
            "Ouverture",
            "Fermeture",
            "Érosion",
            "Dilatation",
            "Inverser le masque",
            "Combiner deux masques",
            "Étiqueter les objets",
            "Contour des objets",
            "Squelette",
            "Carte des distances",
            "Points ultimes",
            "Voronoï",
        ),
    ),
    Menu(
        "compter",
        "Compter et mesurer",
        "ruler",
        "Nombre d'objets, surfaces, intensités, positions",
        ("Analyse de particules", "Mesurer les ROI", "Mesurer", "Trouver les maxima", "Compter les points"),
    ),
    Menu(
        "intensites",
        "Intensités et profils",
        "chart",
        "Histogramme, profil le long d'une ligne, colocalisation",
        ("Histogramme", "Profil d'intensité", "Colocalisation"),
    ),
    Menu(
        "couleurs",
        "Couleurs et canaux",
        "channels",
        "Gris, séparer ou fusionner les canaux, colorants",
        (
            "Niveaux de gris",
            "Séparer les canaux",
            "Fusionner en RGB",
            "Déconvolution des couleurs",
            "Séparer en HSB",
            "Séparer en Lab",
            "Convertir le type",
        ),
    ),
    Menu(
        "roi",
        "Régions dessinées (ROI)",
        "roi_rect",
        "Passer des ROI aux masques et inversement",
        ("ROI vers masque", "Masque vers ROI", "Effacer avec les ROI"),
    ),
    Menu(
        "geometrie",
        "Recadrer, tourner, redimensionner",
        "move",
        "Changer le cadre ou la taille de l'image",
        (
            "Rogner",
            "Tourner",
            "Tourner de 90°",
            "Retourner",
            "Translater",
            "Redimensionner",
            "Regrouper les pixels (bin)",
            "Taille du support",
        ),
    ),
    Menu(
        "piles",
        "Piles et séries (Z, temps)",
        "layers",
        "Projeter, recaler, découper une pile d'images",
        (
            "Projection",
            "Recaler la pile",
            "Sous-pile",
            "Montage",
            "Empiler deux images",
            "Inverser l'ordre de la pile",
        ),
    ),
    Menu(
        "calculs",
        "Calculs entre images",
        "plus",
        "Additionner, soustraire, diviser des images",
        ("Calculatrice d'images", "Mathématiques"),
    ),
    Menu(
        "filtres",
        "Filtres avancés",
        "sliders",
        "Contours, textures, filtres sur mesure",
        (
            "Détecter les contours",
            "Filtre variance",
            "Chapeau haut-de-forme",
            "Filtre minimum",
            "Filtre maximum",
            "Convolution",
        ),
    ),
    Menu(
        "fiji",
        "Fiji / ImageJ",
        "code",
        "Macros et commandes de Fiji, pour tout le reste",
        ("Macro ImageJ", "Commande ImageJ"),
    ),
)

#: Menu des opérations ajoutées par des extensions (absentes des menus ci-dessus).
OTHER_MENU = Menu("autres", "Autres outils", "sparkle", "Ajoutés par des extensions", ())

HELP: dict[str, Help] = {
    # --- améliorer l'image
    "Rehausser le contraste": Help(
        "Étale les niveaux pour utiliser toute la dynamique",
        "Image terne ou sombre que l'on veut lisible (attention : change les valeurs mesurées).",
    ),
    "Contraste local (CLAHE)": Help(
        "Fait ressortir les détails dans les zones sombres et claires",
        "Image très contrastée où l'on ne voit rien dans les ombres ; pour voir, pas pour mesurer.",
    ),
    "Soustraction de fond": Help(
        "Retire un éclairage inégal ou un fond diffus",
        "Avant un seuillage, quand le fond n'est pas uniforme (bords plus sombres, voile).",
    ),
    "Correction de champ plat": Help(
        "Corrige le vignettage avec une image de lame vide",
        "Vous avez une image de référence sans échantillon : la correction la plus juste.",
    ),
    "Balance des blancs": Help(
        "Rend le fond blanc ou gris neutre",
        "Image couleur au fond jaune ou bleuté (lampe halogène, caméra mal réglée).",
    ),
    "Flou gaussien": Help(
        "Adoucit l'image et atténue le bruit",
        "Avant un seuillage ou une recherche de maxima, pour lisser le grain de la caméra.",
    ),
    "Filtre médian": Help(
        "Retire le bruit en gardant les bords nets",
        "Image granuleuse ou avec des pixels isolés ; mieux que le flou pour garder les contours.",
    ),
    "Débruiter (despeckle)": Help(
        "Médian rapide 3×3 contre le bruit « poivre et sel »",
        "Quelques pixels blancs ou noirs isolés à faire disparaître d'un clic.",
    ),
    "Retirer les valeurs aberrantes": Help(
        "Efface les pixels trop clairs ou trop sombres isolés",
        "Pixels chauds de la caméra, poussières : seuls ces pixels sont modifiés.",
    ),
    "Lisser": Help(
        "Moyenne 3×3 : un léger flou",
        "Adoucir très légèrement, comme « Smooth » d'ImageJ.",
    ),
    "Filtre moyenne": Help(
        "Moyenne des pixels voisins dans un disque",
        "Lisser plus ou moins fort selon le rayon ; le flou gaussien donne souvent mieux.",
    ),
    "Accentuer": Help(
        "Renforce un peu la netteté (3×3)",
        "Image légèrement floue, pour une figure ; augmente aussi le bruit.",
    ),
    "Masque flou (netteté)": Help(
        "Renforce la netteté de façon réglable",
        "Rendre une image plus piquante pour une présentation (pas avant de mesurer).",
    ),
    "Filtre passe-bande (FFT)": Help(
        "Garde une gamme de tailles de détails",
        "Retirer à la fois un éclairage inégal (grandes tailles) et le bruit (petites tailles).",
    ),
    "Inverser": Help(
        "Négatif de l'image",
        "Rendre les objets clairs sur fond sombre (ou l'inverse) avant un traitement.",
    ),
    # --- isoler les objets
    "Seuillage automatique": Help(
        "Sépare les objets du fond automatiquement",
        "Première étape pour compter ou mesurer des objets : choisissez « sombres » en fond clair.",
    ),
    "Seuil manuel": Help(
        "Garde les pixels entre deux valeurs choisies",
        "Le seuil automatique se trompe : réglez vous-même avec l'aperçu.",
    ),
    "Seuillage local": Help(
        "Seuil adapté à chaque zone de l'image",
        "Éclairage très inégal que la soustraction de fond ne suffit pas à corriger.",
    ),
    "Seuillage couleur": Help(
        "Garde les pixels d'une certaine couleur",
        "Lames colorées : isoler le marron (DAB), le violet des noyaux…",
    ),
    # --- retoucher un masque
    "Remplir les trous": Help(
        "Bouche les trous à l'intérieur des objets",
        "Après un seuillage, quand les noyaux ou cellules ont des trous.",
    ),
    "Retirer les petits objets": Help(
        "Supprime les objets trop petits (poussières, bruit)",
        "Le masque contient de petites taches qui ne sont pas des objets.",
    ),
    "Séparer les objets (watershed)": Help(
        "Coupe les objets qui se touchent",
        "Cellules ou noyaux collés comptés comme un seul objet.",
    ),
    "Ouverture": Help(
        "Efface les petits détails et les fins ponts entre objets",
        "Nettoyer un masque bruité, détacher des objets reliés par un fil.",
    ),
    "Fermeture": Help(
        "Bouche les petites fissures et creux",
        "Contours d'objets découpés ou en pointillés après le seuillage.",
    ),
    "Érosion": Help(
        "Amincit les objets d'un pixel",
        "Réduire légèrement les objets, ou les détacher les uns des autres.",
    ),
    "Dilatation": Help(
        "Épaissit les objets d'un pixel",
        "Agrandir légèrement les objets, ou combler de petits espaces.",
    ),
    "Inverser le masque": Help(
        "Échange objets et fond",
        "Mesurer le fond plutôt que les objets, ou corriger un seuillage inversé.",
    ),
    "Combiner deux masques": Help(
        "Intersection, union ou différence de deux masques",
        "Objets présents dans deux marquages à la fois, ou dans l'un sans l'autre.",
    ),
    "Étiqueter les objets": Help(
        "Donne un numéro et une couleur à chaque objet",
        "Voir comment les objets sont découpés avant de les mesurer.",
    ),
    "Contour des objets": Help(
        "Ne garde que le bord des objets",
        "Dessiner les contours par-dessus l'image, pour une figure.",
    ),
    "Squelette": Help(
        "Réduit les objets à une ligne d'un pixel",
        "Réseaux, filaments, vaisseaux : longueur et ramifications.",
    ),
    "Carte des distances": Help(
        "Distance de chaque point au bord de l'objet",
        "Épaisseur des objets, ou préparation d'une séparation sur mesure.",
    ),
    "Points ultimes": Help(
        "Le centre de chaque objet ou de chaque lobe",
        "Repérer un point par objet, même quand ils sont collés.",
    ),
    "Voronoï": Help(
        "Découpe l'image en zones d'influence, une par objet",
        "Étudier l'espace autour de chaque cellule, ou les distances entre voisins.",
    ),
    # --- compter et mesurer
    "Analyse de particules": Help(
        "Compte et mesure chaque objet d'un masque",
        "Nombre de cellules, surface et forme de chacune : le tableau de résultats complet.",
    ),
    "Mesurer les ROI": Help(
        "Mesure chaque région dessinée à la main",
        "Surface ou intensité d'une zone précise, longueur d'une ligne tracée.",
    ),
    "Mesurer": Help(
        "Statistiques de toute l'image (ou d'un masque)",
        "Intensité moyenne d'une image, comparaison entre images.",
    ),
    "Trouver les maxima": Help(
        "Repère et compte les points brillants",
        "Spots fluorescents, grains, noyaux : compter sans seuillage.",
    ),
    "Compter les points": Help(
        "Tableau des points posés à la main",
        "Compter à la main ce que l'ordinateur ne sait pas reconnaître.",
    ),
    # --- intensités et profils
    "Histogramme": Help(
        "Répartition des valeurs de l'image",
        "Vérifier l'exposition, choisir un seuil, comparer deux images.",
    ),
    "Profil d'intensité": Help(
        "Intensité le long d'une ligne tracée",
        "Mesurer la largeur d'une structure ou comparer l'intensité d'un bord à l'autre.",
    ),
    "Colocalisation": Help(
        "À quel point deux marquages se superposent",
        "Fluorescence à deux couleurs : les deux protéines sont-elles au même endroit ?",
    ),
    # --- couleurs et canaux
    "Niveaux de gris": Help(
        "Transforme une image couleur en gris",
        "Avant la plupart des mesures et des seuillages sur une image couleur.",
    ),
    "Séparer les canaux": Help(
        "Sépare le rouge, le vert et le bleu",
        "Analyser un seul canal d'une image couleur (ex. le vert d'une fluorescence).",
    ),
    "Fusionner en RGB": Help(
        "Assemble jusqu'à trois images en une image couleur",
        "Superposer des canaux de fluorescence pour une figure.",
    ),
    "Déconvolution des couleurs": Help(
        "Sépare les colorants d'une lame (H&E, DAB…)",
        "Histologie : mesurer l'hématoxyline et l'éosine séparément, quantifier un marquage DAB.",
    ),
    "Séparer en HSB": Help(
        "Teinte, saturation et luminosité",
        "Isoler une couleur par sa teinte, indépendamment de son intensité.",
    ),
    "Séparer en Lab": Help(
        "Clarté et axes de couleur perceptuels",
        "Mesures de couleur proches de la vision humaine.",
    ),
    "Convertir le type": Help(
        "Change le type des pixels (8, 16 ou 32 bits)",
        "Préparer une image pour un logiciel ou un calcul qui exige un type précis.",
    ),
    # --- ROI
    "ROI vers masque": Help(
        "Transforme les régions dessinées en masque",
        "Utiliser vos ROI comme masque d'une autre opération.",
    ),
    "Masque vers ROI": Help(
        "Transforme les objets d'un masque en régions modifiables",
        "Corriger à la main les contours trouvés par le seuillage.",
    ),
    "Effacer avec les ROI": Help(
        "Efface l'intérieur ou l'extérieur des ROI",
        "Ne garder que la zone utile de l'image avant une mesure.",
    ),
    # --- géométrie
    "Rogner": Help(
        "Garde seulement le rectangle de la première ROI",
        "Recadrer sur la zone intéressante.",
    ),
    "Tourner": Help(
        "Tourne l'image d'un angle choisi",
        "Redresser une structure inclinée.",
    ),
    "Tourner de 90°": Help(
        "Quart de tour, sans perte",
        "Mettre l'image dans le bon sens.",
    ),
    "Retourner": Help(
        "Effet miroir horizontal ou vertical",
        "Image inversée par le trajet optique ou la caméra.",
    ),
    "Translater": Help(
        "Décale l'image",
        "Aligner deux images à la main.",
    ),
    "Redimensionner": Help(
        "Agrandit ou réduit l'image",
        "Alléger une très grande image ; l'échelle en µm reste juste.",
    ),
    "Regrouper les pixels (bin)": Help(
        "Réduit en additionnant ou moyennant les pixels",
        "Image bruitée et trop grande : moins de bruit, moins de pixels.",
    ),
    "Taille du support": Help(
        "Agrandit ou réduit le cadre autour de l'image",
        "Ajouter une marge, ou mettre deux images à la même taille.",
    ),
    # --- piles
    "Projection": Help(
        "Résume une pile en une seule image",
        "Pile Z : une image avec tout ce qui est net (maximum) ou la moyenne.",
    ),
    "Recaler la pile": Help(
        "Corrige le glissement entre les images d'une série",
        "Time-lapse qui dérive : aligner avant de mesurer dans le temps.",
    ),
    "Sous-pile": Help(
        "Garde une partie des images de la pile",
        "Ne travailler que sur quelques plans ou quelques temps.",
    ),
    "Montage": Help(
        "Toutes les images de la pile côte à côte",
        "Planche-contact pour une figure ou un rapport.",
    ),
    "Empiler deux images": Help(
        "Met deux images ou piles bout à bout",
        "Regrouper des captures en une série.",
    ),
    "Inverser l'ordre de la pile": Help(
        "Dernière image en premier",
        "Pile Z prise dans le mauvais sens.",
    ),
    # --- calculs
    "Calculatrice d'images": Help(
        "Additionne, soustrait, divise deux images",
        "Retirer une image de fond, faire un rapport de deux canaux.",
    ),
    "Mathématiques": Help(
        "Ajoute, multiplie… une valeur à chaque pixel",
        "Corriger un décalage, passer en logarithme.",
    ),
    # --- filtres avancés
    "Détecter les contours": Help(
        "Fait ressortir les bords des objets",
        "Voir les contours ; préparer une détection d'objets peu contrastés.",
    ),
    "Filtre variance": Help(
        "Fait ressortir les zones texturées",
        "Isoler un tissu texturé d'un fond lisse.",
    ),
    "Chapeau haut-de-forme": Help(
        "Garde les petits détails et retire le reste",
        "Petits spots sur un fond très variable.",
    ),
    "Filtre minimum": Help(
        "Plus petite valeur du voisinage",
        "Rétrécir les zones claires (érosion en gris).",
    ),
    "Filtre maximum": Help(
        "Plus grande valeur du voisinage",
        "Agrandir les zones claires (dilatation en gris).",
    ),
    "Convolution": Help(
        "Filtre sur mesure avec votre propre noyau",
        "Pour les connaisseurs : reproduire un filtre précis.",
    ),
    # --- Fiji
    "Macro ImageJ": Help(
        "Exécute une macro ImageJ dans Fiji",
        "Une macro existante ou une fonction absente de Fuji (nécessite Java).",
    ),
    "Commande ImageJ": Help(
        "Lance n'importe quelle commande de menu de Fiji",
        "Un plugin Fiji précis (nécessite Java).",
    ),
}

RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "compter",
        "Compter et mesurer des cellules ou des noyaux",
        "Le nombre d'objets, et la surface, la forme et l'intensité de chacun.",
        (
            Step("Image couleur ? La passer en gris.", "Niveaux de gris", optional=True),
            Step("Fond inégal ? Le retirer.", "Soustraction de fond", optional=True),
            Step(
                "Séparer les objets du fond. Fuji choisit « sombres » si le fond est clair : "
                "vérifiez avec l'aperçu.",
                "Seuillage automatique",
                params=(("objects", GUESS),),
            ),
            Step("Boucher les trous dans les objets.", "Remplir les trous"),
            Step("Objets collés ? Les séparer.", "Séparer les objets (watershed)", optional=True),
            Step("Compter et mesurer : un tableau avec une ligne par objet.", "Analyse de particules"),
        ),
    ),
    Recipe(
        "mesurer",
        "Mesurer une surface, une longueur ou une intensité",
        "Dessinez la zone à la main, puis mesurez-la.",
        (
            Step("Dessiner la zone : rectangle, ellipse, polygone…", tool=("shapes", "add_ellipse")),
            Step("Ou une ligne pour une longueur.", tool=("shapes", "add_line"), optional=True),
            Step("Mesurer : surface, périmètre, intensités, longueur.", "Mesurer les ROI"),
        ),
    ),
    Recipe(
        "profil",
        "Tracer un profil d'intensité",
        "Comment l'intensité varie le long d'une ligne (largeur d'une structure, bord…).",
        (
            Step("Tracer la ligne sur l'image.", tool=("shapes", "add_line")),
            Step("Afficher le profil (graphique et tableau).", "Profil d'intensité"),
        ),
    ),
    Recipe(
        "spots",
        "Compter des spots ou des grains",
        "Points brillants (fluorescence) ou grains : comptés sans seuillage.",
        (
            Step("Atténuer le grain de la caméra.", "Flou gaussien", optional=True),
            Step(
                "Repérer et compter les points ; ajustez la « proéminence » avec l'aperçu.",
                "Trouver les maxima",
            ),
        ),
    ),
    Recipe(
        "figure",
        "Préparer une belle image (figure)",
        "Pour une présentation ou une publication (pas pour mesurer ensuite).",
        (
            Step("Fond neutre.", "Balance des blancs", optional=True),
            Step("Éclairage plus régulier.", "Soustraction de fond", optional=True),
            Step("Un peu plus de netteté.", "Masque flou (netteté)", optional=True),
            Step("Exporter ce qui est affiché, avec la barre d'échelle : menu Fichier → Exporter la vue."),
        ),
    ),
    Recipe(
        "coloc",
        "Colocalisation de deux marquages",
        "Fluorescence à deux couleurs : se superposent-elles ?",
        (
            Step("Image couleur ? Séparer les canaux.", "Séparer les canaux", optional=True),
            Step("Mesurer la colocalisation des deux canaux.", "Colocalisation"),
        ),
    ),
    Recipe(
        "pile",
        "Résumer une pile Z en une image",
        "Une seule image avec ce qui est net dans chaque plan.",
        (Step("Projection « maximum » de la pile.", "Projection"),),
    ),
)


def menu_of(operation: str) -> Menu:
    """Menu d'une opération (« Autres outils » pour celles des extensions)."""
    return next((m for m in MENUS if operation in m.operations), OTHER_MENU)


def help_of(operation: str, description: str = "") -> Help:
    """Aide d'une opération ; pour une opération d'extension, la première ligne de sa
    description sert de résumé."""
    if operation in HELP:
        return HELP[operation]
    first = description.strip().split("\n")[0] if description else ""
    return Help(first or operation, "")
