# Architecture de Fuji

## Vue d'ensemble

```
┌────────────────────────────────────────────────────────────────────┐
│ Interface (fuji.ui) — Qt + napari                                  │
│  Microscope · Assistants · Acquisition · Analyse · Propriétés      │
├────────────────────────────────────────────────────────────────────┤
│ Assistants (fuji.assistants)   │ Analyse (fuji.analysis)           │
│  Köhler, mise au point,        │  opérations calibrées (définitions│
│  calibration, champ plat       │  d'ImageJ) + pont Fiji (PyImageJ) │
├────────────────────────────────┴───────────────────────────────────┤
│ Cœur (fuji.core) — AUCUNE dépendance à Qt                          │
│  Microscope ← profil YAML · moteur d'acquisition (useq-schema)     │
│  Interfaces : Camera, XYStage, FocusDrive, StateDevice, Shutter,   │
│  LightSource + versions manuelles                                  │
├────────────────────────────────────────────────────────────────────┤
│ Pilotes (fuji.drivers) — registre extensible                       │
│  demo.* │ manual.* │ touptek.camera │ micromanager │ (extensions)  │
└────────────────────────────────────────────────────────────────────┘
          │                                   │
     OME-TIFF (fuji.io)                  Fiji / ImageJ
```

## Principes

1. **L'interface ne parle jamais au matériel.** Elle passe toujours par
   `fuji.core.Microscope`. C'est ce qui permettra plus tard de mettre le matériel
   sur un Raspberry Pi et l'interface sur un ordinateur (voir « Contrôle à distance »).
2. **Un profil YAML décrit un microscope.** Périphériques, pilotes, rôles, objectifs,
   canaux. Le même logiciel sert un Diaplan manuel, un microscope Micro-Manager ou un
   montage maison.
3. **Le manuel est un citoyen de première classe.** Un objectif, un cube, une lampe ou
   une mise au point non motorisés sont des périphériques comme les autres. Quand une
   acquisition a besoin d'un changement, Fuji affiche une consigne et attend. Motoriser
   un élément = changer une ligne du profil.
4. **Propriétés génériques.** Chaque pilote décrit ses réglages (type, bornes, choix,
   unité, mode expert). L'interface construit automatiquement les formulaires, pour
   n'importe quel matériel.
5. **Formats ouverts.** Séquences en `useq-schema` (compatible pymmcore-plus),
   images en OME-TIFF avec métadonnées complètes, profils en YAML.
6. **Mesures identiques à ImageJ.** Périmètre, ellipse, Feret, circularité,
   centroïde : mêmes algorithmes et mêmes noms de colonnes. Un test compare
   directement avec ImageJ (`tests/test_imagej_parity.py`).

## Modules

| Module | Rôle |
|---|---|
| `core/devices.py` | Interfaces abstraites des périphériques (unités : µm, ms, %) |
| `core/manual.py` | Périphériques manuels et consignes à l'utilisateur |
| `core/properties.py` | Description des réglages (pour les formulaires automatiques) |
| `core/profile.py` | Profils YAML (validation pydantic), dossiers utilisateur |
| `core/microscope.py` | Assemble le microscope : rôles, objectifs, taille du pixel, canaux, capture |
| `core/acquisition.py` | Moteur d'acquisition (temps, positions, canaux, Z) + planification |
| `drivers/demo.py` | Microscope simulé (Köhler, défocalisation, fluorescence, micromètre) |
| `drivers/touptek.py` | Caméras ToupTek (SDK toupcam) |
| `drivers/micromanager.py` | Toute configuration Micro-Manager via pymmcore-plus |
| `drivers/gcode.py`, `drivers/gcode_sim.py` | Moteurs pas à pas (GRBL, FluidNC, Marlin) : platine, mise au point, roue, LED ; contrôleur simulé |
| `core/stage.py` | Platine motorisée : orientation par rapport à la caméra, centrage, jeu, mosaïques (plan, recalage, assemblage) |
| `analysis/registration.py` | Déplacement entre deux images et confiance dans cette mesure |
| `remote/controller.py` | Commandes sûres pour un pilotage extérieur (IA, scripts) : garde-fous, mémoire d'images, réponses en texte simple |
| `remote/mcp_server.py` | Serveur MCP : expose le contrôleur à un assistant IA (stdio ou HTTP avec jeton) |
| `assistants/` | Köhler, mise au point, calibration du pixel, champ plat |
| `analysis/` | Images calibrées, registre d'opérations, mesures ImageJ, pont Fiji |
| `io/ome_tiff.py` | Écriture OME-TIFF en continu, métadonnées par plan |
| `ui/theme.py`, `ui/icons.py` | Thème (palettes, accent, feuille de style générée), icônes SVG de Fuji |
| `ui/main_window.py`, `ui/chrome.py`, `ui/quickbar.py` | Fenêtre : navigation, barre supérieure, barre rapide, histogramme |
| `ui/config/` | Éditeur de profil : pages, formulaires générés depuis les options des pilotes, modèles |
| `ui/panels/` | Panneaux : contrôle, assistants, acquisition, analyse, microscopes, réglages |
| `core/catalog.py` | Objectifs et cubes courants, modèles de microscopes |
| `ui/selftest.py` | Essai automatique de l'installation (`fuji selftest`) |
| `packaging/` | Application empaquetée : icônes, script de construction du `.dmg`, contrôle de la version de macOS |

## Ajouter un pilote

```python
# monlabo/platine.py
from fuji.core.devices import XYStage
from fuji.drivers import OptionSpec, register_driver

class MaPlatine(XYStage):
    def __init__(self, name, port="/dev/ttyUSB0", **kw):
        super().__init__(name, **kw)
        ...  # ouvrir le port série

    def get_position(self):
        ...

    def _move_to(self, x, y):
        ...

@register_driver(
    "monlabo.platine",
    title="Platine Mon Labo",
    category="Platines et mise au point",
    kind="xy_stage",
    description="Platine pilotée par Arduino.",
    options=[OptionSpec("port", "str", "Port série", "/dev/ttyUSB0")],
)
def _factory(spec, ctx):
    return MaPlatine(spec.name, label=spec.label, **spec.options)
```

Les `OptionSpec` (type, unité, bornes, aide, option avancée...) suffisent à l'éditeur de
configuration pour générer le formulaire du pilote : rien à écrire côté interface.

Dans le `pyproject.toml` du paquet tiers :

```toml
[project.entry-points."fuji.drivers"]
"monlabo.platine" = "monlabo.platine:_factory"
```

Puis dans le profil : `{name: platine, driver: monlabo.platine, options: {port: /dev/ttyUSB0}}`.

## Ajouter une opération d'analyse

```python
from typing import Annotated
from fuji.analysis import Image, Param, operation

@operation("Mon filtre", category="Filtres", imagej="Process > ...")
def mon_filtre(image: Image, force: Annotated[float, Param(min=0, max=10)] = 1.0) -> Image:
    return image.with_data(image.data * force, name=f"{image.name} (filtré)")
```

L'opération apparaît automatiquement dans le panneau « Analyse » avec son formulaire.

Les entrées annotées `Image`, `Labels`, `Rois` ou `Points` sont des calques (image, masque,
formes, points) ; les autres paramètres deviennent des réglages. Une opération reçoit un
plan 2D : pour une pile, le panneau l'applique au plan affiché ou à chaque plan (case
« Toute la pile »). Une opération qui travaille sur la pile entière (projection,
montage...) le déclare avec `stack=True`.

Pour retrouver les résultats d'ImageJ, utiliser les helpers de `fuji.analysis.ij` : arrondi
d'ImageJ (`restore_dtype`), voisinage circulaire (`disk`), flou gaussien, conversions de
type et de couleur. Voir [IMAGEJ.md](IMAGEJ.md).

## Fluorescence

Ajouter au profil les cubes (périphérique à états), la ou les sources (lampe ou LED),
puis un canal par fluorochrome :

```yaml
devices:
  - {name: cubes, driver: manual.state, options: {labels: ["Vide", "A (UV)", "I3 (bleu)", "N2.1 (vert)"]}}
  - {name: hbo, label: Lampe à mercure, driver: manual.light, options: {initial_on: false}}
channels:
  - name: GFP
    kind: fluorescence
    exposure_ms: 200
    states: {cubes: "I3 (bleu)"}
    lights: {hbo: {on: true}}
    color: green
```

En passant d'un canal à l'autre, Fuji éteint les sources des autres canaux et règle
le cube (ou le demande, s'il est manuel).

## Pilotage extérieur (IA, scripts)

Le paquet `fuji.remote` contient tout ce qu'il faut pour commander Fuji de l'extérieur :

- `MicroscopeController` regroupe les commandes utiles (réglages, images, moteurs,
  acquisition, analyse) et applique une `SafetyPolicy` : lecture seule, moteurs interdits,
  pas maximal par commande, dossier de travail fermé. Il ne dépend ni de Qt ni du
  protocole utilisé ;
- `mcp_server.py` l'expose en **MCP** (le standard des assistants IA), par stdio ou par
  HTTP. Les outils s'exécutent hors de la boucle du serveur, pour qu'un arrêt d'urgence
  passe pendant un long mouvement.

Dans la fenêtre, le même contrôleur partage le microscope affiché : les images produites
par l'IA deviennent des calques, les consignes manuelles s'affichent à l'écran.

## Contrôle à distance

Ce qui existe : les moteurs peuvent être branchés sur un autre ordinateur (port
`socket://…`), et le serveur MCP peut écouter sur le réseau avec un mot de passe.

Ce qui reste prévu : un petit serveur (sur le Raspberry Pi relié au microscope) qui
expose toute l'API `Microscope` sur le réseau, et un pilote « distant » côté ordinateur
qui implémente les mêmes interfaces. Le live sera transmis compressé, les acquisitions
enregistrées sur place puis transférées.

## Tests

```bash
xvfb-run -a pytest              # Linux sans écran
pytest -m "not gui"             # sans interface
FUJI_TEST_IMAGEJ=1 pytest tests/test_imagej_parity.py   # comparaison avec ImageJ
```
