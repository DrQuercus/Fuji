# Fuji

Logiciel de microscopie qui réunit le meilleur de **Micro-Manager** (piloter le
matériel), **napari** (afficher les images) et **Fiji/ImageJ** (analyser), dans une
interface propre qui s'adapte au microscope.

> Projet en développement (version 0.1, premières briques). Voir
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) et [docs/FEUILLE_DE_ROUTE.md](docs/FEUILLE_DE_ROUTE.md).

![Interface de Fuji : assistant Köhler sur le microscope simulé](docs/interface.png)

## L'interface

- **Barre de navigation** à gauche : Contrôle, Assistants, Acquisition, Analyse,
  Calques, Microscope (configuration), Réglages. Recliquer sur la section active
  replie le panneau pour agrandir l'image.
- **Barre rapide** sous l'image : exposition (curseur logarithmique + auto),
  gain, objectif et canal en un clic, **histogramme en direct** avec alerte de
  saturation.
- **Barre supérieure** : choix du microscope (changement à chaud), Live, Capture,
  mode expert, thème clair / sombre.
- **Thèmes** : sombre ou clair, 7 couleurs d'accent, 3 tailles de texte
  (Réglages → Apparence). Raccourcis : Ctrl+L live, Ctrl+K capture, Ctrl+1…7
  sections, Ctrl+B replier le panneau, Ctrl+E modifier le microscope.

![Thème clair, accent turquoise](docs/interface-clair.png)

## Configurer un microscope

Section **Microscope** → « Modifier la configuration… » ou « Nouveau microscope… »
(modèles : microscope manuel + ToupTek, manuel avec fluorescence, Micro-Manager,
simulé, vide, copie de l'actuel). L'éditeur couvre **tout** le profil :

| Page | Contenu |
|---|---|
| Général | nom, description, adaptateur caméra, ON du condenseur, dossier, canal au démarrage |
| Statif et optique | fabricant, modèle, droit/inversé, longueur de tube, oculaires, indice de champ, condenseur ; champ, résolution et profondeur de champ calculés par objectif |
| Périphériques | ajout par catégorie (caméras, platines, tourelles, éclairage, systèmes Micro-Manager, simulation), réglages de chaque pilote générés automatiquement, détection des caméras ToupTek |
| Rôles | caméra, platine XY, mise au point, revolver, lampe de transmission |
| Objectifs | grossissement, ON, immersion, contraste, correction, anneau de phase, distance de travail, pixel calibré, parfocalité (+ modèles courants) |
| Cubes de fluorescence | excitation, dichroïque, émission (+ modèles, dont les blocs Leitz Ploemopak A, I3, N2.1) ; un clic crée la tourelle de cubes, la lampe et un canal par cube |
| Canaux | type, cube, exposition, gain, couleur, longueurs d'onde, position de chaque tourelle, état de chaque lampe, obturateurs, préréglage Micro-Manager |
| Fichier YAML | le profil complet, modifiable directement |

« Vérifier » signale les incohérences (rôle mal attribué, position inexistante...),
« Tester la connexion » connecte le matériel et fait une capture d'essai.

![Éditeur : périphériques](docs/editeur-peripheriques.png)

## Ce que Fuji sait déjà faire

- **Piloter un microscope manuel** (Leitz Diaplan) comme un microscope motorisé :
  objectifs, lampe et mise au point sont déclarés ; quand un protocole a besoin d'un
  changement, Fuji affiche une consigne (« Placez l'objectif 40x ») et attend.
- **Caméras ToupTek** via le SDK officiel (live, capture par déclenchement logiciel,
  12 bits, balance des blancs, réglages du capteur).
- **Tous les microscopes Micro-Manager** (~250 périphériques) avec une seule ligne de
  profil.
- **Microscope simulé** complet pour tout tester sans matériel (y compris un
  diaphragme de champ et un condenseur mal réglés, pour s'entraîner au Köhler).
- **Assistants** : alignement de Köhler guidé, aide à la mise au point, calibration
  automatique de la taille du pixel avec un micromètre objet, uniformité de
  l'éclairage, correction de champ plat.
- **Analyse calibrée en µm** : filtres, soustraction de fond (rolling ball),
  seuillages automatiques (mêmes méthodes qu'ImageJ), séparation des objets
  (watershed), analyse de particules, mesures. Et **Fiji lui-même** à travers
  PyImageJ : macros, commandes et plugins de ton installation Fiji.
- **Acquisition multidimensionnelle** (temps, Z, canaux, positions) décrite avec
  `useq-schema`, enregistrée en **OME-TIFF** avec toutes les métadonnées (lisible
  directement par Fiji/Bio-Formats).

## Installation (macOS, Linux, Windows)

Python 3.10 ou plus récent.

```bash
git clone https://github.com/DrQuercus/Fuji.git
```

```bash
cd Fuji
```

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e ".[all]"
```

Options : `gui` (interface), `micromanager` (compatibilité Micro-Manager), `imagej`
(pont Fiji), `all` (tout), `dev` (tout + outils de test).

### Caméra ToupTek

Le SDK ToupTek n'est pas inclus (licence du fabricant). Télécharge « ToupTek SDK »
(toupcamsdk.zip) sur le site de ToupTek, puis :

```bash
fuji touptek install ~/Downloads/toupcamsdk.zip
```

```bash
fuji touptek info
```

Détails et dépannage : [docs/TOUPTEK.md](docs/TOUPTEK.md).

### Fiji / ImageJ

Le pont Fiji a besoin de Java. Le plus simple est d'utiliser ton Fiji existant :

```bash
export FUJI_FIJI_PATH=/Applications/Fiji.app
```

Sans cette variable, PyImageJ télécharge ImageJ2 automatiquement au premier usage.

## Utilisation

Interface graphique avec le microscope simulé :

```bash
fuji
```

Avec ton microscope :

```bash
fuji --profile leitz-diaplan-touptek
```

Autres commandes :

```bash
fuji profiles
```

```bash
fuji check
```

```bash
fuji -p demo snap -o capture.ome.tif
```

Sans interface (script Python) :

```python
from fuji.core import Microscope
from fuji.analysis import Image, run

with Microscope.from_file("demo") as scope:
    frame = scope.snap()                       # image + métadonnées
image = Image.from_frame(frame)                # calibrée en µm
gray = run("Niveaux de gris", image)
mask = run("Seuillage automatique", gray, method="Otsu", objects="sombres")
table, objets = run("Analyse de particules", mask, intensity=gray, min_area_um2=20)
print(table.summary)                          # nombre, surface totale, %Area...
table.to_csv("particules.csv")
```

## Profils de microscopes

Un profil YAML décrit un setup : périphériques et pilotes, objectifs (grossissement,
ouverture numérique, calibration), canaux (fond clair, DAPI, GFP...). Profils fournis
(`fuji profiles`) :

| Profil | Description |
|---|---|
| `demo` | Microscope simulé motorisé, fond clair + fluorescence |
| `leitz-diaplan-touptek` | Leitz Diaplan manuel + ToupTek E3ISPM08300KPD |
| `micromanager-exemple` | N'importe quelle configuration Micro-Manager |

Les calibrations sont enregistrées dans `~/.fuji/profiles/` sans jamais modifier les
profils fournis.

## État de validation

Le code a été entièrement relu et chaque défaut corrigé est couvert par un test
(plus de 140 tests). Tout cela tourne sur du **matériel simulé** : la vraie caméra ToupTek,
macOS et la comparaison directe avec ImageJ restent à valider — voir
[la feuille de route](docs/FEUILLE_DE_ROUTE.md).

## Tests

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

Sous Linux sans écran : `xvfb-run -a pytest`. Comparaison directe avec ImageJ :
`FUJI_TEST_IMAGEJ=1 pytest tests/test_imagej_parity.py`.

## Licence

Fuji est un logiciel libre sous licence [BSD-3-Clause](LICENSE), comme napari,
pymmcore-plus et scikit-image sur lesquels il s'appuie.

Les composants externes gardent leur propre licence : Micro-Manager, Fiji/ImageJ
(installés séparément) et le SDK ToupTek, propriétaire, que Fuji n'inclut pas et
que chacun télécharge chez le fabricant.

## Contribuer

Les contributions sont bienvenues : nouveaux pilotes de matériel, opérations
d'analyse, profils de microscopes, traductions. Voir
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) pour ajouter un pilote ou une
opération, et lancer `pytest` avant de proposer une modification.
