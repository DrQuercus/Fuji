# Feuille de route

## Version 0.1 (fait)

- [x] Cœur indépendant de l'interface : périphériques, profils YAML, microscope
- [x] Périphériques manuels avec consignes (objectifs, cubes, lampe, mise au point, platine)
- [x] Microscope simulé réaliste (Köhler, défocalisation, fluorescence, micromètre)
- [x] Pilote ToupTek (SDK toupcam) : live, déclenchement logiciel, 12 bits, réglages
- [x] Compatibilité Micro-Manager (pymmcore-plus), canaux importés automatiquement
- [x] Acquisition multidimensionnelle (useq-schema) → OME-TIFF avec métadonnées par plan
- [x] Planification : durée, taille, avertissements, actions manuelles à prévoir
- [x] Assistants : Köhler guidé, mise au point (+ autofocus si motorisé), calibration
      du pixel au micromètre, uniformité de l'éclairage, champ plat
- [x] Analyse : filtres, fond (rolling ball), seuillages ImageJ, binaire, watershed,
      analyse de particules et mesures avec les définitions d'ImageJ
- [x] Pont Fiji : macros, commandes, ouverture dans Fiji
- [x] Interface napari : panneaux adaptés au matériel, mode simple / expert

## Version 0.2 (fait)

- [x] Interface sur mesure : thèmes sombre / clair, couleurs d'accent, icônes Fuji,
      navigation latérale, barre rapide, histogramme en direct avec saturation
- [x] Éditeur complet de microscope : statif, périphériques et options de chaque pilote,
      rôles, objectifs, cubes de fluorescence, canaux, YAML ; vérification et test de connexion
- [x] Assistant « Nouveau microscope » (modèles) et changement de microscope à chaud
- [x] Catalogue : objectifs courants, cubes (dont Leitz Ploemopak)

## Version 0.3 (fait) — relecture complète du code

Relecture de tout le projet, avec un test de non-régression pour chaque défaut corrigé.

- [x] **Caméra ToupTek** : les images 12 bits ne partagent plus la mémoire du SDK (elles
      pouvaient être écrasées par l'image suivante) ; le live ne se fige plus après un
      changement de résolution ; une capture n'accepte qu'une image exposée après la
      demande ; nouvel essai si le déclenchement échoue ; erreur claire si la caméra est
      débranchée ; la taille du pixel suit la résolution choisie
- [x] **Acquisition** : une erreur au démarrage ne laisse plus le moteur « en cours » ;
      les piles Z relatives partent de la mise au point de départ ; l'intervalle du
      time-lapse est respecté position par position ; deux acquisitions du même nom ne
      s'écrasent plus (`essai`, `essai_1`...) ; le plan affiché correspond exactement à ce
      que fait le moteur
- [x] **Mesures identiques à ImageJ** : contour (Wand), périmètre, enveloppe convexe,
      Feret, ellipse, seuil par défaut (arrondi Java), rolling ball, médian, binaire
- [x] **Calibration au micromètre** réécrite : juste à ±0,05 % sur le simulateur de 4x à
      100x, caméra tournée ou non, image 4K ; refuse les images floues et signale les
      mesures peu fiables au lieu de donner un chiffre faux
- [x] **Interface** : plus de blocage possible quand une consigne manuelle arrive pendant
      une acquisition ; l'ancien microscope est fermé avant d'ouvrir le nouveau (même
      caméra) avec retour au précédent en cas d'échec ; réglages verrouillés pendant une
      acquisition ; autofocus, calibration et champ plat ne figent plus la fenêtre ;
      l'analyse d'une pile porte sur le plan affiché ; un champ plat demandé mais absent
      (ou saturé) est signalé
- [x] **Micro-Manager** : capture possible pendant le live ; le live fonctionne avec une
      caméra plus rapide que l'affichage ; une erreur ne laisse plus la caméra « occupée »
- [x] Exposition automatique qui converge depuis une image noire ou saturée ; gros
      fichiers TIFF lus à la demande ; fichiers CSV lisibles par Excel (accents)

## Prochaines étapes

### 1. Valider sur ton matériel (priorité)

Tout ce qui précède est vérifié par plus de 140 tests automatiques, mais **sur du
matériel simulé**. Trois choses n'ont jamais pu être essayées et restent à valider :

- [ ] **La vraie caméra** : `fuji touptek install` puis `fuji touptek info` sur le Mac
      avec la E3ISPM08300KPD
- [ ] Vérifier l'alignement des valeurs 12 bits (0–4095 attendu) et l'ordre RGB
- [ ] Vérifier qu'une capture pendant le live donne bien une image nette et récente
- [ ] **macOS** : l'interface n'a été lancée que sous Linux
- [ ] Renseigner les vrais objectifs et l'adaptateur caméra dans le profil Diaplan
- [ ] Calibrer chaque objectif au micromètre (assistant « Calibration »)
- [ ] **ImageJ réel** : lancer le test de parité (`FUJI_TEST_IMAGEJ=1`). Les calculs
      suivent ceux du code source d'ImageJ, mais ils n'ont pas pu être comparés à ImageJ
      lui-même ici (pas d'accès au téléchargement de Fiji)

### 2. Protocoles (l'équivalent des « journals » de MetaMorph)
- [ ] Enregistrer une suite d'étapes (réglages, acquisition, analyse) dans un fichier
- [ ] Les rejouer, y compris les étapes manuelles
- [ ] Analyse par lots d'un dossier d'images

### 3. Confort
- [ ] Profils d'expérience (« time-lapse cellules vivantes », « lame fixée »...)
- [ ] Installateur macOS (.app)

### 4. Analyse avancée
- [ ] Segmentation par apprentissage (Cellpose, StarDist)
- [ ] Suivi d'objets dans le temps, colocalisation
- [ ] Gestionnaire de ROI compatible avec le ROI Manager d'ImageJ

### 5. Motorisation et grandes acquisitions
- [ ] Pilote pour platine maison (Arduino/ESP32, GRBL ou micrologiciel dédié)
- [ ] Mosaïques et assemblage, autofocus entre positions
- [ ] Écriture OME-Zarr pour les très gros volumes

### 6. Contrôle à distance
- [ ] Serveur « microscope » sur Raspberry Pi (caméra + moteurs)
- [ ] Pilote distant côté ordinateur, live compressé
