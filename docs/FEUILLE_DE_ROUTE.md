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

## Prochaines étapes

### 1. Valider sur ton matériel (priorité)
- [ ] `fuji touptek install` puis `fuji touptek info` sur le Mac avec la E3ISPM08300KPD
- [ ] Vérifier l'alignement des valeurs 12 bits (0–4095 attendu) et l'ordre RGB
- [ ] Renseigner les vrais objectifs et l'adaptateur caméra dans le profil Diaplan
- [ ] Calibrer chaque objectif au micromètre (assistant « Calibration »)
- [ ] Lancer le test de parité avec ImageJ (`FUJI_TEST_IMAGEJ=1`)

### 2. Protocoles (l'équivalent des « journals » de MetaMorph)
- [ ] Enregistrer une suite d'étapes (réglages, acquisition, analyse) dans un fichier
- [ ] Les rejouer, y compris les étapes manuelles
- [ ] Analyse par lots d'un dossier d'images

### 3. Confort
- [ ] Assistant de création de profil (sans éditer le YAML)
- [ ] Histogramme en direct et indicateur de saturation
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
