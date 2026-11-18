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

## Version 0.4 (fait) — l'essentiel d'ImageJ dans Fuji

Objectif : ne plus avoir à basculer sur ImageJ dans la majorité des cas. Détail et
niveau de fidélité de chaque fonction dans [IMAGEJ.md](IMAGEJ.md).

- [x] 76 opérations (contre 19) : types, couleur, filtres, calculs, binaire, piles, ROI,
      mesures — avec les arrondis, voisinages et définitions d'ImageJ
- [x] Les 17 seuillages automatiques d'ImageJ et 8 seuillages locaux
- [x] ROI dessinées sur l'image (rectangle, ellipse, polygone, ligne, points), mesurées
      comme dans ImageJ ; profil d'intensité, histogramme, « Mesurer » au clavier
- [x] Recherche de maxima (règle de proéminence d'ImageJ), colocalisation (Pearson,
      Manders, seuils de Costes), déconvolution des colorants histologiques
- [x] Aperçu en direct, traitement du plan affiché ou de toute la pile, graphiques,
      export de la vue avec barre d'échelle, réglage de l'échelle sur une ligne connue

## Version 0.5 (fait) — motorisation et pilotage par une IA

Détails dans [MOTORISATION.md](MOTORISATION.md) et [IA.md](IA.md).

- [x] **Pilote de moteurs pas à pas** (`gcode`) : cartes GRBL / FluidNC / grblHAL et Marlin ;
      platine XY, mise au point, roue à filtres, LED ; limites de course, rattrapage du
      jeu, origine, prise d'origine, arrêt d'urgence
- [x] Contrôleur simulé et profil `motorise-simulation` pour tout essayer sans matériel
- [x] **Assistant « Platine »** : orientation de la platine par rapport à la caméra, jeu
      mécanique — mesures refusées plutôt que fausses sur une image vide ou sombre
- [x] Centrage par double-clic sur l'image, boutons Stop, déplacements sans figer la fenêtre
- [x] **Mosaïques** : plan en serpentin, recalage des tuiles sur l'image, assemblage
- [x] **Autofocus pendant l'acquisition** (à chaque position ou point de temps) ; sur un
      champ vide la mise au point précédente est conservée
- [x] **Serveur MCP pour IA** : 37 outils (réglages, images, moteurs, acquisition, analyse),
      en local, depuis la fenêtre de Fuji ou à distance avec mot de passe
- [x] **Garde-fous** : lecture seule, moteurs interdits, pas maximal par commande, dossier
      de travail fermé, consignes manuelles jamais supposées faites, arrêt d'urgence
      toujours accepté
- [x] Les erreurs déjà expliquées à l'écran ne déclenchent plus un second message technique
- [x] **Application macOS** : `Fuji.app` dans un `.dmg`, pour processeurs Apple et Intel,
      construite et essayée automatiquement sur des Mac de GitHub (`fuji selftest`) ;
      journal des erreurs dans `~/.fuji/fuji.log` — voir [INSTALLATION_MAC.md](INSTALLATION_MAC.md)
- [x] **Tests de l'interface sur macOS** : les 30 tests y passent, à chaque modification
- [x] Live Micro-Manager : il se figeait au bout de quelques secondes avec une caméra plus
      rapide que la boucle d'affichage (défaut révélé par les machines de GitHub)

## Version 0.6 (fait) — premiers retours sur le vrai matériel

La caméra ToupTek fonctionne sur le Mac : l'image arrive. Corrections et ajouts demandés
après ce premier essai :

- [x] **La fenêtre tient sur un portable** : elle ne pouvait pas descendre sous 1455 pixels
      de large ; désormais ≈ 750 (la barre rapide passe à la ligne, la barre du haut se
      compacte)
- [x] **Deux espaces séparés, Microscope et Analyse**, chacun avec sa vue d'image : les
      outils d'analyse (ROI…) ne restent plus actifs ailleurs ; le live se met en pause
      pendant l'analyse
- [x] **Liste des images prises** : chaque capture est enregistrée tout de suite
      (`~/Fuji/Captures/<date>`, OME-TIFF + vignette) ; on l'ouvre dans Analyse quand on en a
      besoin ; renommer, supprimer (corbeille), changer de dossier. Les acquisitions et les
      mosaïques y arrivent aussi
- [x] **Profondeur de champ étendue en direct**, à la main — voir [EN_DIRECT.md](EN_DIRECT.md)
- [x] **Panorama en direct**, à la main : suivi du déplacement, champ ajouté à l'arrêt,
      position retrouvée si le suivi est perdu, assemblage final en pleine résolution
- [x] Simulateur : échantillon épais (« Épaisseur de l'échantillon ») pour essayer sans
      microscope
- [x] Sur macOS, les tests et l'essai automatique ne touchent plus aux préférences de
      l'utilisateur (elles étaient partagées)

## Version 0.7 (fait) — panneaux toujours ouverts

- [x] Espace Microscope : le **Contrôle** reste ouvert à gauche de l'image, utilisable en
      même temps que les Assistants, l'Acquisition ou la liste des Images (à droite)
- [x] Espace Analyse : la **liste des Images** reste ouverte à gauche, à côté des outils et
      des calques
- [x] Chaque panneau se replie (bouton dans la barre de gauche, Ctrl+Shift+B / Ctrl+B) ;
      Contrôle ouvert, la barre sous l'image ne montre plus que l'histogramme (pas de
      réglages en double)
- [x] Panneau de contrôle plus étroit (lampes, objectif et canal réorganisés) ; outils de
      dessin des ROI qui passent à la ligne

## Version 0.8 (fait) — une analyse plus facile à comprendre

- [x] **Barre d'affichage sous l'image** (espace Analyse), toujours présente : l'image
      affichée (taille, type, échelle), son **histogramme** (plan courant) avec les deux
      bornes du contraste à tirer à la souris, **min / max**, **Auto** (0,35 % saturés,
      comme ImageJ), **Tout**, **gamma** et **couleur** (gris, vert, magenta, chaleur…).
      Elle ne change que l'affichage : les mesures gardent les vraies valeurs
- [x] **Outils rangés par tâche** : « Améliorer l'image », « Isoler les objets »,
      « Retoucher un masque », « Compter et mesurer », « Intensités et profils »,
      « Couleurs et canaux »… (12 menus), avec **une ligne d'explication sous chaque
      outil** ; les mêmes menus dans la barre de menus « Analyse »
- [x] **Une fiche par outil** : ce qu'il fait, quand l'utiliser, son équivalent ImageJ,
      les détails techniques à la demande, puis les réglages et « Exécuter »
- [x] **Démarches guidées**, étape par étape : compter et mesurer des cellules, mesurer
      une surface ou une longueur, profil d'intensité, compter des spots, préparer une
      figure, colocalisation, projection d'une pile Z. Chaque étape ouvre le bon outil
      avec le bon calque ; pour le comptage, Fuji devine si les objets sont sombres (fond
      clair) ou clairs (fluorescence)
- [x] Recherche sans accents ni majuscules, dans les noms, les explications et les noms
      ImageJ (« seuil », « compter », « Gaussian Blur »)

## Version 0.8.1 (fait) — ouvrir et fermer les images

Corrections après essai : une image ouverte dans Analyse ne se fermait plus, et l'ancienne
restait quand on en ouvrait une autre.

- [x] La corbeille du panneau **Calques** ne faisait rien (ni le bouton « recentrer », la
      grille, la 3D) : ces boutons de napari cherchaient une fenêtre napari que Fuji n'a pas.
      Ils sont reliés directement à la vue
- [x] **Une image à la fois** : ouvrir une image de la liste remplace celle qui est ouverte
      (et ses résultats) ; Fuji ne demande que si des masques ou des ROI seraient perdus
      (« Fermer », « Garder ouverte », « Annuler »)
- [x] Bouton **Fermer** sous l'image et Fichier → Fermer l'image (Ctrl+W)
- [x] Plusieurs images ouvertes (clic droit → « Ouvrir à côté ») : une seule est affichée,
      on choisit laquelle sous l'image ; chacune garde ses résultats
- [x] Supprimer une image de la liste la ferme aussi dans Analyse ; la renommer renomme
      l'image ouverte (avant, la rouvrir créait un doublon)

## Version 0.9 (fait) — comme le logiciel de la caméra, et un tour du code

Retour d'essai : le panorama et la profondeur de champ étendue à la main étaient peu
pratiques, et l'image de référence fonctionnait mal ; ceux du logiciel ToupTek sont meilleurs.

- [x] **Image de référence** refaite : la correction se fait à la sortie de la caméra, donc
      **le live est corrigé** (avant : seulement les captures, et le fichier était relu à
      chaque image — plus d'une demi-seconde en 4K, ce qui ralentissait aussi le panorama).
      Commandes dans Contrôle → Caméra ; moyenne de 16 images ; refus d'une référence saturée
      ou trop sombre, alerte si un échantillon est dans le champ ; 15 ms par image 4K
- [x] **Panorama en continu** : plus besoin de s'arrêter sur chaque champ. L'ancien attendait
      une immobilité parfaite que la main n'atteint jamais (2 champs posés sur tout un
      parcours, sur le banc d'essai). Flou de bougé estimé (net / un peu vite / trop vite),
      zones floues remplacées quand on repasse lentement, poussières fixes retirées du suivi,
      éclairage inégal compensé, raccords plus nets, progression de l'assemblage
- [x] **Profondeur de champ étendue** : on ne reste plus bloqué sur « la lame a bougé » ; un
      fond vide ne scintille plus ; plus de halo flou autour des objets ; 2,5 fois plus rapide
      en 4K ; le live reste visible en vignette
- [x] Compositions : état écrit **sur l'image**, cadre coloré, vue qui suit le panorama,
      « Recommencer », menu Microscope (Ctrl+Shift+P, Ctrl+Shift+D)
- [x] **Banc d'essai** qui imite une main (mouvement continu, tremblement, flou de bougé,
      poussières, bruit, images 4K), pour ne plus régler ces outils à l'aveugle
- [x] **Tour du code** (deux relectures complètes, code puis libellés) — corrigé :
      boutons de mode des calques et clic droit sur la liste des calques (ils levaient une
      erreur à chaque clic) ; fichier glissé sur la fenêtre ; centrage par double-clic
      (platine motorisée), jamais déclenché ; images d'une acquisition cachées sous le live ;
      « Exécuter » ignoré pendant un aperçu ; messages effacés par les coordonnées de la
      souris ; caméra débranchée signalée ; exposition automatique hors du fil de l'interface ;
      même affichage d'une image ouverte depuis la mémoire ou le disque
- [x] Libellés harmonisés : « Images », « Outils », « Contrôle », « Configuration »,
      « Préférences » ; boutons Oui / Non / Annuler en français ; réglages des outils en
      français (« rayon du flou », « tolérance »…) ; messages d'erreur qui disent quoi faire ;
      vouvoiement partout ; sur Mac, Mesurer et Histogramme ne prennent plus ⌘M et ⌘H

## Version 1.0 (fait) — des réglages partout dans l'analyse

Retour d'essai : l'outil d'analyse « manquait terriblement de réglages ». État avant : 76
outils dont 24 sans aucun réglage, 129 réglages en tout, saisis dans de simples cases de
nombres ; un affichage limité à min / max, gamma et couleur ; un tableau de résultats figé ;
des ROI sans aucun réglage.

- [x] **Fiches d'outils** : pour chaque nombre un curseur et un champ, une ligne d'explication,
      l'équivalent en µm ; réglages avancés repliés dans la fiche (plus besoin du mode
      expert) ; « Valeurs par défaut » ; derniers réglages retenus ; bornes des curseurs
      d'après l'image (seuils, niveaux) ; aperçu comparable à l'original
- [x] **Quatre outils de retouche** : Réglages de l'image (luminosité, contraste, gamma,
      saturation, teinte, température, dominante, netteté, bruit), Niveaux, Balance des
      couleurs, Réduire le bruit (5 méthodes)
- [x] **Options ajoutées** aux outils existants : seuillages (lisser avant, remplir les trous,
      retirer les petits objets), contours (Sobel, Scharr, Prewitt, Laplacien, Canny), balance
      des blancs (force, cible), profil et histogramme (canal, plage, lissage), accentuation
      (force), lissage (passes), soustraction de fond (voir le fond estimé). Sans y toucher,
      le résultat reste celui d'ImageJ. 80 outils, 175 réglages
- [x] **Barre d'affichage** plus compacte et plus complète : zoom, luminosité, contraste,
      négatif, transparence et contours des masques, part de pixels saturés d'« Auto »,
      lissage à l'écran, barre d'échelle (position, couleur, taille, longueur, fond),
      « appliquer le contraste aux pixels »
- [x] **Tableau de résultats** : colonnes au choix, décimales, virgule décimale, tri,
      statistiques par colonne, distribution d'une colonne, « Cumuler » une série d'images
- [x] **Dessin** : couleur, épaisseur (constante à l'écran), remplissage, taille des points ;
      numéros et mesures affichés sur l'image, en direct ; main levée, trajet, angle,
      modification des sommets ; couleur automatique selon le fond (clair ou sombre)
- [x] Corrigé au passage : erreur au changement de thème après la fermeture d'un tableau
      avec graphique ; l'image se réajuste quand le volet des résultats s'ouvre

Limite connue : la mesure affichée à côté d'une forme est celle de son tracé ; le tableau
compte les pixels entiers. Les deux peuvent différer d'une fraction de pour cent.

## Prochaines étapes

### 1. Valider sur ton matériel (priorité)

Tout ce qui précède est vérifié par environ 290 tests automatiques, mais **sur du
matériel simulé**. Voici ce qui n'a jamais pu être essayé et reste à valider :

- [x] **La vraie caméra** : la E3ISPM08300KPD se connecte et l'image arrive (Mac)
- [ ] Image de référence, panorama et profondeur de champ étendue (refaits en 0.9) sur de
      vraies lames : vitesse permise, seuil de bruit, fidélité des couleurs, fluidité en 4K
- [ ] Vérifier l'alignement des valeurs 12 bits (0–4095 attendu) et l'ordre RGB
- [ ] Vérifier qu'une capture pendant le live donne bien une image nette et récente
- [x] **macOS** : les tests de l'interface et l'essai de l'application empaquetée
      réussissent sur des Mac de GitHub (macOS 15 et 26, processeurs Apple et Intel)
- [ ] **Ton Mac** : télécharger le `.dmg`, autoriser Fuji au premier lancement, vérifier
      l'affichage (aucun œil humain n'a encore vu Fuji sur un écran de Mac)
- [ ] Renseigner les vrais objectifs et l'adaptateur caméra dans le profil Diaplan
- [ ] Calibrer chaque objectif au micromètre (assistant « Calibration »)
- [ ] **ImageJ réel** : lancer le test de parité (`FUJI_TEST_IMAGEJ=1`, environ 90 cas). Les
      calculs suivent ceux du code source d'ImageJ, mais ils n'ont pas pu être comparés à
      ImageJ lui-même ici (pas d'accès au téléchargement de Fiji). Le test indique
      précisément quelle fonction diffère, s'il y en a
- [ ] **De vrais moteurs** : le pilote `gcode` n'a dialogué qu'avec un contrôleur simulé.
      Premier essai conseillé : une carte GRBL seule, sans mécanique, puis un seul axe
- [ ] **Une vraie IA** : le serveur MCP a été essayé avec un client de test, pas encore
      dans une conversation avec Claude Desktop ou Claude Code

### 2. Protocoles (l'équivalent des « journals » de MetaMorph et des macros d'ImageJ)
- [ ] Enregistrer une suite d'étapes (réglages, acquisition, analyse) dans un fichier
- [ ] Les rejouer, y compris les étapes manuelles
- [ ] Analyse par lots d'un dossier d'images
- [ ] Enregistrer et relire les ROI au format d'ImageJ (.roi, .zip)

### 3. Confort
- [ ] Profils d'expérience (« time-lapse cellules vivantes », « lame fixée »...)
- [x] Application macOS (`.app` dans un `.dmg`), construite et essayée automatiquement sur
      des Mac de GitHub — voir [INSTALLATION_MAC.md](INSTALLATION_MAC.md)
- [ ] Signature et notarisation Apple (demande un compte développeur) pour supprimer
      l'étape d'autorisation au premier lancement
- [ ] Installateur Windows (.msi) — la même configuration d'empaquetage le permet

### 4. Analyse avancée
- [ ] Segmentation par apprentissage (Cellpose, StarDist)
- [ ] Suivi d'objets dans le temps
- [ ] Gestionnaire de ROI compatible avec le ROI Manager d'ImageJ

### 5. Motorisation et grandes acquisitions
- [x] Pilote pour platine maison (Arduino/ESP32, GRBL) — version 0.5
- [x] Mosaïques et assemblage, autofocus entre positions — version 0.5
- [ ] Liste de positions mémorisées (revenir sur des points repérés à la main)
- [ ] Carte de mise au point pour les grandes mosaïques (lame inclinée)
- [ ] Égaliser l'éclairage entre les tuiles d'une mosaïque (raccords encore visibles)
- [ ] Tourelle d'objectifs et condenseur motorisés
- [ ] Écriture OME-Zarr pour les très gros volumes

### 6. Contrôle à distance
- [x] Moteurs branchés sur un autre ordinateur du réseau (port `socket://…`) — version 0.5
- [x] Pilotage à distance par une IA (`fuji mcp --http`, avec mot de passe) — version 0.5
- [ ] Serveur « microscope » sur Raspberry Pi (caméra + moteurs)
- [ ] Pilote distant côté ordinateur, live compressé
