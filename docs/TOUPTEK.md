# Caméras ToupTek (E3ISPM08300KPD et autres)

Fuji pilote les caméras ToupTek avec le **SDK officiel** (« toupcam »). Le SDK est
gratuit mais n'est pas redistribuable : chacun le télécharge chez ToupTek.

## Installation sur Mac

1. Télécharger le SDK sur le site de ToupTek (rubrique *Download → SDK*) :
   un fichier `toupcamsdk.zip`.
2. Installer Fuji (voir le README), puis copier le SDK au bon endroit :

   ```bash
   fuji touptek install ~/Downloads/toupcamsdk.zip
   ```

   Fuji copie `toupcam.py` et `libtoupcam.dylib` dans `~/.fuji/touptek`.
3. Si macOS bloque la bibliothèque (fichier téléchargé « mis en quarantaine ») :

   ```bash
   xattr -d com.apple.quarantine ~/.fuji/touptek/libtoupcam.dylib
   ```

4. Brancher la caméra sur un port **USB 3**, fermer ToupView/ToupLite, puis :

   ```bash
   fuji touptek info
   ```

   La caméra doit apparaître avec ses résolutions (3840×2160 pour la E3ISPM08300KPD).

5. Premier test sans interface :

   ```bash
   fuji -p leitz-diaplan-touptek snap -o test.ome.tif
   ```

6. Interface :

   ```bash
   fuji --profile leitz-diaplan-touptek
   ```

## Ce que fait le pilote

| Fonction | Détail |
|---|---|
| Live | mode vidéo de la caméra, chaque image est affichée |
| Capture / acquisition | déclenchement logiciel : une image exposée après la demande (nouvel essai si le déclenchement échoue) |
| Capture pendant le live | la première image entièrement exposée après la demande |
| 12 bits | `bit_depth: 12` dans le profil → RGB 16 bits (valeurs 0–4095) |
| Résolution | propriété « Résolution » (pleine résolution ou binning du capteur) ; la taille du pixel suit |
| Débranchement | erreur claire au lieu d'une attente sans fin |
| Couleur | balance des blancs automatique, température / teinte, saturation |
| Capteur | gain, gain de conversion (HCG/LCG/HDR si disponible), niveau de noir |

## Conseils pour mesurer

Pour des mesures d'intensité fiables, désactiver les « embellissements » de la caméra
(panneau Propriétés, mode expert) :

- **Gamma = 100** (réponse linéaire),
- **Contraste = 0**, **Luminosité = 0**,
- **12 bits**,
- exposition automatique **désactivée** (Fuji la désactive au démarrage),
- correction de champ plat (assistant « Éclairage »).

## Dépannage

| Symptôme | Piste |
|---|---|
| « SDK ToupTek introuvable » | relancer `fuji touptek install` avec le .zip |
| « Aucune caméra ToupTek détectée » | ToupView ouvert ? câble/port USB 3 ? |
| L'image est bleutée / rougeâtre | balance des blancs sur une zone vide de la lame |
| Le live saccade en pleine résolution | normal en 12 bits 4K ; passer en 1920×1080 pour viser |
| Erreur au chargement de la .dylib | voir l'étape 3 (quarantaine) |

## État de validation

Le pilote a été écrit d'après l'API du SDK et testé avec un **SDK simulé**
(`tests/fake_toupcam.py`) : format des lignes, 8/12 bits, déclenchement, live,
réglages, images en retard, échec de déclenchement, débranchement. Il reste à le
valider avec la vraie caméra : `fuji touptek info` et
`fuji -p leitz-diaplan-touptek snap` donnent les informations utiles en cas de souci.

Points à regarder en premier avec la vraie caméra :

1. les valeurs en 12 bits vont-elles bien de 0 à 4095 ? (si la caméra les étale sur
   16 bits, Fuji le détecte tout seul dès la première image : « Microscope →
   Informations matériel » indique alors 16 bits) ;
2. les couleurs sont-elles dans le bon ordre (rouge et bleu non inversés) ?
3. après un changement d'exposition, la capture suivante a-t-elle la bonne luminosité ?
