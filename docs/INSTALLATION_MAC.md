# Installer Fuji sur Mac

Fuji existe en application Mac classique : un fichier `.dmg` à ouvrir, une icône à
glisser dans Applications. Rien d'autre à installer (ni Python, ni Terminal).

## 1. Choisir le bon fichier

Il y a un `.dmg` par type de processeur. Menu Pomme (en haut à gauche) → « À propos de ce Mac » :

| Ligne affichée | Fichier à prendre |
|---|---|
| **Puce** Apple M1, M2, M3, M4… | `Fuji-…-macOS-arm64.dmg` |
| **Processeur** Intel | `Fuji-…-macOS-x86_64.dmg` |

Il faut macOS 13 (Ventura) ou plus récent : c'est ce qu'exige Qt, la bibliothèque qui
dessine la fenêtre. Chaque fichier pèse environ 230 Mo.

## 2. Télécharger

- **Version publiée** : page [Releases](https://github.com/DrQuercus/Fuji/releases) du
  projet, rubrique « Assets ».
- **Dernière construction** : onglet [Actions](https://github.com/DrQuercus/Fuji/actions)
  → « Application macOS » → la dernière exécution réussie → « Artifacts » en bas de page
  (il faut être connecté à GitHub ; le fichier arrive dans une archive `.zip`).

## 3. Installer

1. Double-clic sur le `.dmg`.
2. Glisser **Fuji** sur le dossier **Applications**.
3. Éjecter le disque « Fuji ».

## 4. Premier lancement : autoriser Fuji

Fuji n'est pas « notarisé » par Apple (cela demande un compte développeur payant). Au
premier lancement, macOS refuse donc de l'ouvrir. C'est normal, et il suffit de
l'autoriser **une seule fois** :

- **macOS 15 (Sequoia) et plus récent** : lancer Fuji, fermer le message, puis ouvrir
  Réglages Système → **Confidentialité et sécurité**, descendre jusqu'à « Fuji a été
  bloqué… » et cliquer **Ouvrir quand même**.
- **macOS 13 et 14** : dans Applications, **clic droit** sur Fuji → **Ouvrir** → **Ouvrir**.

Si le message dit que Fuji « est endommagé » (cela arrive sur les Mac à puce Apple), ou si
tu préfères le Terminal, cette commande fait la même chose :

```bash
xattr -dr com.apple.quarantine /Applications/Fuji.app
```

## 5. La caméra ToupTek

Le pilote du fabricant n'est pas inclus dans Fuji (sa licence ne le permet pas). Une fois :

1. Télécharger « ToupTek SDK » (`toupcamsdk.zip`) sur le site de ToupTek.
2. Dans Fuji : **Réglages → Caméras ToupTek → Installer le SDK…** et choisir le fichier.
3. Choisir le microscope « Leitz Diaplan + ToupTek » dans la barre du haut.

## Où sont mes fichiers ?

| Quoi | Où |
|---|---|
| Profils de microscope, calibrations | `~/.fuji/` (dossier caché de ton dossier personnel) |
| Fichiers produits par un assistant IA | `~/Fuji/ia/` |
| Journal des erreurs (utile si quelque chose ne marche pas) | `~/.fuji/fuji.log` |

Pour ouvrir le dossier caché `~/.fuji` : dans le Finder, menu Aller → « Aller au dossier… »
et taper `~/.fuji`.

Désinstaller : mettre Fuji à la corbeille. Les dossiers ci-dessus restent, pour garder tes
calibrations ; supprime-les si tu veux tout effacer.

## Mettre à jour

Télécharger le nouveau `.dmg` et glisser Fuji dans Applications en remplaçant l'ancien.
Les profils et calibrations sont conservés.

## Utiliser les commandes du Terminal avec l'application

Les commandes décrites dans le reste de la documentation (`fuji profiles`, `fuji check`…)
fonctionnent en appelant le programme contenu dans l'application :

```bash
/Applications/Fuji.app/Contents/MacOS/Fuji profiles
```

Pour vérifier d'un coup que tout marche sur ton Mac (une fenêtre s'ouvre quelques
secondes sur le microscope simulé, puis se referme) :

```bash
/Applications/Fuji.app/Contents/MacOS/Fuji selftest
```

## Fabriquer le .dmg soi-même

### Sur GitHub (sans rien installer)

Onglet **Actions** → « Application macOS » → **Run workflow**. Une vingtaine de minutes
plus tard, les deux `.dmg` sont dans « Artifacts ». Chaque construction lance aussi
l'essai automatique ci-dessus sur un vrai Mac : un `.dmg` n'est produit que s'il réussit.

Quand une version est publiée (étiquette `v0.5.0` par exemple), les `.dmg` sont joints
automatiquement à la page de la version.

### Sur ton Mac

Il faut Python 3.12 et les outils de ligne de commande d'Apple (`xcode-select --install`).

```bash
./packaging/macos/build_dmg.sh
```

Le `.dmg` arrive dans le dossier `dist/`. Fabriqué sur place, il s'ouvre sans l'étape
d'autorisation du point 4.

## Limites connues

- Pas de signature Apple : l'étape d'autorisation du premier lancement est inévitable sans
  compte développeur Apple (99 $ par an). Si un jour le projet en a un, la recette de
  construction sait signer et notariser.
- Le pont vers Fiji (macros, plugins) a toujours besoin de Java, à installer à part.
- Une application par processeur : le fichier « arm64 » ne démarre pas sur un Mac Intel.
