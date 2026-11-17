# Images composées en direct, à la main

Deux outils pour un microscope **manuel** (aucun moteur nécessaire) : l'image se construit
à l'écran pendant que tu tournes la mise au point ou que tu déplaces la lame, comme avec le
logiciel fourni par ToupTek. Ils marchent aussi avec des moteurs.

Les deux se trouvent en haut de la section **Acquisition** (espace Microscope), sous
« En direct, à la main ». Le live démarre tout seul. Le résultat arrive dans la liste
**Images**, déjà enregistré.

## Profondeur de champ étendue

Pour un objet plus épais que la profondeur de champ (grains de pollen, coupes épaisses,
insectes, cristaux) : une seule image nette partout.

![Profondeur de champ étendue en cours](edf.png)

1. Fais la mise au point **juste au-dessus** de l'objet (tout est flou).
2. Clique **Profondeur de champ étendue**.
3. Tourne **lentement** la mise au point fine jusqu'à **dépasser** le bas de l'objet. Chaque
   zone de l'image garde l'état le plus net vu jusque-là.
4. La barre « apport de la dernière image » retombe vers 0 % quand plus rien ne devient
   net : clique **Terminer**.

Conseils :

- Ne touche pas à la platine pendant ce temps. Si la lame bouge vraiment, Fuji ignore les
  images (« La lame a bougé ») : remets-la en place ou recommence.
- Lent et régulier vaut mieux que rapide : une zone traversée trop vite peut manquer son
  image la plus nette.
- Repasser dans l'autre sens ne gâche rien : seul ce qui est plus net remplace l'image.

## Panorama

Pour une zone plus grande que le champ de la caméra : déplace la lame à la main, les champs
s'assemblent.

![Panorama en cours](panorama.png)

1. Clique **Panorama**, puis reste immobile un instant sur le premier champ.
2. Pousse la lame **doucement** vers la zone suivante, en gardant environ un quart de champ
   en commun avec ce qui est déjà couvert, puis **arrête-toi** une seconde : le champ
   s'ajoute (il n'est jamais pris pendant le mouvement, donc jamais flou de bougé).
3. Continue de la même façon, dans n'importe quel ordre. **Terminer** assemble le résultat
   en pleine résolution, raccords fondus.

Le cadre autour du live indique l'état du suivi :

| Cadre | Signification |
|---|---|
| vert | déplacement suivi ; à l'arrêt, le champ est ajouté |
| orange | zone déjà couverte (rien à ajouter), ou en attente du premier champ |
| rouge | position perdue (déplacement trop rapide) : **reviens sur une zone déjà couverte**, le suivi reprend tout seul |

**Retirer le dernier champ** enlève un champ raté ; il faut ensuite bouger la lame pour en
reprendre un.

Pour des raccords invisibles :

- fais un **réglage de Köhler** (assistant), qui rend l'éclairage plus uniforme ;
- mieux encore, enregistre une **référence de champ plat** (assistant « Éclairage ») et
  coche « Appliquer » : chaque champ est alors corrigé avant d'être assemblé.

Un très grand panorama est assemblé à une résolution réduite (moitié, tiers…) pour ne pas
dépasser la mémoire de l'ordinateur ; le résumé dans la liste « Images » l'indique.

## Essayer sans microscope

Avec le microscope de démonstration : dans **Contrôle → Toutes les propriétés** (mode
expert), règle « Épaisseur de l'échantillon » sur 40 µm pour simuler un objet épais, puis
fais varier la mise au point. Pour le panorama, déplace la platine simulée par petits pas.
