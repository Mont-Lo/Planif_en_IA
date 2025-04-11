# IFT702 : Planification en IA - Projet

## Rainbow DQN

Structure du code :
- agent : classes d'agents de DQN (différentes combinaisons)
- buffers : classes des buffers (base et PER) et de SegmentTree
- reseaux : classes des réseaux neuronaux (nn.Module)
- utils : fichiers de fonctions annexes

Contenu automatisé :
- agent_saves : fichiers pickle de sauvegarde automatique
- buffer_saves : fichiers numpy contenant les matrices des buffers
- videos : vidéos de tests de jeu

Autre dossier et fichier :
- resultat : vous trouverez tous nos résultats stockés ici. 
    - Dans le dossier 00. Execution 100 000 frames, bous trouverez les résultats pour chaque agent et chaque fonction reward après l'exécution de train_test_inteval.ipynb dont nous avons parlé dans la présentation et le rapport.
    - dans le dossier scores, on a copier les scores afin de faciliter la recherche des fichiers pour tracer la courbe de score reward
    - fichier excel contenant certains résultats que nous avons relevés à la main


## apply_models.ipynb
Vous avez un exemple d'exécution dans ce notebook afin de vous permettre de vous familiariser avec les modèles.

Le notebook permet de créer un modèle Rainbow, de l'entraîner, de le tester et de visualiser la vidéo du test. Durant l'entraînement vous pouvez visualiser le score reward, la perte (ainsi que la diminution d'epsilon pour les autres modèles) toutes les 200 itérations. Vous pouvez modifier ce paramètre en changant la valeur de plotting_interval dans la fonction train.

Les autres modèles sont en commentaire afin de ne pas les initialiser par erreur (lisez bien la section avertissement avant de vous amuser avec).

Quelques remarques :
- On ne trace un point de score qu'une fois qu'on a atteint le but. Vous pouvez voir l'évolution de la moyenne des 10 derniers scores dans le titre du graoh de score reward.
- Les autres modèles sont mis en commentaire. En dehors de l'initialisation qui change mais vous est donné, l'entraînement et le test se déroule exactement pareil qu'à rainbow. Il faut juste changer le nom de l'agent.

Vous pouvez jouer sur les paramètres (case du notebook juste après initialisation) :
- on arrive à obtenir de bons résultats à partir de 10 000 frames

Par défaut la reward fonction est de 2 #1 = reward_V1, 2 = reward_V2, 3 = reward simpliste

Pour exécuter le code avec num_frames = 10 000 itérations, compter au minimum 20 minutes et selon les performances de votre PC jusqu'à 40 min.

buffer_saves enregistre les buffer des modèles. Si le nom existe déjà il ecrase les anciennes valeurs il faut donc soit changer le buffer_dir général ou le paramètre passé à la fonction, par ex pour Rainbow f'{buffer_dir}/rainbow'.

Lors du test changer l'indice du video_path afin de rester cohérent pour l'enregistrement des vidéos.
Affichage des vidéos

Appeler comunément no three celui on enlève 3 composantes pour une meilleure lisibilité

Dans train_test_interval, c'est le fichier qu'on a exécuter sur les 15 PC en faisant varier l'agent utilisé et le nombre de reward.

D

## Avertissement
Il faut utiliser un environment virtuel afin de pourvoir exécuter ce code. 
En outre, exécuter un model créera un buffer en local sur votre machine de taille 5.25GB. Initialiser les 5 modèles en même temps (option Restrat disponible) revient donc à avoir un espace mémoire de 26,25 GB disponible. Assurez vous bien que cette condition soit rempli si vous ne voulez pas avoir d'erreur.


Have fun !