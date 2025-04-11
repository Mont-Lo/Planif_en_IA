# IFT702 : Planification en IA - Projet

##  Avertissement
- Utilisez un environnement virtuel pour garantir la compatibilité des packages (torch, etc.). Il y a un fichier requirements.txt pour cela.
- Chaque modèle génère un buffer de ~5.25 GB. Si vous activez tous les agents en même temps, prévoyez au moins 26.25 GB d’espace disque libre. 
- Afin de supprimer à la main les fichiers buffers générés, il faut auparavant sélection l'option "Restart" du notebook afin de supprimer les variables en mémoire.

## Rainbow DQN

### Structure du code :
- agent/ : contient les différentes classes d'agents DQN avec leurs variantes (Rainbow, NoNoisy, NoNoisyNoCategorical).
- buffers/ : implémente les buffers de replay (classique et Prioritized Experience Replay), ainsi que la structure SegmentTree.
- reseaux/ : contient les architectures de réseaux neuronaux utilisées (torch.nn.Module).
- utils/ : fonctions utilitaires diverses.

#### Contenu généré automatiquement :
- score_agents_* ou nb_crashes_* : fichiers .pkl de sauvegarde (voir notebook *train_test_interval.ipynb*).
- buffer_saves/ : dossier contenant des fichiers .npy pour les matrices des buffers.
- videos/ : vidéos d’agents testant l’environnement.

#### Résultat : vous trouverez tous nos résultats stockés ici
- resultat/ : 
    - 00. Execution 100 000 frames/ : résultats pour chaque agent et fonction de récompense après exécution de train_test_interval.ipynb (cf. rapport et présentation).
    - scores/ : scores extraits pour faciliter la génération des courbes de score reward (cf. notebook *figures.ipynb*)
    - Un fichier .xlsx contenant des résultats collectés manuellement.
    

## apply_models.ipynb
Ce notebook fournit un exemple complet d’exécution d’un modèle (ici no noisy, no dueling). Il permet de :
- créer un agent,
- l'entraîner sur un nombre d’itérations définies,
- le tester,
- générer une vidéo du test.


Paramètres modifiables :
- Le nombre de frames (num_frames) par défaut est de 10 000 (environ 20 à 40 minutes d'exécution selon votre machine).
- La fréquence de traçage (plotting_interval) des graphiques durant l'entraînement peut être ajustée (toutes les 200 itérations par défaut) dans la fonction train.
- Choix de la fonction de récompense :
    1 = reward_V1
    2 = reward_V2 (par défaut)
    3 = reward simpliste

Notes :
- Seul l'atteinte du but permet de tracer un point sur la courbe de score. L’évolution de la moyenne des 10 derniers scores est affichée dans le titre du graphique sur l'évolution des scores. Ce score correspond à la sortie de la fonction reward.
- Les autres modèles (Dueling, Noisy, etc.) sont disponibles mais commentés par défaut. Pour les tester, décommentez et changez le nom de l'agent à chaque étape (entraînement, test) de manière similaire que montrer. Pensez également à modifier video_path pour éviter d’écraser les vidéos précédentes. Avant de réaliser cette opération, il est conseillé de lire la partie "Avertissement".
- Pour éviter d’écraser les buffers sauvegardés si vous instanciez le même type d'agent (deux NoThree par ex.) : modifiez soit buffer_dir, soit le nom donné dans f'{buffer_dir}/NOM'.
- Durant l'entraînement vous pouvez visualiser le score reward, la perte (ainsi que la diminution d'epsilon pour les autres modèles) toutes les 200 itérations. 
- Pour une meilleure lisibilité on appelle "NoThree" le mdèle Rainbow no noisy, no dueling, no categorical.


## train_test_interval.ipynb
Ce notebook a été utilisé pour exécuter massivement les agents sur 15 machines différentes, en variant :
- l’agent (Rainbow, NoNoisy, etc.),
- la fonction de reward.

L’objectif était de minimiser les erreurs et manipulations sur chaque poste puisque le temps d'exécution  nécessaire était autour de 1h30:
- Peu de manipulation nécessaire,
- Nom de variables identique,
- Utilisation de commentaires pour activer/désactiver les agents rapidement.


## figure.ipynb
Ce notebook sert à générer les figures finales du projet :
- Comparaison des nombres de crash entre agents,
- Courbes d’évolution du score reward pour la V2.



## Mot de la fin

Have fun !