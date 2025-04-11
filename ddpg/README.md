# Adaptation DDPG du projet capstone vers notre jeu Freeway

## Contenu

- **main.py**: Permet d'entrainer et de tester l'algorithme
- **ddpg.py**: Comtient la classe principale DDPG_agent.
- **networks.py**: Structure réseaux acteur et critique.
- **noise.py**: Fonction Bruit pour l'exploration
- **utils.py**: Evalue l'algorithme, gère également les graphiques

## Contenu automatisé :
- **model_final.pt** Sauvegarde du modèle (actuellement le poulet qui fait que d'avancer)

## Modification Essentiel fait pour que le code fonctionne

Réduction de la taille des networks
Passage d'actions continue -> discrète

## DDPG_agent (compris dans ddpg.py)

Les modifications d'hyperparamètres des réseaux (gamma, lr_c, lr_a, tau et sigma) se font directement dans l'initialisation
La modification des récompenses se fait directement dans la fonction train, pour l'instant il y a juste la détection des collisions sans pénalisations.
Pour l'instant, le nombre d'épisodes est égale à 15 à modifier dans le train.

## Execution

Lancer le fichier main.py :
Ligne 73 a commenter si on veut créer un nouveau modèle et non continuer l'apprentissage de l'ancien
Ligne 77 a commenter si on veut juste évaluer/tester le modèle chargé à la ligne 73

Lorsque l'on lance l'algorithme en entrainement, nous avons un problème de performance lorsque le réseau critique s'actualise (lorsque le buffer atteint sa capacité empirique et que l'apprentissage commence)