# Adaptation DDPG du projet capstone vers notre jeu Freeway

## Contenu

- **main.py**: Permet d'entrainer et de tester l'algorithme
- **ddpg.py**: Comtient la classe principale DDPG_agent.
- **networks.py**: Structure des réseaux acteur et critique.
- **noise.py**: Fonction Bruit pour l'exploration
- **utils.py**: Evalue l'algorithme, gère également les graphiques

## Contenu automatisé :
- **model_final.pt** : Sauvegarde du modèle (actuellement le poulet qui fait que d'avancer)

## Modification Essentiel fait pour que le code fonctionne

- Réduction de la taille des networks
- Passage d'actions continue -> discrète

## DDPG_agent (compris dans ddpg.py)

- Les modifications d'hyperparamètres des réseaux (gamma, lr_c, lr_a, tau et sigma) se font directement dans l'initialisation
- La modification des récompenses se fait directement dans la fonction train, pour l'instant il y a juste la détection des collisions sans pénalisations.
- Le nombre d'épisodes est égale à 15 à modifier dans le train.

## Execution

Lancer le fichier main.py :
- La ligne 73 permet d'executer le modèle déjà entrainé qui est sauvegardé dans model/model_final.pt, si on veut génerer un nouveau model on doit commenter cette ligne
- La ligne 77 permet d'entrainer le modèle. Si on charge un modèle on peut soit continuer l'entrainement ou bien commenter cette ligne pour passer directement a la phase de test

Lorsque l'on lance l'algorithme en entrainement, nous avons un problème de performance lorsque le réseau critique s'actualise (lorsque le buffer atteint sa capacité empirique et que l'apprentissage commence)
