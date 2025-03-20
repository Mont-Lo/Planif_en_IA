# Ressources :
# https://ale.farama.org/getting-started/
# https://ale.farama.org/gymnasium-interface/ 
# https://blog.coddity.com/articles/ag-atari/


# Installation
# pip install gymnasium[atari]
# pip install pygame==2.5.2 (plus nécessaire je crois)

import gymnasium as gym
import ale_py

# Créer l'environnement
env = gym.make("ALE/Freeway-v5", render_mode="human") #full_action_space=True

# render_mode="rgb_array" (pour récupérer les images du jeu en tant que tableau NumPy)
# Retourne une image de l’état du jeu sous forme d’un tableau NumPy 
# ((hauteur, largeur, 3)) au format RGB.


# Réinitialiser l'environnement
obs, info = env.reset()

"""
env.reset : Relance une partie et de retourne les informations initiales du jeu,
env.step : Lance l'étape suivante de la partie en cours et retourne les informations liées,
env.render : Affiche l'état du jeu à l'instant donné,
env.action_space.sample() : Retourne aléatoirement une des actions possibles du jeu.
action_space = env.action_space.n # Nombre des actions possible
print("Espace d'action :", env.action_space)
"""

game_reward = 0
for _ in range(1000): #ou not done
    env.render()
    #action = env.action_space.sample() # Prendre une action aléatoire
    action = 1 #0 = ne rien faire, 1 = up, 2 = down
    obs, reward, done, truncated, info = env.step(action) #pas truncated dans la version de la source
    # obs = matrice : n x 3
    # Reward 0.0
    # truncated = False
    # Info... {'lives': 0, 'episode_frame_number': 40, 'frame_number': 40}
    # 4 x iteration
    game_reward += reward
env.close()

