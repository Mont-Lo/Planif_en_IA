import pickle

from buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from buffers.ReplayBuffer import ReplayBuffer
from utils.processing import preprocess_observation
from utils.reward_function import instant_reward1, instant_reward2, instant_reward3
from typing import Dict, List, Tuple, Type
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
import numpy as np
import gc

class NoNoisyAgent:
    """
    Agent DQN (sans bruit) interagissant avec un environnement Gym pour le jeu Freeway

    Attributs :
        Network (objet) : architecture du réseau de neurones utilisé pour approximer la fonction de valeur.
        env (gym.Env) : environnement Gym dans lequel l'agent évolue.
        memory (PrioritizedReplayBuffer) : mémoire de rejouabilité avec priorités pour stocker les transitions.
        batch_size (int) : taille des lots pour l'échantillonnage pendant l'entraînement.
        target_update (int) : fréquence (en nombre de pas) de mise à jour du réseau cible.
        epsilon (float) : paramètre de la politique ε-greedy.
        epsilon_decay (float) : taux de décroissance de epsilon à chaque pas.
        max_epsilon (float) : valeur maximale de epsilon.
        min_epsilon (float) : valeur minimale de epsilon.
        gamma (float) : facteur de réduction (discount) pour les récompenses futures.
        dqn (Network) : réseau principal utilisé pour choisir les actions.
        dqn_target (Network) : réseau cible utilisé pour stabiliser l’apprentissage.
        optimizer (torch.optim) : optimiseur pour l’apprentissage du réseau principal.
        transition (list) : transition actuelle composée de (état, action, récompense, état suivant, done).
        v_min (float) : valeur minimale du support pour le DQN catégoriel.
        v_max (float) : valeur maximale du support pour le DQN catégoriel.
        atom_size (int) : nombre d’unités du support (atomes) pour la distribution de valeur.
        support (torch.Tensor) : support discret utilisé dans le DQN catégoriel.
        use_n_step (bool) : indique si l’apprentissage n-étapes est utilisé.
        n_step (int) : nombre d’étapes dans le calcul du TD n-étapes.
        memory_n (ReplayBuffer) : mémoire secondaire pour le TD n-étapes.
        instant_reward (fonction) : fonction utilisée pour calculer la récompense instantanée.
        is_test (bool) : mode test ou entraînement.
        num_video (int) : numéro du dossier de la prochaine vidéo que l'on va crée durant le test
        coord (int) : coordonnée de référence (peut dépendre de la configuration de l’environnement).
        is_crash (bool) : indicateur de collision.
        lst_recent_crashes (List[bool]) : historique des dernières collisions.
        nb_crashes (int) : nombre total de collisions.
        mask : carte du jeu dont on supprimer les informations superflus (mis à jour à chaque frame)
    """

    def __init__(
        self,
        Network : Type[object],
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        seed_test: int,
        buffer_path: str,
        reward_function:int =2,
        num_video: int = 0,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        coord: int = 0, # 187 si on ne fait pas la soustraction
        lst_recent_crashes: List = [False, False],
        nb_crashes: int = 0,
    ):
        """
        Initialise un agent DQN sans bruit.

        Args :
            Network (Type[object]) : classe représentant l’architecture du réseau de neurones.
            env (gym.Env) : environnement Gym dans lequel l’agent évolue.
            memory_size (int) : taille maximale de la mémoire de rejouabilité.
            batch_size (int) : taille des lots pour l’échantillonnage.
            target_update (int) : fréquence des mises à jour du réseau cible.
            epsilon_decay (float) : taux de décroissance de epsilon à chaque étape.
            seed (int) : graine aléatoire pour l’entraînement.
            seed_test (int) : graine aléatoire pour le test.
            buffer_path (str) : chemin pour sauvegarder les tampons de mémoire.
            reward_function (int, optionnel) : identifiant de la fonction de récompense (par défaut = 2).
            num_video (int, optionnel) : initialisation du numéro de la vidéo que l'on va crée durant le test (si on enregistre toutes les vidéos dans le même dossier, il peut y avoir un écrasement)
            max_epsilon (float, optionnel) : valeur maximale de epsilon (par défaut = 1.0).
            min_epsilon (float, optionnel) : valeur minimale de epsilon (par défaut = 0.1).
            gamma (float, optionnel) : facteur de réduction pour les récompenses futures (par défaut = 0.99).
            alpha (float, optionnel) : degré de priorité dans la mémoire PER (par défaut = 0.2).
            beta (float, optionnel) : facteur d’importance dans l’échantillonnage PER (par défaut = 0.6).
            prior_eps (float, optionnel) : petite constante pour garantir un échantillonnage non nul (par défaut = 1e-6).
            v_min (float, optionnel) : valeur minimale du support pour le DQN catégoriel (par défaut = 0.0).
            v_max (float, optionnel) : valeur maximale du support pour le DQN catégoriel (par défaut = 200.0).
            atom_size (int, optionnel) : nombre d’unités discrètes du support (par défaut = 51).
            n_step (int, optionnel) : nombre d’étapes pour le calcul n-step (par défaut = 3).
            coord (int, optionnel) : coordonnée de référence dans l’environnement (par défaut = 0).
        """
        obs_dim = 84 * 84
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.seed_test = seed_test
        self.gamma = gamma

        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("cpu ou gpu ?", self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, f'{buffer_path}_per', batch_size, alpha=alpha, gamma=gamma
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, buffer_path, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # Sélection de la fonction reward
        if reward_function == 1:
            self.instant_reward = instant_reward1
        elif reward_function == 3:
            self.instant_reward = instant_reward3
        else : 
            self.instant_reward = instant_reward2
            
        # mode: train / test
        self.is_test = False
        self.num_video = num_video

        self.coord = coord
        self.is_crash = False
        self.lst_recent_crashes = [False, False]
        self.nb_crashes = 0
        self.mask = None

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
          if state.shape != (7056,):
            state = preprocess_observation(state)
          selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
          selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def find_coord(self, obs) :
        obsShape = obs.shape
        mask = obs.copy()
        min_coord_box = 200
        for i in range (obsShape[0]) :
            for j in range (obsShape[1]) :
                a, b, c = obs[i, j, 0], obs[i, j, 1], obs[i, j, 2]

                if a == b == c or ((a, b, c) == (228, 111, 111) and (i < 25 or i > 195)):
                    mask[i, j] = 0
                elif (a, b, c) == (252, 252, 84) :
                    if (j < 44 or j > 50) :
                        mask[i, j] = 0
                    else :
                        if [i, j] in [[102, 44], [102, 45], [102, 46], [102, 47], [104, 44], [104, 45], [104, 46], [104, 47]] :
                            mask[i, j] = 0
                        else :
                            mask[i, j] = 255
                            if i < min_coord_box :
                                min_coord_box = i
                                if j == 48 :
                                    num_sprite = 1
                                elif j == 49 :
                                    num_sprite = 2
                                else :
                                    num_sprite = 3
                else :
                    mask[i, j] = (a + b + c) / 3 + 50
        self.mask = mask
        self.is_crash = False
        if num_sprite == 1 :
            if not (obs[min_coord_box + 1, 49, 0] == 252 and \
                    \
                    obs[min_coord_box + 1, 48, 0] == 252 and \
                    obs[min_coord_box + 2, 48, 0] == 252 and \
                    obs[min_coord_box + 3, 48, 0] == 252 and \
                    obs[min_coord_box + 4, 48, 0] == 252 and \
                    obs[min_coord_box + 5, 48, 0] == 252 and \
                    obs[min_coord_box + 7, 48, 0] == 252 and \
                    \
                    obs[min_coord_box + 2, 47, 0] == 252 and \
                    obs[min_coord_box + 3, 47, 0] == 252 and \
                    obs[min_coord_box + 4, 47, 0] == 252 and \
                    obs[min_coord_box + 5, 47, 0] == 252 and \
                    obs[min_coord_box + 6, 47, 0] == 252 and \
                    obs[min_coord_box + 7, 47, 0] == 252 and \
                    \
                    obs[min_coord_box + 3, 46, 0] == 252 and \
                    obs[min_coord_box + 4, 46, 0] == 252 and \
                    obs[min_coord_box + 5, 46, 0] == 252 and \
                    obs[min_coord_box + 6, 46, 0] == 252 and \
                    \
                    obs[min_coord_box + 4, 45, 0] == 252 and \
                    obs[min_coord_box + 5, 45, 0] == 252 and \
                    \
                    obs[min_coord_box + 3, 44, 0] == 252 and \
                    obs[min_coord_box + 4, 44, 0] == 252) :
                self.is_crash = True
        elif num_sprite == 2 :
            if not (obs[min_coord_box, 49, 0] == 252 and \
                    obs[min_coord_box + 1, 49, 0] == 252 and \
                    obs[min_coord_box + 2, 49, 0] == 252 and \
                    \
                    obs[min_coord_box + 2, 48, 0] == 252 and \
                    obs[min_coord_box + 3, 48, 0] == 252 and \
                    obs[min_coord_box + 4, 48, 0] == 252 and \
                    obs[min_coord_box + 5, 48, 0] == 252 and \
                    \
                    obs[min_coord_box + 3, 47, 0] == 252 and \
                    obs[min_coord_box + 4, 47, 0] == 252 and \
                    obs[min_coord_box + 5, 47, 0] == 252 and \
                    obs[min_coord_box + 7, 47, 0] == 252 and \
                    \
                    obs[min_coord_box + 3, 46, 0] == 252 and \
                    obs[min_coord_box + 4, 46, 0] == 252 and \
                    obs[min_coord_box + 5, 46, 0] == 252 and \
                    obs[min_coord_box + 6, 46, 0] == 252 and \
                    obs[min_coord_box + 7, 46, 0] == 252 and \
                    \
                    obs[min_coord_box + 4, 45, 0] == 252 and \
                    obs[min_coord_box + 5, 45, 0] == 252 and \
                    obs[min_coord_box + 6, 45, 0] == 252 and \
                    \
                    obs[min_coord_box + 3, 44, 0] == 252 and \
                    obs[min_coord_box + 4, 44, 0] == 252) :
                        self.is_crash = True
        else :
            self.is_crash = True
        self.update_nb_crash()

        self.coord = 187 - min_coord_box

    def update_nb_crash(self) :
        if self.is_crash and not self.lst_recent_crashes[0] and not self.lst_recent_crashes[1]:
            self.nb_crashes += 1
        self.lst_recent_crashes[0], self.lst_recent_crashes[1] = self.lst_recent_crashes[1], self.is_crash
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, game_reward, terminated, truncated, _ = self.env.step(action)
        previous_coord = self.coord
        self.find_coord(next_state)
        reward = self.instant_reward(self.is_crash, game_reward, self.coord - previous_coord)
        next_state = preprocess_observation(self.mask)  # Preprocess image
        done = terminated or truncated

        if not self.is_test:
          if self.transition[0].shape != (7056,):
            self.transition[0] = preprocess_observation(self.transition[0])
          self.transition = [self.transition[0], self.transition[1], reward, next_state, done]
          self.memory.store(*self.transition)

          # N-step transition
          if self.use_n_step:
            one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
          else:
              one_step_transition = self.transition

            # add a single step transition
          if one_step_transition:
              self.memory.store(*one_step_transition)

        return next_state, reward, done


    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200, saving_interval: int = 1000, saving_path: str = None):
        """Train the agent."""
        self.is_test = False
        
        self.seed = np.random.randint(0, 1000)
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        crashes = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                self.seed = np.random.randint(0, 1000)
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0
                crashes.append(self.nb_crashes)
                self.lst_recent_crashes = [False, False]
                self.nb_crashes = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)


                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                # Afin d' afficher un score même si on n'a pas encore atteint l'objectif (ici le bout de la route)
                if len(scores) == 0:
                    self._plot(frame_idx, [score], losses, epsilons)
                else :
                    self._plot(frame_idx, scores, losses, epsilons)

            if saving_path and frame_idx % saving_interval == 0:
                with open(saving_path, 'wb') as f:
                    pickle.dump(self, f)

        self.env.close()

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def test(self, video_folder: str) -> Tuple[float, int]:
        """Test the agent and return test score."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        video_folder = f"{video_folder}{self.num_video}"
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed_test)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        nb_crashes = self.nb_crashes
        self.lst_recent_crashes = [False, False]
        self.nb_crashes = 0
        
        print("score: ", score)
        print("nb_crashes: ", nb_crashes)
        self.env.close()

        # reset
        self.env = naive_env

        self.num_video += 1

        return score, nb_crashes

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
        self,
        frame_idx: int,
        scores: List[float],
        losses: List[float],
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        # On affiche la moyenne des 10 derniers scores
        if len(scores) < 10:
          plt.title('frame %s. score (moyenne) : %s' % (frame_idx, np.mean(scores)))
        else :
          plt.title('frame %s. score (moyenne des 10 derniers): %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.figtext(0.45, -0.1, "Évolution des scores*, pertes et epsilons au fil de l'entraînement \n \n * : Un score est calculé du point de départ à l'atteinte de l'objectif par la fonction reward, il est donc différent du score du jeu.", 
            ha="left", fontsize=12)
        plt.show()