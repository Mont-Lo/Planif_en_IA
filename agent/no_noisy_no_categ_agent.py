import pickle

from buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from buffers.ReplayBuffer import ReplayBuffer
from utils.processing import preprocess_observation
from typing import Dict, List, Tuple, Type
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch.nn.functional as F
import gc

class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        Network (objet): réseau de neurones utilisés pour l'exécution
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self,
        mask,
        Network : Type[object],
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        buffer_path: str,
        num_video: int = 0,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # N-step Learning
        n_step: int = 3,
        coord: int = 0, # 187 si on ne fait pas la soustraction
        is_crash: bool = False
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = 84 * 84
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
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

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False
        self.num_video = num_video

        self.coord = coord
        self.is_crash = is_crash
        self.mask = mask

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
        self.coord = 187 - min_coord_box

    def instant_reward (self, game_reward, coord_variation) :
        score = 0
        if self.is_crash :
            score = -100
        elif coord_variation < -100 :
            score = 100 * game_reward
        else :
            score = game_reward + coord_variation * 2
        self.lst_score.append([score, self.is_crash, game_reward, coord_variation])
        return score

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, game_reward, terminated, truncated, _ = self.env.step(action)
        previous_coord = self.coord
        reward = self.instant_reward (game_reward, self.coord - previous_coord)
        next_state = preprocess_observation(next_state)  # Preprocess image
        done = terminated or truncated

        if action == 1:
          reward += 0.5
        elif action == 0:
          reward -= 0.01

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

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
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
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0

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

    def test(self, video_folder: str) -> float:
        """Test the agent and return test score."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        video_folder = f"{video_folder}{self.num_video}"
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

        self.num_video += 1

        return score

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)

        #Double DQN
        next_q_value = self.dqn_target(next_state).gather(
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()


        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

         # PER : calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

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
    
    def cleanup(self):
        """Libère explicitement la mémoire occupée par l'agent."""

        # Libérer les tensors PyTorch
        del self.dqn
        del self.dqn_target
        del self.optimizer

        # Libérer la mémoire des buffers de replay
        del self.memory
        if self.use_n_step:
            del self.memory_n

        # Nettoyage GPU si applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Forcer la collecte des objets non référencés
        gc.collect()