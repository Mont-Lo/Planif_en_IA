import os
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import noise
import utils
import copy
import numpy as np


class DDPG_agent():
    def __init__(self, env, actor_net, critic_net, target_actor_net, target_critic_net, device):
        """
        Construct the DDPG agent with the given parameters
        :param env: The environment
        :param actor_net: The actor network
        :param critic_net: The critic network
        :param target_actor_net: The target actor network
        :param target_critic_net: The target critic network
        :param device: The device to run
        """
        self.env = env
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.target_actor_net = target_actor_net
        self.target_critic_net = target_critic_net
        self.total_rewards = []
        self.avg_reward = []
        self.a_loss = []
        self.c_loss = []
        self.gamma = 0.99
        self.lr_c = 1e-4
        self.lr_a = 1e-4
        self.tau = 0.01
        self.device = device
        self.env = env

        # init OU noise
        self.noise_o = noise.OrnsteinUhlenbeckNoise(3, sigma=0.2)

        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_c)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_a)
        self.critic_criterion = nn.MSELoss()

        # save variables
        self.save_interval = 20
        self.file_name = "model_final.pt"
        # Learning rate scheduler
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=10000, eta_min=1e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=10000, eta_min=1e-5)

        self.batch_size = 512
        self.replay_buffer = utils.Buffer(self.batch_size)

    def get_action(self, state):

        state = state.float().unsqueeze(0).to(self.device)
        state = state.permute(0, 3, 1, 2)

        with torch.no_grad():
            action = self.actor_net(state)
        return action



    def save(self, path, fname):
        """
        Save the model parameters
        :param path: the file path
        :param fname: the file name
        """
        if not os.path.exists(path):
            os.makedirs(path)
        path_file = os.path.join(path, fname)

        save_dict = {
            'model_actor_state': self.actor_net.state_dict(),
            'model_critic_state': self.critic_net.state_dict(),
            'model_actor_op_state': self.actor_optimizer.state_dict(),
            'model_critic_op_state': self.critic_optimizer.state_dict(),
            'actor_scheduler_state': self.actor_scheduler.state_dict(),
            'critic_scheduler_state': self.critic_scheduler.state_dict(),

            'total_rewards': self.total_rewards,
            'avg_rewards': self.avg_reward,
            'actor_loss': self.a_loss,
            'critic_loss': self.c_loss,

        }
        torch.save(save_dict, path_file)

    def load(self, path, fname):
        """
        Load the model parameters
        :param path: the file path
        :param fname: the file name
        """

        path_file = os.path.join(path, fname)
        load_dict = torch.load(path_file, map_location=self.device, weights_only=False)


        # Load weights and optimizer states
        self.actor_net.load_state_dict(load_dict['model_actor_state'])
        self.critic_net.load_state_dict(load_dict['model_critic_state'])
        self.actor_optimizer.load_state_dict(load_dict['model_actor_op_state'])
        self.critic_optimizer.load_state_dict(load_dict['model_critic_op_state'])
        self.actor_scheduler.load_state_dict(load_dict['actor_scheduler_state'])
        self.critic_scheduler.load_state_dict(load_dict['critic_scheduler_state'])

        # Load other variables
        self.total_rewards = load_dict['total_rewards']
        self.avg_reward = load_dict['avg_rewards']
        self.a_loss = load_dict['actor_loss']
        self.c_loss = load_dict['critic_loss']


        # Creating the target networks
        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        print("Successfully load the model parameters.")

    def train(self):
        """
        Training the agent
        """
        episodes = 15
        
        time_start = time()
        collision_counts = []
        # tqdm permet de voir l'avancer de la boucle
        for i in tqdm.tqdm(range(episodes)):
            collision_counts.append(0)  # Initialize the collision count for each episode
            done = False
            state = self.env.reset()
            ep_reward = 0
            # prepare state for neural network
            state_a = torch.tensor(state[0])

            step = 0
            collision_timer = 0  # Timer pour éviter les collisions en double
            collision_threshold = 15  # Nombre d'itérations à attendre avant de re-détecter une collision
            while not done:
                noise = self.noise_o.sample()       # Sample OU noise
                action = self.get_action(state_a)[0] # Get action from the Actor network
                # Add noise to action to encourage exploration
                action = np.clip(action.cpu() + noise, -1.0, 1.0)
                action = action.detach().cpu().numpy()

                # Move to the next time step with the action
                action = int(action.argmax())  # Prend l'action avec la plus haute valeur
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state_a = torch.tensor(next_state)

                ######################### The Modified Rewards ##############################
                # Correction dans eval_agent :
                is_crash, coord = self.find_coord(next_state)
                if is_crash and collision_timer == 0:  
                    collision_counts[-1] += 1
                    collision_timer = collision_threshold  # Active le délai avant la prochaine détection

                # Réduction du timer (si actif)
                if collision_timer > 0:
                    collision_timer -= 1 
                    
                # Checking if the training reaches the maximum steps on the episode
                if step == 2000:
                    done = True

                #######################################################

                # Push the experience in the batch
                self.replay_buffer.push(state_a, action, reward, next_state_a, done)

                update_frequency = 1  # Effectue la mise à jour toutes les n actions (a augmenter pour accelerer l'entrainement) 

                # Experience replay
                if len(self.replay_buffer.buffer) >= self.batch_size:
                    if step % update_frequency == 0:  # Mise à jour toutes les n actions
                        self.replay()

                step += 1
                state_a = next_state_a
                ep_reward += reward
                #env.render()

            print("Episode reward: ", ep_reward)
            print("Collision counts: ", collision_counts[-1])
            self.total_rewards.append(ep_reward)
            self.avg_reward.append(np.mean(self.total_rewards))

            if i % self.save_interval == 0:
                self.save("model", fname=self.file_name)  # Save the model parameters

            # Update the learning rate
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.save("model", fname=self.file_name)  # Save the model parameters when reaching the end of training
        time_training = (time() - time_start)
        print("Training time: %fs" % time_training)
    
    def find_coord(self, obs):
        obsShape = obs.shape
        mask = obs.copy()
        min_coord_box = 200
        num_sprite = 3  # Par défaut, si aucune condition ne s'applique
        
        for i in range(obsShape[0]):
            for j in range(obsShape[1]):
                a, b, c = obs[i, j, 0], obs[i, j, 1], obs[i, j, 2]

                if a == b == c or ((a, b, c) == (228, 111, 111) and (i < 25 or i > 195)):
                    mask[i, j] = 0
                elif (a, b, c) == (252, 252, 84):
                    if (j < 44 or j > 50):
                        mask[i, j] = 0
                    else:
                        if [i, j] in [[102, 44], [102, 45], [102, 46], [102, 47], [104, 44], [104, 45], [104, 46], [104, 47]]:
                            mask[i, j] = 0
                        else:
                            mask[i, j] = 255
                            if i < min_coord_box:
                                min_coord_box = i
                                if j == 48:
                                    num_sprite = 1
                                elif j == 49:
                                    num_sprite = 2

        # Détection de crash
        is_crash = num_sprite not in [1, 2]  # Par défaut, crash si aucune condition ne s'applique
        
        return is_crash, 187 - min_coord_box

    
    def instant_reward (self, game_reward, coord_variation) :
            score = 0
            if self.is_crash :
                score = -100
            elif coord_variation < -100 :
                score = 100 * game_reward
            else :
                score = coord_variation * 2
            return score

    def replay(self):
        """
        Experience replay on the buffer
        """
        batch = self.replay_buffer.sample()
        state_a_batch, action_batch, reward_batch, next_state_a_batch, done_batch = batch

        # Prepare batch
        state_a_batch = torch.FloatTensor(state_a_batch).unsqueeze(1).to(self.device)
        next_state_a_batch = torch.FloatTensor(next_state_a_batch).unsqueeze(1).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(np.float32(done_batch)).unsqueeze(1).to(self.device)

        next_state_a_batch = next_state_a_batch.squeeze(1)
        next_state_a_batch = next_state_a_batch.permute(0, 3, 1, 2)

        state_a_batch = state_a_batch.squeeze(1)
        state_a_batch = state_a_batch.permute(0, 3, 1, 2)

        # Critic network update
        with torch.no_grad():
            next_action_batch = self.target_actor_net(next_state_a_batch)
            target_value_batch = self.target_critic_net(next_state_a_batch, next_action_batch)
            target_value = reward_batch + (1.0 - done_batch) * self.gamma * target_value_batch

        # Compute Critic loss
        predicted_value = self.critic_net(state_a_batch, action_batch)
        critic_loss = self.critic_criterion(target_value, predicted_value)

        self.c_loss.append(critic_loss.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor network update
        actor_action = self.actor_net(state_a_batch)
        actor_loss = -self.critic_net(state_a_batch, actor_action).mean()

        self.a_loss.append(actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        self.soft_update(self.target_actor_net, self.actor_net)
        self.soft_update(self.target_critic_net, self.critic_net)

    def soft_update(self, target_net, current_net):
        """
        Soft update the target network based on the defined tau parameter and the current network
        :param target_net: The target network
        :param current_net: The current network
        :return:
        """
        for target_param, param in zip(target_net.parameters(), current_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)