
import matplotlib.pyplot as plt
from collections import deque
import random
import numpy as np
import torch

def eval_agent(agent, env, fname, device, load=False):
    """
    Evaluate the model
    :param agent: DDPG_agent object
    :param env: The environment to use in the evaluation
    :param fname: The trained model file name for loading
    :param device: The device to run
    :param load: To load the previous saved model or to use the latest trained model
    :return: test_reward: The list of all episode rewards
             avg_reward: The list of average reward over episodes on each episode
    """

    if load == True:
        # Load the agent
        agent.load("model", fname)  # model is the file path

    test_reward = []
    avg_reward = []
    collision_counts = []
    
    agent.actor_net.eval()
    agent.critic_net.eval()

    for i in range(4):
        state = env.reset()
        state_a = torch.tensor(state[0]).unsqueeze(0).unsqueeze(0).to(device)
        collision_counts.append(0)
        ep_reward = 0
        done = False
        i = 0
        collision_timer = 0  # Timer pour éviter les collisions en double
        collision_threshold = 15  # Nombre d'itérations à attendre avant de re-détecter une collision
        while not done:

            with torch.no_grad():
                state_a = state_a.squeeze(1).permute(0, 3, 1, 2)
                action = agent.actor_net.forward(state_a.float())

            action = action.detach().cpu().numpy()
            action = np.argmax(action)
            next_state, reward, done, truncated, info = env.step(action)


            state_a = torch.tensor(next_state).unsqueeze(0).unsqueeze(0).to(device)
            veh_mat = state_a.squeeze().squeeze()
            num_veh = veh_mat.shape[0]



            ######## The Modified Reward function ###############

            # Correction dans eval_agent :
            is_crash, coord = find_coord(next_state)
            if is_crash and collision_timer == 0:  
                collision_counts[-1] += 1
                collision_timer = collision_threshold  # Active le délai avant la prochaine détection

            # Réduction du timer (si actif)
            if collision_timer > 0:
                collision_timer -= 1 

            ep_reward += reward    
            i += 1
            if i == 2000:
                done = True
            env.render()
        test_reward.append(ep_reward)
        avg_reward.append(np.mean(test_reward))
        print("Collision counts: ", collision_counts[-1])
    return test_reward, avg_reward

def find_coord(obs):
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

class Buffer():
    def __init__(self, batch_size):
        """
        Construct the Buffer used to store the experiences
        :param batch_size: The defined batch size to be used in each replay
        """
        self.batch_size = batch_size
        self.pos = 0
        self.max_size = 100000
        self.buffer = deque(maxlen=self.max_size)

    def push(self, state_a, action, reward, next_state_a, done):
        """
        Store the experience
        :param state_a: The current state
        :param action: The action
        :param reward: The received reward
        :param next_state_a: The next state given by the environment
        :param done: Done condition
        """
        self.buffer.append([state_a, action, reward, next_state_a, done])

    def sample(self):
        """
        Randomly sample the experiences based on the batch size
        :return: state_a: the sampled states
                 action: the sampled actions
                reward: the sampled rewards
                next_state_a: the sampled next state
                done: the sampled Done condition
        """
        batch = random.sample(self.buffer, self.batch_size)

        # unpack and stack each experience in the batch
        state_a, action, reward, next_state_a, done = map(np.stack, zip(*batch))
        return state_a, action, reward, next_state_a, done

def plots(agent):
    """
    Plot the critic loss, actor loss and the training rewards
    :param agent: The trained DPG_agent object to be used for the plots
    """
    plt.plot(agent.c_loss)
    plt.ylabel("Critic Loss")
    plt.xlabel("Updated Steps")
    plt.title("Critic Loss")
    plt.show()

    plt.plot(agent.a_loss)
    plt.ylabel("Actor Loss")
    plt.xlabel("Updated Steps")
    plt.title("Actor Loss")
    plt.show()

    plt.plot(agent.total_rewards, label='Total Reward in the episode')
    plt.ylabel("Rewards")
    plt.title("Training Reward")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()


def plot_eval(reward):
    """
    Plot the evaluation reward
    :param reward:
    :return:
    """
    plt.plot(reward[0], label='Total Reward in the episode')
    plt.plot(reward[1], label='Average Total Reward over episodes')
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.title("Evaluation Reward (Last trial)")
    plt.legend()
    plt.show()