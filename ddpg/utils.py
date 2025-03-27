
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

    if load:
        # Load the agent
        agent.load("model", fname)  # model is the file path

    test_reward = []
    avg_reward = []
    agent.actor_net.eval()
    agent.critic_net.eval()

    for i in range(30):
        state = env.reset()
        state_a = torch.tensor(state[0]).unsqueeze(0).unsqueeze(0).to(device)

        ep_reward = 0
        done = False
        step = 0

        while not done:

            with torch.no_grad():
                state_a = state_a.squeeze(1).permute(0, 3, 1, 2)
                action = agent.actor_net.forward(state_a.float())

            action = action.detach().cpu().numpy()
            action = np.argmax(action)  # Convertir en action discrète (0, 1 ou 2)

            next_state, reward, done, truncated, info = env.step(action)

            ####################### MODIFICATION DE LA RECOMPENSE ########################
            if action == 1:  # Avancer
                reward += 0.5
            elif action == 0:  # Ne rien faire
                reward -= 0.01
            elif action == 2:  # Reculer
                reward -= 0.02
            ##############################################################################

            state_a = torch.tensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

            step += 10
            ep_reward += reward

            if step >= 1500:
                done = True

            env.render()

        test_reward.append(ep_reward)
        avg_reward.append(np.mean(test_reward))

    return test_reward, avg_reward



class Buffer():
    def __init__(self, batch_size):
        """
        Construct the Buffer used to store the experiences
        :param batch_size: The defined batch size to be used in each replay
        """
        self.batch_size = batch_size
        self.pos = 0
        self.max_size = 50
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
    plt.plot(agent.avg_reward, label='Average Total Reward over episodes')
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
