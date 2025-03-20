import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_conv_output(in_shape, kernel, padding=0, stride=1):
    """
    Calculate the CNN layer output dimension
    :param in_shape: The input dimension
    :param kernel: The kernel size
    :param padding: Padding on CNN
    :param stride: The stride value
    :return: A list of height and width of the output
    """

    out_height = int(((in_shape[0] + 2 * padding - kernel) / stride) + 1)
    out_width = int(((in_shape[1] + 2 * padding - kernel) / stride) + 1)
    return out_height, out_width

class Critic_network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):  # Réduction du hidden
        super(Critic_network, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)  # Réduction des filtres et augmentation du stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        out_height, out_width = calculate_conv_output(state_dim, 3, 0, 2)
        out_height, out_width = calculate_conv_output([out_height, out_width], 2, 0, 2)
        out_height, out_width = calculate_conv_output([out_height, out_width], 2, 0, 2)

        flat_dim = out_height * out_width * 64  # Moins de neurones à connecter

        self.fc1 = nn.Linear(flat_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        
        x = torch.cat([x, action.float()], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor_network(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Actor_network, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)  # Moins de filtres
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2)

        out_height, out_width = calculate_conv_output(state_dim, 3, 0, 2)
        out_height, out_width = calculate_conv_output([out_height, out_width], 2, 0, 2)
        out_height, out_width = calculate_conv_output([out_height, out_width], 2, 0, 2)

        self.fc1 = nn.Linear(32 * out_height * out_width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))