from reseaux.noisy import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

# NoisyNet + DuelingNet + Categorical DQN

## NoisyNet + DuelingNet

# NoisyLinear is employed for the last two layers of advantage and
# value layers. The noise should be reset at evey update step.

## DuelingNet + Categorical DQN

# The dueling network architecture is adapted for use with return
# distributions. The network has a shared representation, which is
# then fed into a value stream with atom_size outputs, and into an 
# advantage stream with atom_size × out_dim outputs. For each atom, 
# the value and advantage streams are aggregated, as in dueling DQN,
# and then passed through a softmax layer to obtain the normalized
# parametric distributions used to estimate the returns’ distributions.


# (Please see *04.dueling.ipynb*, *05.noisy_net.ipynb*, 
# *06.categorical_dqn.ipynb* from rainbow-is-all-you-need
# for detailed description of each component's network architecture.)


class Network(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        atom_size: int,
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()