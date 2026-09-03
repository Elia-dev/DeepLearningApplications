import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    """ A simple feedforward neural network that takes in the environment and outputs a probability
        distribution over actions."""
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s

# Same architecture as PolicyNet, but with a single scalar output (estimate of v(s)), no softmax (not a distribution).
class ValueNet(nn.Module):
    """ A simple feedforward neural network that takes in the environment and outputs a single
        scalar value (the estimated value of the state)."""
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        return s

