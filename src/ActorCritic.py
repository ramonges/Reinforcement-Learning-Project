import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.action_bound * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)