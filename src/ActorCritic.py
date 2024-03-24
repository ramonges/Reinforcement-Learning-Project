import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_bound , verbose=False):
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
        print(self.network) if verbose else None
    
    def forward(self, state):
        return self.action_bound * self.network(state)


class Critic(nn.Module):
    def __init__(self, state_size , verbose=False):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output is a single value representing V(s)
        )
        print(self.network) if verbose else None
    
    def forward(self, state):
        return self.network(state)
    

if __name__ == '__main__':
    actor = Actor(5, 3, 1)
    critic = Critic(5)
    