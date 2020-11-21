import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from ..utils.typing import Sequence, Tensor

class FullyConnectedNetwork(nn.Module):
    """
    Maps 1D vector state to Q values of each action using simple feed forward net
    """
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_dims: Sequence[int],
                 seed: int = 0):
        super().__init__()
        self.seed = seed
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_size)
        )
    
    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)

class FullyConnectedDuelingNetwork(nn.Module):
    """
    Just like FullyConnectedNetwork, but uses the dueling architecture
    """
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_dims: Sequence[int],
                 seed: int = 0): 
        super().__init__()
        self.action_size = action_size
        self.seed = seed
        self.network_base = nn.Sequential(
            nn.Linear(state_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.network_A_head = nn.Linear(hidden_dims[1], action_size)
        self.network_V_head = nn.Linear(hidden_dims[1], 1)

    def forward(self, state: Tensor) -> Tensor:
        base = self.network_base(state)
        a = self.network_A_head(base) # 1 x action_size
        v = self.network_V_head(base).expand_as(a) # 1 x 1 -> 1 x action_size

        q = v + (a - a.sum(-1)/self.action_size)

        return q

        

