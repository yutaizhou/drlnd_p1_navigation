import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from ..utils.typing import Sequence, Tensor

class FullyConnectedNetwork(nn.Module):
    """
    Maps state to Q values of each action 
    """
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_dims: Sequence[int],
                 seed: int = 0):
        super().__init__()
        self._seed: int = seed
        self._network = nn.Sequential(
            nn.Linear(state_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_size)
        )
    
    def forward(self, state: Tensor) -> Tensor:
        return self._network(state)

