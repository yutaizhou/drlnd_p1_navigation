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
        self._seed = seed
        self._network = nn.Sequential(
            nn.Linear(state_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_size)
        )
    
    def forward(self, state: Tensor) -> Tensor:
        return self._network(state)

