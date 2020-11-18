import numpy as np
import torch

from .experience_replay import ReplayBuffer
from .model import FullyConnectedNetwork
from ..utils.typing import Sequence, ndarray, Tensor, ExperienceBatch
from ..utils.util import DEVICE

LR = 5e-4
BUFFER_SIZE = 1e5
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
TARGET_NET_UPDATE_FREQ = 4

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, hidden_dims: Sequence[int], seed: int):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.seed: int = seed   

        self.Q_target = FullyConnectedNetwork(state_size, action_size, hidden_dims)
        self.Q_local = FullyConnectedNetwork(state_size, action_size, hidden_dims)
        self.optimizer = torch.optim.Adam(self.Q_local.parameters(), lr=LR)

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.time_step = 0  


    def step(self,
            state: ndarray, 
            action: int, 
            next_state: ndarray,
            reward: float,
            done: bool) -> None:
        self.time_step += 1
        self.buffer.add(state, action, next_state, reward, done)
        
        if len(self.buffer) > BATCH_SIZE:
            experiences = self.buffer.sample()
            self._learn(experiences)
    

    def _learn(self, experiences: ExperienceBatch) -> None:
        states, actions, rewards, next_states, dones = experiences
        target_nexts, _ = self.Q_target(next_states).detach().max(1,keepdim=True)
        targets = rewards + (1-dones) * GAMMA * target_nexts
        currents = self.Q_local(states).gather(1, actions)

        loss = torch.nn.functional.mse_loss(targets, currents)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.time_step % TARGET_NET_UPDATE_FREQ == 0:
            self._update_Q_target()

    def _update_Q_target(self) -> None:
        self.Q_target.load_state_dict(self.Q_local.state_dict())

    def act(self, state: ndarray, eps=0) -> int:
        state: Tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            Qs: Tensor = self.Q_local(state)

        if np.random.uniform() > eps:
            return Qs.argmax().item()
        else:
            return np.random.randint(self.action_size)



