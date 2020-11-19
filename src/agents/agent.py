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
UPDATE_FREQ = 4

class DQNAgent:
    def __init__(self,
                 state_size: int, action_size: int, buffer_size: int = BUFFER_SIZE,
                 hidden_dims: Sequence[int] = [64,64], update_freq = UPDATE_FREQ, tau = TAU,
                 lr: float = LR, batch_size: int = BATCH_SIZE, gamma:float = GAMMA,
                 seed: int = 42):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.seed: int = seed   

        self.Q_target = FullyConnectedNetwork(state_size, action_size, hidden_dims).to(DEVICE)
        self.Q_local = FullyConnectedNetwork(state_size, action_size, hidden_dims).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.Q_local.parameters(), lr=LR)
        self.buffer = ReplayBuffer(int(buffer_size), batch_size)
        self.batch_size = batch_size

        self.time_step = 0
        self.lr = lr
        self.gamma = gamma
        self.update_freq = update_freq
        self.tau = tau

        # exponential decay of epsilons
        self.eps = 1
        self.eps_min = 0.1
        self.eps_decay = 0.9995

    def step(self,
            state: ndarray, 
            action: int, 
            reward: float,
            next_state: ndarray,
            done: bool) -> None:
        self.time_step += 1
        self._update_eps()
        self.buffer.add(state, action, reward, next_state, done)
        
        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            self._learn(experiences)
    
    def _update_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def _learn(self, experiences: ExperienceBatch) -> None:
        states, actions, rewards, next_states, dones = experiences
        target_nexts, _ = self.Q_target(next_states).detach().max(1,keepdim=True)
        targets = rewards + (1-dones) * GAMMA * target_nexts
        currents = self.Q_local(states).gather(1, actions)

        loss = torch.nn.functional.mse_loss(targets, currents)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.time_step % self.update_freq == 0:
            self._update_Q_target()

    def _update_Q_target(self) -> None:
        self.Q_target.load_state_dict(self.Q_local.state_dict())

    def act(self, state: ndarray, is_trainng: bool = True) -> int:
        state: Tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            Qs: Tensor = self.Q_local(state)

        if is_trainng and np.random.uniform() <= self.eps:
            return np.random.randint(self.action_size)
        else:
            return Qs.argmax().item()



