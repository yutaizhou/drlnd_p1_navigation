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
    """
    Basic Deep Q-Learning Agent with target network updated by coping, and uniform replay buffer
    """
    def __init__(self,
                 state_size: int, action_size: int, buffer_size: int = BUFFER_SIZE,
                 hidden_dims: Sequence[int] = [64,64], update_freq: int = UPDATE_FREQ, 
                 lr: float = LR, batch_size: int = BATCH_SIZE, gamma:float = GAMMA,
                 seed: int = 42):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.seed = seed   
        self.time_step: int = 0

        # models, replabuffer, and optimizer
        self.Q_target = FullyConnectedNetwork(state_size, action_size, hidden_dims).to(DEVICE)
        self.Q_local = FullyConnectedNetwork(state_size, action_size, hidden_dims).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.Q_local.parameters(), lr=LR)
        self.buffer = ReplayBuffer(int(buffer_size), batch_size)
        self.batch_size = batch_size

        # important hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.update_freq = update_freq

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
        # schedule eps for exploration, add interaction sample to buffer, and learn from buffer experience
        self.time_step += 1
        self._update_eps()
        self.buffer.add(state, action, reward, next_state, done)
        
        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            self._learn(experiences)
    
    def _update_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def _learn(self, experiences: ExperienceBatch) -> None:
        # compute TD target, retrieve current TD values
        states, actions, rewards, next_states, dones = experiences
        target_nexts, _ = self.Q_target(next_states).detach().max(1,keepdim=True)
        
        targets = rewards + (1-dones) * GAMMA * target_nexts
        currents = self.Q_local(states).gather(1, actions)

        # gradient descent to minimize MSE loss between TD target and current
        loss = torch.nn.functional.mse_loss(targets, currents)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target Q network 
        if self.time_step % self.update_freq == 0:
            self._update_Q_target()

    def _update_Q_target(self) -> None:
        # update with simple copying
        self.Q_target.load_state_dict(self.Q_local.state_dict())

    def act(self, state: ndarray, is_trainng: bool = True) -> int:
        # pass state through q_local to get Q values of each s-a pair
        state: Tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            Qs: Tensor = self.Q_local(state)

        # eps greedy 
        if is_trainng and np.random.uniform() <= self.eps:
            return np.random.randint(self.action_size)
        else:
            return Qs.argmax().item()

class DoubleDQNAgent(DQNAgent):
    def _learn(self, experiences: ExperienceBatch) -> None:
        # use Q_local to select action, use Q_target to evaluate
        states, actions, rewards, next_states, dones = experiences
        _, next_actions = self.Q_local(next_states).detach().max(1,keepdim=True)
        target_nexts = self.Q_target(next_states).gather(1, next_actions)
        
        targets = rewards + (1-dones) * GAMMA * target_nexts
        currents = self.Q_local(states).gather(1, actions)

        # gradient descent to minimize MSE loss between TD target and current
        loss = torch.nn.functional.mse_loss(targets, currents)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target Q network 
        if self.time_step % self.update_freq == 0:
            self._update_Q_target()
