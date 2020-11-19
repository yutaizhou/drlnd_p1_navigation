from collections import namedtuple, deque
import numpy as np
import torch
from ..utils.typing import List, Tuple, Deque, ndarray, ExperienceBatch
from ..utils.util import DEVICE

class ReplayBuffer:
    """
    Most basic experience replay buffer, samples uniformly with replacement
    """
    Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def __init__(self,
                 buffer_size: int,
                 batch_size: int,
                 seed: int = 0):
        self._batch_size: int = batch_size
        self._seed: int = seed
        self._buffer: Deque = deque(buffer_size)
    

    def add(self, 
            state: ndarray, 
            action: int, 
            reward: float,
            next_state: ndarray,
            done: bool) -> None:
        experience = self.Experience(state, action, reward, next_state, done)
        self._buffer.append(experience)
    

    def sample(self) -> ExperienceBatch:
        idc: List[int] = [np.random.randint(len(self.buffer) - 1) for _ in range(self.batch_size)]
        experiences: List[self.Experience] = [self._buffer[idx] for idx in idc]

        states = torch.from_numpy(np.vstack([exp.state for exp in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences]).astype(np.uint8)).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones)


    def __len__(self) -> int:
        return len(self._buffer)


    def empty(self) -> None:
        self._buffer.clear()


