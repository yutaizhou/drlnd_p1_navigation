from collections import namedtuple, deque
import numpy as np
import torch
from ..utils.typing import List, Tuple, Deque, ndarray, ExperienceBatch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    Most basic experience replay buffer, samples uniformly with replacement
    """
    Experience = namedtuple('Experience', ['state', 'action', 'next_state', 'reward', 'done'])

    def __init__(self,
                 buffer_size: int,
                 batch_size: int,
                 seed: int):
        self._batch_size: int = batch_size
        self._seed: int = seed
        self._buffer: Deque = deque(buffer_size)
    

    def add(self, 
            state: ndarray, 
            action: int, 
            next_state: ndarray,
            reward: float,
            done: bool) -> None:
        experience = self.Experience(state, action, next_state, reward, done)
        self._buffer.append(experience)
    

    def sample(self) -> ExperienceBatch:
        idc: List[int] = [np.random.randint(len(self.buffer) - 1) for _ in range(self.batch_size)]
        experiences: List[self.Experience] = [self._buffer[idx] for idx in idc]

        states = torch.from_numpy(np.vstack([exp.state for exp in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)


    def __len__(self) -> int:
        return len(self._buffer)


    def empty(self) -> None:
        self._buffer.clear()


