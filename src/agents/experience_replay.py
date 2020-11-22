from collections import namedtuple, deque
import numpy as np
import torch
from torch import Tensor
from ..utils.typing import List, Tuple, Union, Deque, ndarray, ExperienceBatch
from ..utils.util import DEVICE

class ReplayBuffer:
    """
    Most basic experience replay buffer, samples uniformly with replacement
    """
    Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0):
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.buffer: Deque = deque(maxlen=buffer_size)
    

    def add(self, 
            state: ndarray, 
            action: int, 
            reward: float,
            next_state: ndarray,
            done: bool) -> None:
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def _sample_idc(self, probs = None) -> List[int]:
       return np.random.choice(len(self.buffer), size=self.batch_size, p=probs)


    @staticmethod
    def _process_experience(experiences) -> ExperienceBatch:
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)


    def sample(self) -> ExperienceBatch:
        # idc: indices of the interaction samples to be retrieved from buffer, chosen uniformly at random
        # expereinces: the actual interaction samples retrieved from buffer via idc 
        idc: List[int] = self._sample_idc()
        experiences: List[self.Experience] = [self.buffer[idx] for idx in idc]
        return self._process_experience(experiences)


    def __len__(self) -> int:
        return len(self.buffer)


    def empty(self) -> None:
        self.buffer.clear()

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Experience buffer that samples according to TD error-based priority
    """
    def __init__(self, buffer_size: int, batch_size: int, alpha: float, beta: float, seed: int = 0):
       super().__init__(buffer_size, batch_size, seed)
       self.priorities: Deque = deque(maxlen=buffer_size)
       self.alpha = alpha
       self.beta = beta
       self.eps = 1e-6 # not for exploration, but for ensuring no experineces have prob of zero
    
    def add(self,
            state: ndarray,
            action: int,
            reward: float, 
            next_state: ndarray, 
            done: bool) -> None:
        experience = self.Experience(state, action, reward, next_state, done)
        max_priority = max(self.priorities, default=1)
        self.buffer.append(experience)
        self.priorities.append(max_priority)
    
    @staticmethod
    def _priorities2probs(priorities, eps, alpha) -> ndarray:
        logits = np.array(priorities + eps) ** alpha
        probs = logits / logits.sums()
        return probs

    @staticmethod
    def _get_IS_weight(self, probs: ndarray, buffer_size: int, beta: float) -> ndarray:
        IS_weight = (buffer_size * probs) ** -beta
        IS_weight /= IS_weight.max()
        return IS_weight

    def sample(self) -> Union[ExperienceBatch, Tensor, List[int]]:
        probs: ndarray = self._priorities2probs(self.priorities, self.eps, self.alpha)

        idc: List[int] = self._sample_idc(probs)
        experiences: List[self.Experience] = [self.buffer[idx] for idx in idc]
        experiences = self._process_experience(experiences)
      
        IS_weight = self._get_IS_weight(probs, len(self.buffer), self.beta)
        return experiences, IS_weight, idc

    def update_priorities(self, idc: List[int], td_errors: Tensor) -> None:
        td_errors.abs_()
        for idx, td_error in zip(idc, td_errors):
            self.priorities[idx] = td_error


