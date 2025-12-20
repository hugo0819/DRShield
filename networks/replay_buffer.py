from collections import deque
import random
import numpy as np 

class ReplayBuffer(object):
    
    def __init__(self, buffer_size, device="cpu"):
        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = deque()
        self.device = device

    def __len__(self):
        """Support for len() function."""
        return self._count

    def insert(self, _experience):
        # _experience = (state, action, reward, done, next_state)
        if(self._count <= self._buffer_size):
            self._buffer.append(_experience)
            self._count += 1
        else:
            self._buffer.popleft()
            self._buffer.append(_experience)
    
    def push(self, state, action, reward, next_state, done):
        self.insert((state, action, reward, next_state, done))
    
    def size(self):
        return self._count
    
    def sample_batch(self, batch_size=32):
        _available_batch_length = min(self._count, batch_size)
        batch = random.sample(self._buffer, _available_batch_length)
        
        # Fix: NumPy arrays do not have to() and unsqueeze() methods
        states = np.vstack([e[0] for e in batch])
        actions = np.vstack([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch]).reshape(-1, 1)
        next_states = np.vstack([e[3] for e in batch])
        dones = np.array([e[4] for e in batch]).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    # Add sample method as an alias for sample_batch
    def sample(self, batch_size=32):
        """Compatible with calls in DDPG agents."""
        return self.sample_batch(batch_size)
    
    def clear(self):
        self._buffer.clear()
        self._count = 0