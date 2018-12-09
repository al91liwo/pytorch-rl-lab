from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    A finite sized cache. It is used to sample transitions (s, a, r, s_2).
    """
    
    
    def __init__(self, buffer_size):
        """
        Creates a ReplayBuffer of size buffer_size
        param buffer_size: size of buffer as integer
        """
        self.buffer_size = buffer_size
        self.count = 0
        # list-like container with fast appends and pops on either end
        self.buffer = deque() 

    
    def __str__(self):
        return str(self.buffer_size)

    def add(self, s, a ,r, s_2):
        """
        Store transition (s, a, r, s_2) in replay buffer,
        discards oldest transition when buffer is full
        param s: initial state
        param a: action taken in state s
        param r: reward for taking action a in state s
        param s_2: reached state for given state s with action a
        """
        transition = (s, a, r, s_2)
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else:
            # discard oldest 
            self.buffer.popleft()
            self.buffer.append(transition)


    def sample_batch(self, batch_size):
        """
        Selects a random batch of size batch_size and returns the resulting
        transition model as a tuple of np arrays
        param batch_size: number of batches to take
        return: transitions (s, a, r, s_2) as np arrays
        """
        batch = []

        if self.count < batch_size:
            batch_size = self.count

        batch = random.sample(self.buffer, batch_size)

        s_batch = [b[0] for b in batch]
        a_batch = np.array([[b[1]] for b in batch])
        r_batch = np.array([[b[2]] for b in batch])
        s_2_batch = [b[3] for b in batch]

        return (s_batch, a_batch, r_batch, s_2_batch)


    def clear (self):
        """
        Clears buffer, sets count to 0
        """
        self.buffer.clear()
        self.count = 0








