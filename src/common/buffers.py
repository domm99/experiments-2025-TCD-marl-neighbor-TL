import torch
import numpy as np 

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.uncertainty = np.zeros((capacity,), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, o, a, r, no, d, u):
        i = self.ptr
        self.obs[i] = o; self.act[i] = a; self.rew[i] = r
        self.next_obs[i] = no; self.done[i] = float(d)
        self.uncertainty[i] = u
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], device=device),
            torch.tensor(self.act[idx], device=device),
            torch.tensor(self.rew[idx], device=device),
            torch.tensor(self.next_obs[idx], device=device),
            torch.tensor(self.done[idx], device=device),
            torch.tensor(self.uncertainty[idx], device=device),
        )

    @property
    def uncertainties(self):
       return self.uncertainty[:self.size]