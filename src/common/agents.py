import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from src.spread.config import Config
from src.common.models import DuelingQNet
from src.common.buffers import ReplayBuffer

class IndependentAgent:
    def __init__(self, obs_dim, n_actions, cfg: Config):
        self.cfg = cfg
        self.n_actions = n_actions
        self.policy = DuelingQNet(obs_dim, n_actions).to(cfg.device)
        self.target = DuelingQNet(obs_dim, n_actions).to(cfg.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.replay_size, obs_dim)
        self._opt_steps = 0
        # epsilon scheduling
        self._eps_t = 0

    @property
    def eps(self):
        frac = min(1.0, self._eps_t / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_final - self.cfg.eps_start)

    def act(self, obs):
        self._eps_t += 1
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            q = self.policy(o)
            return int(q.argmax(dim=1).item())

    def optimize(self):
        if self.rb.size < self.cfg.start_learning_after:
            return None
        if self._opt_steps % self.cfg.train_freq != 0:
            self._opt_steps += 1
            return None

        obs, act, rew, next_obs, done = self.rb.sample(self.cfg.batch_size, self.cfg.device)
        q = self.policy(obs).gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_q:
                next_act = self.policy(next_obs).argmax(dim=1, keepdim=True)
                q_next = self.target(next_obs).gather(1, next_act).squeeze(1)
            else:
                q_next = self.target(next_obs).max(dim=1)[0]
            target_q = rew + self.cfg.gamma * (1.0 - done) * q_next

        loss = F.smooth_l1_loss(q, target_q) 
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self._opt_steps += 1
        if self._opt_steps % self.cfg.target_update_freq == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

