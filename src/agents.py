import math
import torch
import random
import torch.nn as nn
from typing import Callable
from src.utils import *
import torch.nn.functional as F
from src.config import Config
from src.models import DuelingQNet
from src.buffers import ReplayBuffer
from src.estimator import UncertaintyEstimator

class IndependentAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: Config):
        self.cfg = cfg
        self.n_actions = n_actions
        self.policy = DuelingQNet(obs_dim, n_actions).to(cfg.device)
        self.target = DuelingQNet(obs_dim, n_actions).to(cfg.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.replay_size, obs_dim)
        self._opt_steps = 0
        self.uncertainty_estimator = UncertaintyEstimator(obs_dim, action_dim=1, reward_dim=1, learning_rate=cfg.lr, cfg=cfg)
        self._eps_t = 0

    @property
    def experience(self):
        return self.rb.get_all()

    # @property
    # def eps(self):
    #     frac = min(1.0, self._eps_t / self.cfg.eps_decay_steps)
    #     return self.cfg.eps_start + frac * (self.cfg.eps_final - self.cfg.eps_start)
    @property
    def eps(self):
        decay_rate = self._eps_t / self.cfg.eps_decay_steps
        decay_rate = min(decay_rate, 1.0)
        return self.cfg.eps_final + (self.cfg.eps_start - self.cfg.eps_final) * math.exp(-decay_rate * self.cfg.eps_decay_rate)

    def aggregated_uncertainty(self, aggregation: Callable[[np.ndarray], float]) -> float:
        return aggregation(self.rb.uncertainties)

    def store_experience(self, obs, act, rew, next_obs, done, uncertainty):
        self.rb.add(obs, act, rew, next_obs, done, uncertainty)

    def compute_uncertainty(self, obs, transferring = False) -> torch.Tensor:
        return self.uncertainty_estimator.compute_uncertainty(obs, transferring)

    def optimize_sars_rnd(self, uncertainty):
        self.uncertainty_estimator.optimize(uncertainty)

    def export_policy(self, time, id):
        torch.save(self.policy.eval().state_dict(), f'{self.cfg.policy_output_dir}/agent_{id}-time_{time}.pth')

    def load_policy_from_snapshot(self, time, id):
        state_dict = torch.load(f'{self.cfg.policy_output_dir}/agent_{id}-time_{time}.pth', map_location=torch.device(self.cfg.device))
        self.policy.load_state_dict(state_dict)

    def act(self, obs):
        self._eps_t += 1
        if random.random() < self.eps:
            return torch.tensor(random.randrange(self.n_actions), device=self.cfg.device).unsqueeze(0)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            q = self.policy(o)
            return torch.tensor(int(q.argmax(dim=-1).item()), device=self.cfg.device).unsqueeze(0)

    def optimize(self, transferring = False, experience = None):
        if not transferring and self.rb.size < self.cfg.start_learning_after:
            return None
        if not transferring and self._opt_steps % self.cfg.train_freq != 0:
            self._opt_steps += 1
            return None

        if not transferring:
            obs, act, rew, next_obs, done, _ = self.rb.sample(self.cfg.batch_size, self.cfg.device)
        else:
            obs, act, rew, next_obs, done, _ = experience

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

    def learn_from_teacher(self, experience):
        obs, act, rew, next_obs, done, uncertainty = experience

        # Computing actual uncertainty
        sars_batch = build_sars_batch(obs, act, rew, next_obs, self.cfg.device)
        actual_uncertainty = self.compute_uncertainty(sars_batch, transferring=True)

        # Computing surprise
        surprise = actual_uncertainty - torch.tensor(uncertainty, device=self.cfg.device)

        # Taking top k surprising SARS tuples
        _, indices = torch.topk(surprise, k=self.cfg.transfer_budget, largest=True, sorted=True)
        indices = indices.cpu().numpy()
        selected_experience = (
            torch.tensor(obs[indices], device=self.cfg.device),
            torch.tensor(act[indices], device=self.cfg.device),
            torch.tensor(rew[indices], device=self.cfg.device),
            torch.tensor(next_obs[indices], device=self.cfg.device),
            torch.tensor(done[indices], device=self.cfg.device),
            torch.tensor(surprise[indices], device=self.cfg.device),
        )

        # Learning on selected received knowledge
        self.optimize(transferring = True, experience = selected_experience)