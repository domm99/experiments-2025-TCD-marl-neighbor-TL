import math
import torch
import random
import pandas as pd
import torch.nn as nn
from typing import Callable
from src.utils import *
import torch.nn.functional as F
from src.config import Config
from tensordict import TensorDict
from src.buffers import ReplayBuffer
from torchrl.data import Categorical
from torchrl.modules import QValueActor
from src.models import DuelingQNet, QNet
from tensordict.nn import TensorDictModule
from src.estimator import UncertaintyEstimator
from torchrl.objectives import DQNLoss, SoftUpdate, HardUpdate


class IndependentAgent:
    def __init__(self, mid, obs_dim: int, n_actions: int, seed: int, cfg: Config):
        self.mid = mid
        self.cfg = cfg
        self.n_actions = n_actions
        self.seed = seed
        self.policy = QNet(obs_dim, n_actions).to(cfg.device)
        action_spec = Categorical(n=n_actions, device=cfg.device)

        self.policy_module = QValueActor(
            module=self.policy,
            in_keys=["observation"],
            spec=action_spec
        )

        self.loss_module = DQNLoss(
            value_network=self.policy_module,
            loss_function="smooth_l1",
            delay_value=True,
            #gamma=cfg.gamma,
            action_space=action_spec
        )
        self.loss_module.make_value_estimator(gamma=cfg.gamma)

        self.target_updater = HardUpdate(
            self.loss_module,
            value_network_update_interval=cfg.target_update_freq
        )

        self.opt = torch.optim.Adam(self.loss_module.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.replay_size, obs_dim)
        self._opt_steps = 0

        self.uncertainty_estimator = UncertaintyEstimator(obs_dim, action_dim=1, reward_dim=1, learning_rate=cfg.lr, cfg=cfg)
        self._eps_t = 0
        self.q_values = []
        self.target_q_values = []

    @property
    def experience(self):
        return self.rb.get_all()

    def get_policy(self):
        return self.policy

    @property
    def eps(self):
        v = self.cfg.eps_start * math.pow(1 - self.cfg.eps_decay_rate, self._eps_t)
        if v < self.cfg.eps_lower_bound:
            return self.cfg.eps_lower_bound
        else:
            return v

    def aggregated_uncertainty(self, aggregation: Callable[[np.ndarray], float]) -> float:
        return aggregation(self.rb.uncertainties)

    def store_experience(self, obs, act, rew, next_obs, done, uncertainty):
        self.rb.add(obs, act, rew, next_obs, done, uncertainty)

    def compute_uncertainty(self, obs, transferring=False) -> torch.Tensor:
        return self.uncertainty_estimator.compute_uncertainty(obs, transferring)

    def optimize_sars_rnd(self, uncertainty):
        self.uncertainty_estimator.optimize(uncertainty)

    def export_policy(self, time, id):
        torch.save(self.policy.eval().state_dict(), f'{self.cfg.policy_output_dir}/agent_{id}-time_{time}.pth')

    def load_policy_from_snapshot(self, time, id):
        state_dict = torch.load(f'{self.cfg.policy_output_dir}/agent_{id}-time_{time}.pth',
                                map_location=torch.device(self.cfg.device))
        self.policy.load_state_dict(state_dict)

    def increment_decay_time(self):
        self._eps_t += 1

    def act(self, obs):
        if random.random() < self.eps:
            return torch.tensor(random.randrange(self.n_actions), device=self.cfg.device).unsqueeze(0)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            q = self.policy(o)
            return torch.tensor(int(q.argmax(dim=-1).item()), device=self.cfg.device).unsqueeze(0)


    def optimize(self, transferring=False, experience=None):
        if not transferring and self.rb.size < self.cfg.start_learning_after:
            return None
        if not transferring and self._opt_steps % self.cfg.train_freq != 0:
            self._opt_steps += 1
            return None

        if not transferring:
            obs, act, rew, next_obs, done, _ = self.rb.sample(self.cfg.batch_size, self.cfg.device)
        else:
            obs, act, rew, next_obs, done, _ = experience

        if rew.ndim == 1:
            rew = rew.unsqueeze(-1)
        if done.ndim == 1:
            done = done.unsqueeze(-1)

        batch_td = TensorDict({
            "observation": obs,
            "action": act.long(),
            "next": {
                "observation": next_obs,
                "reward": rew,
                "done": done.bool(),
                "terminated": done.bool()
            }
        }, batch_size=obs.shape[0], device=self.cfg.device)

        loss_vals = self.loss_module(batch_td)
        loss = loss_vals["loss"]

        if self._opt_steps % 1000 == 0:
            self.q_values.append(loss_vals.get("action_value", torch.tensor(0.)).mean().item())

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.target_updater.step()

        self._opt_steps += 1

        return float(loss.item())

    def dump_logged_qvalues_to_csv(self):
        path = f'{self.cfg.data_output_dir}/qvalues/{self.mid}-seed_{self.seed}.csv'
        try:
            df = pd.read_csv(path)
            if len(self.q_values) > 0:
                df['MeanQ'] = pd.Series(self.q_values)
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"Errore dump csv: {e}")

    def learn_from_teacher(self, experience):
        obs, act, rew, next_obs, done, uncertainty = experience
        sars_batch = build_sars_batch(obs, act, rew, next_obs, self.cfg.device)
        actual_uncertainty = self.compute_uncertainty(sars_batch, transferring=True)
        surprise = actual_uncertainty - torch.tensor(uncertainty, device=self.cfg.device)
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
        self.optimize(transferring=True, experience=selected_experience)

    def learn_from_teacher_model(self, model):
        self.policy.load_state_dict(model.state_dict())
        self.target_updater.init_()