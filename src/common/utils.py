import torch
import numpy as np
import pandas as pd
from src.spread.config import Config
from pettingzoo.sisl import pursuit_v4
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.butterfly import pistonball_v6

def evaluate_parallel(env_fn, agents: dict, n_episodes: int, max_steps: int, device: str):
    env = env_fn()
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(1e9))
        obs = flatten_obs_dict(obs)
        ep_rew = 0.0
        for _ in range(max_steps):
            actions = {}
            with torch.no_grad():
                for aid, o in obs.items():
                    a = agents[aid].act(o)
                    oo = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    q = agents[aid].policy(oo)
                    a = int(q.argmax(dim=1).item())
                    actions[aid] = a
            obs, rew, term, trunc, _ = env.step(actions)
            obs = flatten_obs_dict(obs)
            ep_rew += sum(rew.values())
            if all(term.values()) or all(trunc.values()):
                break
        scores.append(ep_rew)
    env.close()
    return float(np.mean(scores))

def make_env(cfg: Config, env_name = 'SimpleSpread'):

    if env_name == 'SimpleSpread':
        env = simple_spread_v3.parallel_env(
            N=10,
            continuous_actions=cfg.continuous_actions,
            max_cycles=cfg.max_episode_steps
        )
    elif env_name == 'Pursuit':
        env = pursuit_v4.parallel_env(
            n_pursuers=10,
            max_cycles=cfg.max_episode_steps,
        )
    elif env_name == 'Pistonball':
        env = pistonball_v6.parallel_env(
            n_pistons=10,
            max_cycles=cfg.max_episode_steps,
            continuous=cfg.continuous_actions,
        )
    else:
        raise ValueError(f'Unknown env_name: {env_name}')
    return env

def log_uncertainty(ids, agents: dict, logging_path: str):
    for aid in ids:
        mean_u = agents[aid].aggregated_uncertainty(lambda u: np.mean(u))
        df = pd.read_csv(f'{logging_path}{aid}.csv')
        new_line = {'Uncertainty': mean_u}
        df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
        df.to_csv(f'{logging_path}{aid}.csv', index=False)


def flatten_obs_dict(obs_dict: dict) -> dict:
    for k, v in obs_dict.items():
        obs_dict[k] = v.flatten()
    return obs_dict

def ss_average_uncertainty(ids, agents: dict):
    uncertainties = {}
    for aid in ids:
        uncertainties[aid] = agents[aid].aggregated_uncertainty(lambda u: np.mean(u))
    return min(uncertainties, key=uncertainties.get)


def build_sars_batch(obs, act, rew, next_obs, device):
    obs_t      = torch.as_tensor(obs, dtype=torch.float32, device=device)               # [N,60]
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)          # [N,60]
    rew_t      = torch.as_tensor(rew, dtype=torch.float32, device=device).view(-1, 1)   # [N,1]
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device).view(-1, 1)        # [N,1]
    t_sars_batch = torch.cat([obs_t, act_t, rew_t, next_obs_t], dim=1)
    return t_sars_batch