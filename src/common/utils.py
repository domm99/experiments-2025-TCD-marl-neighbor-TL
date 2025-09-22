import torch
import numpy as np
import pandas as pd
from src.spread.config import Config
from pettingzoo.mpe import simple_spread_v3

def evaluate_parallel(env_fn, agents: dict, n_episodes: int, max_steps: int, device: str):
    env = env_fn()
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(1e9))
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
            ep_rew += sum(rew.values())
            if all(term.values()) or all(trunc.values()):
                break
        scores.append(ep_rew)
    env.close()
    return float(np.mean(scores))

def make_env(cfg: Config):
    return simple_spread_v3.parallel_env(
        N=10,
        continuous_actions=cfg.continuous_actions,
        max_cycles=cfg.max_episode_steps
    )

def log_uncertainty(ids, agents: dict, logging_path: str):
    for aid in ids:
        mean_u = agents[aid].aggregated_uncertainty(lambda u: np.mean(u))
        df = pd.read_csv(f'{logging_path}agent-aid.csv')
        new_line = {'Uncertainty': mean_u}
        df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
        df.to_csv(f'{logging_path}agent-aid.csv', index=False)