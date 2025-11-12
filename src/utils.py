import torch
import imageio
import numpy as np
import pandas as pd
from vmas import make_env
from src.config import Config
from moviepy import ImageSequenceClip


def evaluate_parallel(env_fn, agents: dict, n_episodes: int, max_steps: int, current_step: int, device: str, save_gif: bool = False, gif_path: str = None):

    env = env_fn()

    scores = []
    for ep in range(n_episodes):
        obs = env.reset(seed=np.random.randint(1e9))
        ep_rew = 0.0
        frames = []
        for s in range(max_steps):
            actions = {}
            with torch.no_grad():
                for aid, o in obs.items():
                    q = agents[aid].policy(o) # not using act method because I don't want exploration-exploitation
                    a = int(q.argmax(dim=1).item())
                    actions[aid] = torch.tensor(a, device=device).unsqueeze(0)
            obs, rew, term, _ = env.step(actions)
            ep_rew += sum([r.cpu() for r in rew.values()])

            if save_gif:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                )
                if frame is not None:
                    frames.append(frame)

            if term.item():
                break

        scores.append(ep_rew)
        if save_gif and frames:
            clip = ImageSequenceClip(frames, fps=30)
            clip.write_gif(f'{gif_path}step_{current_step}-episode_{ep}.gif', fps=30)

    for agent in agents:
        agents[agent].export_policy(current_step, agent)

    return float(np.mean(scores))


def make_vmas_env(cfg: Config, env_name = 'dispersion', seed: int = 42):
    if env_name in ['dispersion']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,  # Same for agents and landmarks
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=True,
            dict_spaces=True
        )
        return env
    elif env_name in ['discovery']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,
            n_targets=int(cfg.n_agents/2),
            agents_per_target=1,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=True,
            dict_spaces=True,
            covering_range=0.1
        )
        return env
    elif env_name in ['flocking']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=True,
            dict_spaces=True,
            n_obstacles=5,
        )
        return env
    else:
        raise NotImplementedError(f'Env {env_name} not implemented')


def log_uncertainty(ids, agents: dict, logging_path: str, seed: int):
    for aid in ids:
        mean_u = agents[aid].aggregated_uncertainty(lambda u: np.mean(u))
        df = pd.read_csv(f'{logging_path}{aid}-seed_{seed}.csv')
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
    obs_t      = torch.as_tensor(obs, dtype=torch.float32, device=device)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    rew_t      = torch.as_tensor(rew, dtype=torch.float32, device=device).view(-1, 1)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device).view(-1, 1)
    t_sars_batch = torch.cat([obs_t, act_t, rew_t, next_obs_t], dim=1)
    return t_sars_batch


def euclidean_distance(pos1, pos2):
    dist = np.sqrt(np.pow((pos1[0] - pos2[0]), 2) + np.pow((pos1[1] - pos2[1]), 2))
    return dist
