import torch
import imageio
import numpy as np
import pandas as pd
from vmas import make_env
from src.config import Config
from moviepy import ImageSequenceClip
from src.densescenarios import DenseDispersionScenario#, DenseDiscoveryScenario
from vmas.scenarios.mpe.simple_tag import Scenario as SimpleTagScenario

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
            penalise_by_time=False,
            dict_spaces=True
        )
        return env
    elif env_name in ['dropout']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,  # Same for agents and landmarks
            #share_reward=False,  # This way only the agents which reach the goal get the reward
            #penalise_by_time=False,
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
            n_targets=cfg.n_agents,
            agents_per_target=1,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=False,
            dict_spaces=True,
            covering_range=0.1
        )
        return env
    elif env_name in ['balance']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,
            package_mass=1,
            random_package_pos_on_line=False,
            #share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=False,
            dict_spaces=True,
        )
        return env
    elif env_name in ['SimpleTag']:
        scenario = SimpleTagScenario()

        env = make_env(
            scenario=scenario,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            dict_spaces=True,
            num_good_agents=2,
            num_adversaries=cfg.n_agents,
            num_landmarks=2,
            shape_agent_rew=False,
            shape_adversary_rew=False,
            agents_share_rew=False,
            adversaries_share_rew=False,
            observe_same_team=True,
            observe_pos=True,
            observe_vel=True,
            respawn_at_catch=False
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
            penalise_by_time=False,
            dict_spaces=True,
            n_obstacles=5,
        )
        return env
    elif env_name in ['sampling']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,
            n_gaussians=10,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=False,
            dict_spaces=True,
        )
        return env
    elif env_name in ['transport']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=False,
            dict_spaces=True,
        )
        return env
    elif env_name in ['navigation']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=False,
            dict_spaces=True,
        )
        return env
    elif env_name in ['reverse_transport']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            n_agents=cfg.n_agents,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=False,
            dict_spaces=True,
            package_mass=3,
            package_width=0.5,
            package_leght=0.5
        )
        return env
    elif env_name in ['densedispersion']:
        scenario = DenseDispersionScenario(
            n_agents=cfg.n_agents,
            share_reward=False,
            penalise_by_time=False
        )

        env = make_env(
            scenario=scenario,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            dict_spaces=True
        )
        return env
    # elif env_name in ['densediscovery']:
    #     scenario = DenseDiscoveryScenario(
    #         n_agents=1,
    #         n_targets=1,
    #         share_reward=False,
    #         penalise_by_time=False
    #     )
    #
    #     env = make_env(
    #         scenario=scenario,
    #         num_envs=cfg.num_parallel_envs,
    #         device=cfg.device,
    #         continuous_actions=False,
    #         seed=seed,
    #         dict_spaces=True
    #     )
    #     return env
    elif env_name in ['football']:
        env = make_env(
            scenario=env_name,
            num_envs=cfg.num_parallel_envs,
            device=cfg.device,
            continuous_actions=False,
            seed=seed,
            ai_blue_agents=False,
            ai_red_agents=True,
            n_blue_agents=5,
            n_red_agents = 5,
            share_reward=False,  # This way only the agents which reach the goal get the reward
            penalise_by_time=False,
            dense_reward=True,
            dict_spaces=True,
        )
        return env
    else:
        raise NotImplementedError(f'Env {env_name} not implemented')


def log_uncertainty(agents: dict, agents_uncertainty: dict):
    for aid in agents.keys():
        mean_u = agents[aid].aggregated_uncertainty(lambda u: np.mean(u))
        agents_uncertainty[aid].append(mean_u)
    return agents_uncertainty


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
