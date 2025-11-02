import glob
import numpy as np
from src.utils import *
from pathlib import Path
from src.agents import IndependentAgent

if __name__ == '__main__':

    files = glob.glob('policy/*.pth')

    times = [int(f.split('time_')[1].split('.')[0]) for f in files]
    times = sorted(times)

    indices = np.linspace(0, len(times) - 1, 20, dtype=int)
    sample = [times[i] for i in indices]

    hyperparams = {
        'max_seed': int(1),
        'transfer_enabled': False,
        'restricted_communication': False,
        'env_name': 'SimpleSpread',
    }

    gif_path = 'gifs/SimpleSpread/'

    cfg = Config.from_hyperparameters(hyperparams)
    Path(gif_path).mkdir(parents=True, exist_ok=True)

    print(cfg.device)

    for time_to_eval in sample:
        print(f'------------- evaluating {time_to_eval} -------------')
        env = make_env(cfg, 'SimpleSpread')
        obs, _ = env.reset(seed=0)
        obs = flatten_obs_dict(obs)
        agent_ids = env.agents

        agents = {}
        for aid in agent_ids:
            space = env.observation_space(aid)
            o_dim = np.prod(space.shape)
            a_space = env.action_space(aid)
            agents[aid] = IndependentAgent(o_dim, a_space.n, cfg)
            agents[aid].load_policy_from_snapshot(time_to_eval, aid)

        evaluate_parallel(
            lambda: make_env(cfg, 'SimpleSpread', render=True),
            agents,
            1,
            cfg.max_episode_steps,
            time_to_eval,
            cfg.device,
            save_gif = True,
            gif_path = gif_path,
        )