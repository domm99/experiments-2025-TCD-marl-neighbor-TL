import torch
import numpy as np
import random, time
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import imageio.v2 as imageio
from src.common.utils import *
from src.spread.config import Config
from pettingzoo.mpe import simple_spread_v3
from src.common.agents import IndependentAgent

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    
    cfg = Config()
    set_seed(cfg.seed)

    env = make_env(cfg)
    obs, _ = env.reset(seed=cfg.seed)
    agent_ids = env.agents 

    agents = {}
    for aid in agent_ids:
        o_dim = env.observation_space(aid).shape[0]
        a_space = env.action_space(aid)
        agents[aid] = IndependentAgent(o_dim, a_space.n, cfg)

    steps = 0
    t0 = time.time()
    last_log = 0
    while steps < cfg.total_env_steps:
        
        # Independent actions
        actions = {aid: agents[aid].act(obs[aid]) for aid in agent_ids}
        next_obs, rew, term, trunc, _ = env.step(actions)
        
        current_agents = list(next_obs.keys())
        if not current_agents:
            obs, _ = env.reset(seed=cfg.seed)
            continue


        dones = {aid: (term[aid] or trunc[aid]) for aid in current_agents}

        # Memorize each sars tuple 
        for aid in current_agents:
            agents[aid].rb.add(obs[aid], actions[aid], rew[aid], next_obs[aid], dones[aid])

        obs = next_obs
        steps += 1

        # Ottimizzazioni indipendenti
        for aid in current_agents:
            agents[aid].optimize()

        # Logging semplice
        if steps - last_log >= cfg.log_every:
            fps = int(steps / (time.time() - t0 + 1e-9))
            avg_eps = np.mean([a.eps for a in agents.values()])
            rb_sizes = ", ".join(f"{aid}:{agents[aid].rb.size}" for aid in current_agents)
            print(f"[{steps}] fps~{fps} eps~{avg_eps:.3f} rb_sizes=({rb_sizes})")
            last_log = steps

        # Eval
        if steps % cfg.eval_every == 0 and all(a.rb.size >= cfg.start_learning_after for a in agents.values()):
            avg = evaluate_parallel(lambda: make_env(cfg), agents, cfg.eval_episodes, cfg.max_episode_steps, cfg.device)
            print(f"Eval @ {steps}: avg team reward over {cfg.eval_episodes} eps = {avg:.3f}")

    env.close()