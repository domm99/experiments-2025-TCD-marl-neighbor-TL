import random, time
from pathlib import Path
from src.common.utils import *
from src.spread.config import Config
from src.common.agents import IndependentAgent

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    
    cfg = Config()
    set_seed(cfg.seed)

    env_name = 'Pursuit'

    Path(cfg.log_output_dir).mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(columns=['Steps', 'Episodes', 'MeanTeamReward'])
    csv_file_path = f'{cfg.log_output_dir}/results.csv'
    df_results.to_csv(csv_file_path, index=False)

    uncertainty_file_path = f'{cfg.log_output_dir}/uncertainty/'
    Path(uncertainty_file_path).mkdir(parents=True, exist_ok=True)

    env = make_env(cfg, env_name)
    obs, _ = env.reset(seed=cfg.seed)
    obs = flatten_obs_dict(obs)
    agent_ids = env.agents 

    agents = {}
    for aid in agent_ids:
        o_dim = env.observation_space(aid).shape[0]
        a_space = env.action_space(aid)
        agents[aid] = IndependentAgent(o_dim, a_space.n, cfg)
        df_u = pd.DataFrame(columns=['Uncertainty'])
        df_u.to_csv(f'{uncertainty_file_path}{aid}.csv', index=False)

    steps = 0
    t0 = time.time()
    last_log = 0
    while steps < cfg.total_env_steps:
        # Independent actions
        actions = {aid: agents[aid].act(obs[aid]) for aid in agent_ids}
        next_obs, rew, term, trunc, _ = env.step(actions)
        next_obs = flatten_obs_dict(next_obs)

        current_agents = list(next_obs.keys())
        if not current_agents:
            obs, _ = env.reset(seed=cfg.seed)
            obs = flatten_obs_dict(obs)
            continue

        dones = {aid: (term[aid] or trunc[aid]) for aid in current_agents}

        for aid in current_agents:
            o, r, a, next_o = obs[aid], rew[aid], actions[aid], next_obs[aid]
            t_sars = np.concatenate([o, np.array([r]), np.array([a]), next_o], dtype=np.float32)
            t_sars = torch.tensor(t_sars, device=cfg.device)
            uncertainty = agents[aid].compute_uncertainty(t_sars)
            agents[aid].store_experience(obs[aid], actions[aid], rew[aid], next_obs[aid], dones[aid], uncertainty.detach().cpu().item())
            agents[aid].optimize_sars_rnd(uncertainty)

        obs = next_obs
        steps += 1

        for aid in current_agents:
            agents[aid].optimize()

        log_uncertainty(current_agents, agents, uncertainty_file_path)

        # Logging
        if steps - last_log >= cfg.log_every:
            fps = int(steps / (time.time() - t0 + 1e-9))
            avg_eps = np.mean([a.eps for a in agents.values()])
            rb_sizes = ", ".join(f"{aid}:{agents[aid].rb.size}" for aid in current_agents)
            print(f"[{steps}] fps~{fps} eps~{avg_eps:.3f} rb_sizes=({rb_sizes})")
            last_log = steps

        # Eval
        if steps % cfg.eval_every == 0 and all(a.rb.size >= cfg.start_learning_after for a in agents.values()):
            avg = evaluate_parallel(lambda: make_env(cfg, env_name), agents, cfg.eval_episodes, cfg.max_episode_steps, cfg.device)
            df_results = pd.read_csv(csv_file_path)
            print(f"Eval @ {steps}: avg team reward over {cfg.eval_episodes} eps = {avg:.3f}")
            new_line = {'Steps': steps, 'Episodes': cfg.eval_episodes, 'MeanTeamReward': avg}
            df_results = pd.concat([df_results, pd.DataFrame([new_line])], ignore_index=True)
            df_results.to_csv(csv_file_path, index=False)

    env.close()