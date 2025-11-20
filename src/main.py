import time
import random
import argparse
from pathlib import Path
from src.config import Config
from src.transferlearning import *
from src.agents import IndependentAgent

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def str2bool(v):
    return v.lower() in "true"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seed', default='1')
    parser.add_argument('--transfer_enabled', default='False')
    parser.add_argument('--restricted_communication', default='False')
    parser.add_argument('--env_name', default='dispersion')
    args = parser.parse_args()

    hyperparams = {
        'max_seed': int(args.max_seed),
        'transfer_enabled': str2bool(args.transfer_enabled),
        'restricted_communication': str2bool(args.restricted_communication),
        'env_name': args.env_name,
    }

    print(hyperparams)

    cfg = Config.from_hyperparameters(hyperparams)

    max_seed = cfg.max_seed
    for seed in range(max_seed):
        set_seed(seed)

        env_name = cfg.env_name

        print(f'-------------------- USING {cfg.device} --------------------')

        Path(cfg.gif_output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.data_output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.policy_output_dir).mkdir(parents=True, exist_ok=True)

        df_eval_results = pd.DataFrame(columns=['MeanReward'])
        csv_eval_file_path = f'{cfg.data_output_dir}/eval-results-seed_{seed}.csv'
        df_eval_results.to_csv(csv_eval_file_path, index=False)

        csv_train_file_path = f'{cfg.data_output_dir}/train-results-seed_{seed}.csv'
        df_train_loss = pd.DataFrame(columns=['MeanLoss'])
        df_train_reward = pd.DataFrame(columns=['MeanReward'])

        uncertainty_file_path = f'{cfg.data_output_dir}/uncertainty/'
        Path(uncertainty_file_path).mkdir(parents=True, exist_ok=True)

        debug_q_values_path = f'{cfg.data_output_dir}/qvalues/'
        Path(debug_q_values_path).mkdir(parents=True, exist_ok=True)

        train_reward = []
        train_loss = []
        eval_reward = []
        agents_uncertainty = {}

        env = make_vmas_env(cfg, env_name, seed)
        agent_ids = [agent.name for agent in env.agents]

        print(agent_ids)

        agents = {}

        for aid in agent_ids:
            observation_space_dim = np.prod(env.observation_space[aid].shape)
            action_space_dim = env.action_space[aid].n
            agents[aid] = IndependentAgent(aid, observation_space_dim, action_space_dim, cfg)
            agents_uncertainty[aid] = []

            # Create dataframes for each agent
            df_u = pd.DataFrame(columns=['Uncertainty'])
            df_u.to_csv(f'{uncertainty_file_path}{aid}-seed_{seed}.csv', index=False)
            df_debug_q_values = pd.DataFrame(columns=['MeanQ', 'MeanTarget'])
            df_debug_q_values.to_csv(f'{debug_q_values_path}{aid}-seed_{seed}.csv', index=False)

        steps = 0
        t0 = time.time()
        last_log = 0

        obs = env.reset()

        while steps < cfg.total_env_steps:

            actions = {aid: agents[aid].act(obs[aid]) for aid in agent_ids}
            next_obs, rew, term, _ = env.step(actions)

            current_agents = list(next_obs.keys())
            if not current_agents:
                obs = env.reset()
                continue

            for aid in current_agents:
                o, r, a, next_o = obs[aid], rew[aid], actions[aid], next_obs[aid]
                t_sars = torch.cat([o.flatten(), a.flatten(), r.flatten(), next_o.flatten()])
                uncertainty = agents[aid].compute_uncertainty(t_sars)
                agents[aid].store_experience(obs[aid], actions[aid], rew[aid], next_obs[aid], term.item(), uncertainty.detach().cpu().item())
                agents[aid].optimize_sars_rnd(uncertainty)

            obs = next_obs
            steps += 1
            losses = []
            for aid in current_agents:
                loss = agents[aid].optimize()
                losses.append(loss)
            mean_r = np.mean([re.cpu() for re in rew.values()])
            train_reward.append(mean_r)

            if all(l is not None for l in losses):
                mean_loss = np.mean(losses)
                train_loss.append(mean_loss)

            # Transfer learning
            if (cfg.transfer_enabled
                    and steps % cfg.transfer_every == 0
                    and all(a.rb.size >= cfg.start_learning_after for a in agents.values())):
                print('------------------- TRANSFERRING EXPERIENCE -------------------')
                if cfg.restricted_communication:
                    transfer_learning_with_restricted_communication(cfg, current_agents, agents, env)
                else:
                    transfer_learning_all_agents(current_agents, agents)

            # Logging
            if cfg.logging_enabled and steps % cfg.log_every == 0:
                fps = int(steps / (time.time() - t0 + 1e-9))
                avg_eps = np.mean([a.eps for a in agents.values()])
                rb_sizes = ", ".join(f"{aid}:{agents[aid].rb.size}" for aid in current_agents)
                print(f"[{steps}] fps~{fps} eps~{avg_eps:.3f} rb_sizes=({rb_sizes})")
                last_log = steps

            # Eval
            if steps % cfg.eval_every == 0 and all(a.rb.size >= cfg.start_learning_after for a in agents.values()):
                avg = evaluate_parallel(lambda: make_vmas_env(cfg, env_name, seed), agents, cfg.eval_episodes, cfg.eval_steps, steps, cfg.device, save_gif=True, gif_path=cfg.gif_output_dir)
                print(f"Eval @ {steps}: avg team reward over {cfg.eval_episodes} eps = {avg:.3f}")
                eval_reward.append(avg)
                agents_uncertainty = log_uncertainty(current_agents, agents, agents_uncertainty) #uncertainty_file_path, seed)

        # Dumping everything to csv
        df_train_reward['MeanReward'] = train_reward
        df_train_loss['MeanLoss'] = train_loss
        df_eval_results['MeanReward'] = eval_reward

        df_train_loss.to_csv(csv_train_file_path, index=False)
        df_train_reward.to_csv(f'{cfg.data_output_dir}/train-reward-seed_{seed}.csv', index=False)
        df_eval_results.to_csv(csv_eval_file_path, index=False)
        for aid, uncertainties in agents_uncertainty.items():
            df_u = pd.read_csv(f'{uncertainty_file_path}{aid}-seed_{seed}.csv')
            df_u['Uncertainty'] = uncertainties
            df_u.to_csv(f'{uncertainty_file_path}{aid}-seed_{seed}.csv', index=False)

        for aid in agent_ids:
            agents[aid].dump_logged_qvalues_to_csv()