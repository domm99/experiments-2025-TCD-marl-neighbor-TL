import time
import random
import argparse
from pathlib import Path
from src.config import Config
from src.transferlearning import *
from moviepy import ImageSequenceClip
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

    print(f'-------------------- USING {cfg.device} --------------------')

    Path(cfg.gif_output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.policy_output_dir).mkdir(parents=True, exist_ok=True)
    uncertainty_file_path = f'{cfg.data_output_dir}/uncertainty/'
    Path(uncertainty_file_path).mkdir(parents=True, exist_ok=True)
    debug_q_values_path = f'{cfg.data_output_dir}/qvalues/'
    Path(debug_q_values_path).mkdir(parents=True, exist_ok=True)
    training_gif_path = f'{cfg.gif_output_dir}training/'
    Path(training_gif_path).mkdir(parents=True, exist_ok=True)

    for seed in range(max_seed):

        set_seed(seed)
        env_name = cfg.env_name

        df_eval_results = pd.DataFrame(columns=['MeanReward'])
        csv_eval_file_path = f'{cfg.data_output_dir}/eval-results-seed_{seed}.csv'
        df_eval_results.to_csv(csv_eval_file_path, index=False)

        csv_train_file_path = f'{cfg.data_output_dir}/train-results-seed_{seed}.csv'
        df_train_loss = pd.DataFrame(columns=['MeanLoss'])
        df_train_reward = pd.DataFrame(columns=['MeanReward'])

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
            agents[aid] = IndependentAgent(aid, observation_space_dim, action_space_dim, seed, cfg)
            agents_uncertainty[aid] = []

            # Create dataframe for each agent
            df_u = pd.DataFrame(columns=['Uncertainty'])
            df_u.to_csv(f'{uncertainty_file_path}{aid}-seed_{seed}.csv', index=False)
            df_debug_q_values = pd.DataFrame(columns=['MeanQ', 'MeanTarget'])
            df_debug_q_values.to_csv(f'{debug_q_values_path}{aid}-seed_{seed}.csv', index=False)

        steps = 0
        t0 = time.time()
        last_log = 0

        for episode in range(cfg.training_episodes + 1):

            print(f'Starting episode {episode}')
            obs = env.reset()

            episode_losses = []
            episode_rewards = []
            frames = []

            for step in range(cfg.max_training_steps_per_episode):


                #actions = {aid: torch.tensor([0 ,0 ,0]) for aid in agent_ids}#agents[aid].act(obs[aid]) for aid in agent_ids}
                actions = {aid: agents[aid].act(obs[aid]) for aid in agent_ids}
                #print(actions)
                next_obs, rew, term, _ = env.step(actions)

                #print(f'next_obs shape {next_obs["agent_0"].shape}')
                #print(f'rew shape {rew.shape}')
                #print(f'term shape {term.shape}')

                current_agents = list(next_obs.keys())

                if not current_agents or term.item():
                    print(f"Resetting environment at step {step} and frames {len(frames)}")
                    if frames:
                        clip = ImageSequenceClip(frames, fps=30)
                        clip.write_gif(f'{training_gif_path}RESET-episode_{episode}.gif', fps=30)
                    obs = env.reset()
                    break

                for aid in current_agents:
                    o, r, a, next_o = obs[aid].flatten(), rew[aid].flatten(), actions[aid].flatten(), next_obs[aid].flatten()
                    t_sars = torch.cat([o, a, r, next_o])
                    uncertainty = agents[aid].compute_uncertainty(t_sars)
                    o, r, a, next_o = o.cpu().numpy(), r.item(), a.item(), next_o.cpu().numpy()
                    agents[aid].store_experience(o, a, r, next_o, term.item(), uncertainty.detach().cpu().item())
                    agents[aid].optimize_sars_rnd(uncertainty)

                obs = next_obs

                losses = []
                for aid in current_agents:
                    loss = agents[aid].optimize()
                    losses.append(loss)

                mean_r = np.mean([re.cpu() for re in rew.values()])
                episode_rewards.append(mean_r)

                if all(l is not None for l in losses):
                    mean_loss = np.mean(losses)
                    episode_losses.append(mean_loss)


                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                )
                if frame is not None:
                    frames.append(frame)

            ################################### END OF EPISODE LOOP ###################################

            for aid in agent_ids:
                agents[aid].increment_decay_time()

            train_reward.append(np.mean(episode_rewards))
            train_loss.append(np.mean(episode_losses))

            if frames and episode % cfg.export_gif_every == 0:
                clip = ImageSequenceClip(frames, fps=30)
                clip.write_gif(f'{training_gif_path}episode_{episode}.gif', fps=30)

            # Transfer learning
            if (cfg.transfer_enabled
                    and episode % cfg.transfer_every == 0
                    and episode > 0
                    and all(a.rb.size >= cfg.start_learning_after for a in agents.values())):
                print('------------------- TRANSFERRING EXPERIENCE -------------------')
                if cfg.restricted_communication:
                    transfer_learning_with_restricted_communication(cfg, agents, env)
                else:
                    transfer_learning_all_agents(agents)

            # Logging uncertainties
            if episode % cfg.log_uncertainty_every == 0:
                agents_uncertainty = log_uncertainty(agents, agents_uncertainty)

        ################################### END OF TRAINING LOOP ###################################

        # Dumping everything to csv
        df_train_reward['MeanReward'] = train_reward
        df_train_loss['MeanLoss'] = train_loss

        df_train_loss.to_csv(csv_train_file_path, index=False)
        df_train_reward.to_csv(f'{cfg.data_output_dir}/train-reward-seed_{seed}.csv', index=False)
        for aid, uncertainties in agents_uncertainty.items():
            df_u = pd.read_csv(f'{uncertainty_file_path}{aid}-seed_{seed}.csv')
            df_u['Uncertainty'] = uncertainties
            df_u.to_csv(f'{uncertainty_file_path}{aid}-seed_{seed}.csv', index=False)

        for aid in agent_ids:
            agents[aid].dump_logged_qvalues_to_csv()

    ################################### END OF SEEDS LOOP ###################################