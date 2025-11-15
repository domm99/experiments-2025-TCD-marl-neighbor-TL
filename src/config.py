import os
import yaml
import torch
from dataclasses import dataclass

def get_current_device():
    device: str = 'cpu'
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            device = current_accelerator.type
    return device

@dataclass
class Config:
    
    max_seed: int = 1
    device: str = get_current_device()

    # env
    continuous_actions: bool = False  # When using DQN actions must be Discrete
    total_env_steps: int = 400_000    # global steps
    num_parallel_envs = 1
    n_agents: int = 3

    # DQN
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 512
    replay_size: int = 300_000
    start_learning_after: int = 1_000
    train_freq: int = 5
    target_update_freq: int = 2_000
    double_q: bool = True
    grad_clip: float = 10.0

    # eps-greedy (per-agent)
    eps_start: float = 1.0
    eps_final: float = 0.08
    eps_decay_steps: int = 400_000
    eps_decay_rate: float = 6.0

    # transfer learning
    transfer_enabled: bool = False
    restricted_communication: bool = False
    transfer_every: int = 10
    transfer_budget: int = 5
    K: int = 3 # Number of nearest neighbors

    # logging
    logging_enabled: bool = False
    log_every: int = 2000
    eval_every: int = 2_000
    eval_episodes: int = 5
    eval_steps: int = 50
    env_name: str = ''
    data_output_dir: str = ''
    policy_output_dir: str = ''
    gif_output_dir: str = ''

    @classmethod
    def from_hyperparameters(cls, hyperparams):
        env_name = hyperparams['env_name']
        transfer_enabled = hyperparams['transfer_enabled']
        restricted_communication = hyperparams['restricted_communication']

        return cls(
            max_seed=hyperparams['max_seed'],
            transfer_enabled=transfer_enabled,
            restricted_communication=restricted_communication,
            env_name=env_name,
            data_output_dir=f'data/{env_name}/transfer_{transfer_enabled}-restricted_{restricted_communication}/',
            policy_output_dir=f'policy/{env_name}/transfer_{transfer_enabled}-restricted_{restricted_communication}/',
            gif_output_dir=f'gifs/{env_name}/transfer_{transfer_enabled}-restricted_{restricted_communication}/',
        )
