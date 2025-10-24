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
    max_episode_steps: int = 300
    continuous_actions: bool = False  # When using DQN actions must be Discrete
    total_env_steps: int = 500_000    # global steps

    # DQN
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 256
    replay_size: int = 100_000
    start_learning_after: int = 2000
    train_freq: int = 1
    target_update_freq: int = 2000
    double_q: bool = True
    grad_clip: float = 10.0

    # eps-greedy (per-agent)
    eps_start: float = 1.0
    eps_final: float = 0.05
    eps_decay_steps: int = 150_000

    # transfer learning
    transfer_enabled: bool = False
    restricted_communication: bool = False
    transfer_every: int = 2000
    transfer_budget: int = 1000
    K: int = 3 # Number of nearest neighbors
    communication_range: float = 2

    # logging
    logging_enabled: bool = False
    log_every: int = 2000
    eval_every: int = 10_000
    eval_episodes: int = 10
    env_name: str = ''
    data_output_dir: str = 'data/'
    log_output_dir: str = ''

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
            log_output_dir=f"data/{env_name}-transfer_{transfer_enabled}-restricted_{restricted_communication}/"
        )