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
    
    seed: int = 42
    device: str = get_current_device()

    # env
    max_episode_steps: int = 200
    continuous_actions: bool = False  # When using DQN actions must be Discrete
    total_env_steps: int = 300_000    # global step (per parallel env)

    # DQN
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 256
    replay_size: int = 100_000
    start_learning_after: int = 5_000
    train_freq: int = 1
    target_update_freq: int = 2000
    double_q: bool = True
    grad_clip: float = 10.0

    # eps-greedy (per-agent)
    eps_start: float = 1.0
    eps_final: float = 0.05
    eps_decay_steps: int = 150_000

    # logging
    log_every: int = 2000
    eval_every: int = 25_000
    eval_episodes: int = 10
    log_output_dir = 'data/simple-spread/'
