import os
import torch
from pathlib import Path
from benchmarl.algorithms import IqlConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.environments import PettingZooTask
from benchmarl.experiment import Experiment, ExperimentConfig

def get_current_device():
    device: str = 'cpu'
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            device = current_accelerator.type
    return device

if __name__ == '__main__':

    device = get_current_device()
    print(f'Using device: {device}')

    algorithm_config = IqlConfig.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = device
    experiment_config.train_device= device
    experiment_config.buffer_device = device
    experiment_config.save_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    experiment_config.loggers = ['csv']

    task = PettingZooTask.SIMPLE_SPREAD.get_from_yaml()
    task.config['N'] = 10

    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [256, 128]
    model_config.activation_class = torch.nn.ReLU

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=42,
        config=experiment_config,
    )

    print('Starting experiment')
    experiment.run()
