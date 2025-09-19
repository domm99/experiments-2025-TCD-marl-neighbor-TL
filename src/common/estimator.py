import torch
import torch.nn.functional as F
from src.common.models import SarsRND

class UncertaintyEstimator:

    def __init__(self, observation_dim: int, action_dim: int, reward_dim: int, learning_rate: float):

        ## Input of SarsRND --> (s_t, a, r, s_{t+1})
        ## Input size --> obs_dim + action_size + 1 + obs_dim
        rnd_input_size = observation_dim * 2 + action_dim + reward_dim

        self.predictor = SarsRND(rnd_input_size)
        self.target = SarsRND(rnd_input_size)
        self.target.eval()
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    def compute_uncertainty(self, x: torch.Tensor):
        with torch.no_grad():
            y_target = self.target(x)
        y_pred = self.predictor(x)
        return F.mse_loss(y_pred, y_target)

    def optimize(self, uncertainty: torch.Tensor):
        self.predictor.train()
        self.optimizer.zero_grad(set_to_none=True)
        uncertainty.backward()
        self.optimizer.step()