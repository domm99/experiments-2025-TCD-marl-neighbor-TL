import torch
import torch.nn as nn

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv   = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        z = self.feat(x)
        v = self.value(z)
        a = self.adv(z)
        return v + a - a.mean(dim=1, keepdim=True)



class SarsRND(nn.Module):

    def __init__(self, obs_dim: int, hidden=None, embedding_size: int = 1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hidden is None:
            hidden = [512, 256, 128]
        self.hidden = hidden
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h

        layers.append(nn.Linear(in_dim, embedding_size))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)