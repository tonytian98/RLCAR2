from torch import nn
import torch


class DuelingDQN(nn.Module):
    def __init__(self, obs_size: int, hidden_sizes: list[int], n_actions: int):
        super().__init__()
        hidden_layers = []
        if len(hidden_sizes) > 1:
            for i in range(len(hidden_sizes) - 1):
                hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                hidden_layers.append(nn.ReLU())

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_sizes[0]), nn.ReLU(), *hidden_layers
        )

        self.fc_value = nn.Linear(hidden_sizes[-1], 1)
        self.fc_adv = nn.Linear(hidden_sizes[-1], n_actions)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        x = x.to(torch.float)
        adv = self.fc_avd(x)
        value = self.fc_value(x)
        return value + adv - torch.mean(adv, dim=1, keepdim=True)

    def get_net(self):
        return self.net
