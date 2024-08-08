from torch import nn
import torch


class DQN(nn.Module):
    def __init__(self, obs_size: int, hidden_sizes: list[int], n_actions: int):
        super().__init__()
        hidden_layers = []
        if len(hidden_sizes) > 1:
            for i in range(len(hidden_sizes) - 1):
                hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                hidden_layers.append(nn.ReLU())

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_sizes[0]),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_sizes[-1], n_actions),
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        return self.net(x.to(torch.float))

    def get_net(self):
        return self.net
