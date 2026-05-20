import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> None:
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridPolicyNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, parameter_dim: int, hidden_dim: int = 256, action_embedding_dim: int = 32) -> None:
        super().__init__()
        self.encoder = MLP(observation_dim, hidden_dim, hidden_dim, depth=2)
        self.discrete_head = nn.Linear(hidden_dim, action_dim)
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        self.continuous_head = MLP(hidden_dim + action_embedding_dim, hidden_dim, 2 * parameter_dim, depth=2)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def discrete_logits(self, observation: torch.Tensor) -> torch.Tensor:
        return self.discrete_head(self.encode(observation))

    def continuous_raw(self, observation: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(observation)
        action_embed = self.action_embedding(action_ids)
        if action_embed.dim() == 1:
            action_embed = action_embed.unsqueeze(0)
        return self.continuous_head(torch.cat([encoded, action_embed], dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = MLP(observation_dim, hidden_dim, 1, depth=2)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)
