from typing import Optional

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
        self.action_dim = int(action_dim)
        self.observation_encoder = MLP(observation_dim, hidden_dim, hidden_dim, depth=2)
        self.action_mask_encoder = MLP(action_dim, hidden_dim, hidden_dim, depth=1)
        self.fusion = MLP(2 * hidden_dim, hidden_dim, hidden_dim, depth=1)
        self.discrete_head = nn.Linear(hidden_dim, action_dim)
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        self.continuous_head = MLP(hidden_dim + action_embedding_dim, hidden_dim, 2 * parameter_dim, depth=2)

    def _coerce_action_mask(self, observation: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if action_mask is None:
            return torch.ones(
                (observation.shape[0], self.action_dim),
                dtype=observation.dtype,
                device=observation.device,
            )
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        return action_mask.to(device=observation.device, dtype=observation.dtype)

    def encode(self, observation: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        obs_encoded = self.observation_encoder(observation)
        mask_encoded = self.action_mask_encoder(self._coerce_action_mask(observation, action_mask))
        return self.fusion(torch.cat([obs_encoded, mask_encoded], dim=-1))

    def discrete_logits(self, observation: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.discrete_head(self.encode(observation, action_mask=action_mask))

    def continuous_raw(self, observation: torch.Tensor, action_ids: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoded = self.encode(observation, action_mask=action_mask)
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
