from typing import Optional

import torch
from torch.distributions import Beta


class AffineBeta:
    """Independent Beta distributions mapped to bounded physical ranges."""

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-6) -> None:
        self.base = Beta(alpha, beta)
        self.low = low
        self.high = high
        self.scale = torch.clamp(high - low, min=eps)
        self.eps = float(eps)

    @property
    def mean(self) -> torch.Tensor:
        return self.low + self.scale * self.base.mean

    def sample(self) -> torch.Tensor:
        return self.low + self.scale * self.base.sample()

    def rsample(self) -> torch.Tensor:
        if hasattr(self.base, "rsample"):
            return self.low + self.scale * self.base.rsample()
        return self.sample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((value - self.low) / self.scale, self.eps, 1.0 - self.eps)
        return self.base.log_prob(normalized) - torch.log(self.scale)

    def entropy(self) -> torch.Tensor:
        return self.base.entropy() + torch.log(self.scale)

    def masked_log_prob(self, value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        log_prob = self.log_prob(value)
        if mask is None:
            return log_prob.sum(dim=-1)
        return (log_prob * mask.to(dtype=log_prob.dtype)).sum(dim=-1)

    def masked_entropy(self, mask: Optional[torch.Tensor]) -> torch.Tensor:
        entropy = self.entropy()
        if mask is None:
            return entropy.sum(dim=-1)
        return (entropy * mask.to(dtype=entropy.dtype)).sum(dim=-1)
