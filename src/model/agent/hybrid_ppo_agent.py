from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from common.config import HybridPPOConfig
from common.types import MacroAction, MacroTransition
from primitives.library import ParameterizedPrimitiveLibrary
from ..buffer import SMDPBatch, SMDPRolloutBuffer
from ..distributions import AffineBeta
from ..networks import HybridPolicyNetwork, ValueNetwork
from ..smdp import compute_smdp_targets


@dataclass(frozen=True)
class ActionSelection:
    macro_action: MacroAction
    log_prob: float
    value: float
    discrete_log_prob: float
    continuous_log_prob: float


class HybridPPOAgent:
    def __init__(self, config: HybridPPOConfig, primitive_library: ParameterizedPrimitiveLibrary, device: Optional[torch.device] = None) -> None:
        self.config = config
        self.primitive_library = primitive_library
        self.device = device or torch.device("cpu")
        self.policy = HybridPolicyNetwork(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            parameter_dim=config.parameter_dim,
            hidden_dim=config.hidden_dim,
            action_embedding_dim=config.action_embedding_dim,
        ).to(self.device)
        self.value_net = ValueNetwork(config.observation_dim, hidden_dim=config.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.critic_lr)
        self.buffer = SMDPRolloutBuffer()
        self._low = torch.as_tensor(self.primitive_library.low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(self.primitive_library.high, dtype=torch.float32, device=self.device)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> ActionSelection:
        obs_tensor = self._obs_tensor(observation)
        logits = self.policy.discrete_logits(obs_tensor)
        discrete_dist = Categorical(logits=logits)
        action_ids = torch.argmax(discrete_dist.probs, dim=-1) if deterministic else discrete_dist.sample()
        continuous_dist = self._continuous_dist(obs_tensor, action_ids)
        parameters = continuous_dist.mean if deterministic else continuous_dist.sample()
        active_mask = self._active_mask_tensor(action_ids)
        discrete_log_prob = discrete_dist.log_prob(action_ids)
        continuous_log_prob = continuous_dist.masked_log_prob(parameters, active_mask)
        total_log_prob = discrete_log_prob + continuous_log_prob
        value = self.value_net(obs_tensor).squeeze(-1)
        parameter_vector = parameters.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return ActionSelection(
            macro_action=MacroAction(primitive_id=int(action_ids.item()), parameters=parameter_vector),
            log_prob=float(total_log_prob.item()),
            value=float(value.item()),
            discrete_log_prob=float(discrete_log_prob.item()),
            continuous_log_prob=float(continuous_log_prob.item()),
        )

    def store_transition(self, transition: MacroTransition) -> None:
        self.buffer.add(transition)

    def update(self) -> Dict[str, float]:
        batch = self.buffer.as_batch()
        metrics = self.update_from_batch(batch)
        self.buffer.clear()
        return metrics

    def update_from_batch(self, batch: SMDPBatch) -> Dict[str, float]:
        observations = self._obs_tensor(batch.observations)
        next_observations = self._obs_tensor(batch.next_observations)
        action_ids = torch.as_tensor(batch.action_ids, dtype=torch.int64, device=self.device)
        parameters = torch.as_tensor(batch.parameters, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_values = self.value_net(next_observations).squeeze(-1).cpu().numpy()
        targets = compute_smdp_targets(
            rewards=batch.rewards,
            taus=batch.taus,
            dones=batch.dones,
            values=batch.values,
            next_values=next_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        advantages = torch.as_tensor(targets.advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(targets.returns, dtype=torch.float32, device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_samples = observations.shape[0]
        indices = np.arange(num_samples)
        last_actor_loss = 0.0
        last_value_loss = 0.0
        last_entropy_d = 0.0
        last_entropy_c = 0.0

        for _ in range(int(self.config.update_epochs)):
            np.random.shuffle(indices)
            for start in range(0, num_samples, int(self.config.mini_batch_size)):
                stop = min(start + int(self.config.mini_batch_size), num_samples)
                mb = indices[start:stop]
                obs_mb = observations[mb]
                action_mb = action_ids[mb]
                param_mb = parameters[mb]
                adv_mb = advantages[mb]
                return_mb = returns[mb]
                old_log_prob_mb = old_log_probs[mb]
                total_log_prob, entropy_d, entropy_c = self.evaluate_actions(obs_mb, action_mb, param_mb)
                ratio = torch.exp(total_log_prob - old_log_prob_mb)
                surrogate_1 = ratio * adv_mb
                surrogate_2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                actor_loss -= self.config.entropy_coef_discrete * entropy_d.mean()
                actor_loss -= self.config.entropy_coef_continuous * entropy_c.mean()
                predicted_values = self.value_net(obs_mb).squeeze(-1)
                value_loss = F.mse_loss(predicted_values, return_mb)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                last_actor_loss = float(actor_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy_d = float(entropy_d.mean().item())
                last_entropy_c = float(entropy_c.mean().item())

        return {
            "actor_loss": last_actor_loss,
            "value_loss": last_value_loss,
            "entropy_discrete": last_entropy_d,
            "entropy_continuous": last_entropy_c,
            "advantage_mean": float(advantages.mean().item()),
            "return_mean": float(returns.mean().item()),
        }

    def evaluate_actions(self, observations: torch.Tensor, action_ids: torch.Tensor, parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy.discrete_logits(observations)
        discrete_dist = Categorical(logits=logits)
        continuous_dist = self._continuous_dist(observations, action_ids)
        active_mask = self._active_mask_tensor(action_ids)
        discrete_log_prob = discrete_dist.log_prob(action_ids)
        continuous_log_prob = continuous_dist.masked_log_prob(parameters, active_mask)
        total_log_prob = discrete_log_prob + continuous_log_prob
        return total_log_prob, discrete_dist.entropy(), continuous_dist.masked_entropy(active_mask)

    def estimate_value(self, observation: np.ndarray) -> float:
        with torch.no_grad():
            value = self.value_net(self._obs_tensor(observation)).squeeze(-1)
        return float(value.item())

    def checkpoint_state(self) -> Dict[str, object]:
        return {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_checkpoint_state(self, checkpoint_state: Mapping[str, object]) -> None:
        self.policy.load_state_dict(checkpoint_state["policy"])
        self.value_net.load_state_dict(checkpoint_state["value_net"])
        self.actor_optimizer.load_state_dict(checkpoint_state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint_state["critic_optimizer"])

    def _continuous_dist(self, observations: torch.Tensor, action_ids: torch.Tensor) -> AffineBeta:
        raw = self.policy.continuous_raw(observations, action_ids)
        alpha_raw, beta_raw = torch.chunk(raw, 2, dim=-1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0
        return AffineBeta(alpha=alpha, beta=beta, low=self._low, high=self._high)

    def _active_mask_tensor(self, action_ids: torch.Tensor) -> torch.Tensor:
        action_list = action_ids.detach().cpu().tolist()
        masks = [self.primitive_library.active_mask(int(action_id)).astype(np.float32) for action_id in action_list]
        return torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32, device=self.device)

    def _obs_tensor(self, observation: np.ndarray) -> torch.Tensor:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs
