from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from common.config import HybridPPOConfig
from common.runtime_config import BASE_OBSERVATION_FEATURE_DIM, GUIDANCE_FEATURE_DIM
from common.types import MacroAction, MacroTransition
from primitives.library import ParameterizedPrimitiveLibrary, SemanticPrimitive
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
    _OBSERVED_FEATURE_DIM = int(BASE_OBSERVATION_FEATURE_DIM)
    _RECOVER_ENTER_ARTICULATION_RAD = float(np.deg2rad(24.0))
    _STOP_CHECK_GOAL_DISTANCE_NORM = 0.08
    _STOP_CHECK_HEADING_ERROR_RAD = float(np.deg2rad(12.0))
    _STOP_CHECK_MIN_CLEARANCE_NORM = 0.10

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
        self._action_active_mask = torch.as_tensor(
            np.stack([self.primitive_library.active_mask(action_id).astype(np.float32) for action_id in range(self.primitive_library.action_dim)], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        self._fallback_action_bias = self._build_fallback_action_bias()
        self._recover_action_ids = self._action_ids_for_semantic(SemanticPrimitive.ARTICULATION_RECOVER)
        self._stop_action_ids = self._action_ids_for_semantic(SemanticPrimitive.STOP_CHECK)
        self._proxy_cache_token = None
        self._proxy_parameter_centers = None
        self._proxy_parameter_scales = None
        self._proxy_valid_mask = None

    def act(self, observation: np.ndarray, deterministic: bool = False) -> ActionSelection:
        obs_tensor = self._obs_tensor(observation)
        policy_context = self._masked_policy_context(obs_tensor)
        discrete_dist = Categorical(logits=policy_context["masked_logits"])
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
        last_safety_loss = 0.0
        last_sampled_safety_score = 0.0

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
                total_log_prob, entropy_d, entropy_c, safety_loss, sampled_safety_score = self.evaluate_actions(obs_mb, action_mb, param_mb)
                ratio = torch.exp(total_log_prob - old_log_prob_mb)
                surrogate_1 = ratio * adv_mb
                surrogate_2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                actor_loss -= self.config.entropy_coef_discrete * entropy_d.mean()
                actor_loss -= self.config.entropy_coef_continuous * entropy_c.mean()
                actor_loss += float(self.config.safety_loss_coef) * safety_loss.mean()
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
                last_safety_loss = float(safety_loss.mean().item())
                last_sampled_safety_score = float(sampled_safety_score.mean().item())

        return {
            "actor_loss": last_actor_loss,
            "value_loss": last_value_loss,
            "entropy_discrete": last_entropy_d,
            "entropy_continuous": last_entropy_c,
            "safety_loss": last_safety_loss,
            "sampled_safety_score": last_sampled_safety_score,
            "advantage_mean": float(advantages.mean().item()),
            "return_mean": float(returns.mean().item()),
        }

    def evaluate_actions(self, observations: torch.Tensor, action_ids: torch.Tensor, parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_context = self._masked_policy_context(observations)
        discrete_dist = Categorical(logits=policy_context["masked_logits"])
        continuous_dist = self._continuous_dist(observations, action_ids)
        active_mask = self._active_mask_tensor(action_ids)
        discrete_log_prob = discrete_dist.log_prob(action_ids)
        continuous_log_prob = continuous_dist.masked_log_prob(parameters, active_mask)
        total_log_prob = discrete_log_prob + continuous_log_prob
        safety_loss, sampled_safety_score = self._continuous_safety_terms(policy_context, action_ids, parameters)
        return total_log_prob, discrete_dist.entropy(), continuous_dist.masked_entropy(active_mask), safety_loss, sampled_safety_score

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

    def _masked_policy_context(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_logits = self.policy.discrete_logits(observations)
        state_gate_valid_mask = self._state_gate_valid_mask(observations)
        context = {
            "raw_logits": raw_logits,
            "masked_logits": self._apply_state_gate_mask(raw_logits, state_gate_valid_mask),
            "semantic_scores": torch.ones_like(raw_logits),
            "state_gate_valid_mask": state_gate_valid_mask,
            "proxy_scores": None,
            "all_action_means": None,
            "all_action_stds": None,
        }
        if not self._soft_mask_enabled():
            return context

        self._ensure_proxy_tensors()
        proxy_scores, _articulation_bins = self._query_proxy_scores(observations)
        proxy_scores_tensor = torch.as_tensor(proxy_scores, dtype=torch.float32, device=self.device)
        all_action_means, all_action_stds = self._all_action_distribution_moments(observations)
        semantic_scores = self._distribution_aware_semantic_scores(
            all_action_means=all_action_means,
            all_action_stds=all_action_stds,
            proxy_scores=proxy_scores_tensor,
        )
        masked_logits = self._apply_state_gate_mask(
            self._apply_semantic_soft_mask(raw_logits=raw_logits, semantic_scores=semantic_scores),
            state_gate_valid_mask,
        )
        context.update(
            {
                "masked_logits": masked_logits,
                "semantic_scores": semantic_scores,
                "proxy_scores": proxy_scores_tensor,
                "all_action_means": all_action_means,
                "all_action_stds": all_action_stds,
            }
        )
        return context

    def _action_ids_for_semantic(self, semantic: SemanticPrimitive) -> Tuple[int, ...]:
        return tuple(int(spec.primitive_id) for spec in self.primitive_library.specs if spec.semantic == semantic)

    def _base_feature_offset(self, observations: torch.Tensor) -> int:
        sidecar = self.primitive_library.proxy_sidecar
        if sidecar is not None:
            return int(sidecar.num_rays)

        appended_guidance_offset = observations.shape[1] - self._OBSERVED_FEATURE_DIM - int(GUIDANCE_FEATURE_DIM)
        if appended_guidance_offset >= 4:
            return int(appended_guidance_offset)
        return int(observations.shape[1] - self._OBSERVED_FEATURE_DIM)

    def _state_gate_valid_mask(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        valid_mask = torch.ones((batch_size, self.primitive_library.action_dim), dtype=torch.bool, device=self.device)
        feature_offset = self._base_feature_offset(observations)
        if feature_offset < 0 or observations.shape[1] < feature_offset + self._OBSERVED_FEATURE_DIM:
            return valid_mask

        features = observations[:, feature_offset : feature_offset + self._OBSERVED_FEATURE_DIM]
        goal_distance = torch.clamp(features[:, 0], min=0.0)
        relative_heading = torch.atan2(features[:, 4], features[:, 3])
        articulation = torch.atan2(features[:, 6], features[:, 5])
        if feature_offset > 0:
            min_clearance = torch.min(observations[:, :feature_offset], dim=-1).values
        else:
            min_clearance = torch.ones((batch_size,), dtype=observations.dtype, device=self.device)

        if self._recover_action_ids:
            recover_allowed = torch.abs(articulation) >= float(self._RECOVER_ENTER_ARTICULATION_RAD)
            valid_mask[:, list(self._recover_action_ids)] = recover_allowed.unsqueeze(-1)

        if self._stop_action_ids:
            stop_allowed = (
                (goal_distance <= float(self._STOP_CHECK_GOAL_DISTANCE_NORM))
                & (torch.abs(relative_heading) <= float(self._STOP_CHECK_HEADING_ERROR_RAD))
                & (min_clearance <= float(self._STOP_CHECK_MIN_CLEARANCE_NORM))
            )
            valid_mask[:, list(self._stop_action_ids)] = stop_allowed.unsqueeze(-1)

        return valid_mask

    def _apply_state_gate_mask(self, logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask > 0.5
        if torch.all(valid_mask):
            return logits
        blocked_logits = torch.full_like(logits, -1e9)
        masked_logits = torch.where(valid_mask, logits, blocked_logits)
        all_invalid = ~torch.any(valid_mask, dim=-1, keepdim=True)
        return torch.where(all_invalid, logits, masked_logits)

    def _soft_mask_enabled(self) -> bool:
        return bool(self.config.soft_mask_enabled and self.primitive_library.proxy_sidecar is not None)

    def _ensure_proxy_tensors(self) -> None:
        sidecar = self.primitive_library.proxy_sidecar
        if sidecar is None:
            self._proxy_cache_token = None
            self._proxy_parameter_centers = None
            self._proxy_parameter_scales = None
            self._proxy_valid_mask = None
            return
        token = id(sidecar)
        if token == self._proxy_cache_token:
            return
        self._proxy_cache_token = token
        self._proxy_parameter_centers = torch.as_tensor(sidecar.parameter_centers, dtype=torch.float32, device=self.device)
        self._proxy_parameter_scales = torch.as_tensor(np.maximum(sidecar.parameter_scales, float(self.config.soft_mask_eps)), dtype=torch.float32, device=self.device)
        self._proxy_valid_mask = torch.as_tensor(sidecar.proxy_valid_mask.astype(np.float32), dtype=torch.float32, device=self.device)

    def _query_proxy_scores(self, observations: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        sidecar = self.primitive_library.proxy_sidecar
        if sidecar is None:
            raise RuntimeError("proxy safety sidecar is required when soft mask is enabled")
        observation_np = observations.detach().cpu().numpy()
        feature_offset = self._base_feature_offset(observations)
        if observation_np.shape[1] < feature_offset + self._OBSERVED_FEATURE_DIM:
            raise ValueError("observation does not contain enough post-lidar features to reconstruct articulation angle")
        scores = []
        bins = []
        for row in observation_np:
            articulation_angle = float(np.arctan2(row[feature_offset + 6], row[feature_offset + 5]))
            query = sidecar.compute_proxy_scores(
                lidar_observation=row[:feature_offset],
                articulation_angle=articulation_angle,
                gamma=float(self.config.soft_mask_gamma),
                eps=float(self.config.soft_mask_eps),
            )
            scores.append(query.proxy_scores)
            bins.append(query.articulation_bin_index)
        return np.stack(scores, axis=0).astype(np.float32), np.asarray(bins, dtype=np.int64)

    def _all_action_distribution_moments(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means = []
        stds = []
        batch_size = observations.shape[0]
        for action_id in range(self.primitive_library.action_dim):
            action_tensor = torch.full((batch_size,), int(action_id), dtype=torch.int64, device=self.device)
            dist = self._continuous_dist(observations, action_tensor)
            means.append(dist.mean)
            stds.append(dist.stddev)
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)

    def _distribution_aware_semantic_scores(self, all_action_means: torch.Tensor, all_action_stds: torch.Tensor, proxy_scores: torch.Tensor) -> torch.Tensor:
        active_mask = self._action_active_mask.unsqueeze(0).unsqueeze(2)
        proxy_centers = self._proxy_parameter_centers.unsqueeze(0)
        proxy_scales = self._proxy_parameter_scales.unsqueeze(0)
        means = all_action_means.unsqueeze(2)
        stds = all_action_stds.unsqueeze(2)
        denom = torch.clamp(proxy_scales + stds, min=float(self.config.soft_mask_eps))
        squared_distance = torch.square((means - proxy_centers) / denom) * active_mask
        active_count = torch.clamp(active_mask.sum(dim=-1), min=1.0)
        normalized_distance = squared_distance.sum(dim=-1) / active_count
        temperature = max(float(self.config.soft_mask_temperature), 1e-6)
        proxy_similarity = torch.exp(-0.5 * normalized_distance / (temperature * temperature)) * self._proxy_valid_mask.unsqueeze(0)
        similarity_sum = proxy_similarity.sum(dim=-1)
        semantic_scores = (proxy_similarity * proxy_scores).sum(dim=-1) / torch.clamp(similarity_sum, min=float(self.config.soft_mask_eps))
        semantic_scores = torch.where(similarity_sum > 0.0, semantic_scores, torch.zeros_like(semantic_scores))
        return torch.clamp(semantic_scores, 0.0, 1.0)

    def _apply_semantic_soft_mask(self, raw_logits: torch.Tensor, semantic_scores: torch.Tensor) -> torch.Tensor:
        floor = min(1.0, max(float(self.config.soft_mask_floor), float(self.config.soft_mask_eps)))
        effective_scores = torch.clamp(semantic_scores, min=floor, max=1.0)
        masked_logits = raw_logits + float(self.config.soft_mask_logit_scale) * torch.log(effective_scores)
        all_invalid = torch.all(semantic_scores <= floor + 1e-6, dim=-1)
        if torch.any(all_invalid):
            masked_logits = masked_logits + all_invalid.to(dtype=masked_logits.dtype).unsqueeze(-1) * self._fallback_action_bias.unsqueeze(0)
        return masked_logits

    def _continuous_safety_terms(self, policy_context: Dict[str, torch.Tensor], action_ids: torch.Tensor, parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = action_ids.shape[0]
        if not self._soft_mask_enabled() or policy_context["proxy_scores"] is None:
            zeros = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
            return zeros, zeros

        batch_indices = torch.arange(batch_size, device=self.device)
        selected_proxy_scores = policy_context["proxy_scores"][batch_indices, action_ids]
        selected_proxy_centers = self._proxy_parameter_centers[action_ids]
        selected_proxy_scales = self._proxy_parameter_scales[action_ids]
        selected_proxy_valid = self._proxy_valid_mask[action_ids]
        selected_active_mask = self._action_active_mask[action_ids]
        selected_means = policy_context["all_action_means"][batch_indices, action_ids]
        selected_stds = policy_context["all_action_stds"][batch_indices, action_ids]

        sampled_similarity = self._proxy_parameter_similarity(
            parameter_values=parameters,
            proxy_centers=selected_proxy_centers,
            proxy_scales=selected_proxy_scales,
            active_mask=selected_active_mask,
        ) * selected_proxy_valid
        sampled_score = self._weighted_proxy_score(selected_proxy_scores, sampled_similarity)

        mean_similarity = self._proxy_parameter_similarity(
            parameter_values=selected_means,
            proxy_centers=selected_proxy_centers,
            proxy_scales=selected_proxy_scales + selected_stds.unsqueeze(1),
            active_mask=selected_active_mask,
        ) * selected_proxy_valid
        mean_score = self._weighted_proxy_score(selected_proxy_scores, mean_similarity)
        safety_loss = 1.0 - mean_score
        return safety_loss, sampled_score

    def _proxy_parameter_similarity(self, parameter_values: torch.Tensor, proxy_centers: torch.Tensor, proxy_scales: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = active_mask.unsqueeze(1)
        values = parameter_values.unsqueeze(1)
        denom = torch.clamp(proxy_scales, min=float(self.config.soft_mask_eps))
        squared_distance = torch.square((values - proxy_centers) / denom) * expanded_mask
        active_count = torch.clamp(expanded_mask.sum(dim=-1), min=1.0)
        normalized_distance = squared_distance.sum(dim=-1) / active_count
        temperature = max(float(self.config.continuous_safety_temperature), 1e-6)
        return torch.exp(-0.5 * normalized_distance / (temperature * temperature))

    def _weighted_proxy_score(self, proxy_scores: torch.Tensor, proxy_similarity: torch.Tensor) -> torch.Tensor:
        similarity_sum = proxy_similarity.sum(dim=-1)
        weighted = (proxy_similarity * proxy_scores).sum(dim=-1) / torch.clamp(similarity_sum, min=float(self.config.soft_mask_eps))
        weighted = torch.where(similarity_sum > 0.0, weighted, torch.zeros_like(weighted))
        return torch.clamp(weighted, 0.0, 1.0)

    def _build_fallback_action_bias(self) -> torch.Tensor:
        bias = np.zeros((self.primitive_library.action_dim,), dtype=np.float32)
        fallback_bonus = float(self.config.soft_mask_fallback_bonus)
        if fallback_bonus <= 0.0:
            return torch.zeros((self.primitive_library.action_dim,), dtype=torch.float32, device=self.device)
        fallback_map = {
            SemanticPrimitive.STOP_CHECK: fallback_bonus,
            SemanticPrimitive.ARTICULATION_RECOVER: 0.75 * fallback_bonus,
            SemanticPrimitive.STRAIGHT_ADJUST: 0.5 * fallback_bonus,
        }
        matched = False
        for spec in self.primitive_library.specs:
            if spec.semantic in fallback_map:
                bias[int(spec.primitive_id)] = float(fallback_map[spec.semantic])
                matched = True
        if not matched and bias.size > 0:
            bias[0] = fallback_bonus
        return torch.as_tensor(bias, dtype=torch.float32, device=self.device)

    def _obs_tensor(self, observation: np.ndarray) -> torch.Tensor:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs
