from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

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
    proxy_scores: Optional[np.ndarray] = None
    proxy_prefix_lengths: Optional[np.ndarray] = None


class HybridPPOAgent:
    _OBSERVED_FEATURE_DIM = int(BASE_OBSERVATION_FEATURE_DIM)
    _RECOVER_ENTER_ARTICULATION_RAD = float(np.deg2rad(24.0))
    _STOP_CHECK_GOAL_DISTANCE_NORM = 0.08
    _STOP_CHECK_HEADING_ERROR_RAD = float(np.deg2rad(12.0))
    _STOP_CHECK_MIN_CLEARANCE_NORM = 0.10
    _TEACHER_DISCRETE_LOSS_COEF = 0.25
    _TEACHER_PARAMETER_LOSS_COEF = 0.10

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
        self.buffer = SMDPRolloutBuffer(action_dim=self.primitive_library.action_dim, parameter_dim=self.primitive_library.parameter_dim)
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

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        teacher_action_probs: Optional[np.ndarray] = None,
        teacher_weight: float = 0.0,
    ) -> ActionSelection:
        obs_tensor = self._obs_tensor(observation)
        policy_context = self._masked_policy_context(obs_tensor)
        selection_probs = self._selection_probs_from_context(
            policy_context,
            teacher_action_probs=teacher_action_probs,
            teacher_weights=teacher_weight,
        )
        discrete_dist = Categorical(probs=selection_probs)
        action_ids = torch.argmax(discrete_dist.probs, dim=-1) if deterministic else discrete_dist.sample()
        continuous_dist = self._continuous_dist(obs_tensor, action_ids, action_mask=policy_context["network_action_mask"])
        parameters = continuous_dist.mean if deterministic else continuous_dist.sample()
        active_mask = self._active_mask_tensor(action_ids)
        discrete_log_prob = discrete_dist.log_prob(action_ids)
        continuous_log_prob = continuous_dist.masked_log_prob(parameters, active_mask)
        total_log_prob = discrete_log_prob + continuous_log_prob
        value = self.value_net(obs_tensor).squeeze(-1)
        parameter_vector = parameters.squeeze(0).detach().cpu().numpy().astype(np.float32)
        proxy_scores_np: Optional[np.ndarray] = None
        proxy_prefix_np: Optional[np.ndarray] = None
        if policy_context.get("proxy_scores") is not None:
            proxy_scores_np = policy_context["proxy_scores"].squeeze(0).detach().cpu().numpy().astype(np.float32)
            proxy_prefix_np = policy_context["proxy_prefix_lengths"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        return ActionSelection(
            macro_action=MacroAction(primitive_id=int(action_ids.item()), parameters=parameter_vector),
            log_prob=float(total_log_prob.item()),
            value=float(value.item()),
            discrete_log_prob=float(discrete_log_prob.item()),
            continuous_log_prob=float(continuous_log_prob.item()),
            proxy_scores=proxy_scores_np,
            proxy_prefix_lengths=proxy_prefix_np,
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
        teacher_action_probs = torch.as_tensor(batch.teacher_action_probs, dtype=torch.float32, device=self.device)
        teacher_parameter_targets = torch.as_tensor(batch.teacher_parameter_targets, dtype=torch.float32, device=self.device)
        teacher_weights = torch.as_tensor(batch.teacher_weights, dtype=torch.float32, device=self.device)
        cached_proxy_scores: Optional[torch.Tensor] = None
        cached_proxy_prefix: Optional[torch.Tensor] = None
        if self._soft_mask_enabled() and batch.proxy_scores.size > 0 and batch.proxy_scores.shape[0] == num_samples:
            cached_proxy_scores = torch.as_tensor(batch.proxy_scores, dtype=torch.float32, device=self.device)
            cached_proxy_prefix = torch.as_tensor(batch.proxy_prefix_lengths, dtype=torch.float32, device=self.device)
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
        last_teacher_discrete_loss = 0.0
        last_teacher_parameter_loss = 0.0
        last_teacher_weight = 0.0

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
                teacher_prob_mb = teacher_action_probs[mb]
                teacher_target_mb = teacher_parameter_targets[mb]
                teacher_weight_mb = teacher_weights[mb]
                eval_bundle = self._evaluate_action_bundle(
                    observations=obs_mb,
                    action_ids=action_mb,
                    parameters=param_mb,
                    teacher_action_probs=teacher_prob_mb,
                    teacher_weights=teacher_weight_mb,
                    cached_proxy_scores=cached_proxy_scores[mb] if cached_proxy_scores is not None else None,
                    cached_proxy_prefix_lengths=cached_proxy_prefix[mb] if cached_proxy_prefix is not None else None,
                )
                total_log_prob = eval_bundle["total_log_prob"]
                entropy_d = eval_bundle["discrete_entropy"]
                entropy_c = eval_bundle["continuous_entropy"]
                safety_loss = eval_bundle["safety_loss"]
                sampled_safety_score = eval_bundle["sampled_safety_score"]
                teacher_discrete_loss, teacher_parameter_loss, teacher_weight_mean = self._teacher_guidance_losses(
                    policy_context=eval_bundle["policy_context"],
                    continuous_dist=eval_bundle["continuous_dist"],
                    action_ids=action_mb,
                    teacher_action_probs=teacher_prob_mb,
                    teacher_parameter_targets=teacher_target_mb,
                    teacher_weights=teacher_weight_mb,
                )
                ratio = torch.exp(total_log_prob - old_log_prob_mb)
                surrogate_1 = ratio * adv_mb
                surrogate_2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                actor_loss -= self.config.entropy_coef_discrete * entropy_d.mean()
                actor_loss -= self.config.entropy_coef_continuous * entropy_c.mean()
                actor_loss += float(self.config.safety_loss_coef) * safety_loss.mean()
                actor_loss += float(self._TEACHER_DISCRETE_LOSS_COEF) * teacher_discrete_loss
                actor_loss += float(self._TEACHER_PARAMETER_LOSS_COEF) * teacher_parameter_loss
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
                last_teacher_discrete_loss = float(teacher_discrete_loss.item())
                last_teacher_parameter_loss = float(teacher_parameter_loss.item())
                last_teacher_weight = float(teacher_weight_mean.item())

        return {
            "actor_loss": last_actor_loss,
            "value_loss": last_value_loss,
            "entropy_discrete": last_entropy_d,
            "entropy_continuous": last_entropy_c,
            "safety_loss": last_safety_loss,
            "sampled_safety_score": last_sampled_safety_score,
            "teacher_discrete_kl": last_teacher_discrete_loss,
            "teacher_parameter_loss": last_teacher_parameter_loss,
            "teacher_weight_mean": last_teacher_weight,
            "advantage_mean": float(advantages.mean().item()),
            "return_mean": float(returns.mean().item()),
        }

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        action_ids: torch.Tensor,
        parameters: torch.Tensor,
        teacher_action_probs: Optional[torch.Tensor] = None,
        teacher_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bundle = self._evaluate_action_bundle(
            observations=observations,
            action_ids=action_ids,
            parameters=parameters,
            teacher_action_probs=teacher_action_probs,
            teacher_weights=teacher_weights,
        )
        return (
            bundle["total_log_prob"],
            bundle["discrete_entropy"],
            bundle["continuous_entropy"],
            bundle["safety_loss"],
            bundle["sampled_safety_score"],
        )

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

    def _continuous_dist(self, observations: torch.Tensor, action_ids: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> AffineBeta:
        raw = self.policy.continuous_raw(observations, action_ids, action_mask=action_mask)
        alpha_raw, beta_raw = torch.chunk(raw, 2, dim=-1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0
        return AffineBeta(alpha=alpha, beta=beta, low=self._low, high=self._high)

    def _active_mask_tensor(self, action_ids: torch.Tensor) -> torch.Tensor:
        action_list = action_ids.detach().cpu().tolist()
        masks = [self.primitive_library.active_mask(int(action_id)).astype(np.float32) for action_id in action_list]
        return torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32, device=self.device)

    def _evaluate_action_bundle(
        self,
        observations: torch.Tensor,
        action_ids: torch.Tensor,
        parameters: torch.Tensor,
        teacher_action_probs: Optional[torch.Tensor] = None,
        teacher_weights: Optional[torch.Tensor] = None,
        cached_proxy_scores: Optional[torch.Tensor] = None,
        cached_proxy_prefix_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        policy_context = self._masked_policy_context(
            observations,
            cached_proxy_scores=cached_proxy_scores,
            cached_proxy_prefix_lengths=cached_proxy_prefix_lengths,
        )
        selection_probs = self._selection_probs_from_context(
            policy_context,
            teacher_action_probs=teacher_action_probs,
            teacher_weights=teacher_weights,
        )
        discrete_dist = Categorical(probs=selection_probs)
        continuous_dist = self._continuous_dist(observations, action_ids, action_mask=policy_context["network_action_mask"])
        active_mask = self._active_mask_tensor(action_ids)
        discrete_log_prob = discrete_dist.log_prob(action_ids)
        continuous_log_prob = continuous_dist.masked_log_prob(parameters, active_mask)
        safety_loss, sampled_safety_score = self._continuous_safety_terms(observations, policy_context, action_ids, parameters)
        return {
            "policy_context": policy_context,
            "continuous_dist": continuous_dist,
            "total_log_prob": discrete_log_prob + continuous_log_prob,
            "discrete_entropy": discrete_dist.entropy(),
            "continuous_entropy": continuous_dist.masked_entropy(active_mask),
            "safety_loss": safety_loss,
            "sampled_safety_score": sampled_safety_score,
        }

    def _selection_probs_from_context(
        self,
        policy_context: Dict[str, torch.Tensor],
        teacher_action_probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        teacher_weights: Optional[Union[torch.Tensor, np.ndarray, float]] = None,
    ) -> torch.Tensor:
        if teacher_action_probs is None or teacher_weights is None:
            return policy_context["masked_probs"]
        return self._blend_teacher_action_probs(
            policy_probs=policy_context["masked_probs"],
            selection_action_mask=policy_context["selection_action_mask"],
            teacher_action_probs=teacher_action_probs,
            teacher_weights=teacher_weights,
        )

    def _blend_teacher_action_probs(
        self,
        policy_probs: torch.Tensor,
        selection_action_mask: torch.Tensor,
        teacher_action_probs: Union[torch.Tensor, np.ndarray],
        teacher_weights: Union[torch.Tensor, np.ndarray, float],
    ) -> torch.Tensor:
        teacher_probs = torch.as_tensor(teacher_action_probs, dtype=policy_probs.dtype, device=self.device)
        if teacher_probs.dim() == 1:
            teacher_probs = teacher_probs.unsqueeze(0)
        teacher_probs = torch.clamp(teacher_probs, min=0.0)
        teacher_probs = teacher_probs * torch.clamp(selection_action_mask.to(dtype=policy_probs.dtype), min=0.0, max=1.0)
        teacher_sum = teacher_probs.sum(dim=-1, keepdim=True)
        normalized_teacher = teacher_probs / torch.clamp(teacher_sum, min=float(self.config.soft_mask_eps))
        normalized_teacher = torch.where(teacher_sum > float(self.config.soft_mask_eps), normalized_teacher, policy_probs)

        weight = torch.as_tensor(teacher_weights, dtype=policy_probs.dtype, device=self.device)
        if weight.dim() == 0:
            weight = weight.unsqueeze(0)
        if weight.dim() == 1:
            weight = weight.unsqueeze(-1)
        if weight.shape[0] == 1 and policy_probs.shape[0] != 1:
            weight = weight.expand(policy_probs.shape[0], 1)
        weight = torch.clamp(weight, min=0.0, max=1.0)

        mixed = (1.0 - weight) * policy_probs + weight * normalized_teacher
        return mixed / torch.clamp(mixed.sum(dim=-1, keepdim=True), min=float(self.config.soft_mask_eps))

    def _teacher_guidance_losses(
        self,
        policy_context: Dict[str, torch.Tensor],
        continuous_dist: AffineBeta,
        action_ids: torch.Tensor,
        teacher_action_probs: torch.Tensor,
        teacher_parameter_targets: torch.Tensor,
        teacher_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros((), dtype=torch.float32, device=self.device)
        if teacher_action_probs.numel() == 0 or teacher_weights.numel() == 0:
            return zeros, zeros, zeros

        weights = torch.clamp(teacher_weights.reshape(-1), min=0.0, max=1.0)
        if not torch.any(weights > float(self.config.soft_mask_eps)):
            return zeros, zeros, zeros

        teacher_probs = self._normalize_teacher_action_probs(
            teacher_action_probs=teacher_action_probs,
            selection_action_mask=policy_context["selection_action_mask"],
        )
        student_probs = torch.clamp(policy_context["masked_probs"], min=float(self.config.soft_mask_eps), max=1.0)
        teacher_log_probs = torch.log(torch.clamp(teacher_probs, min=float(self.config.soft_mask_eps), max=1.0))
        student_log_probs = torch.log(student_probs)
        discrete_kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)

        active_mask = self._active_mask_tensor(action_ids)
        active_count = torch.clamp(active_mask.sum(dim=-1), min=1.0)
        parameter_error = torch.square((continuous_dist.mean - teacher_parameter_targets) * active_mask).sum(dim=-1) / active_count
        weight_sum = torch.clamp(weights.sum(), min=float(self.config.soft_mask_eps))
        return (
            torch.sum(discrete_kl * weights) / weight_sum,
            torch.sum(parameter_error * weights) / weight_sum,
            torch.mean(weights),
        )

    def _normalize_teacher_action_probs(self, teacher_action_probs: torch.Tensor, selection_action_mask: torch.Tensor) -> torch.Tensor:
        teacher_probs = torch.clamp(teacher_action_probs, min=0.0)
        teacher_probs = teacher_probs * torch.clamp(selection_action_mask.to(dtype=teacher_probs.dtype), min=0.0, max=1.0)
        teacher_sum = teacher_probs.sum(dim=-1, keepdim=True)
        normalized = teacher_probs / torch.clamp(teacher_sum, min=float(self.config.soft_mask_eps))
        fallback = policy_context_probs = selection_action_mask.to(dtype=teacher_probs.dtype)
        fallback = fallback / torch.clamp(fallback.sum(dim=-1, keepdim=True), min=float(self.config.soft_mask_eps))
        return torch.where(teacher_sum > float(self.config.soft_mask_eps), normalized, fallback)

    def _masked_policy_context(
        self,
        observations: torch.Tensor,
        cached_proxy_scores: Optional[torch.Tensor] = None,
        cached_proxy_prefix_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        state_gate_valid_mask = self._state_gate_valid_mask(observations)
        proxy_scores_tensor = None
        proxy_prefix_lengths_tensor = None
        proxy_action_mask = torch.ones((observations.shape[0], self.primitive_library.action_dim), dtype=torch.float32, device=self.device)
        proxy_action_has_safe_prefix = torch.ones_like(state_gate_valid_mask)
        if self._soft_mask_enabled():
            if cached_proxy_scores is not None and cached_proxy_prefix_lengths is not None:
                proxy_scores_tensor = cached_proxy_scores.to(device=self.device, dtype=torch.float32)
                proxy_prefix_lengths_tensor = cached_proxy_prefix_lengths.to(device=self.device, dtype=torch.float32)
            else:
                self._ensure_proxy_tensors()
                proxy_scores, proxy_prefix_lengths, _articulation_bins = self._query_proxy_scores(observations)
                proxy_scores_tensor = torch.as_tensor(proxy_scores, dtype=torch.float32, device=self.device)
                proxy_prefix_lengths_tensor = torch.as_tensor(proxy_prefix_lengths, dtype=torch.float32, device=self.device)
            proxy_action_mask, proxy_action_has_safe_prefix = self._aggregate_proxy_action_mask(
                proxy_scores=proxy_scores_tensor,
                proxy_prefix_lengths=proxy_prefix_lengths_tensor,
            )

        network_action_mask = proxy_action_mask * state_gate_valid_mask.to(dtype=proxy_action_mask.dtype)
        selection_action_mask = self._selection_action_mask(
            proxy_action_mask=proxy_action_mask,
            proxy_action_has_safe_prefix=proxy_action_has_safe_prefix,
            state_gate_valid_mask=state_gate_valid_mask,
        )
        raw_logits = self.policy.discrete_logits(observations, action_mask=network_action_mask)
        raw_probs = torch.softmax(raw_logits, dim=-1)
        masked_probs = self._reweighted_action_probs(
            raw_logits=raw_logits,
            action_mask=selection_action_mask,
            state_gate_valid_mask=state_gate_valid_mask,
        )
        context = {
            "raw_logits": raw_logits,
            "raw_probs": raw_probs,
            "masked_logits": self._probs_to_logits(masked_probs),
            "masked_probs": masked_probs,
            "semantic_scores": proxy_action_mask,
            "state_gate_valid_mask": state_gate_valid_mask,
            "network_action_mask": network_action_mask,
            "selection_action_mask": selection_action_mask,
            "proxy_action_mask": proxy_action_mask,
            "proxy_action_has_safe_prefix": proxy_action_has_safe_prefix,
            "proxy_scores": None,
            "proxy_prefix_lengths": None,
            "all_action_means": None,
            "all_action_stds": None,
        }
        if self._soft_mask_enabled():
            context.update(
                {
                    "proxy_scores": proxy_scores_tensor,
                    "proxy_prefix_lengths": proxy_prefix_lengths_tensor,
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

    def _aggregate_proxy_action_mask(self, proxy_scores: torch.Tensor, proxy_prefix_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_scores = torch.max(torch.clamp(proxy_scores, 0.0, 1.0), dim=-1).values
        has_safe_prefix = torch.max(proxy_prefix_lengths, dim=-1).values > 0.0
        action_scores = torch.where(has_safe_prefix, action_scores, torch.zeros_like(action_scores))
        return action_scores, has_safe_prefix

    def _selection_action_mask(
        self,
        proxy_action_mask: torch.Tensor,
        proxy_action_has_safe_prefix: torch.Tensor,
        state_gate_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        gate_weights = state_gate_valid_mask.to(dtype=proxy_action_mask.dtype)
        if not self._soft_mask_enabled():
            return gate_weights
        floor = min(1.0, max(float(self.config.soft_mask_floor), float(self.config.soft_mask_eps)))
        clipped = torch.clamp(proxy_action_mask, 0.0, 1.0)
        softened = torch.where(
            proxy_action_has_safe_prefix,
            torch.clamp(clipped, min=floor, max=1.0),
            torch.zeros_like(clipped),
        )
        return softened * gate_weights

    def _reweighted_action_probs(
        self,
        raw_logits: torch.Tensor,
        action_mask: torch.Tensor,
        state_gate_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        raw_probs = torch.softmax(raw_logits, dim=-1)
        weights = torch.clamp(action_mask.to(dtype=raw_probs.dtype), min=0.0, max=1.0)
        scale = max(float(self.config.soft_mask_logit_scale), 0.0)
        if scale != 1.0:
            positive = weights > 0.0
            weights = torch.where(positive, torch.pow(weights, scale), weights)
        weighted_probs = raw_probs * weights
        weight_sum = weighted_probs.sum(dim=-1, keepdim=True)
        fallback_probs = self._fallback_action_probs(raw_logits, state_gate_valid_mask)
        normalized = weighted_probs / torch.clamp(weight_sum, min=float(self.config.soft_mask_eps))
        return torch.where(weight_sum > float(self.config.soft_mask_eps), normalized, fallback_probs)

    def _fallback_action_probs(self, raw_logits: torch.Tensor, state_gate_valid_mask: torch.Tensor) -> torch.Tensor:
        fallback_logits = raw_logits + self._fallback_action_bias.unsqueeze(0)
        fallback_probs = torch.softmax(fallback_logits, dim=-1)
        fallback_weights = state_gate_valid_mask.to(dtype=fallback_probs.dtype)
        valid_count = fallback_weights.sum(dim=-1, keepdim=True)
        fallback_weights = torch.where(valid_count > 0.0, fallback_weights, torch.ones_like(fallback_weights))
        weighted = fallback_probs * fallback_weights
        return weighted / torch.clamp(weighted.sum(dim=-1, keepdim=True), min=float(self.config.soft_mask_eps))

    def _probs_to_logits(self, probs: torch.Tensor) -> torch.Tensor:
        zero_logits = torch.full_like(probs, -1e9)
        min_positive = torch.finfo(probs.dtype).tiny
        safe_logits = torch.log(torch.clamp(probs, min=min_positive))
        return torch.where(probs > 0.0, safe_logits, zero_logits)

    def _action_mask_screening_metrics(self, policy_context: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reference_invalid = ~policy_context["state_gate_valid_mask"]
        if policy_context["proxy_action_has_safe_prefix"] is not None:
            reference_invalid = torch.logical_or(reference_invalid, ~policy_context["proxy_action_has_safe_prefix"])
        predicted_invalid = policy_context["selection_action_mask"] <= float(self.config.soft_mask_eps)
        reference_invalid_f = reference_invalid.to(dtype=policy_context["raw_probs"].dtype)
        return {
            "reference_invalid": reference_invalid,
            "predicted_invalid": predicted_invalid,
            "false_negative": torch.logical_and(reference_invalid, ~predicted_invalid),
            "false_positive": torch.logical_and(~reference_invalid, predicted_invalid),
            "invalid_prob_before": (policy_context["raw_probs"] * reference_invalid_f).sum(dim=-1),
            "invalid_prob_after": (policy_context["masked_probs"] * reference_invalid_f).sum(dim=-1),
        }

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

    def _query_proxy_scores(self, observations: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sidecar = self.primitive_library.proxy_sidecar
        if sidecar is None:
            raise RuntimeError("proxy safety sidecar is required when soft mask is enabled")
        observation_np = observations.detach().cpu().numpy()
        feature_offset = self._base_feature_offset(observations)
        if observation_np.shape[1] < feature_offset + self._OBSERVED_FEATURE_DIM:
            raise ValueError("observation does not contain enough post-lidar features to reconstruct articulation angle")
        articulation_angles = np.arctan2(
            observation_np[:, feature_offset + 6],
            observation_np[:, feature_offset + 5],
        ).astype(np.float32)
        lidar = observation_np[:, :feature_offset].astype(np.float32)
        scores, prefix_lengths, bins = sidecar.compute_proxy_scores_batch(
            lidar_observations=lidar,
            articulation_angles=articulation_angles,
            gamma=float(self.config.soft_mask_gamma),
            eps=float(self.config.soft_mask_eps),
        )
        return (
            scores.astype(np.float32),
            prefix_lengths.astype(np.float32),
            bins.astype(np.int64),
        )

    def _all_action_distribution_moments(self, observations: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        means = []
        stds = []
        batch_size = observations.shape[0]
        for action_id in range(self.primitive_library.action_dim):
            action_tensor = torch.full((batch_size,), int(action_id), dtype=torch.int64, device=self.device)
            dist = self._continuous_dist(observations, action_tensor, action_mask=action_mask)
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

    def _continuous_safety_terms(
        self,
        observations: torch.Tensor,
        policy_context: Dict[str, torch.Tensor],
        action_ids: torch.Tensor,
        parameters: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = action_ids.shape[0]
        if not self._soft_mask_enabled() or policy_context["proxy_scores"] is None:
            zeros = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
            return zeros, zeros

        all_action_means = policy_context["all_action_means"]
        all_action_stds = policy_context["all_action_stds"]
        if all_action_means is None or all_action_stds is None:
            all_action_means, all_action_stds = self._all_action_distribution_moments(
                observations,
                action_mask=policy_context["network_action_mask"],
            )
            policy_context["all_action_means"] = all_action_means
            policy_context["all_action_stds"] = all_action_stds

        batch_indices = torch.arange(batch_size, device=self.device)
        selected_proxy_scores = policy_context["proxy_scores"][batch_indices, action_ids]
        selected_proxy_centers = self._proxy_parameter_centers[action_ids]
        selected_proxy_scales = self._proxy_parameter_scales[action_ids]
        selected_proxy_valid = self._proxy_valid_mask[action_ids]
        selected_active_mask = self._action_active_mask[action_ids]
        selected_means = all_action_means[batch_indices, action_ids]
        selected_stds = all_action_stds[batch_indices, action_ids]

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
