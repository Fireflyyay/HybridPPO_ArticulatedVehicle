import numpy as np
import torch

from common.config import HybridPPOConfig
from common.types import MacroTransition
from model.agent import HybridPPOAgent
from primitives import ProxySafetySidecar, build_default_primitive_library


def _make_proxy_sidecar(library, lidar_rays: int = 4) -> ProxySafetySidecar:
    action_dim = library.action_dim
    parameter_dim = library.parameter_dim
    required_clearance = np.full((1, action_dim, 2, 3, lidar_rays), 9.0, dtype=np.float32)
    required_clearance[0, 0, :, :, :] = 1.0
    parameter_centers = np.zeros((action_dim, 2, parameter_dim), dtype=np.float32)
    parameter_centers[:, 0, :] = -0.4
    parameter_centers[:, 1, :] = 0.4
    parameter_scales = np.full_like(parameter_centers, 0.25, dtype=np.float32)
    nominal_horizons = np.full((action_dim, 2), 3, dtype=np.int64)
    proxy_valid_mask = np.ones((action_dim, 2), dtype=np.bool_)
    return ProxySafetySidecar(
        required_clearance=required_clearance,
        articulation_bin_centers=np.array([0.0], dtype=np.float32),
        parameter_centers=parameter_centers,
        parameter_scales=parameter_scales,
        nominal_horizons=nominal_horizons,
        proxy_valid_mask=proxy_valid_mask,
        lidar_range=10.0,
        library_signature=library.signature,
    )


def test_hybrid_agent_action_and_update_smoke():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=10,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        mini_batch_size=2,
        update_epochs=2,
    )
    agent = HybridPPOAgent(config, library)

    selection = agent.act(np.zeros((10,), dtype=np.float32))
    assert 0 <= selection.macro_action.primitive_id < library.action_dim
    assert selection.macro_action.parameters.shape == (library.parameter_dim,)
    assert np.all(selection.macro_action.parameters >= library.low - 1e-6)
    assert np.all(selection.macro_action.parameters <= library.high + 1e-6)

    for index in range(4):
        obs = np.linspace(0.0, 1.0, 10, dtype=np.float32) + index * 0.05
        sample = agent.act(obs)
        agent.store_transition(
            MacroTransition(
                observation=obs,
                action_id=sample.macro_action.primitive_id,
                parameters=sample.macro_action.parameters,
                reward=1.0 - 0.1 * index,
                tau=1 + (index % 2),
                next_observation=obs + 0.1,
                done=bool(index == 3),
                log_prob=sample.log_prob,
                value=sample.value,
            )
        )

    metrics = agent.update()
    assert "actor_loss" in metrics
    assert "value_loss" in metrics


def test_distribution_aware_semantic_scores_do_not_reduce_by_proxy_max():
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(build_default_primitive_library()))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_enabled=True,
    )
    agent = HybridPPOAgent(config, library)
    agent._ensure_proxy_tensors()

    all_action_means = torch.zeros((1, library.action_dim, library.parameter_dim), dtype=torch.float32)
    all_action_stds = torch.full_like(all_action_means, 0.05)
    proxy_scores = torch.zeros((1, library.action_dim, 2), dtype=torch.float32)
    proxy_scores[0, 0] = torch.tensor([0.0, 1.0], dtype=torch.float32)
    proxy_scores[0, 1] = torch.tensor([0.4, 0.4], dtype=torch.float32)

    all_action_means[0, 0, :] = -0.4
    all_action_means[0, 1, :] = 0.0
    semantic_scores = agent._distribution_aware_semantic_scores(
        all_action_means=all_action_means.to(agent.device),
        all_action_stds=all_action_stds.to(agent.device),
        proxy_scores=proxy_scores.to(agent.device),
    )

    assert float(semantic_scores[0, 0].item()) < float(semantic_scores[0, 1].item())


def test_hybrid_agent_soft_mask_keeps_act_and_evaluate_consistent():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        mini_batch_size=2,
        update_epochs=1,
        soft_mask_enabled=True,
        soft_mask_logit_scale=4.0,
    )
    agent = HybridPPOAgent(config, library)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()

    observation = np.zeros((13,), dtype=np.float32)
    observation[:4] = 0.5
    observation[9] = 1.0
    selection = agent.act(observation, deterministic=True)
    obs_tensor = agent._obs_tensor(observation)
    action_tensor = torch.as_tensor([selection.macro_action.primitive_id], dtype=torch.int64, device=agent.device)
    parameter_tensor = torch.as_tensor(selection.macro_action.parameters.reshape(1, -1), dtype=torch.float32, device=agent.device)
    total_log_prob, _entropy_d, _entropy_c, safety_loss, sampled_safety_score = agent.evaluate_actions(obs_tensor, action_tensor, parameter_tensor)

    assert selection.macro_action.primitive_id == 0
    assert np.isclose(float(total_log_prob.item()), float(selection.log_prob))
    assert float(safety_loss.item()) >= 0.0
    assert 0.0 <= float(sampled_safety_score.item()) <= 1.0


def test_continuous_safety_loss_does_not_modify_ppo_log_prob():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_enabled=True,
        safety_loss_coef=1.0,
    )
    agent = HybridPPOAgent(config, library)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()

    observation = np.zeros((13,), dtype=np.float32)
    observation[:4] = 0.5
    observation[9] = 1.0
    obs_tensor = agent._obs_tensor(observation)
    selection = agent.act(observation, deterministic=True)
    action_tensor = torch.as_tensor([selection.macro_action.primitive_id], dtype=torch.int64, device=agent.device)
    parameter_tensor = torch.as_tensor(selection.macro_action.parameters.reshape(1, -1), dtype=torch.float32, device=agent.device)
    context = agent._masked_policy_context(obs_tensor)
    discrete_dist = torch.distributions.Categorical(logits=context["masked_logits"])
    continuous_dist = agent._continuous_dist(obs_tensor, action_tensor)
    active_mask = agent._active_mask_tensor(action_tensor)
    manual_total = discrete_dist.log_prob(action_tensor) + continuous_dist.masked_log_prob(parameter_tensor, active_mask)

    total_log_prob, _entropy_d, _entropy_c, safety_loss, _sampled_safety_score = agent.evaluate_actions(obs_tensor, action_tensor, parameter_tensor)

    assert np.isclose(float(total_log_prob.item()), float(manual_total.item()))
    assert float(safety_loss.item()) >= 0.0


def test_soft_mask_defaults_keep_low_scored_actions_available():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=10,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
    )
    agent = HybridPPOAgent(config, library)
    raw_logits = torch.zeros((1, library.action_dim), dtype=torch.float32, device=agent.device)
    semantic_scores = torch.zeros_like(raw_logits)
    semantic_scores[0, 0] = 1.0

    masked_logits = agent._apply_semantic_soft_mask(raw_logits, semantic_scores)
    probs = torch.softmax(masked_logits, dim=-1)
    low_action_prob = float(probs[0, 1].item())
    best_action_prob = float(probs[0, 0].item())

    assert low_action_prob >= 0.05
    assert best_action_prob <= 0.5
