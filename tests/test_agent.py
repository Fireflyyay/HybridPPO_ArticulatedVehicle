import numpy as np
import torch

from common.config import HybridPPOConfig
from common.types import MacroTransition
from model import HybridPolicyNetwork
from model.agent import HybridPPOAgent
from primitives import ProxySafetySidecar, SemanticPrimitive, build_default_primitive_library


def _make_observation(
    lidar_values,
    goal_distance: float,
    relative_angle: float = 0.0,
    relative_heading: float = 0.0,
    articulation: float = 0.0,
    speed: float = 0.0,
    articulation_rate: float = 0.0,
    guidance_features=None,
) -> np.ndarray:
    lidar = np.asarray(lidar_values, dtype=np.float32).reshape(-1)
    features = np.array(
        [
            goal_distance,
            np.cos(relative_angle),
            np.sin(relative_angle),
            np.cos(relative_heading),
            np.sin(relative_heading),
            np.cos(articulation),
            np.sin(articulation),
            speed,
            articulation_rate,
        ],
        dtype=np.float32,
    )
    guidance = np.zeros((0,), dtype=np.float32)
    if guidance_features is not None:
        guidance = np.asarray(guidance_features, dtype=np.float32).reshape(-1)
    return np.concatenate([lidar, features, guidance], axis=0)


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
    assert "hard_invalid_mass" in metrics
    assert "invalid_loss" in metrics


def test_hybrid_policy_network_responds_to_action_mask_input():
    torch.manual_seed(7)
    policy = HybridPolicyNetwork(
        observation_dim=6,
        action_dim=4,
        parameter_dim=3,
        hidden_dim=16,
        action_embedding_dim=8,
    )
    observation = torch.zeros((1, 6), dtype=torch.float32)
    action_ids = torch.tensor([2], dtype=torch.int64)
    mask_a = torch.zeros((1, 4), dtype=torch.float32)
    mask_b = torch.tensor([[1.0, 0.3, 0.8, 0.5]], dtype=torch.float32)

    logits_a = policy.discrete_logits(observation, action_mask=mask_a)
    logits_b = policy.discrete_logits(observation, action_mask=mask_b)
    continuous_a = policy.continuous_raw(observation, action_ids, action_mask=mask_a)
    continuous_b = policy.continuous_raw(observation, action_ids, action_mask=mask_b)

    assert not torch.allclose(logits_a, logits_b)
    assert not torch.allclose(continuous_a, continuous_b)


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


def test_masked_policy_context_uses_distribution_aware_semantic_scores_for_selection():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_enabled=True,
    )
    agent = HybridPPOAgent(config, library)

    def _fake_query_proxy_scores(_observations):
        proxy_scores = np.zeros((1, library.action_dim, 2), dtype=np.float32)
        proxy_prefix_lengths = np.ones((1, library.action_dim, 2), dtype=np.float32)
        proxy_scores[0, 0] = np.array([0.0, 1.0], dtype=np.float32)
        proxy_scores[0, 1] = np.array([0.45, 0.45], dtype=np.float32)
        return proxy_scores, proxy_prefix_lengths, np.zeros((1,), dtype=np.int64)

    def _fake_distribution_moments(observations, action_mask=None):
        del observations, action_mask
        means = torch.zeros((1, library.action_dim, library.parameter_dim), dtype=torch.float32, device=agent.device)
        stds = torch.full_like(means, 0.05)
        means[0, 0, :] = -0.4
        means[0, 1, :] = 0.0
        return means, stds

    agent._query_proxy_scores = _fake_query_proxy_scores
    agent._all_action_distribution_moments = _fake_distribution_moments

    context = agent._masked_policy_context(agent._obs_tensor(np.zeros((13,), dtype=np.float32)))

    assert float(context["proxy_action_mask"][0, 0].item()) > float(context["proxy_action_mask"][0, 1].item())
    assert float(context["semantic_scores"][0, 0].item()) < float(context["semantic_scores"][0, 1].item())
    assert float(context["selection_action_mask"][0, 0].item()) < float(context["selection_action_mask"][0, 1].item())


def test_reweighted_action_probs_match_manual_soft_mask_formula():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=10,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_logit_scale=1.0,
    )
    agent = HybridPPOAgent(config, library)
    raw_logits = torch.tensor([[2.0, 1.0, -1.0, 0.5, -0.2, -0.7, -0.9, -1.1]], dtype=torch.float32, device=agent.device)
    action_mask = torch.tensor([[1.0, 0.5, 0.0, 0.25, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32, device=agent.device)
    state_gate = torch.ones_like(raw_logits, dtype=torch.bool, device=agent.device)

    probs = agent._reweighted_action_probs(raw_logits, action_mask, state_gate)
    manual = torch.softmax(raw_logits, dim=-1) * action_mask
    manual = manual / manual.sum(dim=-1, keepdim=True)

    assert torch.allclose(probs, manual)
    assert float(probs[0, 0].item()) > float(probs[0, 1].item())
    assert float(probs[0, 1].item()) > float(probs[0, 2].item())


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


def test_hybrid_agent_reports_fallback_diagnostics_when_mask_degenerates():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_enabled=True,
    )
    agent = HybridPPOAgent(config, library)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()

    observation = np.zeros((13,), dtype=np.float32)
    selection = agent.act(observation, deterministic=True, return_diagnostics=True)

    assert selection.diagnostics is not None
    assert selection.diagnostics["fallback_triggered"] is True
    assert selection.diagnostics["all_invalid_fallback"] is True
    assert selection.diagnostics["selected_action_hard_valid"] is False
    assert selection.diagnostics["selected_action_execution_valid"] is True
    assert selection.diagnostics["selection_mask_positive_count"] > 0


def test_masked_policy_context_fail_open_keeps_execution_distribution_non_empty():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_enabled=True,
    )
    agent = HybridPPOAgent(config, library)

    observation = np.zeros((13,), dtype=np.float32)
    context = agent._masked_policy_context(agent._obs_tensor(observation))

    assert not bool(torch.any(context["hard_valid_mask"]).item())
    assert bool(torch.any(context["execution_valid_mask"]).item())
    assert bool(context["all_invalid_fallback"][0].item()) is True
    assert float(context["masked_probs"].sum().item()) == 1.0


def test_invalid_mass_loss_reports_raw_invalid_probability_mass():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        mini_batch_size=1,
        update_epochs=1,
        soft_mask_enabled=True,
        invalid_loss_coef=0.05,
    )
    agent = HybridPPOAgent(config, library)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()

    observation = np.zeros((13,), dtype=np.float32)
    observation[:4] = 0.5
    observation[9] = 1.0
    selection = agent.act(observation, deterministic=True)
    agent.store_transition(
        MacroTransition(
            observation=observation,
            action_id=selection.macro_action.primitive_id,
            parameters=selection.macro_action.parameters,
            reward=1.0,
            tau=1,
            next_observation=observation.copy(),
            done=False,
            log_prob=selection.log_prob,
            value=selection.value,
            proxy_scores=selection.proxy_scores,
            proxy_prefix_lengths=selection.proxy_prefix_lengths,
            hard_valid_mask=selection.hard_valid_mask,
            execution_valid_mask=selection.execution_valid_mask,
            soft_score=selection.soft_score,
        )
    )

    metrics = agent.update()

    assert float(metrics["hard_invalid_mass"]) > 0.0
    assert float(metrics["invalid_loss"]) > 0.0
    assert float(metrics["selected_action_hard_valid"]) == 1.0


def test_action_mask_screening_metrics_have_no_leak_or_false_block_on_proxy_invalid_actions():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_enabled=True,
    )
    agent = HybridPPOAgent(config, library)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()

    observation = np.zeros((13,), dtype=np.float32)
    observation[:4] = 0.5
    observation[9] = 1.0
    context = agent._masked_policy_context(agent._obs_tensor(observation))
    metrics = agent._action_mask_screening_metrics(context)

    assert not torch.any(metrics["false_negative"])
    assert not torch.any(metrics["false_positive"])
    assert float(metrics["invalid_prob_before"].item()) > 0.0
    assert float(metrics["invalid_prob_after"].item()) == 0.0


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
    discrete_dist = torch.distributions.Categorical(probs=context["masked_probs"])
    continuous_dist = agent._continuous_dist(obs_tensor, action_tensor, action_mask=context["network_action_mask"])
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


def test_state_gate_blocks_articulation_recover_when_articulation_is_small():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
    )
    agent = HybridPPOAgent(config, library)
    recover_id = int(library.spec(SemanticPrimitive.ARTICULATION_RECOVER).primitive_id)
    forward_id = int(library.spec(SemanticPrimitive.FORWARD_LEFT).primitive_id)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()
        agent.policy.discrete_head.bias[recover_id] = 5.0
        agent.policy.discrete_head.bias[forward_id] = 1.0

    observation = _make_observation([1.0, 1.0, 1.0, 1.0], goal_distance=0.5, articulation=0.1)
    selection = agent.act(observation, deterministic=True)
    context = agent._masked_policy_context(agent._obs_tensor(observation))

    assert selection.macro_action.primitive_id == forward_id
    assert float(context["masked_logits"][0, recover_id].item()) < -1e8


def test_state_gate_allows_articulation_recover_when_articulation_is_large():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
    )
    agent = HybridPPOAgent(config, library)
    recover_id = int(library.spec(SemanticPrimitive.ARTICULATION_RECOVER).primitive_id)
    forward_id = int(library.spec(SemanticPrimitive.FORWARD_LEFT).primitive_id)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()
        agent.policy.discrete_head.bias[recover_id] = 5.0
        agent.policy.discrete_head.bias[forward_id] = 1.0

    observation = _make_observation([1.0, 1.0, 1.0, 1.0], goal_distance=0.5, articulation=0.5)
    selection = agent.act(observation, deterministic=True)

    assert selection.macro_action.primitive_id == recover_id


def test_state_gate_reads_base_features_when_guidance_is_appended():
    base_library = build_default_primitive_library()
    library = build_default_primitive_library(proxy_sidecar=_make_proxy_sidecar(base_library))
    config = HybridPPOConfig(
        observation_dim=17,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        soft_mask_enabled=True,
    )
    agent = HybridPPOAgent(config, library)
    recover_id = int(library.spec(SemanticPrimitive.ARTICULATION_RECOVER).primitive_id)
    forward_id = int(library.spec(SemanticPrimitive.FORWARD_LEFT).primitive_id)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()
        agent.policy.discrete_head.bias[recover_id] = 5.0
        agent.policy.discrete_head.bias[forward_id] = 1.0

    observation = _make_observation(
        [1.0, 1.0, 1.0, 1.0],
        goal_distance=0.5,
        articulation=0.1,
        guidance_features=[0.0, 1.0, 0.0, 0.95],
    )
    selection = agent.act(observation, deterministic=True)

    assert selection.macro_action.primitive_id == forward_id


def test_state_gate_blocks_stop_check_far_from_goal():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
    )
    agent = HybridPPOAgent(config, library)
    stop_id = int(library.spec(SemanticPrimitive.STOP_CHECK).primitive_id)
    forward_id = int(library.spec(SemanticPrimitive.FORWARD_LEFT).primitive_id)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()
        agent.policy.discrete_head.bias[stop_id] = 5.0
        agent.policy.discrete_head.bias[forward_id] = 1.0

    observation = _make_observation([0.7, 0.8, 0.9, 1.0], goal_distance=0.5, relative_heading=0.0)
    selection = agent.act(observation, deterministic=True)
    context = agent._masked_policy_context(agent._obs_tensor(observation))

    assert selection.macro_action.primitive_id == forward_id
    assert float(context["masked_logits"][0, stop_id].item()) < -1e8


def test_state_gate_allows_stop_check_near_goal_in_tight_space():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=13,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
    )
    agent = HybridPPOAgent(config, library)
    stop_id = int(library.spec(SemanticPrimitive.STOP_CHECK).primitive_id)
    forward_id = int(library.spec(SemanticPrimitive.FORWARD_LEFT).primitive_id)
    with torch.no_grad():
        agent.policy.discrete_head.weight.zero_()
        agent.policy.discrete_head.bias.zero_()
        agent.policy.discrete_head.bias[stop_id] = 5.0
        agent.policy.discrete_head.bias[forward_id] = 1.0

    observation = _make_observation([0.03, 0.04, 0.05, 0.06], goal_distance=0.04, relative_heading=0.05)
    selection = agent.act(observation, deterministic=True)

    assert selection.macro_action.primitive_id == stop_id
