from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from common.types import MacroTransition
from env.adapter import UnifiedArticulatedEnvProtocol
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
from training.soft_teacher import CoarseGuidanceSoftTeacher


@dataclass(frozen=True)
class EpisodeSummary:
    total_reward: float
    macro_steps: int
    low_level_steps: int
    success: bool
    collision: bool
    terminated: bool
    truncated: bool
    done_reason: str
    final_goal_distance: float
    last_info: Dict[str, object]
    action_diagnostics: Dict[str, object]
    start_min_lidar_norm: float = 1.0


def _mean_scalar(step_diagnostics, key: str) -> Optional[float]:
    values = [float(item[key]) for item in step_diagnostics if item.get(key) is not None]
    if len(values) == 0:
        return None
    return float(sum(values) / float(len(values)))


def _aggregate_action_diagnostics(step_diagnostics) -> Dict[str, object]:
    if len(step_diagnostics) == 0:
        return {}

    steps = int(len(step_diagnostics))
    primitive_counts: Dict[str, int] = {}
    semantic_counts: Dict[str, int] = {}
    macro_termination_reason_counts: Dict[str, int] = {}
    termination_by_action: Dict[str, int] = {}
    fallback_trigger_count = 0
    all_invalid_fallback_count = 0
    mask_degenerate_count = 0
    selected_matches_raw_argmax_count = 0
    selected_matches_semantic_argmax_count = 0
    selected_action_hard_valid_count = 0
    selected_stop_check_count = 0
    selected_articulation_recover_count = 0
    selected_straight_adjust_count = 0
    macro_articulation_limit_count = 0
    macro_short_articulation_limit_count = 0
    articulation_limit_taus = []

    for item in step_diagnostics:
        action_id = int(item.get("selected_action_id", -1))
        semantic = str(item.get("selected_semantic", "unknown"))
        primitive_key = f"{action_id}:{semantic}"
        primitive_counts[primitive_key] = int(primitive_counts.get(primitive_key, 0) + 1)
        semantic_counts[semantic] = int(semantic_counts.get(semantic, 0) + 1)
        macro_reason = item.get("macro_termination_reason")
        if macro_reason is not None:
            macro_reason = str(macro_reason)
            macro_termination_reason_counts[macro_reason] = int(macro_termination_reason_counts.get(macro_reason, 0) + 1)
            cross_key = f"{action_id}:{macro_reason}"
            termination_by_action[cross_key] = int(termination_by_action.get(cross_key, 0) + 1)
        fallback_trigger_count += int(bool(item.get("fallback_triggered", False)))
        all_invalid_fallback_count += int(bool(item.get("all_invalid_fallback", False)))
        mask_degenerate_count += int(int(item.get("selection_mask_positive_count", 0)) <= 0)
        selected_matches_raw_argmax_count += int(bool(item.get("selected_matches_raw_argmax", False)))
        selected_matches_semantic_argmax_count += int(bool(item.get("selected_matches_semantic_argmax", False)))
        selected_action_hard_valid_count += int(bool(item.get("selected_action_hard_valid", False)))
        selected_stop_check_count += int(semantic == "stop-check")
        selected_articulation_recover_count += int(semantic == "articulation-recover")
        selected_straight_adjust_count += int(semantic == "straight-adjust")
        macro_articulation_limit_count += int(bool(item.get("macro_articulation_limit_hit", False)))
        macro_short_articulation_limit_count += int(bool(item.get("macro_short_articulation_limit_hit", False)))
        if bool(item.get("macro_articulation_limit_hit", False)) and item.get("macro_tau") is not None:
            articulation_limit_taus.append(float(item.get("macro_tau")))

    dominant_semantic = max(semantic_counts.items(), key=lambda pair: pair[1])[0]
    summary: Dict[str, object] = {
        "steps": int(steps),
        "dominant_semantic": str(dominant_semantic),
        "primitive_counts": primitive_counts,
        "semantic_counts": semantic_counts,
        "macro_termination_reason_counts": macro_termination_reason_counts,
        "termination_by_action": termination_by_action,
        "fallback_trigger_rate": float(fallback_trigger_count / float(steps)),
        "all_invalid_fallback_rate": float(all_invalid_fallback_count / float(steps)),
        "mask_degenerate_rate": float(mask_degenerate_count / float(steps)),
        "selected_matches_raw_argmax_rate": float(selected_matches_raw_argmax_count / float(steps)),
        "selected_matches_semantic_argmax_rate": float(selected_matches_semantic_argmax_count / float(steps)),
        "selected_action_hard_valid_rate": float(selected_action_hard_valid_count / float(steps)),
        "selected_stop_check_rate": float(selected_stop_check_count / float(steps)),
        "selected_articulation_recover_rate": float(selected_articulation_recover_count / float(steps)),
        "selected_straight_adjust_rate": float(selected_straight_adjust_count / float(steps)),
        "macro_articulation_limit_rate": float(macro_articulation_limit_count / float(steps)),
        "macro_short_articulation_limit_rate": float(macro_short_articulation_limit_count / float(steps)),
    }
    if articulation_limit_taus:
        summary["macro_articulation_limit_tau_mean"] = float(sum(articulation_limit_taus) / float(len(articulation_limit_taus)))
    for metric_key in (
        "hard_valid_ratio",
        "soft_score_mean",
        "soft_score_min",
        "selected_raw_prob",
        "selected_prob",
        "selected_soft_score",
        "selected_selection_mask",
        "selected_proxy_action_mask",
        "selected_semantic_score",
        "selected_proxy_score_max",
        "selected_proxy_prefix_max",
    ):
        mean_value = _mean_scalar(step_diagnostics, metric_key)
        if mean_value is not None:
            summary[f"{metric_key}_mean"] = float(mean_value)
    return summary


class MacroRolloutDriver:
    def __init__(
        self,
        env: UnifiedArticulatedEnvProtocol,
        macro_env: ParameterizedMacroActionWrapper,
        agent: HybridPPOAgent,
        max_macro_steps: int,
        soft_teacher: Optional[CoarseGuidanceSoftTeacher] = None,
    ) -> None:
        self.env = env
        self.macro_env = macro_env
        self.agent = agent
        self.max_macro_steps = int(max_macro_steps)
        self.soft_teacher = soft_teacher

    def collect_episode(
        self,
        level: str,
        seed: Optional[int] = None,
        deterministic: bool = False,
        store_transition: bool = True,
        reset_options: Optional[Mapping[str, object]] = None,
    ) -> EpisodeSummary:
        options = {"level": str(level)}
        if reset_options is not None:
            options.update(dict(reset_options))
            options["level"] = str(level)
        observation, reset_info = self.env.reset(seed=seed, options=options)
        start_min_lidar_norm = float(reset_info.get("start_min_lidar_norm", 1.0))
        observation = np.asarray(observation, dtype=np.float32)
        total_reward = 0.0
        macro_steps = 0
        low_level_steps = 0
        last_info: Dict[str, object] = self.env.current_info()
        terminated = False
        truncated = False
        previous_action_id: Optional[int] = None
        action_diagnostics = []

        while macro_steps < self.max_macro_steps and not (terminated or truncated):
            if hasattr(self.env, "clear_reference_override"):
                self.env.clear_reference_override()
            teacher_advice = None if self.soft_teacher is None else self.soft_teacher.advise(observation, previous_action_id=previous_action_id)
            if (
                teacher_advice is not None
                and hasattr(self.env, "set_reference_override")
                and teacher_advice.reference_goal_position is not None
                and teacher_advice.reference_goal_heading is not None
            ):
                self.env.set_reference_override(teacher_advice.reference_goal_position, float(teacher_advice.reference_goal_heading))
            context = self.env.make_primitive_context()
            selection = self.agent.act(
                observation,
                deterministic=deterministic,
                teacher_action_probs=None if teacher_advice is None else teacher_advice.action_probs,
                teacher_weight=0.0 if teacher_advice is None else float(teacher_advice.weight),
                return_diagnostics=True,
            )
            step_diagnostics = None if selection.diagnostics is None else dict(selection.diagnostics)
            next_observation, reward, terminated, truncated, step_info = self.macro_env.step(
                selection.macro_action,
                start_state=self.env.get_articulated_state(),
                context=context,
            )
            if next_observation is None:
                next_observation = self.env.build_observation()
            next_observation = np.asarray(next_observation, dtype=np.float32)
            tau = int(step_info.get("tau", 1))
            macro_steps += 1
            low_level_steps += tau
            forced_truncation = bool((not terminated) and (not truncated) and macro_steps >= self.max_macro_steps)
            final_done = bool(terminated or truncated or forced_truncation)

            if forced_truncation:
                truncated = True
                step_info = dict(step_info)
                step_info["done_reason"] = "macro_budget"
                step_info["truncated"] = True
                step_info["done"] = True

            if step_diagnostics is not None:
                macro_termination_reason = str(step_info.get("macro_termination_reason", "unknown"))
                step_diagnostics.update(
                    {
                        "macro_termination_reason": macro_termination_reason,
                        "macro_tau": int(tau),
                        "macro_articulation_limit_hit": bool(macro_termination_reason == "articulation_limit"),
                        "macro_short_articulation_limit_hit": bool(macro_termination_reason == "articulation_limit" and int(tau) <= 2),
                    }
                )
                action_diagnostics.append(step_diagnostics)

            if store_transition:
                self.agent.store_transition(
                    MacroTransition(
                        observation=observation,
                        action_id=int(selection.macro_action.primitive_id),
                        parameters=np.asarray(selection.macro_action.parameters, dtype=np.float32).copy(),
                        reward=float(reward),
                        tau=int(tau),
                        next_observation=next_observation,
                        done=bool(final_done),
                        log_prob=float(selection.log_prob),
                        value=float(selection.value),
                        teacher_action_probs=None if teacher_advice is None else np.asarray(teacher_advice.action_probs, dtype=np.float32).copy(),
                        teacher_parameter_target=None
                        if teacher_advice is None
                        else np.asarray(teacher_advice.parameter_targets[int(selection.macro_action.primitive_id)], dtype=np.float32).copy(),
                        teacher_weight=0.0 if teacher_advice is None else float(teacher_advice.weight),
                        proxy_scores=None if selection.proxy_scores is None else np.asarray(selection.proxy_scores, dtype=np.float32).copy(),
                        proxy_prefix_lengths=None if selection.proxy_prefix_lengths is None else np.asarray(selection.proxy_prefix_lengths, dtype=np.float32).copy(),
                        hard_valid_mask=None if selection.hard_valid_mask is None else np.asarray(selection.hard_valid_mask, dtype=np.bool_).copy(),
                        execution_valid_mask=None if selection.execution_valid_mask is None else np.asarray(selection.execution_valid_mask, dtype=np.bool_).copy(),
                        soft_score=None if selection.soft_score is None else np.asarray(selection.soft_score, dtype=np.float32).copy(),
                    )
                )

            total_reward += float(reward)
            observation = next_observation
            last_info = dict(step_info)
            previous_action_id = int(selection.macro_action.primitive_id)

        return EpisodeSummary(
            total_reward=float(total_reward),
            macro_steps=int(macro_steps),
            low_level_steps=int(low_level_steps),
            success=bool(last_info.get("goal_reached", False) or last_info.get("success", False)),
            collision=bool(last_info.get("collision", False)),
            terminated=bool(terminated),
            truncated=bool(truncated),
            done_reason=str(last_info.get("done_reason", "running")),
            final_goal_distance=float(self.env.distance_to_goal()),
            last_info=dict(last_info),
            action_diagnostics=_aggregate_action_diagnostics(action_diagnostics),
            start_min_lidar_norm=float(start_min_lidar_norm),
        )