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
        observation, _ = self.env.reset(seed=seed, options=options)
        observation = np.asarray(observation, dtype=np.float32)
        total_reward = 0.0
        macro_steps = 0
        low_level_steps = 0
        last_info: Dict[str, object] = self.env.current_info()
        terminated = False
        truncated = False
        previous_action_id: Optional[int] = None

        while macro_steps < self.max_macro_steps and not (terminated or truncated):
            context = self.env.make_primitive_context()
            teacher_advice = None if self.soft_teacher is None else self.soft_teacher.advise(observation, previous_action_id=previous_action_id)
            selection = self.agent.act(
                observation,
                deterministic=deterministic,
                teacher_action_probs=None if teacher_advice is None else teacher_advice.action_probs,
                teacher_weight=0.0 if teacher_advice is None else float(teacher_advice.weight),
            )
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
                step_info.setdefault("done_reason", "macro_budget")
                step_info["truncated"] = True
                step_info["done"] = True

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
        )