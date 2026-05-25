from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from common.runtime_config import ExperimentConfig
from common.types import MacroAction
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library
from training.soft_teacher import CoarseGuidanceSoftTeacher


@dataclass(frozen=True)
class TeacherEvaluationSummary:
    success_rate: float
    collision_rate: float
    mean_return: float
    mean_macro_steps: float
    mean_goal_distance: float
    mean_teacher_weight: float
    teacher_coverage_rate: float
    done_reason_counts: Dict[str, int]


def evaluate_soft_teacher(
    config: ExperimentConfig,
    level: str,
    episodes: int,
    seed_offset: int = 0,
    reset_options: Optional[Mapping[str, object]] = None,
    deterministic: bool = True,
) -> TeacherEvaluationSummary:
    env = create_env_adapter(
        env_config=config.env,
        vehicle_config=config.vehicle,
        observation_config=config.observation,
        reward_config=config.reward,
    )
    primitive_library = build_default_primitive_library()
    executor = ParameterizedPrimitiveExecutor(
        primitive_library,
        vehicle_config=config.vehicle,
        executor_config=config.primitive_executor,
    )
    macro_env = ParameterizedMacroActionWrapper(env, executor, gamma=config.agent.gamma)
    teacher = CoarseGuidanceSoftTeacher(
        env=env,
        executor=executor,
        primitive_library=primitive_library,
        vehicle_config=config.vehicle,
        observation_config=config.observation,
    )

    successes = 0
    collisions = 0
    total_reward = 0.0
    total_macro_steps = 0
    total_goal_distance = 0.0
    total_teacher_weight = 0.0
    total_teacher_steps = 0
    done_reason_counts: Dict[str, int] = {}
    rng = np.random.default_rng(int(config.seed + seed_offset))

    for episode_idx in range(max(0, int(episodes))):
        options = {"level": str(level)}
        if reset_options is not None:
            options.update(dict(reset_options))
            options["level"] = str(level)
        observation, info = env.reset(seed=int(config.seed + seed_offset + episode_idx), options=options)
        observation = np.asarray(observation, dtype=np.float32)
        previous_action_id = None
        macro_steps = 0
        episode_reward = 0.0
        terminated = False
        truncated = False
        last_info = dict(info)

        while macro_steps < int(config.schedule.max_macro_steps_per_episode) and not (terminated or truncated):
            advice = teacher.advise(observation, previous_action_id=previous_action_id)
            if advice is None:
                last_info = dict(last_info)
                last_info.setdefault("done_reason", "teacher_none")
                break
            total_teacher_weight += float(advice.weight)
            total_teacher_steps += 1
            if deterministic:
                action_id = int(np.argmax(advice.action_probs))
            else:
                action_id = int(rng.choice(np.arange(advice.action_probs.shape[0]), p=advice.action_probs))
            parameters = np.asarray(advice.parameter_targets[action_id], dtype=np.float32).copy()
            next_observation, reward, terminated, truncated, step_info = macro_env.step(
                MacroAction(primitive_id=action_id, parameters=parameters),
                start_state=env.get_articulated_state(),
                context=env.make_primitive_context(),
            )
            if next_observation is None:
                next_observation = env.build_observation()
            observation = np.asarray(next_observation, dtype=np.float32)
            episode_reward += float(reward)
            macro_steps += 1
            previous_action_id = action_id
            last_info = dict(step_info)

        if not terminated and not truncated and macro_steps >= int(config.schedule.max_macro_steps_per_episode):
            last_info = dict(last_info)
            last_info["done_reason"] = "macro_budget"

        success = bool(last_info.get("goal_reached", False) or last_info.get("success", False))
        collision = bool(last_info.get("collision", False))
        done_reason = str(last_info.get("done_reason", "running"))
        successes += int(success)
        collisions += int(collision)
        total_reward += float(episode_reward)
        total_macro_steps += int(macro_steps)
        total_goal_distance += float(env.distance_to_goal())
        done_reason_counts[done_reason] = int(done_reason_counts.get(done_reason, 0) + 1)

    count = max(1, int(episodes))
    macro_denom = max(1, total_macro_steps)
    teacher_step_denom = max(1, total_teacher_steps)
    return TeacherEvaluationSummary(
        success_rate=float(successes / count),
        collision_rate=float(collisions / count),
        mean_return=float(total_reward / count),
        mean_macro_steps=float(total_macro_steps / count),
        mean_goal_distance=float(total_goal_distance / count),
        mean_teacher_weight=float(total_teacher_weight / teacher_step_denom),
        teacher_coverage_rate=float(total_teacher_steps / macro_denom),
        done_reason_counts=dict(sorted(done_reason_counts.items())),
    )