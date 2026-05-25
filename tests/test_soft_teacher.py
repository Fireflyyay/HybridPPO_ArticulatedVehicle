from typing import Optional

from common.runtime_config import ExperimentConfig, ObservationConfig, TrainingScheduleConfig
from env.adapter import create_env_adapter
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library
from training.soft_teacher import CoarseGuidanceSoftTeacher
from training.teacher_eval import evaluate_soft_teacher


def _build_teacher(observation_config: Optional[ObservationConfig] = None):
    config = ExperimentConfig(observation=observation_config or ObservationConfig(lidar_num_beams=32))
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
    teacher = CoarseGuidanceSoftTeacher(
        env=env,
        executor=executor,
        primitive_library=primitive_library,
        vehicle_config=config.vehicle,
        observation_config=config.observation,
    )
    return config, env, teacher


def test_soft_teacher_weight_tracks_warmup_corridor_width():
    _config, env, teacher = _build_teacher()

    wide_observation, _ = env.reset(seed=11, options={"level": "Warmup", "warmup_progress": 0.0})
    wide_advice = teacher.advise(wide_observation)

    narrow_observation, _ = env.reset(seed=11, options={"level": "Warmup", "warmup_progress": 1.0})
    narrow_advice = teacher.advise(narrow_observation)

    assert wide_advice is not None
    assert narrow_advice is not None
    assert 0.0 < float(wide_advice.weight) < float(narrow_advice.weight) <= float(teacher._MAX_WEIGHT)
    assert float(narrow_advice.diagnostics["corridor_width"]) < float(wide_advice.diagnostics["corridor_width"])


def test_soft_teacher_evaluation_smoke_across_warmup_and_normal():
    config = ExperimentConfig(
        observation=ObservationConfig(lidar_num_beams=32),
        schedule=TrainingScheduleConfig(max_macro_steps_per_episode=64),
    )

    warmup_wide = evaluate_soft_teacher(
        config,
        level="Warmup",
        episodes=4,
        seed_offset=200,
        reset_options={"warmup_progress": 0.0},
    )
    warmup_narrow = evaluate_soft_teacher(
        config,
        level="Warmup",
        episodes=4,
        seed_offset=400,
        reset_options={"warmup_progress": 1.0},
    )
    normal = evaluate_soft_teacher(
        config,
        level="Normal",
        episodes=4,
        seed_offset=600,
    )

    assert warmup_wide.mean_macro_steps <= float(config.schedule.max_macro_steps_per_episode)
    assert warmup_narrow.mean_macro_steps <= float(config.schedule.max_macro_steps_per_episode)
    assert normal.mean_macro_steps <= float(config.schedule.max_macro_steps_per_episode)
    assert warmup_wide.teacher_coverage_rate > 0.0
    assert warmup_narrow.teacher_coverage_rate > 0.0
    assert normal.teacher_coverage_rate > 0.0
    assert warmup_wide.collision_rate == 0.0
    assert warmup_narrow.collision_rate == 0.0
    assert normal.collision_rate == 0.0
    assert warmup_wide.mean_goal_distance < 12.0
    assert warmup_narrow.mean_goal_distance < 12.0
    assert normal.mean_goal_distance < 20.0
    assert warmup_narrow.mean_teacher_weight >= warmup_wide.mean_teacher_weight