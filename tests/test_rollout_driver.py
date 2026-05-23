import numpy as np

from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.config import VehicleConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library
from training.rollout import MacroRolloutDriver
from training.soft_teacher import CoarseGuidanceSoftTeacher


def test_macro_rollout_driver_collects_episode_summary():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=30),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    library = build_default_primitive_library()
    agent = HybridPPOAgent(
        config=__import__("common.config", fromlist=["HybridPPOConfig"]).HybridPPOConfig(
            observation_dim=env.observation_dim,
            action_dim=library.action_dim,
            parameter_dim=library.parameter_dim,
            mini_batch_size=2,
            update_epochs=1,
        ),
        primitive_library=library,
    )
    executor = ParameterizedPrimitiveExecutor(library)
    macro_env = ParameterizedMacroActionWrapper(env, executor, gamma=agent.config.gamma)
    driver = MacroRolloutDriver(env, macro_env, agent, max_macro_steps=4)

    summary = driver.collect_episode(level="Debug", seed=3, deterministic=True, store_transition=False)
    assert summary.macro_steps > 0
    assert summary.low_level_steps >= summary.macro_steps
    assert isinstance(summary.total_reward, float)


def test_macro_rollout_driver_stores_soft_teacher_labels_in_narrow_scene():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=30),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    library = build_default_primitive_library()
    agent = HybridPPOAgent(
        config=__import__("common.config", fromlist=["HybridPPOConfig"]).HybridPPOConfig(
            observation_dim=env.observation_dim,
            action_dim=library.action_dim,
            parameter_dim=library.parameter_dim,
            mini_batch_size=2,
            update_epochs=1,
        ),
        primitive_library=library,
    )
    executor = ParameterizedPrimitiveExecutor(library)
    teacher = CoarseGuidanceSoftTeacher(
        env=env,
        executor=executor,
        primitive_library=library,
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
    )
    macro_env = ParameterizedMacroActionWrapper(env, executor, gamma=agent.config.gamma)
    driver = MacroRolloutDriver(env, macro_env, agent, max_macro_steps=4, soft_teacher=teacher)

    reset_options = {"warmup_progress": 1.0}
    guidance_seed = None
    for seed in range(20):
        _observation, info = env.reset(seed=seed, options={"level": "Warmup", **reset_options})
        if bool(info.get("guidance_available", False)):
            guidance_seed = seed
            break
    assert guidance_seed is not None

    summary = driver.collect_episode(level="Warmup", seed=guidance_seed, deterministic=True, store_transition=True, reset_options=reset_options)
    batch = agent.buffer.as_batch()

    assert summary.macro_steps > 0
    assert batch.teacher_action_probs.shape[1] == library.action_dim
    assert batch.teacher_parameter_targets.shape[1] == library.parameter_dim
    assert np.any(batch.teacher_weights > 0.0)