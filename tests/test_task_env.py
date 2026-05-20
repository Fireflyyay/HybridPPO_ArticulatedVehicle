import numpy as np

from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.config import VehicleConfig
from env.adapter import create_env_adapter


def test_task_env_reset_and_step_produce_expected_fields():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )

    observation, info = env.reset(seed=7, options={"level": "Debug"})
    assert observation.shape == (17,)
    assert info["done_reason"] == "reset"

    next_observation, reward, terminated, truncated, step_info = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert next_observation.shape == (17,)
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert "collision" in step_info
    assert "goal_reached" in step_info
    assert "reward_info" in step_info


def test_task_env_default_observation_uses_108_beams():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(),
        reward_config=RewardConfig(),
    )

    observation, _ = env.reset(seed=7, options={"level": "Debug"})
    assert observation.shape == (117,)
    assert env.observation_dim == 117


def test_scene_resets_are_collision_free_for_training_levels():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )

    for level in ("Debug", "Warmup", "Normal"):
        for seed in range(25):
            env.reset(seed=seed, options={"level": level})
            assert not env.env.predict_collision(env.get_articulated_state())
            assert not env.env.predict_collision(env.get_goal_state())