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
    assert observation.shape == (ObservationConfig(lidar_num_beams=8).observation_dim,)
    assert info["done_reason"] == "reset"
    assert "guidance_available" in info
    assert "guidance_path_confidence" in info
    assert "scene_metadata" in info
    assert "corridor_width" in info

    next_observation, reward, terminated, truncated, step_info = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert next_observation.shape == (ObservationConfig(lidar_num_beams=8).observation_dim,)
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert "collision" in step_info
    assert "goal_reached" in step_info
    assert "reward_info" in step_info
    assert "topology_progress" in step_info["reward_info"]
    assert "distance_progress" in step_info["reward_info"]
    assert "near_goal_factor" in step_info["reward_info"]
    assert "rear_overlap_progress" in step_info["reward_info"]


def test_task_env_default_observation_uses_108_beams():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(),
        reward_config=RewardConfig(),
    )

    observation, _ = env.reset(seed=7, options={"level": "Debug"})
    assert observation.shape == (ObservationConfig().observation_dim,)
    assert env.observation_dim == ObservationConfig().observation_dim


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


def test_warmup_reset_options_control_corridor_curriculum_width():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )

    env.reset(seed=7, options={"level": "Warmup", "warmup_progress": 0.0})
    wide_width = float(env.env._scene.metadata["corridor_width"])
    assert not env.env.predict_collision(env.get_articulated_state())
    assert not env.env.predict_collision(env.get_goal_state())

    env.reset(seed=7, options={"level": "Warmup", "warmup_progress": 1.0})
    narrow_width = float(env.env._scene.metadata["corridor_width"])
    assert not env.env.predict_collision(env.get_articulated_state())
    assert not env.env.predict_collision(env.get_goal_state())
    assert wide_width > narrow_width


def test_global_guidance_is_exposed_in_observation_and_info():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )

    observation, info = env.reset(seed=7, options={"level": "Warmup", "warmup_progress": 1.0})
    guidance = observation[-4:]

    assert observation.shape == (ObservationConfig(lidar_num_beams=8).observation_dim,)
    assert isinstance(bool(info["guidance_available"]), bool)
    assert 0.0 <= float(info["guidance_path_confidence"]) <= 1.0
    assert guidance.shape == (4,)
    assert 0.0 <= float(guidance[-1]) <= 1.0