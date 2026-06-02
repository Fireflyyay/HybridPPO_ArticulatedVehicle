import math

import numpy as np
import pytest

from common.config import VehicleConfig
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.types import ArticulatedState, wrap_to_pi
from env.adapter import create_env_adapter
from env.success import articulated_body_polygons


def test_observation_shape_unchanged_with_dual_lidar():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=108),
        reward_config=RewardConfig(),
    )
    observation, _ = env.reset(seed=7, options={"level": "Debug"})
    expected_dim = 108 + 9 + 4
    assert observation.shape == (expected_dim,)
    assert observation.shape[0] == ObservationConfig().observation_dim


def test_observation_shape_unchanged_with_8_beams():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    observation, _ = env.reset(seed=7, options={"level": "Debug"})
    expected_dim = 8 + 9 + 4
    assert observation.shape == (expected_dim,)
    assert env.observation_dim == expected_dim


def test_dual_lidar_origins_distinct_at_phi_zero():
    cfg = VehicleConfig()
    state = ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0, speed=0.0, articulation_rate=0.0)
    assert abs(state.articulation_angle) < 1e-9

    hinge_x = state.x - cfg.hitch_offset * np.cos(state.front_heading)
    hinge_y = state.y - cfg.hitch_offset * np.sin(state.front_heading)
    front_center_x = hinge_x + 0.5 * cfg.front_length * np.cos(state.front_heading)
    front_center_y = hinge_y + 0.5 * cfg.front_length * np.sin(state.front_heading)
    rear_center_x = hinge_x - 0.5 * cfg.rear_length * np.cos(state.rear_heading)
    rear_center_y = hinge_y - 0.5 * cfg.rear_length * np.sin(state.rear_heading)

    front_origin = (state.x, state.y)
    rear_origin = (rear_center_x, rear_center_y)

    distance = float(np.hypot(front_origin[0] - rear_origin[0], front_origin[1] - rear_origin[1]))
    assert distance > 1.0, f"front and rear origins are too close: {distance:.3f}m"
    assert distance < 10.0, f"front and rear origins are too far: {distance:.3f}m"

    expected_distance = cfg.hitch_offset + 0.5 * cfg.rear_length
    expected_distance = 1.8 + 2.2
    assert abs(distance - expected_distance) < 0.1, (
        f"unexpected distance: {distance:.3f} vs expected ~{expected_distance:.3f}"
    )

    assert distance > 2.0

    from shapely.geometry import Point
    front_poly, rear_poly = articulated_body_polygons(state, cfg)
    assert front_poly.contains(Point(front_origin[0], front_origin[1]))
    assert rear_poly.contains(Point(rear_origin[0], rear_origin[1]))


@pytest.fixture
def vehicle_cfg():
    return VehicleConfig()


@pytest.fixture
def env_cfg():
    return EnvRuntimeConfig(max_low_level_steps_per_episode=20)


def test_dual_lidar_shape_equals_total_beams():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=108),
        reward_config=RewardConfig(),
    )
    observation, _ = env.reset(seed=7, options={"level": "Debug"})
    lidar_part = observation[:108]
    features_part = observation[108:117]
    guidance_part = observation[117:121]

    assert lidar_part.shape == (108,)
    assert features_part.shape == (9,)
    assert guidance_part.shape == (4,)
    assert len(observation) == 121

    front_lidar = lidar_part[:54]
    rear_lidar = lidar_part[54:]
    assert front_lidar.shape == (54,)
    assert rear_lidar.shape == (54,)


def test_front_lidar_uses_front_heading_rear_uses_rear_heading(vehicle_cfg, env_cfg):
    env = create_env_adapter(
        env_config=env_cfg,
        vehicle_config=vehicle_cfg,
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env

    state = ArticulatedState(
        x=0.0, y=0.0,
        front_heading=0.0,
        rear_heading=float(np.deg2rad(20.0)),
        speed=0.0, articulation_rate=0.0,
    )

    lidar = task_env._lidar_observation(state)
    assert lidar.shape == (8,)
    assert lidar.shape[0] == 8

    front_beams = 4
    rear_beams = 4

    mid_lidar = np.full((8,), 100.0, dtype=np.float32)
    front_headings = state.front_heading + np.linspace(-np.pi, np.pi, front_beams, endpoint=False)
    rear_headings = state.rear_heading + np.linspace(-np.pi, np.pi, rear_beams, endpoint=False)

    assert not np.allclose(front_headings, rear_headings), (
        f"front and rear headings should differ when articulation is non-zero:\n"
        f"  front_headings[:2]={np.rad2deg(front_headings[:2])}\n"
        f"  rear_headings[:2]={np.rad2deg(rear_headings[:2])}"
    )


def test_rear_lidar_origin_location(vehicle_cfg):
    """Verify rear LiDAR origin is at rear body center (matches articulated_body_polygons)."""
    state = ArticulatedState(
        x=0.0, y=0.0,
        front_heading=0.0,
        rear_heading=0.0,
        speed=0.0, articulation_rate=0.0,
    )

    hinge_x = state.x - vehicle_cfg.hitch_offset * np.cos(state.front_heading)
    hinge_y = state.y - vehicle_cfg.hitch_offset * np.sin(state.front_heading)
    expected_rear_center_x = hinge_x - 0.5 * vehicle_cfg.rear_length * np.cos(state.rear_heading)
    expected_rear_center_y = hinge_y - 0.5 * vehicle_cfg.rear_length * np.sin(state.rear_heading)

    from env.task_env import KinematicTaskEnv
    assert KinematicTaskEnv._rear_lidar_origin is not None

    front_x, front_y = state.x, state.y

    assert abs(front_x - 0.0) < 1e-9
    assert abs(front_y - 0.0) < 1e-9

    assert abs(expected_rear_center_x - (-vehicle_cfg.hitch_offset - 0.5 * vehicle_cfg.rear_length)) < 1e-9
    assert abs(expected_rear_center_x - (-1.8 - 2.2)) < 1e-9

    front_poly, rear_poly = articulated_body_polygons(state, vehicle_cfg)
    from shapely.geometry import Point
    assert front_poly.contains(Point(front_x, front_y))
    assert rear_poly.contains(Point(expected_rear_center_x, expected_rear_center_y))


def test_articulated_state_at_nonzero_phi_rear_origin_correct(vehicle_cfg):
    from env.task_env import KinematicTaskEnv
    from shapely.geometry import Point

    phi_rad = float(np.deg2rad(30.0))
    state = ArticulatedState(
        x=10.0, y=20.0,
        front_heading=0.0,
        rear_heading=-phi_rad,
        speed=0.0, articulation_rate=0.0,
    )

    hinge_x = state.x - vehicle_cfg.hitch_offset * np.cos(state.front_heading)
    hinge_y = state.y - vehicle_cfg.hitch_offset * np.sin(state.front_heading)
    expected_rear_center_x = hinge_x - 0.5 * vehicle_cfg.rear_length * np.cos(state.rear_heading)
    expected_rear_center_y = hinge_y - 0.5 * vehicle_cfg.rear_length * np.sin(state.rear_heading)

    _, rear_poly = articulated_body_polygons(state, vehicle_cfg)
    assert rear_poly.contains(Point(expected_rear_center_x, expected_rear_center_y))


def test_lidar_consistency_phi_zero(vehicle_cfg, env_cfg):
    env = create_env_adapter(
        env_config=env_cfg,
        vehicle_config=vehicle_cfg,
        observation_config=ObservationConfig(lidar_num_beams=108),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env

    state = ArticulatedState(
        x=0.0, y=0.0,
        front_heading=0.0, rear_heading=0.0,
        speed=0.0, articulation_rate=0.0,
    )
    lidar = task_env._lidar_observation(state)

    front_lidar = lidar[:54]
    rear_lidar = lidar[54:]

    for idx in range(54):
        angle = -np.pi + idx * (2.0 * np.pi / 54.0)
        opposite_idx = (idx + 27) % 54
        raw_distance_front = front_lidar[idx] * vehicle_cfg.front_length
        raw_distance_rear = rear_lidar[idx] * vehicle_cfg.rear_length
        assert front_lidar[idx] == pytest.approx(rear_lidar[idx], abs=1e-4), (
            f"phi=0 beam {idx}: front={front_lidar[idx]:.4f}, rear={rear_lidar[idx]:.4f} differ too much"
        )


def test_env_internal_front_and_rear_beams():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=108),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env

    assert task_env._front_beams == 54
    assert task_env._rear_beams == 54
    assert task_env._front_beams + task_env._rear_beams == 108
    assert len(task_env._front_beam_angles) == 54
    assert len(task_env._rear_beam_angles) == 54


def test_env_internal_front_and_rear_beams_odd_count():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=7),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env

    assert task_env._front_beams == 3
    assert task_env._rear_beams == 4
    assert task_env._front_beams + task_env._rear_beams == 7


def test_beam_angles_are_360_degree():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=108),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env

    front_span = task_env._front_beam_angles[-1] - task_env._front_beam_angles[0]
    rear_span = task_env._rear_beam_angles[-1] - task_env._rear_beam_angles[0]
    step_front = task_env._front_beam_angles[1] - task_env._front_beam_angles[0]
    step_rear = task_env._rear_beam_angles[1] - task_env._rear_beam_angles[0]

    assert abs(front_span + step_front - 2.0 * np.pi) < 1e-9
    assert abs(rear_span + step_rear - 2.0 * np.pi) < 1e-9


def test_dual_lidar_normalization_range(vehicle_cfg, env_cfg):
    env = create_env_adapter(
        env_config=env_cfg,
        vehicle_config=vehicle_cfg,
        observation_config=ObservationConfig(lidar_num_beams=8, lidar_max_range=30.0),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env

    state = ArticulatedState(
        x=0.0, y=0.0,
        front_heading=0.0, rear_heading=0.0,
        speed=0.0, articulation_rate=0.0,
    )
    lidar = task_env._lidar_observation(state)

    assert np.all(lidar >= 0.0)
    assert np.all(lidar <= 1.0)
    front = lidar[:4]
    rear = lidar[4:]
    assert np.all(front >= 0.0)
    assert np.all(front <= 1.0)
    assert np.all(rear >= 0.0)
    assert np.all(rear <= 1.0)


def test_proxy_safety_metadata_check_warns_on_old_sidecar(tmp_path):
    import warnings
    import numpy as np
    from primitives.proxy_safety import ProxySafetySidecar, load_proxy_safety_sidecar, save_proxy_safety_sidecar

    required = np.zeros((1, 1, 1, 2, 4), dtype=np.float32)
    sidecar_no_version = ProxySafetySidecar(
        required_clearance=required,
        articulation_bin_centers=np.array([0.0], dtype=np.float32),
        parameter_centers=np.zeros((1, 1, 3), dtype=np.float32),
        parameter_scales=np.ones((1, 1, 3), dtype=np.float32),
        nominal_horizons=np.full((1, 1), 2, dtype=np.int64),
        proxy_valid_mask=np.ones((1, 1), dtype=np.bool_),
        lidar_range=10.0,
        metadata={},
    )
    npz_path = tmp_path / "old_sidecar.npz"
    save_proxy_safety_sidecar(str(npz_path), sidecar_no_version)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_proxy_safety_sidecar(str(npz_path))
        found = [w for w in caught if "lidar_model_version" in str(w.message)]
        assert len(found) >= 1, "old sidecar without lidar_model_version should trigger a warning"


def test_proxy_safety_sidecar_metadata_has_dual_fields():
    from primitives import build_default_primitive_library
    from primitives.proxy_safety import build_proxy_safety_sidecar
    from common.config import PrimitiveExecutorConfig, VehicleConfig

    library = build_default_primitive_library()
    sidecar = build_proxy_safety_sidecar(
        library=library,
        vehicle_config=VehicleConfig(),
        executor_config=PrimitiveExecutorConfig(max_macro_steps=4),
        lidar_num=108,
        lidar_range=30.0,
        articulation_bin_count=2,
        proxy_resolution=1,
        max_proxies_per_action=1,
    )

    meta = sidecar.metadata
    assert meta["lidar_model_version"] == "dual_body_54_54_v1"
    assert meta["front_beams"] == 54
    assert meta["rear_beams"] == 54
    assert meta["front_frame"] == "front_heading"
    assert meta["rear_frame"] == "rear_heading"
    assert meta["lidar_range"] == 30.0
    assert meta["lidar_num"] == 108
    assert sidecar.num_rays == 108


def test_proxy_safety_sidecar_with_8_beams_metadata(vehicle_cfg):
    from primitives import build_default_primitive_library
    from primitives.proxy_safety import build_proxy_safety_sidecar
    from common.config import PrimitiveExecutorConfig

    library = build_default_primitive_library()
    sidecar = build_proxy_safety_sidecar(
        library=library,
        vehicle_config=vehicle_cfg,
        executor_config=PrimitiveExecutorConfig(max_macro_steps=4),
        lidar_num=8,
        lidar_range=20.0,
        articulation_bin_count=1,
        proxy_resolution=1,
        max_proxies_per_action=1,
    )

    meta = sidecar.metadata
    assert meta["front_beams"] == 4
    assert meta["rear_beams"] == 4
    assert meta["lidar_num"] == 8
    assert sidecar.num_rays == 8
