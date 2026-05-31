import math

import numpy as np

from common.types import ArticulatedState
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.config import VehicleConfig
from env.adapter import create_env_adapter


def _state_with_pose(base: ArticulatedState, x: float, y: float) -> ArticulatedState:
    return ArticulatedState(
        x=float(x),
        y=float(y),
        front_heading=float(base.front_heading),
        rear_heading=float(base.rear_heading),
        speed=0.0,
        articulation_rate=0.0,
    )


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


def test_escape_reference_can_be_disabled_and_reference_override_is_used():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20, escape_reference_enabled=False),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )

    env.reset(seed=7, options={"level": "Warmup", "warmup_progress": 1.0})
    task_env = env.env
    goal = env.get_goal_state()
    reference_position, reference_heading = task_env._reference_target(env.get_articulated_state())

    assert np.allclose(np.asarray(reference_position, dtype=np.float64), np.asarray([goal.x, goal.y], dtype=np.float64))
    assert np.isclose(reference_heading, float(goal.front_heading))

    override_position = (float(goal.x) - 1.0, float(goal.y) + 1.0)
    override_heading = float(goal.front_heading) - 0.2
    env.set_reference_override(override_position, override_heading)
    reference_position, reference_heading = task_env._reference_target(env.get_articulated_state())

    assert np.allclose(np.asarray(reference_position, dtype=np.float64), np.asarray(override_position, dtype=np.float64))
    assert np.isclose(reference_heading, override_heading)


def test_topology_progress_only_rewards_new_frontier_advance():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(reverse_penalty_coef=0.0),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env
    base = env.get_articulated_state()
    goal = env.get_goal_state()

    previous_state = _state_with_pose(base, x=0.0, y=float(base.y))
    advanced_state = _state_with_pose(base, x=1.0, y=float(base.y))
    reversed_state = _state_with_pose(base, x=0.0, y=float(base.y))

    task_env._best_rewarded_topology_cost = 10.0
    task_env._best_rewarded_goal_distance = float(task_env.distance_to_goal(previous_state))
    task_env._global_guidance.query_cost_to_go_details = lambda x, y: (10.0 - float(x), "direct")
    task_env._path_progress_m = lambda state: None

    previous_metrics = task_env.success_checker.evaluate(previous_state, goal, collision_free=True)
    advanced_metrics = task_env.success_checker.evaluate(advanced_state, goal, collision_free=True)
    reversed_metrics = task_env.success_checker.evaluate(reversed_state, goal, collision_free=True)

    _, forward_info = task_env._compute_reward(
        previous_state=previous_state,
        next_state=advanced_state,
        previous_metrics=previous_metrics,
        current_metrics=advanced_metrics,
        collision=False,
        out_of_bounds=False,
        success=False,
        timeout=False,
    )
    _, reverse_info = task_env._compute_reward(
        previous_state=advanced_state,
        next_state=reversed_state,
        previous_metrics=advanced_metrics,
        current_metrics=reversed_metrics,
        collision=False,
        out_of_bounds=False,
        success=False,
        timeout=False,
    )
    _, repeat_info = task_env._compute_reward(
        previous_state=reversed_state,
        next_state=advanced_state,
        previous_metrics=reversed_metrics,
        current_metrics=advanced_metrics,
        collision=False,
        out_of_bounds=False,
        success=False,
        timeout=False,
    )

    assert forward_info["topology_progress"] > 0.0
    assert np.isclose(reverse_info["topology_progress"], 0.0)
    assert np.isclose(repeat_info["topology_progress"], 0.0)
    assert forward_info["progress_source"] == "topology"


def test_progress_fallback_uses_goal_frontier_and_lower_weight():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env
    base = env.get_articulated_state()
    goal = env.get_goal_state()

    dx = float(goal.x - base.x)
    dy = float(goal.y - base.y)
    norm = max(float(math.hypot(dx, dy)), 1e-6)
    step = 1.0
    closer_state = _state_with_pose(base, x=float(base.x) + step * dx / norm, y=float(base.y) + step * dy / norm)

    base_metrics = task_env.success_checker.evaluate(base, goal, collision_free=True)
    closer_metrics = task_env.success_checker.evaluate(closer_state, goal, collision_free=True)

    task_env._best_rewarded_topology_cost = None
    task_env._best_rewarded_goal_distance = float(task_env.distance_to_goal(base))
    task_env._global_guidance.query_cost_to_go_details = lambda x, y: (None, "unavailable")
    task_env._local_clearance_factor = lambda state: 0.5
    task_env._path_progress_m = lambda state: None

    _, forward_info = task_env._compute_reward(
        previous_state=base,
        next_state=closer_state,
        previous_metrics=base_metrics,
        current_metrics=closer_metrics,
        collision=False,
        out_of_bounds=False,
        success=False,
        timeout=False,
    )
    _, repeat_info = task_env._compute_reward(
        previous_state=base,
        next_state=closer_state,
        previous_metrics=base_metrics,
        current_metrics=closer_metrics,
        collision=False,
        out_of_bounds=False,
        success=False,
        timeout=False,
    )

    assert forward_info["progress_source"] == "clearance_distance"
    assert forward_info["topology_progress"] > 0.0
    assert forward_info["progress_weight"] < float(task_env.reward_config.progress_weight)
    assert np.isclose(repeat_info["topology_progress"], 0.0)


def test_reverse_penalty_is_applied_when_moving_against_topology_potential():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(reverse_penalty_coef=1.0),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env
    base = env.get_articulated_state()
    goal = env.get_goal_state()

    forward_state = _state_with_pose(base, x=float(base.x) + 1.0, y=float(base.y))
    reverse_state = _state_with_pose(base, x=float(base.x), y=float(base.y))
    far_forward_state = _state_with_pose(base, x=float(base.x) + 2.0, y=float(base.y))

    base_metrics = task_env.success_checker.evaluate(base, goal, collision_free=True)
    forward_metrics = task_env.success_checker.evaluate(forward_state, goal, collision_free=True)
    reverse_metrics = task_env.success_checker.evaluate(reverse_state, goal, collision_free=True)
    far_forward_metrics = task_env.success_checker.evaluate(far_forward_state, goal, collision_free=True)

    task_env._best_rewarded_topology_cost = 10.0
    task_env._best_rewarded_goal_distance = float(task_env.distance_to_goal(base))
    task_env._global_guidance.query_cost_to_go_details = lambda x, y: (10.0 - float(x), "direct")
    task_env._path_progress_m = lambda state: None

    _, advance_info = task_env._compute_reward(
        previous_state=base,
        next_state=forward_state,
        previous_metrics=base_metrics,
        current_metrics=forward_metrics,
        collision=False, out_of_bounds=False, success=False, timeout=False,
    )
    _, regress_info = task_env._compute_reward(
        previous_state=forward_state,
        next_state=reverse_state,
        previous_metrics=forward_metrics,
        current_metrics=reverse_metrics,
        collision=False, out_of_bounds=False, success=False, timeout=False,
    )
    _, re_advance_info = task_env._compute_reward(
        previous_state=reverse_state,
        next_state=far_forward_state,
        previous_metrics=reverse_metrics,
        current_metrics=far_forward_metrics,
        collision=False, out_of_bounds=False, success=False, timeout=False,
    )

    assert advance_info["topology_progress"] > 0.0
    assert regress_info["topology_progress"] < 0.0
    assert re_advance_info["topology_progress"] > 0.0


def test_make_primitive_context_uses_local_guidance_reference_during_escape_phase():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20, escape_reference_enabled=True),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env
    base = env.get_articulated_state()
    task_env._guidance_available = True
    task_env._global_guidance.path_points_world = np.asarray(
        [
            [float(base.x), float(base.y)],
            [float(base.x), float(base.y) + 4.0],
            [float(base.x), float(base.y) + 8.0],
            [float(base.x) + 4.0, float(base.y) + 8.0],
        ],
        dtype=np.float64,
    )
    task_env._global_guidance.path_s = task_env._global_guidance._polyline_arc_length(task_env._global_guidance.path_points_world)
    task_env._global_guidance.progress_idx = 0
    task_env._goal_state = ArticulatedState(
        x=float(base.x) + 4.0,
        y=float(base.y) + 8.0,
        front_heading=0.0,
        rear_heading=0.0,
    )

    context = task_env.make_primitive_context()

    assert context.goal_position is not None
    assert context.goal_heading is not None
    assert abs(float(context.goal_position[0]) - float(base.x)) < 1e-6
    assert float(context.goal_position[1]) > float(base.y) + 2.0
    assert np.isclose(float(context.goal_heading), float(np.pi / 2.0), atol=0.15)


def test_make_primitive_context_returns_final_goal_reference_after_escape_phase():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20, escape_reference_enabled=True),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env
    base = env.get_articulated_state()
    task_env._guidance_available = True
    task_env._global_guidance.path_points_world = np.asarray(
        [
            [float(base.x), float(base.y)],
            [float(base.x), float(base.y) + 4.0],
            [float(base.x), float(base.y) + 8.0],
            [float(base.x) + 4.0, float(base.y) + 8.0],
        ],
        dtype=np.float64,
    )
    task_env._global_guidance.path_s = task_env._global_guidance._polyline_arc_length(task_env._global_guidance.path_points_world)
    task_env._global_guidance.progress_idx = 0
    task_env._goal_state = ArticulatedState(
        x=float(base.x) + 4.0,
        y=float(base.y) + 8.0,
        front_heading=0.0,
        rear_heading=0.0,
    )
    task_env._state = ArticulatedState(
        x=float(base.x) + 0.1,
        y=float(base.y) + 9.0,
        front_heading=0.0,
        rear_heading=0.0,
    )

    context = task_env.make_primitive_context()

    assert context.goal_position == (float(task_env._goal_state.x), float(task_env._goal_state.y))
    assert np.isclose(float(context.goal_heading), float(task_env._goal_state.front_heading))


def test_heading_progress_uses_cached_escape_reference_during_escape_phase():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    env.reset(seed=7, options={"level": "Debug"})
    task_env = env.env
    base = env.get_articulated_state()
    goal = env.get_goal_state()
    task_env._progress_reward = lambda previous_state, next_state, current_goal_distance: (0.0, 0.0, 0.0, "topology", 1.0)
    task_env._set_active_reference(
        goal_position=(float(base.x), float(base.y) + 3.0),
        goal_heading=float(np.pi / 2.0),
    )
    previous_state = ArticulatedState(
        x=float(base.x),
        y=float(base.y),
        front_heading=0.2,
        rear_heading=0.2,
    )
    next_state = ArticulatedState(
        x=float(base.x),
        y=float(base.y) + 0.2,
        front_heading=0.9,
        rear_heading=0.9,
    )
    previous_metrics = task_env.success_checker.evaluate(previous_state, goal, collision_free=True)
    current_metrics = task_env.success_checker.evaluate(next_state, goal, collision_free=True)

    _, reward_info = task_env._compute_reward(
        previous_state=previous_state,
        next_state=next_state,
        previous_metrics=previous_metrics,
        current_metrics=current_metrics,
        collision=False,
        out_of_bounds=False,
        success=False,
        timeout=False,
    )

    assert reward_info["heading_progress"] > 0.0


def test_stage_potential_rewards_bay_exit_progress():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=20),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(reverse_penalty_coef=0.0),
    )
    env.reset(seed=7, options={"level": "Warmup", "warmup_task": "BayExit", "warmup_progress": 0.0})
    task_env = env.env
    base = env.get_articulated_state()
    goal = env.get_goal_state()
    mouth = np.asarray(task_env._scene.metadata["start_bay_mouth"], dtype=np.float64)
    start_xy = np.asarray([float(base.x), float(base.y)], dtype=np.float64)
    direction = mouth - start_xy
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)

    prev_state = _state_with_pose(base, x=float(base.x), y=float(base.y))
    next_state = _state_with_pose(
        base,
        x=float(base.x) + float(direction[0]),
        y=float(base.y) + float(direction[1]),
    )
    regress_state = prev_state

    prev_metrics = task_env.success_checker.evaluate(prev_state, goal, collision_free=True)
    next_metrics = task_env.success_checker.evaluate(next_state, goal, collision_free=True)
    regress_metrics = task_env.success_checker.evaluate(regress_state, goal, collision_free=True)

    _, advance_info = task_env._compute_reward(
        previous_state=prev_state,
        next_state=next_state,
        previous_metrics=prev_metrics,
        current_metrics=next_metrics,
        collision=False, out_of_bounds=False, success=False, timeout=False,
    )
    assert advance_info["progress_source"] == "bay_exit_potential"
    assert advance_info["topology_progress"] > 0.0
    assert advance_info["progress_weight"] == float(task_env.reward_config.progress_weight)
    assert advance_info["stage_exit_progress"] > 0.0

    _, regress_info = task_env._compute_reward(
        previous_state=next_state,
        next_state=regress_state,
        previous_metrics=next_metrics,
        current_metrics=regress_metrics,
        collision=False, out_of_bounds=False, success=False, timeout=False,
    )
    assert np.isclose(regress_info["topology_progress"], 0.0)

    _, repeat_info = task_env._compute_reward(
        previous_state=regress_state,
        next_state=next_state,
        previous_metrics=regress_metrics,
        current_metrics=next_metrics,
        collision=False, out_of_bounds=False, success=False, timeout=False,
    )
    assert np.isclose(repeat_info["topology_progress"], 0.0)
