import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from common.config import VehicleConfig
from common.runtime_config import EnvRuntimeConfig, GUIDANCE_FEATURE_DIM, ObservationConfig, RewardConfig
from common.types import ArticulatedState, LowLevelControl, PrimitiveExecutionContext, wrap_to_pi
from env.dynamics import ArticulatedKinematics
from env.global_guidance import CoarseGlobalGuidance
from env.scenes import BaselineInspiredSceneFactory, SceneSpec
from env.success import ParkingSuccessChecker, articulated_body_polygons


@dataclass(frozen=True)
class TaskStepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class KinematicTaskEnv:
    _PROJECTED_TOPOLOGY_WEIGHT_SCALE = 0.60
    _CLEARANCE_DISTANCE_WEIGHT_SCALE = 0.35
    _EUCLIDEAN_DISTANCE_WEIGHT_SCALE = 0.20

    def __init__(
        self,
        scene_factory: BaselineInspiredSceneFactory,
        vehicle_config: VehicleConfig,
        observation_config: ObservationConfig,
        reward_config: RewardConfig,
        env_config: EnvRuntimeConfig,
    ) -> None:
        self.scene_factory = scene_factory
        self.vehicle_config = vehicle_config
        self.observation_config = observation_config
        self.reward_config = reward_config
        self.env_config = env_config
        self.kinematics = ArticulatedKinematics(vehicle_config)
        self.success_checker = ParkingSuccessChecker(vehicle_config)
        self._rng = np.random.default_rng()
        self._scene: Optional[SceneSpec] = None
        self._state: Optional[ArticulatedState] = None
        self._goal_state: Optional[ArticulatedState] = None
        self._obstacle_union = None
        self._world_box = None
        self._beam_angle_offsets = np.linspace(
            -np.pi,
            np.pi,
            int(self.observation_config.lidar_num_beams),
            endpoint=False,
            dtype=np.float64,
        )
        self._obstacle_segment_x1 = np.empty((0,), dtype=np.float64)
        self._obstacle_segment_y1 = np.empty((0,), dtype=np.float64)
        self._obstacle_segment_dx = np.empty((0,), dtype=np.float64)
        self._obstacle_segment_dy = np.empty((0,), dtype=np.float64)
        self._step_count = 0
        self._last_info: Dict[str, object] = {}
        self._global_guidance = CoarseGlobalGuidance()
        self._guidance_available = False
        self._best_rewarded_topology_cost: Optional[float] = None
        self._best_rewarded_goal_distance: Optional[float] = None
        self._initial_topology_cost: Optional[float] = None
        self._best_rewarded_stage_potential: Optional[float] = None
        self._active_reference_goal_position: Optional[Tuple[float, float]] = None
        self._active_reference_goal_heading: Optional[float] = None
        self._reference_override_goal_position: Optional[Tuple[float, float]] = None
        self._reference_override_goal_heading: Optional[float] = None

    @property
    def observation_dim(self) -> int:
        return int(self.observation_config.observation_dim)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        level = str((options or {}).get("level", self.env_config.default_level))
        self._scene = self.scene_factory.generate(level, self._rng, options=options)
        self._state = self._scene.start_state
        self._goal_state = self._scene.goal_state
        self._obstacle_union = None if len(self._scene.obstacles) == 0 else unary_union(list(self._scene.obstacles))
        self._cache_obstacle_segments()
        xmin, xmax, ymin, ymax = self._scene.world_bounds
        self._world_box = box(xmin, ymin, xmax, ymax)
        self._guidance_available = self._global_guidance.plan_scene_path(
            scene=self._scene,
            vehicle_config=self.vehicle_config,
            start_xy=(float(self._state.x), float(self._state.y)),
            goal_xy=(float(self._goal_state.x), float(self._goal_state.y)),
        )
        self.clear_reference_override()
        self._set_active_reference(
            goal_position=(float(self._goal_state.x), float(self._goal_state.y)),
            goal_heading=float(self._goal_state.front_heading),
        )
        initial_topology_cost, _ = self._global_guidance.query_cost_to_go_details(float(self._state.x), float(self._state.y))
        self._best_rewarded_topology_cost = None if initial_topology_cost is None else float(initial_topology_cost)
        self._initial_topology_cost = None if initial_topology_cost is None else float(initial_topology_cost)
        self._best_rewarded_goal_distance = float(self.distance_to_goal(self._state))
        initial_stage = self._stage_potential(self._state)
        self._best_rewarded_stage_potential = None if initial_stage is None else float(initial_stage["total"])
        self._step_count = 0
        initial_lidar = self._lidar_observation(self._state)
        start_min_lidar_norm = float(np.min(initial_lidar)) if len(initial_lidar) > 0 else 1.0
        self._last_info = self._build_info(
            collision=False,
            goal_reached=False,
            terminated=False,
            truncated=False,
            done_reason="reset",
            reward_info={},
        )
        self._last_info["start_min_lidar_norm"] = float(start_min_lidar_norm)
        return self.build_observation(), dict(self._last_info)

    def step(self, action: np.ndarray):
        if self._state is None:
            raise RuntimeError("environment must be reset before step")
        control = LowLevelControl(articulation_rate=float(action[0]), speed=float(action[1]))
        previous_state = self._state
        previous_metrics = self.success_checker.evaluate(previous_state, self._goal_state, collision_free=True)
        next_state, _ = self.kinematics.step_with_diagnostics(previous_state, control)
        self._step_count += 1

        collision = self._intersects_obstacles(next_state)
        out_of_bounds = self._is_out_of_bounds(next_state)
        collision_free = (not collision) and (not out_of_bounds)
        success_metrics = self.success_checker.evaluate(next_state, self._goal_state, collision_free=collision_free)
        timeout = self._step_count >= int(self.env_config.max_low_level_steps_per_episode)

        terminated = False
        truncated = False
        done_reason = "running"
        if success_metrics.success:
            terminated = True
            done_reason = "goal"
        elif collision:
            terminated = True
            done_reason = "collision"
        elif out_of_bounds:
            terminated = True
            done_reason = "out_of_bounds"
        elif timeout:
            truncated = True
            done_reason = "timeout"

        reward, reward_info = self._compute_reward(
            previous_state=previous_state,
            next_state=next_state,
            previous_metrics=previous_metrics,
            current_metrics=success_metrics,
            collision=collision,
            out_of_bounds=out_of_bounds,
            success=success_metrics.success,
            timeout=timeout,
        )

        self._state = next_state
        self._last_info = self._build_info(
            collision=bool(collision),
            goal_reached=bool(success_metrics.success),
            terminated=bool(terminated),
            truncated=bool(truncated),
            done_reason=done_reason,
            reward_info=reward_info,
            success_metrics=success_metrics,
        )
        observation = self.build_observation()
        return observation, float(reward), bool(terminated), bool(truncated), dict(self._last_info)

    def build_observation(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("environment must be reset before observation generation")
        lidar = self._lidar_observation(self._state)
        dx = float(self._goal_state.x - self._state.x)
        dy = float(self._goal_state.y - self._state.y)
        distance = float(np.hypot(dx, dy)) / max(float(self.observation_config.goal_distance_scale), 1e-6)
        angle_to_goal = float(np.arctan2(dy, dx))
        relative_angle = wrap_to_pi(angle_to_goal - float(self._state.front_heading))
        relative_heading = wrap_to_pi(float(self._goal_state.front_heading) - float(self._state.front_heading))
        articulation = float(self._state.articulation_angle)
        speed_scale = max(abs(float(self.vehicle_config.speed_min)), abs(float(self.vehicle_config.speed_max)), 1e-6)
        articulation_rate_scale = max(
            abs(float(self.vehicle_config.articulation_rate_min)),
            abs(float(self.vehicle_config.articulation_rate_max)),
            1e-6,
        )
        features = np.array(
            [
                distance,
                np.cos(relative_angle),
                np.sin(relative_angle),
                np.cos(relative_heading),
                np.sin(relative_heading),
                np.cos(articulation),
                np.sin(articulation),
                float(self._state.speed) / speed_scale,
                float(self._state.articulation_rate) / articulation_rate_scale,
            ],
            dtype=np.float32,
        )
        guidance = self._guidance_observation(lidar)
        return np.concatenate([lidar, features, guidance], axis=0).astype(np.float32)

    def get_articulated_state(self) -> ArticulatedState:
        if self._state is None:
            raise RuntimeError("environment must be reset before state access")
        return self._state

    def get_goal_state(self) -> ArticulatedState:
        if self._goal_state is None:
            raise RuntimeError("environment must be reset before goal access")
        return self._goal_state

    def current_info(self) -> Dict[str, object]:
        return dict(self._last_info)

    def distance_to_goal(self, state: Optional[ArticulatedState] = None) -> float:
        if self._goal_state is None:
            raise RuntimeError("environment must be reset before goal distance access")
        current = self.get_articulated_state() if state is None else state
        return float(np.hypot(float(self._goal_state.x - current.x), float(self._goal_state.y - current.y)))

    def query_cost_to_go(self, state: Optional[ArticulatedState] = None) -> Optional[float]:
        current = self.get_articulated_state() if state is None else state
        return self._global_guidance.query_cost_to_go(float(current.x), float(current.y))

    def make_primitive_context(self) -> PrimitiveExecutionContext:
        goal_position, goal_heading = self._reference_target(self.get_articulated_state())
        self._set_active_reference(goal_position=goal_position, goal_heading=goal_heading)
        return PrimitiveExecutionContext(
            goal_position=goal_position,
            goal_heading=goal_heading,
            collision_checker=self.predict_collision,
        )

    def set_reference_override(self, goal_position, goal_heading: float) -> None:
        self._reference_override_goal_position = (float(goal_position[0]), float(goal_position[1]))
        self._reference_override_goal_heading = float(goal_heading)

    def clear_reference_override(self) -> None:
        self._reference_override_goal_position = None
        self._reference_override_goal_heading = None

    def _set_active_reference(self, goal_position: Tuple[float, float], goal_heading: float) -> None:
        self._active_reference_goal_position = (float(goal_position[0]), float(goal_position[1]))
        self._active_reference_goal_heading = float(goal_heading)

    def _reference_target(self, state: ArticulatedState) -> Tuple[Tuple[float, float], float]:
        if self._reference_override_goal_position is not None and self._reference_override_goal_heading is not None:
            return self._reference_override_goal_position, float(self._reference_override_goal_heading)

        if bool(self.env_config.escape_reference_enabled):
            local_reference = self._escape_reference_target(state)
            if local_reference is not None:
                return local_reference

        if bool(self.env_config.phase_reference_enabled):
            phase_reference = self._phase_reference_target(state)
            if phase_reference is not None:
                return phase_reference

        goal = self.get_goal_state()
        return (float(goal.x), float(goal.y)), float(goal.front_heading)

    def _phase_reference_target(self, state: ArticulatedState) -> Optional[Tuple[Tuple[float, float], float]]:
        guidance_ref = self._guidance_reference(state)
        if guidance_ref is None:
            return None
        target_pos, target_heading, progress_m = guidance_ref
        if self._is_docking_phase(state, progress_m, target_heading):
            goal = self.get_goal_state()
            return (float(goal.x), float(goal.y)), float(goal.front_heading)
        return target_pos, target_heading

    def _is_docking_phase(self, state: ArticulatedState, progress_m: float, tangent_heading: float) -> bool:
        path_s = self._global_guidance.path_s
        if path_s is None or len(path_s) == 0:
            return False
        total_path_m = float(path_s[-1])
        if total_path_m < 1e-6:
            return False
        if progress_m / total_path_m < 0.9:
            return False
        goal_distance = self.distance_to_goal(state)
        near_goal_threshold = float(self.reward_config.near_goal_radius) * 1.5
        if goal_distance > near_goal_threshold:
            return False
        goal = self.get_goal_state()
        heading_diff = abs(wrap_to_pi(tangent_heading - float(goal.front_heading)))
        if heading_diff > math.radians(30.0):
            return False
        return True

    def _escape_reference_target(self, state: ArticulatedState) -> Optional[Tuple[Tuple[float, float], float]]:
        if not bool(self._guidance_available):
            return None
        guidance_reference = self._guidance_reference(state, lookahead_override_m=self._escape_progress_radius_m())
        if guidance_reference is None:
            return None
        goal_position, goal_heading, progress_m = guidance_reference
        if not self._use_escape_reference(state, progress_m):
            return None
        return goal_position, goal_heading

    def _guidance_reference(
        self,
        state: ArticulatedState,
        lookahead_override_m: Optional[float] = None,
    ) -> Optional[Tuple[Tuple[float, float], float, float]]:
        path_points = self._global_guidance.path_points_world
        path_s = self._global_guidance.path_s
        if path_points is None or path_s is None or len(path_points) < 2:
            return None

        point = np.array([float(state.x), float(state.y)], dtype=np.float64)
        lo = int(max(0, self._global_guidance.progress_idx - 2))
        hi = int(min(len(path_points), self._global_guidance.progress_idx + self._global_guidance.progress_search_window + 1))
        segment = path_points[lo:hi]
        if len(segment) == 0:
            segment = path_points
            lo = 0

        d2 = np.sum((segment - point) ** 2, axis=1)
        best_local_idx = int(np.argmin(d2))
        best_idx = int(max(self._global_guidance.progress_idx, lo + best_local_idx))
        self._global_guidance.progress_idx = best_idx

        progress_m = float(path_s[best_idx])
        path_end_m = float(path_s[-1])
        lookahead_m = self._guidance_lookahead_m(state)
        if lookahead_override_m is not None:
            lookahead_m = float(min(float(lookahead_m), max(float(lookahead_override_m), 0.0)))
        target_s = float(min(progress_m + lookahead_m, path_end_m))
        target_point = self._global_guidance._interp_on_path(target_s)
        if target_point is None:
            return None

        current_point = self._global_guidance._interp_on_path(progress_m)
        heading_vector = None
        if lookahead_override_m is None:
            half_window = 2.0 * float(self._global_guidance.grid_resolution)
            s0 = max(progress_m - half_window, 0.0)
            s1 = min(progress_m + half_window, path_end_m)
            p0 = self._global_guidance._interp_on_path(s0)
            p1 = self._global_guidance._interp_on_path(s1)
            if p0 is not None and p1 is not None and float(s1 - s0) > 1e-6:
                heading_vector = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
            else:
                heading_probe = self._global_guidance._interp_on_path(
                    float(min(progress_m + self._global_guidance.grid_resolution, path_end_m))
                )
                if heading_probe is not None:
                    heading_vector = np.asarray(heading_probe, dtype=np.float64) - np.asarray(current_point, dtype=np.float64)
        if heading_vector is None or float(np.linalg.norm(heading_vector)) < 1e-6:
            if current_point is not None:
                heading_vector = np.asarray(target_point, dtype=np.float64) - np.asarray(current_point, dtype=np.float64)
            else:
                heading_vector = None
        if heading_vector is None or float(np.linalg.norm(heading_vector)) < 1e-6:
            next_idx = min(best_idx + 1, len(path_points) - 1)
            prev_idx = max(best_idx - 1, 0)
            heading_vector = np.asarray(path_points[next_idx], dtype=np.float64) - np.asarray(path_points[prev_idx], dtype=np.float64)
        if float(np.linalg.norm(heading_vector)) < 1e-6:
            return None

        goal_heading = float(math.atan2(float(heading_vector[1]), float(heading_vector[0])))
        return (float(target_point[0]), float(target_point[1])), goal_heading, progress_m

    def _guidance_lookahead_m(self, state: ArticulatedState) -> float:
        lookahead = float(self._global_guidance.lookahead_base + self._global_guidance.lookahead_speed_gain * abs(float(state.speed)))
        return float(np.clip(lookahead, self._global_guidance.lookahead_min, self._global_guidance.lookahead_max))

    def _use_escape_reference(self, state: ArticulatedState, progress_m: float) -> bool:
        if self._scene is None:
            return False
        start = self._scene.start_state
        start_displacement = float(np.hypot(float(state.x - start.x), float(state.y - start.y)))
        escape_progress_m = self._escape_progress_radius_m()
        extended_escape_progress_m = 1.5 * escape_progress_m
        min_clearance = self._min_clearance_m(state)
        clearance_trigger = float(max(self._global_guidance.near_obs_dist_m, 0.5 * float(self.vehicle_config.body_width) + 0.35))
        low_clearance = min_clearance is not None and float(min_clearance) <= clearance_trigger
        near_start = bool(start_displacement <= escape_progress_m or progress_m <= escape_progress_m)
        extended_near_start = bool(start_displacement <= extended_escape_progress_m or progress_m <= extended_escape_progress_m)
        return bool(near_start or (low_clearance and extended_near_start))

    def _escape_progress_radius_m(self) -> float:
        return float(max(float(self.vehicle_config.front_length) + 0.5 * float(self.vehicle_config.rear_length), 2.0))

    def _min_clearance_m(self, state: ArticulatedState) -> Optional[float]:
        if self._scene is None:
            return None
        lidar_norm = np.asarray(self._lidar_observation(state), dtype=np.float64).reshape(-1)
        if lidar_norm.size == 0:
            return None
        return float(np.min(lidar_norm)) * max(float(self.observation_config.lidar_max_range), 1e-6)

    def _path_progress_m(self, state: ArticulatedState) -> Optional[float]:
        path_points = self._global_guidance.path_points_world
        path_s = self._global_guidance.path_s
        if path_points is None or path_s is None or len(path_points) < 2:
            return None

        point = np.array([float(state.x), float(state.y)], dtype=np.float64)
        lo = int(max(0, self._global_guidance.progress_idx - 2))
        hi = int(min(len(path_points), self._global_guidance.progress_idx + self._global_guidance.progress_search_window + 1))
        if hi - lo < 2:
            lo = 0
            hi = len(path_points)

        best_dist2 = float("inf")
        best_s = float(path_s[lo])
        for idx in range(lo, hi - 1):
            seg_start = np.asarray(path_points[idx], dtype=np.float64)
            seg_end = np.asarray(path_points[idx + 1], dtype=np.float64)
            seg_vec = seg_end - seg_start
            seg_len2 = float(np.dot(seg_vec, seg_vec))
            if seg_len2 < 1e-12:
                proj_dist2 = float(np.sum((point - seg_start) ** 2))
                if proj_dist2 < best_dist2:
                    best_dist2 = proj_dist2
                    best_s = float(path_s[idx])
                continue
            t = float(np.dot(point - seg_start, seg_vec) / seg_len2)
            t = float(np.clip(t, 0.0, 1.0))
            proj_point = seg_start + t * seg_vec
            proj_dist2 = float(np.sum((point - proj_point) ** 2))
            if proj_dist2 < best_dist2:
                best_dist2 = proj_dist2
                best_s = float(path_s[idx] + t * (float(path_s[idx + 1]) - float(path_s[idx])))
        return float(best_s)

    def predict_collision(self, state: ArticulatedState) -> bool:
        return bool(self._intersects_obstacles(state) or self._is_out_of_bounds(state))

    def _compute_reward(
        self,
        previous_state: ArticulatedState,
        next_state: ArticulatedState,
        previous_metrics,
        current_metrics,
        collision: bool,
        out_of_bounds: bool,
        success: bool,
        timeout: bool,
    ) -> Tuple[float, Dict[str, object]]:
        previous_goal_distance = float(np.hypot(float(self._goal_state.x - previous_state.x), float(self._goal_state.y - previous_state.y)))
        current_goal_distance = float(np.hypot(float(self._goal_state.x - next_state.x), float(self._goal_state.y - next_state.y)))
        distance_progress = float(previous_goal_distance - current_goal_distance)
        topology_progress, delta_j, progress_weight, progress_source, clearance_factor = self._progress_reward(
            previous_state=previous_state,
            next_state=next_state,
            current_goal_distance=current_goal_distance,
        )
        reference_heading = float(self._goal_state.front_heading)
        if self._active_reference_goal_heading is not None:
            reference_heading = float(self._active_reference_goal_heading)
        previous_heading_error = abs(wrap_to_pi(reference_heading - float(previous_state.front_heading))) / np.pi
        current_heading_error = abs(wrap_to_pi(reference_heading - float(next_state.front_heading))) / np.pi
        heading_progress = previous_heading_error - current_heading_error
        front_overlap_progress = float(current_metrics.front_overlap_ratio - previous_metrics.front_overlap_ratio)
        rear_overlap_progress = float(current_metrics.rear_overlap_ratio - previous_metrics.rear_overlap_ratio)
        overlap_progress = 0.75 * front_overlap_progress + 0.25 * rear_overlap_progress
        near_goal_radius = max(float(self.reward_config.near_goal_radius), 1e-6)
        near_goal_factor = float(np.clip(1.0 - min(previous_goal_distance, current_goal_distance) / near_goal_radius, 0.0, 1.0))
        shaped_heading_weight = float(self.reward_config.heading_weight) * float(1.0 + 0.5 * near_goal_factor)
        shaped_overlap_weight = float(self.reward_config.overlap_weight) * float(1.0 + near_goal_factor)
        reward = float(self.reward_config.step_penalty)
        reward += float(progress_weight) * float(topology_progress)
        reward += float(self.reward_config.distance_weight) * float(near_goal_factor) * float(distance_progress)
        reward += float(shaped_heading_weight) * float(heading_progress)
        reward += float(shaped_overlap_weight) * float(overlap_progress)
        terminal_bonus = 0.0
        if success:
            terminal_bonus += float(self.reward_config.success_reward)
        if collision:
            terminal_bonus += float(self.reward_config.collision_penalty)
        if out_of_bounds:
            terminal_bonus += float(self.reward_config.out_of_bounds_penalty)
        if timeout:
            terminal_bonus += float(self.reward_config.timeout_penalty)
        reward += terminal_bonus
        stage_prev = self._stage_potential(previous_state)
        stage_next = self._stage_potential(next_state)
        stage_info: Dict[str, object] = {}
        if stage_prev is not None and stage_next is not None:
            stage_info = {
                "stage_potential": float(stage_next["total"]),
                "stage_potential_delta": float(stage_next["total"]) - float(stage_prev["total"]),
                "stage_exit_progress": float(stage_next["exit"]),
                "stage_route_progress": float(stage_next["route"]),
                "stage_entry_progress": float(stage_next["entry"]),
            }
        return float(reward), {
            "topology_progress": float(topology_progress),
            "topology_delta_j": float(delta_j),
            "progress_weight": float(progress_weight),
            "distance_progress": float(distance_progress),
            "near_goal_factor": float(near_goal_factor),
            "heading_progress": float(heading_progress),
            "front_overlap_progress": float(front_overlap_progress),
            "rear_overlap_progress": float(rear_overlap_progress),
            "overlap_progress": float(overlap_progress),
            "distance_weight": float(self.reward_config.distance_weight),
            "heading_weight": float(shaped_heading_weight),
            "overlap_weight": float(shaped_overlap_weight),
            "clearance_factor": float(clearance_factor),
            "step_penalty": float(self.reward_config.step_penalty),
            "terminal_bonus": float(terminal_bonus),
            "progress_source": str(progress_source),
            **stage_info,
        }

    def _metadata_point(self, key: str) -> Optional[Tuple[float, float]]:
        if self._scene is None:
            return None
        value = self._scene.metadata.get(str(key))
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.shape[0] < 2 or not bool(np.all(np.isfinite(arr[:2]))):
            return None
        return float(arr[0]), float(arr[1])

    def _metadata_float(self, key: str) -> Optional[float]:
        if self._scene is None:
            return None
        value = self._scene.metadata.get(str(key))
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return float(parsed)

    @staticmethod
    def _distance_to_point(state: ArticulatedState, point: Tuple[float, float]) -> float:
        return float(math.hypot(float(state.x) - float(point[0]), float(state.y) - float(point[1])))

    def _bay_stage_span(
        self,
        metadata_key: str,
        endpoint: ArticulatedState,
        mouth: Tuple[float, float],
    ) -> float:
        configured = self._metadata_float(metadata_key)
        if configured is not None and configured > 1e-6:
            return float(configured)
        return float(max(self._distance_to_point(endpoint, mouth), 1e-6))

    @staticmethod
    def _stage_progress_from_distance(distance_m: float, span_m: float) -> float:
        span = max(float(span_m), 1e-6)
        return float(np.clip(span - float(distance_m), 0.0, span))

    def _stage_potential(self, state: ArticulatedState) -> Optional[Dict[str, object]]:
        if self._scene is None or self._goal_state is None:
            return None

        metadata = self._scene.metadata
        start_role = str(metadata.get("start_role", "")).strip().lower()
        goal_role = str(metadata.get("goal_role", "")).strip().lower()
        start_mouth = self._metadata_point("start_bay_mouth")
        goal_mouth = self._metadata_point("goal_bay_mouth")
        if start_mouth is None and start_role == "bay":
            start_mouth = self._metadata_point("bay_mouth")
        if goal_mouth is None and goal_role == "bay":
            goal_mouth = self._metadata_point("bay_mouth")

        active = False
        exit_progress = 0.0
        if start_role == "bay" and start_mouth is not None:
            exit_span = self._bay_stage_span("bay_exit_progress_m", self._scene.start_state, start_mouth)
            exit_progress = self._stage_progress_from_distance(self._distance_to_point(state, start_mouth), exit_span)
            active = True

        route_progress = 0.0
        route_mode = "unavailable"
        if self._initial_topology_cost is not None:
            current_cost, route_mode = self._global_guidance.query_cost_to_go_details(float(state.x), float(state.y))
            if current_cost is not None:
                route_progress = max(0.0, float(self._initial_topology_cost) - float(current_cost))
        if route_progress <= 0.0:
            path_progress = self._path_progress_m(state)
            if path_progress is not None:
                route_progress = max(0.0, float(path_progress))
                route_mode = "path"

        entry_progress = 0.0
        if goal_role == "bay" and goal_mouth is not None:
            entry_span = self._bay_stage_span("bay_entry_progress_m", self._goal_state, goal_mouth)
            entry_progress = self._stage_progress_from_distance(float(self.distance_to_goal(state)), entry_span)
            active = True

        if not active:
            return None
        total = float(exit_progress + route_progress + entry_progress)
        return {
            "total": total,
            "exit": float(exit_progress),
            "route": float(route_progress),
            "entry": float(entry_progress),
            "route_mode": str(route_mode),
        }

    @staticmethod
    def _stage_progress_source(previous: Dict[str, object], current: Dict[str, object]) -> str:
        exit_delta = float(current["exit"]) - float(previous["exit"])
        entry_delta = float(current["entry"]) - float(previous["entry"])
        route_delta = float(current["route"]) - float(previous["route"])
        if exit_delta > 1e-9:
            return "bay_exit_potential"
        if entry_delta > 1e-9:
            return "bay_entry_potential"
        deltas = {
            "stage_route": route_delta,
            "bay_exit_potential": exit_delta,
            "bay_entry_potential": entry_delta,
        }
        positive = {key: value for key, value in deltas.items() if value > 1e-9}
        if positive:
            source = max(positive.items(), key=lambda item: item[1])[0]
        else:
            source = max(deltas.items(), key=lambda item: item[1])[0]
        if source == "stage_route":
            if str(previous.get("route_mode")) == "projected" or str(current.get("route_mode")) == "projected":
                return "projected_stage_route"
            return "stage_route"
        return source

    def _progress_reward(
        self,
        previous_state: ArticulatedState,
        next_state: ArticulatedState,
        current_goal_distance: float,
    ) -> Tuple[float, float, float, str, float]:
        base_weight = float(self.reward_config.progress_weight)
        reverse_coef = float(self.reward_config.reverse_penalty_coef)
        sigma = max(float(self.reward_config.topology_sigma), 1e-6)

        stage_prev = self._stage_potential(previous_state)
        stage_next = self._stage_potential(next_state)
        if stage_prev is not None and stage_next is not None:
            prev_total = float(stage_prev["total"])
            next_total = float(stage_next["total"])
            if self._best_rewarded_stage_potential is None:
                self._best_rewarded_stage_potential = prev_total
            best_stage = max(float(self._best_rewarded_stage_potential), prev_total)
            frontier_delta = max(0.0, next_total - best_stage)
            if next_total > best_stage + 1e-9:
                self._best_rewarded_stage_potential = next_total
            raw_delta = float(next_total - prev_total)
            forward = float(math.tanh(frontier_delta / sigma))
            regression = float(reverse_coef * min(0.0, float(math.tanh(raw_delta / sigma))))
            progress_source = self._stage_progress_source(stage_prev, stage_next)
            weight_scale = (
                float(self._PROJECTED_TOPOLOGY_WEIGHT_SCALE)
                if progress_source == "projected_stage_route"
                else 1.0
            )
            return forward + regression, raw_delta, base_weight * weight_scale, progress_source, 1.0

        j_prev, prev_mode = self._global_guidance.query_cost_to_go_details(float(previous_state.x), float(previous_state.y))
        j_next, next_mode = self._global_guidance.query_cost_to_go_details(float(next_state.x), float(next_state.y))
        if j_prev is not None and j_next is not None:
            if self._best_rewarded_topology_cost is None:
                self._best_rewarded_topology_cost = float(j_prev)
            best_topology_cost = min(float(self._best_rewarded_topology_cost), float(j_prev))
            frontier_delta = max(0.0, best_topology_cost - float(j_next))
            if float(j_next) + 1e-9 < best_topology_cost:
                self._best_rewarded_topology_cost = float(j_next)
            raw_delta_j = float(j_prev - j_next)
            forward = float(math.tanh(frontier_delta / sigma))
            regression = float(reverse_coef * min(0.0, float(math.tanh(raw_delta_j / sigma))))
            topology_progress = forward + regression
            weight_scale = (
                1.0
                if prev_mode == "direct" and next_mode == "direct"
                else float(self._PROJECTED_TOPOLOGY_WEIGHT_SCALE)
            )
            progress_source = "topology" if weight_scale >= 1.0 else "projected_topology"
            return topology_progress, float(raw_delta_j), base_weight * weight_scale, progress_source, 1.0

        previous_goal_distance = float(self.distance_to_goal(previous_state))
        raw_distance_delta = float(previous_goal_distance - current_goal_distance)
        distance_frontier = self._distance_frontier_improvement(current_goal_distance)
        regression = float(reverse_coef * min(0.0, raw_distance_delta / sigma))
        clearance_factor = self._local_clearance_factor(next_state)
        if clearance_factor is not None:
            forward = float(distance_frontier * clearance_factor)
            topology_progress = forward + float(regression * clearance_factor)
            return (
                float(topology_progress),
                0.0,
                base_weight * float(self._CLEARANCE_DISTANCE_WEIGHT_SCALE),
                "clearance_distance",
                float(clearance_factor),
            )
        forward = float(distance_frontier)
        topology_progress = forward + regression
        return (
            float(topology_progress),
            0.0,
            base_weight * float(self._EUCLIDEAN_DISTANCE_WEIGHT_SCALE),
            "euclidean_distance",
            1.0,
        )

    def _distance_frontier_improvement(self, current_goal_distance: float) -> float:
        if self._best_rewarded_goal_distance is None:
            self._best_rewarded_goal_distance = float(current_goal_distance)
            return 0.0
        best_goal_distance = float(self._best_rewarded_goal_distance)
        improvement = max(0.0, best_goal_distance - float(current_goal_distance))
        if float(current_goal_distance) + 1e-9 < best_goal_distance:
            self._best_rewarded_goal_distance = float(current_goal_distance)
        return float(improvement)

    def _local_clearance_factor(self, state: ArticulatedState) -> Optional[float]:
        if self._scene is None:
            return None
        lidar_norm = np.asarray(self._lidar_observation(state), dtype=np.float64).reshape(-1)
        if lidar_norm.size == 0:
            return None
        lidar_range = float(self.observation_config.lidar_max_range)
        lidar_values = np.clip(lidar_norm, 0.0, 1.0) * lidar_range
        min_lidar = float(np.min(lidar_values))
        dense_ratio = float(np.mean(lidar_values < float(self._global_guidance.near_obs_dist_m)))
        clear_span = max(1e-6, float(self._global_guidance.full_clearance_m) - float(self._global_guidance.min_clearance_m))
        clear_factor = float(
            np.clip((min_lidar - float(self._global_guidance.min_clearance_m)) / clear_span, 0.0, 1.0)
        )
        dense_factor = float(
            1.0
            - np.clip(
                dense_ratio / max(1e-6, float(self._global_guidance.max_dense_ratio)),
                0.0,
                1.0,
            )
        )
        return float(np.clip(clear_factor * dense_factor, 0.0, 1.0))

    def _build_info(
        self,
        collision: bool,
        goal_reached: bool,
        terminated: bool,
        truncated: bool,
        done_reason: str,
        reward_info: Dict[str, object],
        success_metrics=None,
    ) -> Dict[str, object]:
        scene_metadata = {} if self._scene is None else dict(self._scene.metadata)
        info = {
            "collision": bool(collision),
            "goal_reached": bool(goal_reached),
            "done": bool(terminated or truncated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done_reason": str(done_reason),
            "reward_info": dict(reward_info),
            "level": None if self._scene is None else str(self._scene.level),
            "step_count": int(self._step_count),
            "guidance_available": bool(self._guidance_available),
            "guidance_path_confidence": float(self._global_guidance.path_confidence),
            "escape_reference_enabled": bool(self.env_config.escape_reference_enabled),
            "reference_override_active": bool(
                self._reference_override_goal_position is not None and self._reference_override_goal_heading is not None
            ),
            "scene_metadata": scene_metadata,
            "scene_type": scene_metadata.get("scene_type"),
            "corridor_width": scene_metadata.get("corridor_width"),
            "corridor_min_width": scene_metadata.get("corridor_min_width"),
            "warmup_progress": scene_metadata.get("warmup_progress"),
        }
        if success_metrics is not None:
            info.update(
                {
                    "front_overlap_ratio": float(success_metrics.front_overlap_ratio),
                    "rear_overlap_ratio": float(success_metrics.rear_overlap_ratio),
                    "heading_error_rad": float(success_metrics.heading_error_rad),
                    "success": bool(success_metrics.success),
                }
            )
        return info

    def _guidance_observation(self, lidar: np.ndarray) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("environment must be reset before observation generation")
        if not self._guidance_available:
            return np.zeros((GUIDANCE_FEATURE_DIM,), dtype=np.float32)
        return self._global_guidance.get_soft_hint(
            state_x=float(self._state.x),
            state_y=float(self._state.y),
            heading=float(self._state.front_heading),
            speed=float(self._state.speed),
            lidar_norm=np.asarray(lidar, dtype=np.float32),
            lidar_range=float(self.observation_config.lidar_max_range),
        )

    def _lidar_observation(self, state: ArticulatedState) -> np.ndarray:
        beam_count = int(self.observation_config.lidar_num_beams)
        max_range = float(self.observation_config.lidar_max_range)
        headings = float(state.front_heading) + self._beam_angle_offsets
        ray_dx = np.cos(headings)
        ray_dy = np.sin(headings)
        bound_distance = self._distance_to_scene_bounds(float(state.x), float(state.y), ray_dx, ray_dy, max_range)
        obstacle_distance = self._distance_to_obstacle_segments(float(state.x), float(state.y), ray_dx, ray_dy, max_range)
        values = np.minimum(bound_distance, obstacle_distance)
        if values.shape[0] != beam_count:
            raise RuntimeError("lidar distance computation returned unexpected beam count")
        return (values / max(max_range, 1e-6)).astype(np.float32)

    def _cache_obstacle_segments(self) -> None:
        if self._scene is None or len(self._scene.obstacles) == 0:
            self._obstacle_segment_x1 = np.empty((0,), dtype=np.float64)
            self._obstacle_segment_y1 = np.empty((0,), dtype=np.float64)
            self._obstacle_segment_dx = np.empty((0,), dtype=np.float64)
            self._obstacle_segment_dy = np.empty((0,), dtype=np.float64)
            return

        x1s = []
        y1s = []
        x2s = []
        y2s = []
        for polygon in self._scene.obstacles:
            self._append_ring_segments(np.asarray(polygon.exterior.coords, dtype=np.float64), x1s, y1s, x2s, y2s)
            for interior in polygon.interiors:
                self._append_ring_segments(np.asarray(interior.coords, dtype=np.float64), x1s, y1s, x2s, y2s)

        if len(x1s) == 0:
            self._obstacle_segment_x1 = np.empty((0,), dtype=np.float64)
            self._obstacle_segment_y1 = np.empty((0,), dtype=np.float64)
            self._obstacle_segment_dx = np.empty((0,), dtype=np.float64)
            self._obstacle_segment_dy = np.empty((0,), dtype=np.float64)
            return

        self._obstacle_segment_x1 = np.asarray(x1s, dtype=np.float64)
        self._obstacle_segment_y1 = np.asarray(y1s, dtype=np.float64)
        self._obstacle_segment_dx = np.asarray(x2s, dtype=np.float64) - self._obstacle_segment_x1
        self._obstacle_segment_dy = np.asarray(y2s, dtype=np.float64) - self._obstacle_segment_y1

    @staticmethod
    def _append_ring_segments(
        coords: np.ndarray,
        x1s: list,
        y1s: list,
        x2s: list,
        y2s: list,
    ) -> None:
        if coords.ndim != 2 or coords.shape[0] < 2:
            return
        x1s.extend(coords[:-1, 0].tolist())
        y1s.extend(coords[:-1, 1].tolist())
        x2s.extend(coords[1:, 0].tolist())
        y2s.extend(coords[1:, 1].tolist())

    def _distance_to_scene_bounds(
        self,
        x: float,
        y: float,
        ray_dx: np.ndarray,
        ray_dy: np.ndarray,
        max_range: float,
    ) -> np.ndarray:
        if self._scene is None:
            return np.full(ray_dx.shape, float(max_range), dtype=np.float64)
        xmin, xmax, ymin, ymax = self._scene.world_bounds
        epsilon = 1e-9
        tolerance = 1e-6
        tx = np.full(ray_dx.shape, np.inf, dtype=np.float64)
        positive_x = ray_dx > epsilon
        negative_x = ray_dx < -epsilon
        tx[positive_x] = (float(xmax) - x) / ray_dx[positive_x]
        tx[negative_x] = (float(xmin) - x) / ray_dx[negative_x]
        y_hit = y + tx * ray_dy
        invalid_tx = (tx <= 0.0) | (y_hit < float(ymin) - tolerance) | (y_hit > float(ymax) + tolerance)
        tx[invalid_tx] = np.inf

        ty = np.full(ray_dy.shape, np.inf, dtype=np.float64)
        positive_y = ray_dy > epsilon
        negative_y = ray_dy < -epsilon
        ty[positive_y] = (float(ymax) - y) / ray_dy[positive_y]
        ty[negative_y] = (float(ymin) - y) / ray_dy[negative_y]
        x_hit = x + ty * ray_dx
        invalid_ty = (ty <= 0.0) | (x_hit < float(xmin) - tolerance) | (x_hit > float(xmax) + tolerance)
        ty[invalid_ty] = np.inf

        distances = np.minimum(tx, ty)
        distances[~np.isfinite(distances)] = float(max_range)
        return np.minimum(distances, float(max_range))

    def _distance_to_obstacle_segments(
        self,
        x: float,
        y: float,
        ray_dx: np.ndarray,
        ray_dy: np.ndarray,
        max_range: float,
    ) -> np.ndarray:
        if self._obstacle_segment_x1.size == 0:
            return np.full(ray_dx.shape, float(max_range), dtype=np.float64)

        epsilon = 1e-9
        qmp_x = self._obstacle_segment_x1.reshape(1, -1) - float(x)
        qmp_y = self._obstacle_segment_y1.reshape(1, -1) - float(y)
        seg_dx = self._obstacle_segment_dx.reshape(1, -1)
        seg_dy = self._obstacle_segment_dy.reshape(1, -1)
        ray_dx_2d = ray_dx.reshape(-1, 1)
        ray_dy_2d = ray_dy.reshape(-1, 1)

        determinant = ray_dx_2d * seg_dy - ray_dy_2d * seg_dx
        parallel = np.abs(determinant) <= epsilon
        safe_determinant = np.where(parallel, 1.0, determinant)
        distance_along_ray = (qmp_x * seg_dy - qmp_y * seg_dx) / safe_determinant
        edge_position = (qmp_x * ray_dy_2d - qmp_y * ray_dx_2d) / safe_determinant
        valid = (
            (~parallel)
            & (distance_along_ray >= 0.0)
            & (distance_along_ray <= float(max_range))
            & (edge_position >= -epsilon)
            & (edge_position <= 1.0 + epsilon)
        )
        distance_along_ray = np.where(valid, distance_along_ray, np.inf)
        distances = np.min(distance_along_ray, axis=1)
        distances[~np.isfinite(distances)] = float(max_range)
        return distances

    def _intersects_obstacles(self, state: ArticulatedState) -> bool:
        if self._obstacle_union is None:
            return False
        front_poly, rear_poly = articulated_body_polygons(state, self.vehicle_config)
        return bool(front_poly.intersects(self._obstacle_union) or rear_poly.intersects(self._obstacle_union))

    def _is_out_of_bounds(self, state: ArticulatedState) -> bool:
        front_poly, rear_poly = articulated_body_polygons(state, self.vehicle_config)
        return not bool(self._world_box.covers(front_poly) and self._world_box.covers(rear_poly))
