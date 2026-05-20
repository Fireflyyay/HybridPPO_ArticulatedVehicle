from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from common.config import VehicleConfig
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.types import ArticulatedState, LowLevelControl, PrimitiveExecutionContext, wrap_to_pi
from env.dynamics import ArticulatedKinematics
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
        self._step_count = 0
        self._last_info: Dict[str, object] = {}

    @property
    def observation_dim(self) -> int:
        return int(self.observation_config.observation_dim)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        level = str((options or {}).get("level", self.env_config.default_level))
        self._scene = self.scene_factory.generate(level, self._rng)
        self._state = self._scene.start_state
        self._goal_state = self._scene.goal_state
        self._obstacle_union = None if len(self._scene.obstacles) == 0 else unary_union(list(self._scene.obstacles))
        xmin, xmax, ymin, ymax = self._scene.world_bounds
        self._world_box = box(xmin, ymin, xmax, ymax)
        self._step_count = 0
        self._last_info = self._build_info(
            collision=False,
            goal_reached=False,
            terminated=False,
            truncated=False,
            done_reason="reset",
            reward_info={},
        )
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
        return np.concatenate([lidar, features], axis=0).astype(np.float32)

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

    def make_primitive_context(self) -> PrimitiveExecutionContext:
        goal = self.get_goal_state()
        return PrimitiveExecutionContext(
            goal_position=(float(goal.x), float(goal.y)),
            goal_heading=float(goal.front_heading),
            collision_checker=self.predict_collision,
        )

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
    ) -> Tuple[float, Dict[str, float]]:
        progress = (self.distance_to_goal(previous_state) - self.distance_to_goal(next_state)) / max(float(self.observation_config.goal_distance_scale), 1e-6)
        previous_heading_error = abs(wrap_to_pi(float(self._goal_state.front_heading) - float(previous_state.front_heading))) / np.pi
        current_heading_error = abs(wrap_to_pi(float(self._goal_state.front_heading) - float(next_state.front_heading))) / np.pi
        heading_progress = previous_heading_error - current_heading_error
        overlap_progress = float(current_metrics.front_overlap_ratio - previous_metrics.front_overlap_ratio)
        reward = float(self.reward_config.step_penalty)
        reward += float(self.reward_config.progress_weight) * float(progress)
        reward += float(self.reward_config.heading_weight) * float(heading_progress)
        reward += float(self.reward_config.overlap_weight) * float(overlap_progress)
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
        return float(reward), {
            "progress": float(progress),
            "heading_progress": float(heading_progress),
            "overlap_progress": float(overlap_progress),
            "step_penalty": float(self.reward_config.step_penalty),
            "terminal_bonus": float(terminal_bonus),
        }

    def _build_info(
        self,
        collision: bool,
        goal_reached: bool,
        terminated: bool,
        truncated: bool,
        done_reason: str,
        reward_info: Dict[str, float],
        success_metrics=None,
    ) -> Dict[str, object]:
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

    def _lidar_observation(self, state: ArticulatedState) -> np.ndarray:
        beam_count = int(self.observation_config.lidar_num_beams)
        max_range = float(self.observation_config.lidar_max_range)
        origin = Point(float(state.x), float(state.y))
        values = np.full((beam_count,), max_range, dtype=np.float32)
        beam_angles = np.linspace(-np.pi, np.pi, beam_count, endpoint=False)
        for index, beam_angle in enumerate(beam_angles):
            heading = float(state.front_heading + beam_angle)
            bound_distance = self._distance_to_bounds(float(state.x), float(state.y), heading, max_range)
            obstacle_distance = self._distance_to_obstacles(origin, heading, max_range)
            values[index] = float(min(max_range, bound_distance, obstacle_distance) / max(max_range, 1e-6))
        return values.astype(np.float32)

    def _distance_to_bounds(self, x: float, y: float, heading: float, max_range: float) -> float:
        if self._scene is None:
            return float(max_range)
        xmin, xmax, ymin, ymax = self._scene.world_bounds
        cos_heading = float(np.cos(heading))
        sin_heading = float(np.sin(heading))
        distances = []
        if abs(cos_heading) > 1e-9:
            tx = (xmax - x) / cos_heading if cos_heading > 0.0 else (xmin - x) / cos_heading
            if tx > 0.0:
                y_hit = y + tx * sin_heading
                if ymin - 1e-6 <= y_hit <= ymax + 1e-6:
                    distances.append(float(tx))
        if abs(sin_heading) > 1e-9:
            ty = (ymax - y) / sin_heading if sin_heading > 0.0 else (ymin - y) / sin_heading
            if ty > 0.0:
                x_hit = x + ty * cos_heading
                if xmin - 1e-6 <= x_hit <= xmax + 1e-6:
                    distances.append(float(ty))
        return float(min(distances) if distances else max_range)

    def _distance_to_obstacles(self, origin: Point, heading: float, max_range: float) -> float:
        if self._obstacle_union is None:
            return float(max_range)
        ray = LineString(
            [
                (float(origin.x), float(origin.y)),
                (float(origin.x + max_range * np.cos(heading)), float(origin.y + max_range * np.sin(heading))),
            ]
        )
        intersection = ray.intersection(self._obstacle_union)
        if intersection.is_empty:
            return float(max_range)
        return float(min(max_range, origin.distance(intersection)))

    def _intersects_obstacles(self, state: ArticulatedState) -> bool:
        if self._obstacle_union is None:
            return False
        front_poly, rear_poly = articulated_body_polygons(state, self.vehicle_config)
        return bool(front_poly.intersects(self._obstacle_union) or rear_poly.intersects(self._obstacle_union))

    def _is_out_of_bounds(self, state: ArticulatedState) -> bool:
        front_poly, rear_poly = articulated_body_polygons(state, self.vehicle_config)
        return not bool(self._world_box.covers(front_poly) and self._world_box.covers(rear_poly))