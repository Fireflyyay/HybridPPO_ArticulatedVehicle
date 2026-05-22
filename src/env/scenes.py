from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from common.config import VehicleConfig
from common.runtime_config import SceneLevelConfig
from common.types import ArticulatedState, wrap_to_pi
from env.success import articulated_body_polygons


@dataclass(frozen=True)
class SceneSpec:
    level: str
    world_bounds: Tuple[float, float, float, float]
    obstacles: Tuple[Polygon, ...]
    start_state: ArticulatedState
    goal_state: ArticulatedState
    metadata: Dict[str, object]


@dataclass(frozen=True)
class _PoseCandidate:
    x: float
    y: float
    heading: float


def _extract_polygons(geometry) -> List[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    polygons: List[Polygon] = []
    for sub_geometry in getattr(geometry, "geoms", []):
        polygons.extend(_extract_polygons(sub_geometry))
    return polygons


def _transform_polygon(polygon: Polygon, rotation_deg: float, dx: float, dy: float) -> Polygon:
    return translate(rotate(polygon, rotation_deg, origin=(0.0, 0.0), use_radians=False), xoff=dx, yoff=dy)


def _transform_state(state: ArticulatedState, rotation_deg: float, dx: float, dy: float) -> ArticulatedState:
    theta = float(np.deg2rad(rotation_deg))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    x = cos_theta * float(state.x) - sin_theta * float(state.y) + float(dx)
    y = sin_theta * float(state.x) + cos_theta * float(state.y) + float(dy)
    heading = float(state.front_heading + theta)
    rear_heading = float(state.rear_heading + theta)
    return ArticulatedState(
        x=x,
        y=y,
        front_heading=heading,
        rear_heading=rear_heading,
        speed=float(state.speed),
        articulation_rate=float(state.articulation_rate),
    )


def _sample_distance(rng: np.random.Generator, low_high: Tuple[float, float]) -> float:
    low, high = float(low_high[0]), float(low_high[1])
    if high < low:
        low, high = high, low
    return float(rng.uniform(low, high))


def _warmup_progress(options: Optional[Mapping[str, object]]) -> Optional[float]:
    if not options or "warmup_progress" not in options:
        return None
    return float(options["warmup_progress"])


class BaselineInspiredSceneFactory:
    def __init__(
        self,
        presets: Union[Mapping[str, SceneLevelConfig], Sequence[Tuple[str, SceneLevelConfig]]],
        vehicle_config: Optional[VehicleConfig] = None,
    ) -> None:
        self.presets = dict(presets)
        self.vehicle_config = vehicle_config or VehicleConfig()

    def _front_reach(self) -> float:
        return max(0.0, float(self.vehicle_config.front_length) - float(self.vehicle_config.hitch_offset))

    def _rear_reach(self) -> float:
        return max(0.0, float(self.vehicle_config.hitch_offset) + float(self.vehicle_config.rear_length))

    def _candidate_state(self, candidate: _PoseCandidate) -> ArticulatedState:
        return ArticulatedState(candidate.x, candidate.y, candidate.heading, candidate.heading)

    def _filter_valid_candidates(
        self,
        candidates: Iterable[_PoseCandidate],
        obstacles: Sequence[Polygon],
        world_bounds: Tuple[float, float, float, float],
    ) -> Tuple[_PoseCandidate, ...]:
        xmin, xmax, ymin, ymax = world_bounds
        world_box = box(float(xmin), float(ymin), float(xmax), float(ymax))
        obstacle_union = None if len(obstacles) == 0 else unary_union(list(obstacles))
        valid: List[_PoseCandidate] = []
        for candidate in candidates:
            front_poly, rear_poly = articulated_body_polygons(self._candidate_state(candidate), self.vehicle_config)
            if not bool(world_box.covers(front_poly) and world_box.covers(rear_poly)):
                continue
            if obstacle_union is not None and bool(front_poly.intersects(obstacle_union) or rear_poly.intersects(obstacle_union)):
                continue
            valid.append(candidate)
        return tuple(valid)

    def generate(
        self,
        level: str,
        rng: np.random.Generator,
        options: Optional[Mapping[str, object]] = None,
    ) -> SceneSpec:
        level_name = str(level)
        if level_name not in self.presets:
            raise KeyError(f"unknown level: {level_name}")
        config = self.presets[level_name]
        if level_name == "Debug":
            return self._generate_debug(level_name, config, rng)
        if level_name == "Warmup":
            return self._generate_warmup(level_name, config, rng, options=options)
        return self._generate_block_mixing(level_name, config, rng)

    def _generate_debug(self, level: str, config: SceneLevelConfig, rng: np.random.Generator) -> SceneSpec:
        world_min = float(config.world_min)
        world_max = float(config.world_max)
        margin = float(config.boundary_margin)
        for _ in range(256):
            start_x = float(rng.uniform(world_min + margin, world_max - margin))
            start_y = float(rng.uniform(world_min + margin, world_max - margin))
            goal_x = float(rng.uniform(world_min + margin, world_max - margin))
            goal_y = float(rng.uniform(world_min + margin, world_max - margin))
            dist = float(np.hypot(goal_x - start_x, goal_y - start_y))
            low, high = config.pair_distance_range
            if dist < float(low) or dist > float(high):
                continue
            path_heading = float(np.arctan2(goal_y - start_y, goal_x - start_x))
            jitter = float(config.heading_jitter_rad)
            start_heading = path_heading + float(rng.uniform(-jitter, jitter))
            goal_heading = path_heading + float(rng.uniform(-jitter, jitter))
            return SceneSpec(
                level=level,
                world_bounds=(world_min, world_max, world_min, world_max),
                obstacles=tuple(),
                start_state=ArticulatedState(start_x, start_y, start_heading, start_heading),
                goal_state=ArticulatedState(goal_x, goal_y, goal_heading, goal_heading),
                metadata={"scene_type": "debug_blank"},
            )
        raise RuntimeError("failed to sample debug scene")

    def _generate_warmup(
        self,
        level: str,
        config: SceneLevelConfig,
        rng: np.random.Generator,
        options: Optional[Mapping[str, object]] = None,
    ) -> SceneSpec:
        corridor_width = float(config.resolve_warmup_corridor_width(_warmup_progress(options)))
        half_width = 0.5 * corridor_width
        bay_depth = 12.0
        corridor = box(-30.0, -half_width, 14.0, half_width)
        bay = box(8.0, half_width, 20.0, half_width + bay_depth)
        free_space = unary_union([corridor, bay]).buffer(0)
        world = box(config.world_min, config.world_min, config.world_max, config.world_max)
        obstacles = tuple(_extract_polygons(world.difference(free_space).buffer(0)))
        head_clearance = float(config.parking_head_wall_clearance)
        start_state = ArticulatedState(x=-12.0, y=0.0, front_heading=0.0, rear_heading=0.0)
        goal_state = ArticulatedState(
            x=14.0,
            y=half_width + bay_depth - head_clearance - self._front_reach(),
            front_heading=float(np.pi / 2.0),
            rear_heading=float(np.pi / 2.0),
        )
        world_bounds = (float(config.world_min), float(config.world_max), float(config.world_min), float(config.world_max))

        for _ in range(64):
            rotation_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
            dx = float(rng.uniform(-4.0, 4.0))
            dy = float(rng.uniform(-4.0, 4.0))
            transformed_obstacles = tuple(_transform_polygon(poly, rotation_deg, dx, dy) for poly in obstacles)
            transformed_start = _transform_state(start_state, rotation_deg, dx, dy)
            transformed_goal = _transform_state(goal_state, rotation_deg, dx, dy)
            valid = self._filter_valid_candidates(
                [
                    _PoseCandidate(transformed_start.x, transformed_start.y, transformed_start.front_heading),
                    _PoseCandidate(transformed_goal.x, transformed_goal.y, transformed_goal.front_heading),
                ],
                transformed_obstacles,
                world_bounds,
            )
            if len(valid) != 2:
                continue
            return SceneSpec(
                level=level,
                world_bounds=world_bounds,
                obstacles=transformed_obstacles,
                start_state=transformed_start,
                goal_state=transformed_goal,
                metadata={
                    "scene_type": "warmup_bay",
                    "aligned_to": "ppo_articulated_vehicle",
                    "corridor_width": float(corridor_width),
                    "corridor_min_width": float(config.warmup_corridor_min_width()),
                    "warmup_progress": _warmup_progress(options),
                },
            )
        raise RuntimeError("failed to sample warmup scene")

    def _generate_block_mixing(self, level: str, config: SceneLevelConfig, rng: np.random.Generator) -> SceneSpec:
        world_min = float(config.world_min)
        world_max = float(config.world_max)
        margin = float(config.boundary_margin)
        rear_clearance = self._rear_reach() + 0.25
        front_clearance = self._front_reach() + float(config.parking_head_wall_clearance)
        world_bounds = (world_min, world_max, world_min, world_max)

        for _ in range(128):
            world = box(world_min, world_min, world_max, world_max)
            corridor_width = float(rng.integers(config.corridor_width_range[0], config.corridor_width_range[1] + 1))
            center_x = float(rng.uniform(world_min * 0.2, world_max * 0.2))
            free_shapes: List[Polygon] = []
            candidates: List[_PoseCandidate] = []

            main_corridor = box(center_x - corridor_width * 0.5, world_min + margin, center_x + corridor_width * 0.5, world_max - margin)
            free_shapes.append(main_corridor)
            candidates.extend(
                [
                    _PoseCandidate(center_x, world_min + margin + rear_clearance, float(np.pi / 2.0)),
                    _PoseCandidate(center_x, world_max - margin - rear_clearance, float(-np.pi / 2.0)),
                ]
            )

            branch_count = int(rng.integers(config.branch_count_range[0], config.branch_count_range[1] + 1))
            bay_count = int(rng.integers(config.parking_bay_count_range[0], config.parking_bay_count_range[1] + 1))
            branch_ys = rng.uniform(world_min + margin + 8.0, world_max - margin - 8.0, size=max(1, branch_count))

            for branch_y in branch_ys[:branch_count]:
                direction = int(rng.choice([-1, 1]))
                length = float(_sample_distance(rng, config.segment_length_range))
                end_x = float(np.clip(center_x + direction * length, world_min + margin + 4.0, world_max - margin - 4.0))
                branch = box(min(center_x, end_x), branch_y - corridor_width * 0.5, max(center_x, end_x), branch_y + corridor_width * 0.5)
                free_shapes.append(branch)
                heading = 0.0 if direction > 0 else float(np.pi)
                candidates.append(_PoseCandidate(end_x - direction * front_clearance, branch_y, heading))

                if bay_count > 0:
                    bay_count -= 1
                    bay_length = float(_sample_distance(rng, config.parking_bay_length_range))
                    bay_depth = float(_sample_distance(rng, config.parking_bay_depth_range))
                    bay_dir = int(rng.choice([-1, 1]))
                    bay = box(
                        end_x - bay_length * 0.5,
                        branch_y,
                        end_x + bay_length * 0.5,
                        branch_y + bay_dir * bay_depth,
                    ) if bay_dir > 0 else box(
                        end_x - bay_length * 0.5,
                        branch_y + bay_dir * bay_depth,
                        end_x + bay_length * 0.5,
                        branch_y,
                    )
                    free_shapes.append(bay)
                    bay_heading = float(np.pi / 2.0) if bay_dir > 0 else float(-np.pi / 2.0)
                    bay_goal_y = branch_y + bay_dir * max(front_clearance, bay_depth - front_clearance)
                    candidates.append(_PoseCandidate(end_x, bay_goal_y, bay_heading))

            free_space = unary_union(free_shapes).buffer(0)
            obstacles = tuple(_extract_polygons(world.difference(free_space).buffer(0)))
            valid_candidates = self._filter_valid_candidates(candidates, obstacles, world_bounds)
            if len(valid_candidates) < 2:
                continue
            try:
                start, goal = self._sample_pose_pair(
                    rng,
                    valid_candidates,
                    config.pair_distance_range,
                    config.pair_heading_diff_range_deg,
                )
            except RuntimeError:
                continue
            return SceneSpec(
                level=level,
                world_bounds=world_bounds,
                obstacles=obstacles,
                start_state=ArticulatedState(start.x, start.y, start.heading, start.heading),
                goal_state=ArticulatedState(goal.x, goal.y, goal.heading, goal.heading),
                metadata={
                    "scene_type": "block_mixing_plant",
                    "free_shape_count": len(free_shapes),
                    "valid_candidate_count": len(valid_candidates),
                    "aligned_to": "ppo_articulated_vehicle",
                },
            )
        raise RuntimeError("failed to sample block mixing scene")

    def _sample_pose_pair(
        self,
        rng: np.random.Generator,
        candidates: Iterable[_PoseCandidate],
        pair_distance_range: Tuple[float, float],
        pair_heading_diff_range_deg: Tuple[float, float] = (0.0, 180.0),
    ) -> Tuple[_PoseCandidate, _PoseCandidate]:
        candidate_list = list(candidates)
        if len(candidate_list) < 2:
            raise RuntimeError("insufficient pose candidates")
        low, high = float(pair_distance_range[0]), float(pair_distance_range[1])
        min_heading_deg, max_heading_deg = float(pair_heading_diff_range_deg[0]), float(pair_heading_diff_range_deg[1])
        for _ in range(256):
            start = candidate_list[int(rng.integers(0, len(candidate_list)))]
            goal = candidate_list[int(rng.integers(0, len(candidate_list)))]
            if start is goal:
                continue
            dist = float(np.hypot(goal.x - start.x, goal.y - start.y))
            if dist < low or dist > high:
                continue
            heading_diff_deg = abs(float(np.rad2deg(wrap_to_pi(goal.heading - start.heading))))
            if heading_diff_deg < min_heading_deg or heading_diff_deg > max_heading_deg:
                continue
            return start, goal
        start = candidate_list[0]
        goal = max(candidate_list[1:], key=lambda item: float(np.hypot(item.x - start.x, item.y - start.y)))
        return start, goal