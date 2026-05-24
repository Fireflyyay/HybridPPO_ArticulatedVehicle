from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import LineString, Polygon, box
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

    _CARDINAL_DIRS: Tuple[Tuple[float, float], ...] = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

    def _generate_warmup(
        self,
        level: str,
        config: SceneLevelConfig,
        rng: np.random.Generator,
        options: Optional[Mapping[str, object]] = None,
    ) -> SceneSpec:
        corridor_width = float(config.resolve_warmup_corridor_width(_warmup_progress(options)))
        half_width = 0.5 * corridor_width
        xmin = float(config.world_min)
        xmax = float(config.world_max)
        ymin = float(config.world_min)
        ymax = float(config.world_max)
        margin = float(config.boundary_margin)
        world_bounds = (xmin, xmax, ymin, ymax)
        world = box(xmin, ymin, xmax, ymax)

        centerline, turn_count = self._random_centerline(config, rng, xmin, xmax, ymin, ymax, margin)
        end_pad = max(self._front_reach(), self._rear_reach()) + 2.0
        padded_centerline = self._pad_centerline_ends(centerline, end_pad)
        corridor = LineString(padded_centerline).buffer(half_width, cap_style=2, join_style=2)
        bay_polygon, bay_heading = self._random_bay_on_centerline(
            centerline, half_width, corridor_width, config, rng, xmin, xmax, ymin, ymax, margin
        )

        free_space = unary_union([corridor, bay_polygon]).buffer(0)
        # Conservative shrink-then-expand to eliminate boundary touching artifacts
        # that would cause intersects()-based collision checks to false-positive.
        free_space = free_space.buffer(-0.05).buffer(0.05)
        obstacles = tuple(_extract_polygons(world.difference(free_space).buffer(0)))

        sx, sy = centerline[0]
        sdx = centerline[1][0] - sx
        sdy = centerline[1][1] - sy
        start_heading = float(np.arctan2(sdy, sdx))
        # Place goal conservatively inside the bay, nudging toward corridor if needed.
        bay_centroid = bay_polygon.centroid
        gx, gy = float(bay_centroid.x), float(bay_centroid.y)
        start_state = ArticulatedState(x=sx, y=sy, front_heading=start_heading, rear_heading=start_heading)
        goal_state = ArticulatedState(x=gx, y=gy, front_heading=float(bay_heading), rear_heading=float(bay_heading))
        obs_union = unary_union(list(obstacles)) if obstacles else None
        if obs_union is not None:
            for nudge in range(6):
                front_p, rear_p = articulated_body_polygons(goal_state, self.vehicle_config)
                if not front_p.intersects(obs_union) and not rear_p.intersects(obs_union):
                    break
                # Nudge toward corridor (opposite of bay heading) by 0.4 + nudge*0.4 meters
                step = 0.4 + float(nudge) * 0.4
                gx -= np.cos(float(bay_heading)) * step
                gy -= np.sin(float(bay_heading)) * step
                goal_state = ArticulatedState(x=gx, y=gy, front_heading=float(bay_heading), rear_heading=float(bay_heading))
        start_state = ArticulatedState(x=sx, y=sy, front_heading=start_heading, rear_heading=start_heading)
        goal_state = ArticulatedState(x=gx, y=gy, front_heading=float(bay_heading), rear_heading=float(bay_heading))

        return SceneSpec(
            level=level,
            world_bounds=world_bounds,
            obstacles=obstacles,
            start_state=start_state,
            goal_state=goal_state,
            metadata={
                "scene_type": "warmup_centerline",
                "aligned_to": "ppo_articulated_vehicle",
                "corridor_width": float(corridor_width),
                "corridor_min_width": float(config.warmup_corridor_min_width()),
                "warmup_progress": _warmup_progress(options),
                "turn_count": int(turn_count),
                "centerline_points": len(centerline),
            },
        )

    def _random_centerline(
        self,
        config: SceneLevelConfig,
        rng: np.random.Generator,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        margin: float,
    ) -> Tuple[List[Tuple[float, float]], int]:
        inner_xmin = xmin + margin + 6.0
        inner_xmax = xmax - margin - 6.0
        inner_ymin = ymin + margin + 6.0
        inner_ymax = ymax - margin - 6.0
        sx = float(rng.uniform(inner_xmin, inner_xmax))
        sy = float(rng.uniform(inner_ymin, inner_ymax))
        dir_idx = int(rng.integers(0, 4))
        dx, dy = self._CARDINAL_DIRS[dir_idx]
        turn_count = int(rng.integers(config.warmup_turn_count_range[0], config.warmup_turn_count_range[1] + 1))
        seg_min = max(4.0, float(config.segment_length_range[0]))
        seg_max = max(seg_min, float(config.segment_length_range[1]))
        points: List[Tuple[float, float]] = [(float(sx), float(sy))]
        cx, cy = sx, sy
        for seg_i in range(turn_count + 1):
            seg_len = float(rng.uniform(seg_min, seg_max))
            ex = float(np.clip(cx + dx * seg_len, inner_xmin, inner_xmax))
            ey = float(np.clip(cy + dy * seg_len, inner_ymin, inner_ymax))
            if float(np.hypot(ex - cx, ey - cy)) < 4.0:
                ex = float(np.clip(cx + dx * seg_max, inner_xmin, inner_xmax))
                ey = float(np.clip(cy + dy * seg_max, inner_ymin, inner_ymax))
            points.append((ex, ey))
            if seg_i < turn_count:
                turn = int(rng.choice([-1, 1]))
                if turn == -1:
                    dx, dy = -dy, dx
                else:
                    dx, dy = dy, -dx
                cx, cy = ex, ey
        return points, turn_count

    @staticmethod
    def _pad_centerline_ends(
        centerline: List[Tuple[float, float]],
        extend_by: float,
    ) -> List[Tuple[float, float]]:
        if len(centerline) < 2 or extend_by <= 0.0:
            return list(centerline)
        padded = list(centerline)
        fx, fy = padded[0]
        sx, sy = padded[1]
        fdx = fx - sx
        fdy = fy - sy
        f_len = float(np.hypot(fdx, fdy))
        if f_len > 1e-6:
            padded[0] = (float(fx + (fdx / f_len) * extend_by), float(fy + (fdy / f_len) * extend_by))
        lx, ly = padded[-1]
        plx, ply = padded[-2]
        ldx = lx - plx
        ldy = ly - ply
        l_len = float(np.hypot(ldx, ldy))
        if l_len > 1e-6:
            padded[-1] = (float(lx + (ldx / l_len) * extend_by), float(ly + (ldy / l_len) * extend_by))
        return padded

    def _random_bay_on_centerline(
        self,
        centerline: List[Tuple[float, float]],
        half_width: float,
        corridor_width: float,
        config: SceneLevelConfig,
        rng: np.random.Generator,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        margin: float,
    ) -> Tuple[Polygon, float]:
        n_segs = len(centerline) - 1
        seg_idx = int(rng.integers(0, max(1, n_segs)))
        x1, y1 = centerline[seg_idx]
        x2, y2 = centerline[seg_idx + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = float(np.hypot(dx, dy))
        if seg_len < 1e-6:
            seg_idx = 0
            x1, y1 = centerline[0]
            x2, y2 = centerline[1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len = float(np.hypot(dx, dy))
        ndx = dx / max(seg_len, 1e-6)
        ndy = dy / max(seg_len, 1e-6)
        side = int(rng.choice([-1, 1]))
        if side == -1:
            pdx, pdy = -ndy, ndx
        else:
            pdx, pdy = ndy, -ndx
        bay_length = max(1.0, float(rng.uniform(config.parking_bay_length_range[0], config.parking_bay_length_range[1])))
        bay_depth = max(1.0, float(rng.uniform(config.parking_bay_depth_range[0], config.parking_bay_depth_range[1])))
        t = float(rng.uniform(0.25, 0.75))
        attach_x = x1 + t * dx
        attach_y = y1 + t * dy
        edge_x = attach_x + pdx * half_width
        edge_y = attach_y + pdy * half_width
        half_len = 0.5 * bay_length
        corners = [
            (edge_x - ndx * half_len, edge_y - ndy * half_len),
            (edge_x + ndx * half_len, edge_y + ndy * half_len),
            (edge_x + ndx * half_len + pdx * bay_depth, edge_y + ndy * half_len + pdy * bay_depth),
            (edge_x - ndx * half_len + pdx * bay_depth, edge_y - ndy * half_len + pdy * bay_depth),
        ]
        bay_polygon = Polygon(corners)
        world_box = box(xmin + margin, ymin + margin, xmax - margin, ymax - margin)
        if not world_box.contains(bay_polygon):
            bay_polygon = bay_polygon.intersection(world_box)
            if bay_polygon.is_empty or bay_polygon.area < 4.0:
                side = -side
                if side == -1:
                    pdx, pdy = -ndy, ndx
                else:
                    pdx, pdy = ndy, -ndx
                edge_x = attach_x + pdx * half_width
                edge_y = attach_y + pdy * half_width
                corners = [
                    (edge_x - ndx * half_len, edge_y - ndy * half_len),
                    (edge_x + ndx * half_len, edge_y + ndy * half_len),
                    (edge_x + ndx * half_len + pdx * bay_depth, edge_y + ndy * half_len + pdy * bay_depth),
                    (edge_x - ndx * half_len + pdx * bay_depth, edge_y - ndy * half_len + pdy * bay_depth),
                ]
                bay_polygon = Polygon(corners)
                if not world_box.contains(bay_polygon):
                    bay_polygon = bay_polygon.intersection(world_box)
        bay_heading = float(np.arctan2(pdy, pdx))
        return bay_polygon, bay_heading

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
                    "corridor_width": float(corridor_width),
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