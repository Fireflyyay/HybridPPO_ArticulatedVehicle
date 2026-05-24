import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from common.config import VehicleConfig
from common.runtime_config import SceneLevelConfig
from common.types import ArticulatedState, wrap_to_pi
from env.success import articulated_body_polygons


_CARDINAL_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)


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


@dataclass(frozen=True)
class _GridSegment:
    start: Tuple[int, int]
    end: Tuple[int, int]
    width_cells: int
    orientation: str


@dataclass(frozen=True)
class _ParkingBay:
    orientation: str
    heading: float
    rect: Tuple[int, int, int, int]
    center: Tuple[float, float]
    length: float
    depth: float
    access_cells: Tuple[Tuple[int, int], ...]
    polygon: Polygon


def _extract_polygons(geometry) -> List[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    polygons: List[Polygon] = []
    for sub_geometry in getattr(geometry, "geoms", []):
        polygons.extend(_extract_polygons(sub_geometry))
    return polygons


def _range_sample(rng: np.random.Generator, low_high: Tuple[int, int]) -> int:
    low = int(min(low_high[0], low_high[1]))
    high = int(max(low_high[0], low_high[1]))
    return int(rng.integers(low, high + 1))


def _warmup_progress(options: Optional[Mapping[str, object]]) -> Optional[float]:
    if not options or "warmup_progress" not in options:
        return None
    return float(options["warmup_progress"])


def _orthogonal_directions(direction: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    dx, dy = int(direction[0]), int(direction[1])
    if dx != 0:
        return (0, 1), (0, -1)
    return (1, 0), (-1, 0)


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

    def _world_bounds(self, config: SceneLevelConfig) -> Tuple[float, float, float, float]:
        return (
            float(config.world_min),
            float(config.world_max),
            float(config.world_min),
            float(config.world_max),
        )

    def _grid_origin(self, config: SceneLevelConfig) -> Tuple[float, float]:
        return float(config.world_min), float(config.world_min)

    def _grid_shape(self, config: SceneLevelConfig) -> Tuple[int, int]:
        return int(config.grid_width), int(config.grid_height)

    def _block_size(self, config: SceneLevelConfig) -> float:
        return float(config.block_size)

    def _boundary_margin_cells(self, config: SceneLevelConfig) -> int:
        return max(1, int(math.ceil(float(config.boundary_margin) / max(self._block_size(config), 1e-6))))

    def _cell_center_world(self, config: SceneLevelConfig, gx: int, gy: int) -> Tuple[float, float]:
        origin_x, origin_y = self._grid_origin(config)
        block_size = self._block_size(config)
        return (
            origin_x + (float(gx) + 0.5) * block_size,
            origin_y + (float(gy) + 0.5) * block_size,
        )

    def _cell_bounds_world(
        self,
        config: SceneLevelConfig,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Tuple[float, float, float, float]:
        origin_x, origin_y = self._grid_origin(config)
        block_size = self._block_size(config)
        return (
            origin_x + float(x0) * block_size,
            origin_y + float(y0) * block_size,
            origin_x + float(x1) * block_size,
            origin_y + float(y1) * block_size,
        )

    def _world_box(self, config: SceneLevelConfig):
        world_min, world_max, _, _ = self._world_bounds(config)
        return box(world_min, world_min, world_max, world_max)

    def _empty_occupancy(self, config: SceneLevelConfig) -> np.ndarray:
        grid_width, grid_height = self._grid_shape(config)
        return np.ones((grid_height, grid_width), dtype=np.uint8)

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

    def _largest_component_mask(self, free_grid: np.ndarray) -> Tuple[np.ndarray, int, int]:
        height, width = free_grid.shape
        visited = np.zeros_like(free_grid, dtype=bool)
        keep_mask = np.zeros_like(free_grid, dtype=bool)
        best_component: List[Tuple[int, int]] = []
        total_free = int(np.count_nonzero(free_grid))

        for gy in range(height):
            for gx in range(width):
                if visited[gy, gx] or not bool(free_grid[gy, gx]):
                    continue
                stack = [(gx, gy)]
                visited[gy, gx] = True
                component: List[Tuple[int, int]] = []
                while stack:
                    cell_x, cell_y = stack.pop()
                    component.append((cell_x, cell_y))
                    for step_x, step_y in _CARDINAL_DIRECTIONS:
                        next_x = cell_x + int(step_x)
                        next_y = cell_y + int(step_y)
                        if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                            continue
                        if visited[next_y, next_x] or not bool(free_grid[next_y, next_x]):
                            continue
                        visited[next_y, next_x] = True
                        stack.append((next_x, next_y))
                if len(component) > len(best_component):
                    best_component = component

        for gx, gy in best_component:
            keep_mask[gy, gx] = True
        return keep_mask, int(len(best_component)), int(total_free)

    def _cleanup_free_islands(self, occupancy: np.ndarray) -> Tuple[int, float, int, int]:
        free_grid = np.asarray(occupancy == 0, dtype=bool)
        keep_mask, component_size, total_free = self._largest_component_mask(free_grid)
        removed_mask = np.logical_and(free_grid, np.logical_not(keep_mask))
        removed_cells = int(np.count_nonzero(removed_mask))
        occupancy[removed_mask] = 1
        keep_ratio = float(component_size) / float(max(total_free, 1))
        return removed_cells, keep_ratio, component_size, total_free

    def _free_ratio(self, occupancy: np.ndarray) -> float:
        return float(np.count_nonzero(occupancy == 0)) / float(max(1, occupancy.size))

    def _free_space_geometry(self, occupancy: np.ndarray, config: SceneLevelConfig):
        block_size = self._block_size(config)
        origin_x, origin_y = self._grid_origin(config)
        rectangles = []
        height, width = occupancy.shape
        for gy in range(height):
            gx = 0
            while gx < width:
                if int(occupancy[gy, gx]) != 0:
                    gx += 1
                    continue
                run_start = gx
                while gx < width and int(occupancy[gy, gx]) == 0:
                    gx += 1
                rectangles.append(
                    box(
                        origin_x + float(run_start) * block_size,
                        origin_y + float(gy) * block_size,
                        origin_x + float(gx) * block_size,
                        origin_y + float(gy + 1) * block_size,
                    )
                )
        if not rectangles:
            return None
        return unary_union(rectangles).buffer(0)

    def _occupancy_to_obstacles(self, occupancy: np.ndarray, config: SceneLevelConfig) -> Tuple[Polygon, ...]:
        free_space = self._free_space_geometry(occupancy, config)
        world = self._world_box(config)
        if free_space is None or getattr(free_space, "is_empty", True):
            return (world,)
        return tuple(_extract_polygons(world.difference(free_space).buffer(0)))

    def _carve_rect(self, occupancy: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        height, width = occupancy.shape
        xx0 = max(0, min(int(x0), int(x1)))
        yy0 = max(0, min(int(y0), int(y1)))
        xx1 = min(width, max(int(x0), int(x1)))
        yy1 = min(height, max(int(y0), int(y1)))
        if xx0 >= xx1 or yy0 >= yy1:
            return
        occupancy[yy0:yy1, xx0:xx1] = 0

    def _carve_axis_segment(
        self,
        occupancy: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        width_cells: int,
    ) -> Optional[_GridSegment]:
        start_x, start_y = int(start[0]), int(start[1])
        end_x, end_y = int(end[0]), int(end[1])
        half_low = (int(width_cells) - 1) // 2
        half_high = int(width_cells) // 2
        if start_x == end_x:
            y0 = min(start_y, end_y)
            y1 = max(start_y, end_y) + 1
            self._carve_rect(occupancy, start_x - half_low, y0, start_x + half_high + 1, y1)
            return _GridSegment(
                start=(start_x, start_y),
                end=(end_x, end_y),
                width_cells=int(width_cells),
                orientation="vertical",
            )
        if start_y == end_y:
            x0 = min(start_x, end_x)
            x1 = max(start_x, end_x) + 1
            self._carve_rect(occupancy, x0, start_y - half_low, x1, start_y + half_high + 1)
            return _GridSegment(
                start=(start_x, start_y),
                end=(end_x, end_y),
                width_cells=int(width_cells),
                orientation="horizontal",
            )
        return None

    def _carve_geometry(self, occupancy: np.ndarray, config: SceneLevelConfig, geometry) -> None:
        if geometry is None or getattr(geometry, "is_empty", True):
            return
        min_x, min_y, max_x, max_y = geometry.bounds
        origin_x, origin_y = self._grid_origin(config)
        block_size = max(self._block_size(config), 1e-6)
        grid_width, grid_height = self._grid_shape(config)
        gx0 = max(0, int(math.floor((float(min_x) - origin_x) / block_size)) - 1)
        gy0 = max(0, int(math.floor((float(min_y) - origin_y) / block_size)) - 1)
        gx1 = min(grid_width - 1, int(math.ceil((float(max_x) - origin_x) / block_size)) + 1)
        gy1 = min(grid_height - 1, int(math.ceil((float(max_y) - origin_y) / block_size)) + 1)
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                world_x0, world_y0, world_x1, world_y1 = self._cell_bounds_world(config, gx, gy, gx + 1, gy + 1)
                if geometry.intersects(box(world_x0, world_y0, world_x1, world_y1)):
                    occupancy[gy, gx] = 0

    def _clip_world_point(self, config: SceneLevelConfig, point: np.ndarray, margin_m: float) -> np.ndarray:
        xmin, xmax, ymin, ymax = self._world_bounds(config)
        point[0] = float(np.clip(point[0], xmin + margin_m, xmax - margin_m))
        point[1] = float(np.clip(point[1], ymin + margin_m, ymax - margin_m))
        return point

    def _clip_cell(self, config: SceneLevelConfig, cell: Tuple[int, int], margin_cells: int) -> Tuple[int, int]:
        grid_width, grid_height = self._grid_shape(config)
        min_x = int(margin_cells)
        min_y = int(margin_cells)
        max_x = int(grid_width - margin_cells - 1)
        max_y = int(grid_height - margin_cells - 1)
        return (
            min(max(int(cell[0]), min_x), max_x),
            min(max(int(cell[1]), min_y), max_y),
        )

    def _sample_route_points(
        self,
        rng: np.random.Generator,
        config: SceneLevelConfig,
        start_cell: Tuple[int, int],
        primary_direction: Tuple[int, int],
        margin_cells: int,
        segment_count_range: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        segment_count = _range_sample(rng, segment_count_range)
        current = (int(start_cell[0]), int(start_cell[1]))
        direction = (int(primary_direction[0]), int(primary_direction[1]))
        route = [current]

        for segment_idx in range(segment_count):
            length_cells = max(4, _range_sample(rng, config.segment_length_range))
            end_cell = self._clip_cell(
                config,
                (
                    current[0] + int(direction[0]) * length_cells,
                    current[1] + int(direction[1]) * length_cells,
                ),
                margin_cells,
            )
            manhattan = abs(int(end_cell[0]) - int(current[0])) + abs(int(end_cell[1]) - int(current[1]))
            if manhattan < 4:
                break
            route.append(end_cell)
            current = end_cell
            if segment_idx == segment_count - 1:
                continue
            if rng.random() < float(config.turn_probability):
                next_directions = _orthogonal_directions(direction)
                direction = next_directions[int(rng.integers(0, len(next_directions)))]
        return route

    def _carve_route(
        self,
        occupancy: np.ndarray,
        route: Sequence[Tuple[int, int]],
        width_cells: int,
    ) -> List[_GridSegment]:
        segments: List[_GridSegment] = []
        for start, end in zip(route[:-1], route[1:]):
            segment = self._carve_axis_segment(occupancy, start, end, width_cells)
            if segment is not None:
                segments.append(segment)
        return segments

    def _segment_length_world(self, config: SceneLevelConfig, segment: _GridSegment) -> float:
        dx = int(segment.end[0]) - int(segment.start[0])
        dy = int(segment.end[1]) - int(segment.start[1])
        return float(abs(dx) + abs(dy)) * self._block_size(config)

    def _endpoint_candidate(self, config: SceneLevelConfig, segment: _GridSegment, use_start: bool) -> Optional[_PoseCandidate]:
        length_world = self._segment_length_world(config, segment)
        if length_world < self._rear_reach() + 1.5:
            return None
        start_xy = np.asarray(self._cell_center_world(config, int(segment.start[0]), int(segment.start[1])), dtype=np.float64)
        end_xy = np.asarray(self._cell_center_world(config, int(segment.end[0]), int(segment.end[1])), dtype=np.float64)
        axis = end_xy - start_xy
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return None
        axis /= norm
        clearance = self._rear_reach() + 0.35
        if use_start:
            point = start_xy + axis * clearance
            heading = float(math.atan2(axis[1], axis[0]))
        else:
            point = end_xy - axis * clearance
            heading = float(math.atan2(-axis[1], -axis[0]))
        return _PoseCandidate(float(point[0]), float(point[1]), heading)

    def _segment_anchor_cell(self, segment: _GridSegment, rng: np.random.Generator) -> Optional[Tuple[int, int]]:
        start_x, start_y = int(segment.start[0]), int(segment.start[1])
        end_x, end_y = int(segment.end[0]), int(segment.end[1])
        if str(segment.orientation) == "horizontal":
            left = min(start_x, end_x) + 2
            right = max(start_x, end_x) - 2
            if left > right:
                return None
            return int(rng.integers(left, right + 1)), int(start_y)
        bottom = min(start_y, end_y) + 2
        top = max(start_y, end_y) - 2
        if bottom > top:
            return None
        return int(start_x), int(rng.integers(bottom, top + 1))

    def _build_parking_bay(
        self,
        config: SceneLevelConfig,
        orientation: str,
        heading: float,
        rect: Tuple[int, int, int, int],
        access_cells: Sequence[Tuple[int, int]],
    ) -> _ParkingBay:
        x0, y0, x1, y1 = rect
        world_x0, world_y0, world_x1, world_y1 = self._cell_bounds_world(config, x0, y0, x1, y1)
        polygon = box(world_x0, world_y0, world_x1, world_y1)
        if str(orientation) in {"east", "west"}:
            length = float(y1 - y0) * self._block_size(config)
            depth = float(x1 - x0) * self._block_size(config)
        else:
            length = float(x1 - x0) * self._block_size(config)
            depth = float(y1 - y0) * self._block_size(config)
        return _ParkingBay(
            orientation=str(orientation),
            heading=float(heading),
            rect=(int(x0), int(y0), int(x1), int(y1)),
            center=(0.5 * (world_x0 + world_x1), 0.5 * (world_y0 + world_y1)),
            length=float(length),
            depth=float(depth),
            access_cells=tuple((int(cell_x), int(cell_y)) for cell_x, cell_y in access_cells),
            polygon=polygon,
        )

    def _goal_candidate_from_bay(self, bay: _ParkingBay, config: SceneLevelConfig) -> _PoseCandidate:
        axis = np.asarray([math.cos(float(bay.heading)), math.sin(float(bay.heading))], dtype=np.float64)
        center = np.asarray([float(bay.center[0]), float(bay.center[1])], dtype=np.float64)
        coords = np.asarray(bay.polygon.exterior.coords[:-1], dtype=np.float64)
        far_wall_projection = float(np.max(coords @ axis))
        center_projection = float(center @ axis)
        reference_projection = far_wall_projection - float(config.parking_head_wall_clearance) - self._front_reach()
        point = center + axis * (reference_projection - center_projection)
        return _PoseCandidate(float(point[0]), float(point[1]), float(bay.heading))

    def _sample_pose_pair_from_groups(
        self,
        rng: np.random.Generator,
        start_candidates: Sequence[_PoseCandidate],
        goal_candidates: Sequence[_PoseCandidate],
        pair_distance_range: Tuple[float, float],
        pair_heading_diff_range_deg: Tuple[float, float],
    ) -> Tuple[_PoseCandidate, _PoseCandidate]:
        if len(start_candidates) == 0 or len(goal_candidates) == 0:
            raise RuntimeError("insufficient pose candidates")

        min_distance = float(min(pair_distance_range[0], pair_distance_range[1]))
        max_distance = float(max(pair_distance_range[0], pair_distance_range[1]))
        min_heading = float(min(pair_heading_diff_range_deg[0], pair_heading_diff_range_deg[1]))
        max_heading = float(max(pair_heading_diff_range_deg[0], pair_heading_diff_range_deg[1]))

        valid_pairs: List[Tuple[float, _PoseCandidate, _PoseCandidate]] = []
        fallback_pairs: List[Tuple[float, _PoseCandidate, _PoseCandidate]] = []
        for start in start_candidates:
            for goal in goal_candidates:
                distance = float(math.hypot(float(goal.x) - float(start.x), float(goal.y) - float(start.y)))
                heading_diff_deg = abs(float(np.rad2deg(wrap_to_pi(float(goal.heading) - float(start.heading)))))
                fallback_pairs.append((distance, start, goal))
                if distance < min_distance or distance > max_distance:
                    continue
                if heading_diff_deg < min_heading or heading_diff_deg > max_heading:
                    continue
                valid_pairs.append((distance, start, goal))

        if len(valid_pairs) > 0:
            top_k = sorted(valid_pairs, key=lambda item: item[0], reverse=True)[: min(4, len(valid_pairs))]
            chosen = top_k[int(rng.integers(0, len(top_k)))]
            return chosen[1], chosen[2]
        if len(fallback_pairs) > 0:
            fallback = max(fallback_pairs, key=lambda item: item[0])
            return fallback[1], fallback[2]
        raise RuntimeError("failed to sample pose pair")

    def _generate_random_warmup_bay(
        self,
        occupancy: np.ndarray,
        config: SceneLevelConfig,
        rng: np.random.Generator,
    ) -> _ParkingBay:
        grid_width, grid_height = self._grid_shape(config)
        boundary_margin = self._boundary_margin_cells(config)
        bay_margin = boundary_margin + 8
        length_cells = _range_sample(rng, config.parking_bay_length_range)
        depth_cells = _range_sample(rng, config.parking_bay_depth_range)
        orientations = ["east", "west", "north", "south"]
        rng.shuffle(orientations)

        for orientation in orientations:
            if orientation == "east":
                x0_low = boundary_margin + bay_margin + 1
                x0_high = grid_width - boundary_margin - bay_margin - depth_cells
                y0_low = boundary_margin + bay_margin
                y0_high = grid_height - boundary_margin - bay_margin - length_cells
                if x0_low > x0_high or y0_low > y0_high:
                    continue
                x0 = int(rng.integers(x0_low, x0_high + 1))
                y0 = int(rng.integers(y0_low, y0_high + 1))
                x1 = x0 + depth_cells
                y1 = y0 + length_cells
                access_cells = tuple((x0 - 1, gy) for gy in range(y0, y1))
                heading = 0.0
            elif orientation == "west":
                x1_low = boundary_margin + bay_margin + depth_cells
                x1_high = grid_width - boundary_margin - bay_margin - 1
                y0_low = boundary_margin + bay_margin
                y0_high = grid_height - boundary_margin - bay_margin - length_cells
                if x1_low > x1_high or y0_low > y0_high:
                    continue
                x1 = int(rng.integers(x1_low, x1_high + 1))
                y0 = int(rng.integers(y0_low, y0_high + 1))
                x0 = x1 - depth_cells
                y1 = y0 + length_cells
                access_cells = tuple((x1, gy) for gy in range(y0, y1))
                heading = float(np.pi)
            elif orientation == "north":
                x0_low = boundary_margin + bay_margin
                x0_high = grid_width - boundary_margin - bay_margin - length_cells
                y0_low = boundary_margin + bay_margin + 1
                y0_high = grid_height - boundary_margin - bay_margin - depth_cells
                if x0_low > x0_high or y0_low > y0_high:
                    continue
                x0 = int(rng.integers(x0_low, x0_high + 1))
                y0 = int(rng.integers(y0_low, y0_high + 1))
                x1 = x0 + length_cells
                y1 = y0 + depth_cells
                access_cells = tuple((gx, y0 - 1) for gx in range(x0, x1))
                heading = float(0.5 * np.pi)
            else:
                x0_low = boundary_margin + bay_margin
                x0_high = grid_width - boundary_margin - bay_margin - length_cells
                y1_low = boundary_margin + bay_margin + depth_cells
                y1_high = grid_height - boundary_margin - bay_margin - 1
                if x0_low > x0_high or y1_low > y1_high:
                    continue
                x0 = int(rng.integers(x0_low, x0_high + 1))
                y1 = int(rng.integers(y1_low, y1_high + 1))
                x1 = x0 + length_cells
                y0 = y1 - depth_cells
                access_cells = tuple((gx, y1) for gx in range(x0, x1))
                heading = float(-0.5 * np.pi)

            self._carve_rect(occupancy, x0, y0, x1, y1)
            for access_x, access_y in access_cells:
                if 0 <= int(access_x) < grid_width and 0 <= int(access_y) < grid_height:
                    occupancy[int(access_y), int(access_x)] = 0
            return self._build_parking_bay(
                config,
                orientation=orientation,
                heading=heading,
                rect=(x0, y0, x1, y1),
                access_cells=access_cells,
            )

        raise RuntimeError("failed to place warmup parking bay")

    def _build_warmup_corridor(
        self,
        config: SceneLevelConfig,
        goal_candidate: _PoseCandidate,
        access_cells: Sequence[Tuple[int, int]],
        corridor_width: float,
        rng: np.random.Generator,
    ) -> Tuple[ArticulatedState, object]:
        access_index = int(rng.integers(0, len(access_cells))) if len(access_cells) > 1 else 0
        access_world = np.asarray(self._cell_center_world(config, *access_cells[access_index]), dtype=np.float64)
        forward = np.asarray([math.cos(float(goal_candidate.heading)), math.sin(float(goal_candidate.heading))], dtype=np.float64)
        lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        lateral_sign = -1.0 if rng.random() < 0.5 else 1.0
        world_margin = max(corridor_width, self._rear_reach()) + 2.0

        approach_distance = float(rng.uniform(4.0, 8.0))
        start_distance = float(rng.uniform(18.0, 28.0))
        lateral_span = float(rng.uniform(max(3.0, corridor_width * 0.8), corridor_width * 2.2))

        approach_point = access_world - forward * approach_distance
        bend_point = access_world - forward * (0.45 * start_distance) + lateral * lateral_sign * (0.65 * lateral_span)
        start_point = access_world - forward * start_distance + lateral * lateral_sign * lateral_span
        launch_point = start_point - forward * (self._rear_reach() + 2.0)

        points = [launch_point, start_point, bend_point, approach_point, access_world]
        clipped_points: List[np.ndarray] = []
        for point in points:
            clipped_points.append(self._clip_world_point(config, np.asarray(point, dtype=np.float64), world_margin))

        start_vector = clipped_points[1] - clipped_points[0]
        norm = float(np.linalg.norm(start_vector))
        if norm < 1e-6:
            raise RuntimeError("failed to build warmup start heading")
        start_heading = float(math.atan2(start_vector[1], start_vector[0]))
        start_state = ArticulatedState(
            x=float(clipped_points[1][0]),
            y=float(clipped_points[1][1]),
            front_heading=start_heading,
            rear_heading=start_heading,
        )

        corridor_geometry = LineString([(float(point[0]), float(point[1])) for point in clipped_points]).buffer(
            0.5 * float(corridor_width),
            cap_style=1,
            join_style=1,
        )
        return start_state, corridor_geometry

    def _try_place_bay_on_segment(
        self,
        occupancy: np.ndarray,
        config: SceneLevelConfig,
        segment: _GridSegment,
        rng: np.random.Generator,
    ) -> Optional[_ParkingBay]:
        grid_width, grid_height = self._grid_shape(config)
        boundary_margin = self._boundary_margin_cells(config)
        width_cells = int(segment.width_cells)
        half_low = (width_cells - 1) // 2
        half_high = width_cells // 2
        length_cells = _range_sample(rng, config.parking_bay_length_range)
        depth_cells = _range_sample(rng, config.parking_bay_depth_range)

        if str(segment.orientation) == "horizontal":
            min_x = min(int(segment.start[0]), int(segment.end[0]))
            max_x = max(int(segment.start[0]), int(segment.end[0]))
            if max_x - min_x < length_cells + 2:
                return None
            x0 = int(rng.integers(min_x + 1, max_x - length_cells + 2))
            x1 = x0 + length_cells
            side = "north" if rng.random() < 0.5 else "south"
            center_y = int(segment.start[1])
            if side == "north":
                y0 = center_y + half_high + 1
                y1 = y0 + depth_cells
                access_cells = tuple((gx, y0 - 1) for gx in range(x0, x1))
                heading = float(0.5 * np.pi)
                orientation = "north"
            else:
                y1 = center_y - half_low
                y0 = y1 - depth_cells
                access_cells = tuple((gx, y1) for gx in range(x0, x1))
                heading = float(-0.5 * np.pi)
                orientation = "south"
            if y0 < boundary_margin or y1 > grid_height - boundary_margin:
                return None
        else:
            min_y = min(int(segment.start[1]), int(segment.end[1]))
            max_y = max(int(segment.start[1]), int(segment.end[1]))
            if max_y - min_y < length_cells + 2:
                return None
            y0 = int(rng.integers(min_y + 1, max_y - length_cells + 2))
            y1 = y0 + length_cells
            side = "east" if rng.random() < 0.5 else "west"
            center_x = int(segment.start[0])
            if side == "east":
                x0 = center_x + half_high + 1
                x1 = x0 + depth_cells
                access_cells = tuple((x0 - 1, gy) for gy in range(y0, y1))
                heading = 0.0
                orientation = "east"
            else:
                x1 = center_x - half_low
                x0 = x1 - depth_cells
                access_cells = tuple((x1, gy) for gy in range(y0, y1))
                heading = float(np.pi)
                orientation = "west"
            if x0 < boundary_margin or x1 > grid_width - boundary_margin:
                return None

        if np.any(occupancy[y0:y1, x0:x1] == 0):
            return None
        self._carve_rect(occupancy, x0, y0, x1, y1)
        return self._build_parking_bay(
            config,
            orientation=orientation,
            heading=heading,
            rect=(x0, y0, x1, y1),
            access_cells=access_cells,
        )

    def _sample_branch_bays(
        self,
        occupancy: np.ndarray,
        config: SceneLevelConfig,
        rng: np.random.Generator,
        segments: Sequence[_GridSegment],
        target_count: int,
    ) -> List[_ParkingBay]:
        if target_count <= 0:
            return []
        bays: List[_ParkingBay] = []
        attempts = 0
        while len(bays) < int(target_count) and attempts < 8 * max(1, int(target_count)):
            attempts += 1
            if len(segments) == 0:
                break
            segment = segments[int(rng.integers(0, len(segments)))]
            bay = self._try_place_bay_on_segment(occupancy, config, segment, rng)
            if bay is None:
                continue
            bays.append(bay)
        return bays

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
        margin = float(config.boundary_margin) + self._rear_reach() + 1.0
        for _ in range(256):
            start_x = float(rng.uniform(world_min + margin, world_max - margin))
            start_y = float(rng.uniform(world_min + margin, world_max - margin))
            goal_x = float(rng.uniform(world_min + margin, world_max - margin))
            goal_y = float(rng.uniform(world_min + margin, world_max - margin))
            distance = float(np.hypot(goal_x - start_x, goal_y - start_y))
            low, high = config.pair_distance_range
            if distance < float(min(low, high)) or distance > float(max(low, high)):
                continue
            path_heading = float(np.arctan2(goal_y - start_y, goal_x - start_x))
            jitter = float(config.heading_jitter_rad)
            start_heading = path_heading + float(rng.uniform(-jitter, jitter))
            goal_heading = path_heading + float(rng.uniform(-jitter, jitter))
            return SceneSpec(
                level=level,
                world_bounds=self._world_bounds(config),
                obstacles=tuple(),
                start_state=ArticulatedState(start_x, start_y, start_heading, start_heading),
                goal_state=ArticulatedState(goal_x, goal_y, goal_heading, goal_heading),
                metadata={"scene_type": "debug_empty"},
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
        corridor_width_cells = max(1, int(round(corridor_width / max(self._block_size(config), 1e-6))))
        world_bounds = self._world_bounds(config)

        for _ in range(96):
            occupancy = self._empty_occupancy(config)
            bay = self._generate_random_warmup_bay(occupancy, config, rng)
            goal_candidate = self._goal_candidate_from_bay(bay, config)
            try:
                start_state, corridor_geometry = self._build_warmup_corridor(
                    config,
                    goal_candidate,
                    bay.access_cells,
                    corridor_width,
                    rng,
                )
            except RuntimeError:
                continue
            self._carve_geometry(occupancy, config, corridor_geometry)
            self._carve_geometry(occupancy, config, bay.polygon)
            removed_islands, keep_ratio, component_size, total_free = self._cleanup_free_islands(occupancy)
            obstacles = self._occupancy_to_obstacles(occupancy, config)
            goal_state = ArticulatedState(goal_candidate.x, goal_candidate.y, goal_candidate.heading, goal_candidate.heading)
            valid = self._filter_valid_candidates(
                [
                    _PoseCandidate(start_state.x, start_state.y, start_state.front_heading),
                    goal_candidate,
                ],
                obstacles,
                world_bounds,
            )
            if len(valid) != 2:
                continue
            return SceneSpec(
                level=level,
                world_bounds=world_bounds,
                obstacles=obstacles,
                start_state=start_state,
                goal_state=goal_state,
                metadata={
                    "scene_type": "block_mixing_plant",
                    "scene_variant": "warmup_curriculum",
                    "corridor_generation_mode": "polyline_warmup",
                    "corridor_width": float(corridor_width),
                    "corridor_width_cells": int(corridor_width_cells),
                    "corridor_min_width": float(config.warmup_corridor_min_width()),
                    "warmup_progress": _warmup_progress(options),
                    "grid_width": int(config.grid_width),
                    "grid_height": int(config.grid_height),
                    "block_size": float(config.block_size),
                    "parking_bay_count": 1,
                    "free_ratio": self._free_ratio(occupancy),
                    "largest_component_free_cells": int(component_size),
                    "total_free_cells": int(total_free),
                    "island_cleanup": (int(removed_islands), float(keep_ratio)),
                },
            )
        raise RuntimeError("failed to sample warmup scene")

    def _generate_block_mixing(self, level: str, config: SceneLevelConfig, rng: np.random.Generator) -> SceneSpec:
        world_bounds = self._world_bounds(config)
        margin_cells = self._boundary_margin_cells(config)
        grid_width, grid_height = self._grid_shape(config)

        for _ in range(128):
            occupancy = self._empty_occupancy(config)
            corridor_width = float(_range_sample(rng, config.corridor_width_range))
            width_cells = max(2, int(round(corridor_width / max(self._block_size(config), 1e-6))))
            hub_cell = (
                int(rng.integers(margin_cells + 12, grid_width - margin_cells - 12)),
                int(rng.integers(margin_cells + 12, grid_height - margin_cells - 12)),
            )
            all_segments: List[_GridSegment] = []

            primary_directions = list(_CARDINAL_DIRECTIONS)
            rng.shuffle(primary_directions)
            for direction in primary_directions[: max(1, int(config.main_corridor_count))]:
                route = self._sample_route_points(
                    rng,
                    config,
                    hub_cell,
                    direction,
                    margin_cells,
                    segment_count_range=(2, 3),
                )
                all_segments.extend(self._carve_route(occupancy, route, width_cells))

            branch_count = _range_sample(rng, config.branch_count_range)
            for _branch_idx in range(branch_count):
                if len(all_segments) == 0:
                    break
                anchor_segment = all_segments[int(rng.integers(0, len(all_segments)))]
                anchor_cell = self._segment_anchor_cell(anchor_segment, rng)
                if anchor_cell is None:
                    continue
                branch_dirs = _orthogonal_directions((1, 0) if str(anchor_segment.orientation) == "horizontal" else (0, 1))
                branch_dir = branch_dirs[int(rng.integers(0, len(branch_dirs)))]
                route = self._sample_route_points(
                    rng,
                    config,
                    anchor_cell,
                    branch_dir,
                    margin_cells,
                    segment_count_range=(1, 2),
                )
                all_segments.extend(self._carve_route(occupancy, route, width_cells))

            if len(all_segments) == 0:
                continue

            target_bay_count = _range_sample(rng, config.parking_bay_count_range)
            bays = self._sample_branch_bays(occupancy, config, rng, all_segments, target_bay_count)
            if len(bays) < max(1, target_bay_count):
                continue

            removed_islands, keep_ratio, component_size, total_free = self._cleanup_free_islands(occupancy)
            obstacles = self._occupancy_to_obstacles(occupancy, config)

            raw_start_candidates = [
                candidate
                for segment in all_segments
                for candidate in (
                    self._endpoint_candidate(config, segment, use_start=True),
                    self._endpoint_candidate(config, segment, use_start=False),
                )
                if candidate is not None
            ]
            raw_goal_candidates = [self._goal_candidate_from_bay(bay, config) for bay in bays]

            start_candidates = self._filter_valid_candidates(raw_start_candidates, obstacles, world_bounds)
            goal_candidates = self._filter_valid_candidates(raw_goal_candidates, obstacles, world_bounds)
            if len(start_candidates) == 0 or len(goal_candidates) == 0:
                continue

            try:
                start_candidate, goal_candidate = self._sample_pose_pair_from_groups(
                    rng,
                    start_candidates,
                    goal_candidates,
                    config.pair_distance_range,
                    config.pair_heading_diff_range_deg,
                )
            except RuntimeError:
                continue

            return SceneSpec(
                level=level,
                world_bounds=world_bounds,
                obstacles=obstacles,
                start_state=ArticulatedState(
                    start_candidate.x,
                    start_candidate.y,
                    start_candidate.heading,
                    start_candidate.heading,
                ),
                goal_state=ArticulatedState(
                    goal_candidate.x,
                    goal_candidate.y,
                    goal_candidate.heading,
                    goal_candidate.heading,
                ),
                metadata={
                    "scene_type": "block_mixing_plant",
                    "scene_variant": "constructive_grid",
                    "corridor_generation_mode": "constructive_attachment",
                    "corridor_width": float(corridor_width),
                    "corridor_width_cells": int(width_cells),
                    "grid_width": int(config.grid_width),
                    "grid_height": int(config.grid_height),
                    "block_size": float(config.block_size),
                    "hub_cell": (int(hub_cell[0]), int(hub_cell[1])),
                    "main_corridor_count": int(config.main_corridor_count),
                    "branch_count": int(branch_count),
                    "parking_bay_count": int(len(bays)),
                    "free_ratio": self._free_ratio(occupancy),
                    "largest_component_free_cells": int(component_size),
                    "total_free_cells": int(total_free),
                    "island_cleanup": (int(removed_islands), float(keep_ratio)),
                    "valid_start_candidate_count": int(len(start_candidates)),
                    "valid_goal_candidate_count": int(len(goal_candidates)),
                },
            )

        raise RuntimeError("failed to sample block mixing scene")