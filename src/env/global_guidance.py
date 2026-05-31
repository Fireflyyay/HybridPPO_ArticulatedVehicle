import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from common.config import VehicleConfig
from env.scenes import SceneSpec


class CoarseGlobalGuidance:
    def __init__(
        self,
        grid_resolution: float = 1.0,
        lookahead_base: float = 6.0,
        lookahead_speed_gain: float = 2.5,
        lookahead_min: float = 3.0,
        lookahead_max: float = 12.0,
        progress_search_window: int = 40,
        min_clearance_m: float = 1.2,
        full_clearance_m: float = 4.0,
        near_obs_dist_m: float = 2.0,
        max_dense_ratio: float = 0.35,
    ) -> None:
        self.grid_resolution = float(grid_resolution)
        self.lookahead_base = float(lookahead_base)
        self.lookahead_speed_gain = float(lookahead_speed_gain)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_max = float(lookahead_max)
        self.progress_search_window = int(progress_search_window)
        self.min_clearance_m = float(min_clearance_m)
        self.full_clearance_m = float(full_clearance_m)
        self.near_obs_dist_m = float(near_obs_dist_m)
        self.max_dense_ratio = float(max_dense_ratio)

        self.path_points_world: Optional[np.ndarray] = None
        self.path_s: Optional[np.ndarray] = None
        self.progress_idx = 0
        self.path_confidence = 0.0
        self.inflation_radius = 0.0

        self.cost_to_go_map: Optional[np.ndarray] = None
        self.cost_to_go_bounds: Optional[Tuple[float, float, float, float]] = None
        self.cost_to_go_goal_radius: float = 2.0

    def clear_path(self) -> None:
        self.path_points_world = None
        self.path_s = None
        self.progress_idx = 0
        self.path_confidence = 0.0
        self.inflation_radius = 0.0

    def _clear_cost_to_go(self) -> None:
        self.cost_to_go_map = None
        self.cost_to_go_bounds = None

    def _clear_all(self) -> None:
        self.clear_path()
        self._clear_cost_to_go()

    def plan_scene_path(
        self,
        scene: SceneSpec,
        vehicle_config: VehicleConfig,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> bool:
        occupancy, bounds, inflation_radius = self._build_occupancy(scene, vehicle_config)
        start = self._world_to_cell(float(start_xy[0]), float(start_xy[1]), bounds, occupancy.shape)
        goal = self._world_to_cell(float(goal_xy[0]), float(goal_xy[1]), bounds, occupancy.shape)
        if start is None or goal is None:
            self._clear_all()
            return False

        occupancy[start[0], start[1]] = 0
        occupancy[goal[0], goal[1]] = 0
        cell_path = self._astar(occupancy, start, goal)

        # Compute the cost-to-go map regardless of A* success — it provides a
        # cheap global topology prior that is useful even without a viable path.
        self._compute_backward_cost_to_go(occupancy, goal, bounds)

        if cell_path is None or len(cell_path) == 0:
            self.clear_path()
            return False

        path_points = np.asarray([self._cell_to_world(i, j, bounds) for i, j in cell_path], dtype=np.float64)
        self.path_points_world = path_points
        self.path_s = self._polyline_arc_length(path_points)
        self.progress_idx = 0
        self.inflation_radius = float(inflation_radius)
        self.path_confidence = self._estimate_path_confidence(
            path_points=path_points,
            start_xy=start_xy,
            goal_xy=goal_xy,
            inflation_radius=inflation_radius,
            vehicle_config=vehicle_config,
        )
        return True

    def get_soft_hint(
        self,
        state_x: float,
        state_y: float,
        heading: float,
        speed: float,
        lidar_norm: Optional[np.ndarray] = None,
        lidar_range: float = 30.0,
    ) -> np.ndarray:
        if self.path_points_world is None or self.path_s is None or len(self.path_points_world) < 2:
            return np.zeros((4,), dtype=np.float32)

        point = np.array([float(state_x), float(state_y)], dtype=np.float64)
        lo = int(max(0, self.progress_idx - 2))
        hi = int(min(len(self.path_points_world), self.progress_idx + self.progress_search_window + 1))
        segment = self.path_points_world[lo:hi]
        if len(segment) == 0:
            segment = self.path_points_world
            lo = 0

        d2 = np.sum((segment - point) ** 2, axis=1)
        best_local_idx = int(np.argmin(d2))
        best_idx = int(lo + best_local_idx)
        self.progress_idx = max(self.progress_idx, best_idx)

        lookahead = self.lookahead_base + self.lookahead_speed_gain * abs(float(speed))
        lookahead = float(np.clip(lookahead, self.lookahead_min, self.lookahead_max))

        waypoint = self._interp_on_path(float(self.path_s[self.progress_idx]) + lookahead)
        if waypoint is None:
            return np.zeros((4,), dtype=np.float32)

        dx = float(waypoint[0] - point[0])
        dy = float(waypoint[1] - point[1])
        cos_heading = math.cos(float(heading))
        sin_heading = math.sin(float(heading))
        x_ego = cos_heading * dx + sin_heading * dy
        y_ego = -sin_heading * dx + cos_heading * dy
        norm = math.hypot(x_ego, y_ego)
        if norm < 1e-6:
            return np.zeros((4,), dtype=np.float32)

        local_hint_strength = self._hint_strength(lidar_norm=lidar_norm, lidar_range=lidar_range)
        guidance_strength = float(np.clip(local_hint_strength * self.path_confidence, 0.0, 1.0))
        ux = x_ego / norm
        uy = y_ego / norm
        lateral_error = float(np.clip(y_ego / max(lookahead, 1e-6), -1.0, 1.0))
        return np.array(
            [
                ux * guidance_strength,
                uy * guidance_strength,
                lateral_error * guidance_strength,
                self.path_confidence,
            ],
            dtype=np.float32,
        )

    def _build_occupancy(
        self,
        scene: SceneSpec,
        vehicle_config: VehicleConfig,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
        xmin, xmax, ymin, ymax = scene.world_bounds
        nx = int(math.ceil((float(xmax) - float(xmin)) / self.grid_resolution)) + 1
        ny = int(math.ceil((float(ymax) - float(ymin)) / self.grid_resolution)) + 1
        bounds = (float(xmin), float(xmax), float(ymin), float(ymax))
        occupancy = np.zeros((nx, ny), dtype=np.uint8)

        inflation_radius = self._resolve_inflation_radius(scene, vehicle_config)
        half_diag = math.sqrt(2.0) * 0.5 * self.grid_resolution
        for obstacle in scene.obstacles:
            expanded = obstacle.buffer(inflation_radius + half_diag + 1e-6)
            if expanded.is_empty:
                continue
            prepared = prep(expanded)
            grid_bounds = self._world_to_grid_bounds(expanded.bounds, bounds, nx, ny)
            if grid_bounds is None:
                continue
            i0, i1, j0, j1 = grid_bounds
            for i in range(i0, i1 + 1):
                x = float(xmin) + float(i) * self.grid_resolution
                for j in range(j0, j1 + 1):
                    y = float(ymin) + float(j) * self.grid_resolution
                    if prepared.intersects(Point(x, y)):
                        occupancy[i, j] = 1
        return occupancy, bounds, inflation_radius

    def _resolve_inflation_radius(self, scene: SceneSpec, vehicle_config: VehicleConfig) -> float:
        vehicle_half_width = 0.5 * float(vehicle_config.body_width)
        corridor_width = scene.metadata.get("corridor_width")
        if corridor_width is None:
            return float(vehicle_half_width)

        # Keep coarse guidance conservative, but cap by the vehicle half-width so wide bays stay traversable.
        residual_half_clearance = 0.5 * max(float(corridor_width) - float(vehicle_config.body_width), 0.0)
        return float(max(0.0, min(vehicle_half_width, residual_half_clearance)))

    def _estimate_path_confidence(
        self,
        path_points: np.ndarray,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        inflation_radius: float,
        vehicle_config: VehicleConfig,
    ) -> float:
        path_s = self._polyline_arc_length(path_points)
        path_length = float(path_s[-1]) if len(path_s) > 0 else 0.0
        direct_distance = float(np.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1])))
        if path_length <= 1e-6:
            efficiency = 1.0
        else:
            efficiency = float(np.clip(direct_distance / max(path_length, direct_distance, 1e-6), 0.0, 1.0))
        vehicle_half_width = max(0.5 * float(vehicle_config.body_width), 1e-6)
        safety_ratio = float(np.clip(float(inflation_radius) / vehicle_half_width, 0.0, 1.0))
        return float(np.clip(0.5 * efficiency + 0.5 * safety_ratio, 0.0, 1.0))

    def _world_to_grid_bounds(
        self,
        geometry_bounds: Tuple[float, float, float, float],
        bounds: Tuple[float, float, float, float],
        nx: int,
        ny: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        xmin, _, ymin, _ = bounds
        gxmin, gymin, gxmax, gymax = geometry_bounds
        i0 = max(0, int(math.floor((float(gxmin) - float(xmin)) / self.grid_resolution)))
        i1 = min(nx - 1, int(math.ceil((float(gxmax) - float(xmin)) / self.grid_resolution)))
        j0 = max(0, int(math.floor((float(gymin) - float(ymin)) / self.grid_resolution)))
        j1 = min(ny - 1, int(math.ceil((float(gymax) - float(ymin)) / self.grid_resolution)))
        if i0 > i1 or j0 > j1:
            return None
        return i0, i1, j0, j1

    def _world_to_cell(
        self,
        x: float,
        y: float,
        bounds: Tuple[float, float, float, float],
        shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        xmin, xmax, ymin, ymax = bounds
        nx, ny = shape
        if x < float(xmin) or x > float(xmax) or y < float(ymin) or y > float(ymax):
            return None
        i = int(round((float(x) - float(xmin)) / self.grid_resolution))
        j = int(round((float(y) - float(ymin)) / self.grid_resolution))
        i = min(max(i, 0), nx - 1)
        j = min(max(j, 0), ny - 1)
        return i, j

    def _cell_to_world(self, i: int, j: int, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
        xmin, _, ymin, _ = bounds
        return float(xmin) + float(i) * self.grid_resolution, float(ymin) + float(j) * self.grid_resolution

    def _compute_backward_cost_to_go(
        self,
        occupancy: np.ndarray,
        goal_cell: Tuple[int, int],
        bounds: Tuple[float, float, float, float],
    ) -> None:
        """Run backward Dijkstra from the goal area to compute J(x,y) for all free cells.

        The cost-to-go map assigns 0 to the goal area (cells within ``cost_to_go_goal_radius``
        of the goal cell) and then propagates outward using 8-connected grid steps.
        """
        nx, ny = occupancy.shape
        cost_map = np.full((nx, ny), np.inf, dtype=np.float64)

        radius_cells = int(self.cost_to_go_goal_radius / max(self.grid_resolution, 1e-6))
        gi, gj = goal_cell
        goal_cells: List[Tuple[int, int]] = []
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                ni, nj = gi + di, gj + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    if math.hypot(float(di), float(dj)) <= float(radius_cells) and occupancy[ni, nj] == 0:
                        cost_map[ni, nj] = 0.0
                        goal_cells.append((ni, nj))

        if not goal_cells:
            self.cost_to_go_map = None
            self.cost_to_go_bounds = None
            return

        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, cell) for cell in goal_cells]
        heapq.heapify(heap)

        while heap:
            cost, (ci, cj) = heapq.heappop(heap)
            if float(cost) > float(cost_map[ci, cj]) + 1e-12:
                continue
            for di, dj, step_cost in moves:
                ni, nj = ci + di, cj + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                if occupancy[ni, nj] != 0:
                    continue
                new_cost = float(cost + step_cost)
                if new_cost + 1e-12 < float(cost_map[ni, nj]):
                    cost_map[ni, nj] = new_cost
                    heapq.heappush(heap, (new_cost, (ni, nj)))

        self.cost_to_go_map = cost_map.astype(np.float32)
        self.cost_to_go_bounds = bounds

    def query_cost_to_go(self, x: float, y: float) -> Optional[float]:
        value, _ = self.query_cost_to_go_details(x, y)
        return value

    def query_cost_to_go_details(self, x: float, y: float) -> Tuple[Optional[float], str]:
        """Query J(x,y) and report whether the value is direct or projected.

        Returns ``(value, "direct")`` for a valid bilinear interpolation,
        ``(value, "projected")`` when the nearest finite free-cell is used as a
        fallback, and ``(None, "unavailable")`` when neither is available.
        """
        if self.cost_to_go_map is None or self.cost_to_go_bounds is None:
            return None, "unavailable"
        xmin, xmax, ymin, ymax = self.cost_to_go_bounds
        nx, ny = int(self.cost_to_go_map.shape[0]), int(self.cost_to_go_map.shape[1])
        resolution = max(self.grid_resolution, 1e-6)
        inside_bounds = bool(float(xmin) <= float(x) <= float(xmax) and float(ymin) <= float(y) <= float(ymax))

        if inside_bounds:
            grid_x = float(np.clip((float(x) - float(xmin)) / resolution, 0.0, max(nx - 1, 0)))
            grid_y = float(np.clip((float(y) - float(ymin)) / resolution, 0.0, max(ny - 1, 0)))
            value = self._query_cost_to_go_grid(grid_x, grid_y)
            if value is not None:
                return value, "direct"
        else:
            grid_x = float(np.clip((float(x) - float(xmin)) / resolution, 0.0, max(nx - 1, 0)))
            grid_y = float(np.clip((float(y) - float(ymin)) / resolution, 0.0, max(ny - 1, 0)))

        projected_value = self._project_cost_to_go_to_nearest_valid_cell(grid_x, grid_y)
        if projected_value is not None:
            return projected_value, "projected"
        return None, "unavailable"

    def _query_cost_to_go_grid(self, grid_x: float, grid_y: float) -> Optional[float]:
        if self.cost_to_go_map is None:
            return None
        nx, ny = int(self.cost_to_go_map.shape[0]), int(self.cost_to_go_map.shape[1])
        i0 = int(math.floor(grid_x))
        j0 = int(math.floor(grid_y))
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        tx = float(grid_x - i0)
        ty = float(grid_y - j0)

        weighted_value = 0.0
        total_weight = 0.0
        for i, wx in ((i0, 1.0 - tx), (i1, tx)):
            for j, wy in ((j0, 1.0 - ty), (j1, ty)):
                weight = float(wx * wy)
                if weight <= 0.0:
                    continue
                val = float(self.cost_to_go_map[i, j])
                if not np.isfinite(val):
                    continue
                weighted_value += weight * val
                total_weight += weight

        if total_weight <= 1e-9:
            return None
        return float(weighted_value / total_weight)

    def _project_cost_to_go_to_nearest_valid_cell(self, grid_x: float, grid_y: float) -> Optional[float]:
        if self.cost_to_go_map is None:
            return None
        nx, ny = int(self.cost_to_go_map.shape[0]), int(self.cost_to_go_map.shape[1])
        center_i = int(round(float(np.clip(grid_x, 0.0, max(nx - 1, 0)))))
        center_j = int(round(float(np.clip(grid_y, 0.0, max(ny - 1, 0)))))
        max_radius = max(nx, ny)

        for radius in range(max_radius + 1):
            best_value = None
            best_dist2 = float("inf")
            i0 = max(0, center_i - radius)
            i1 = min(nx - 1, center_i + radius)
            j0 = max(0, center_j - radius)
            j1 = min(ny - 1, center_j + radius)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    if radius > 0 and abs(i - center_i) < radius and abs(j - center_j) < radius:
                        continue
                    value = float(self.cost_to_go_map[i, j])
                    if not np.isfinite(value):
                        continue
                    dist2 = float((float(i) - grid_x) ** 2 + (float(j) - grid_y) ** 2)
                    if dist2 + 1e-12 < best_dist2:
                        best_dist2 = dist2
                        best_value = value
            if best_value is not None:
                return float(best_value)
        return None

    def _astar(
        self,
        occupancy: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        nx, ny = occupancy.shape

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))

        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
        parents = {start: None}
        g_cost = {start: 0.0}

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current == goal:
                path: List[Tuple[int, int]] = []
                cursor = current
                while cursor is not None:
                    path.append(cursor)
                    cursor = parents[cursor]
                path.reverse()
                return path

            if current_cost > g_cost.get(current, 1e18) + 1e-12:
                continue

            ci, cj = current
            for di, dj, step_cost in moves:
                ni, nj = ci + di, cj + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                if occupancy[ni, nj] != 0:
                    continue
                next_cost = float(current_cost + step_cost)
                nxt = (ni, nj)
                if next_cost + 1e-12 >= g_cost.get(nxt, 1e18):
                    continue
                g_cost[nxt] = next_cost
                parents[nxt] = current
                heapq.heappush(open_heap, (next_cost + heuristic(nxt, goal), next_cost, nxt))

        return None

    def _polyline_arc_length(self, points: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        if points.shape[0] == 1:
            return np.zeros((1,), dtype=np.float64)
        deltas = np.linalg.norm(points[1:] - points[:-1], axis=1)
        path_s = np.zeros((points.shape[0],), dtype=np.float64)
        path_s[1:] = np.cumsum(deltas)
        return path_s

    def _interp_on_path(self, arc_query: float) -> Optional[np.ndarray]:
        if self.path_points_world is None or self.path_s is None or len(self.path_points_world) == 0:
            return None
        if arc_query <= float(self.path_s[0]):
            return self.path_points_world[0].copy()
        if arc_query >= float(self.path_s[-1]):
            return self.path_points_world[-1].copy()

        idx = int(np.searchsorted(self.path_s, arc_query))
        idx0 = max(0, idx - 1)
        idx1 = min(len(self.path_s) - 1, idx)
        s0 = float(self.path_s[idx0])
        s1 = float(self.path_s[idx1])
        if s1 - s0 < 1e-9:
            return self.path_points_world[idx1].copy()
        ratio = float((arc_query - s0) / (s1 - s0))
        return (1.0 - ratio) * self.path_points_world[idx0] + ratio * self.path_points_world[idx1]

    def _hint_strength(self, lidar_norm: Optional[np.ndarray], lidar_range: float) -> float:
        if lidar_norm is None or len(lidar_norm) == 0:
            return 1.0
        lidar_values = np.clip(np.asarray(lidar_norm, dtype=np.float64).reshape(-1), 0.0, 1.0) * float(lidar_range)
        min_lidar = float(np.min(lidar_values))
        dense_ratio = float(np.mean(lidar_values < self.near_obs_dist_m))
        clear_span = max(1e-6, self.full_clearance_m - self.min_clearance_m)
        clear_factor = float(np.clip((min_lidar - self.min_clearance_m) / clear_span, 0.0, 1.0))
        dense_factor = float(1.0 - np.clip(dense_ratio / max(1e-6, self.max_dense_ratio), 0.0, 1.0))
        return float(np.clip(clear_factor * dense_factor, 0.0, 1.0))