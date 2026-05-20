from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..common.config import SuccessCriteriaConfig, VehicleConfig
from ..common.types import ArticulatedState, wrap_to_pi

try:
    from shapely.geometry import Polygon
except Exception as exc:  # pragma: no cover
    Polygon = None
    _SHAPELY_IMPORT_ERROR = exc
else:
    _SHAPELY_IMPORT_ERROR = None


@dataclass(frozen=True)
class SuccessMetrics:
    front_overlap_ratio: float
    rear_overlap_ratio: float
    heading_error_rad: float
    success: bool


def _require_shapely() -> None:
    if Polygon is None:
        raise RuntimeError("shapely is required for success metrics") from _SHAPELY_IMPORT_ERROR


def _rectangle_polygon(center_x: float, center_y: float, heading: float, length: float, width: float):
    dx = 0.5 * float(length)
    dy = 0.5 * float(width)
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float64)
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]], dtype=np.float64)
    points = corners @ rotation.T
    points[:, 0] += float(center_x)
    points[:, 1] += float(center_y)
    return Polygon(points)


def articulated_body_polygons(state: ArticulatedState, vehicle_config: VehicleConfig):
    _require_shapely()
    hinge_x = float(state.x) - float(vehicle_config.hitch_offset) * np.cos(float(state.front_heading))
    hinge_y = float(state.y) - float(vehicle_config.hitch_offset) * np.sin(float(state.front_heading))
    front_center_x = hinge_x + 0.5 * float(vehicle_config.front_length) * np.cos(float(state.front_heading))
    front_center_y = hinge_y + 0.5 * float(vehicle_config.front_length) * np.sin(float(state.front_heading))
    rear_center_x = hinge_x - 0.5 * float(vehicle_config.rear_length) * np.cos(float(state.rear_heading))
    rear_center_y = hinge_y - 0.5 * float(vehicle_config.rear_length) * np.sin(float(state.rear_heading))
    front_poly = _rectangle_polygon(front_center_x, front_center_y, state.front_heading, vehicle_config.front_length, vehicle_config.body_width)
    rear_poly = _rectangle_polygon(rear_center_x, rear_center_y, state.rear_heading, vehicle_config.rear_length, vehicle_config.body_width)
    return front_poly, rear_poly


class ParkingSuccessChecker:
    def __init__(self, vehicle_config: VehicleConfig, criteria: SuccessCriteriaConfig = SuccessCriteriaConfig()) -> None:
        self.vehicle_config = vehicle_config
        self.criteria = criteria

    def evaluate(self, state: ArticulatedState, goal_state: ArticulatedState, collision_free: bool = True) -> SuccessMetrics:
        current_front, current_rear = articulated_body_polygons(state, self.vehicle_config)
        goal_front, goal_rear = articulated_body_polygons(goal_state, self.vehicle_config)
        front_overlap = float(current_front.intersection(goal_front).area) / (float(goal_front.area) + 1e-9)
        rear_overlap = float(current_rear.intersection(goal_rear).area) / (float(goal_rear.area) + 1e-9)
        heading_error = abs(wrap_to_pi(float(state.front_heading) - float(goal_state.front_heading)))
        success = (collision_free or not self.criteria.require_collision_free) and heading_error <= float(self.criteria.heading_threshold_rad) and front_overlap >= float(self.criteria.front_overlap_threshold)
        return SuccessMetrics(front_overlap_ratio=float(front_overlap), rear_overlap_ratio=float(rear_overlap), heading_error_rad=float(heading_error), success=bool(success))

    def spec(self) -> Dict[str, float]:
        return {
            "heading_threshold_rad": float(self.criteria.heading_threshold_rad),
            "front_overlap_threshold": float(self.criteria.front_overlap_threshold),
            "require_collision_free": bool(self.criteria.require_collision_free),
        }