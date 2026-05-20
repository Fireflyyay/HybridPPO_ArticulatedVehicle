from hybridppo_articulated_vehicle.config import VehicleConfig
from hybridppo_articulated_vehicle.success import ParkingSuccessChecker
from hybridppo_articulated_vehicle.types import ArticulatedState


def test_success_checker_matches_identical_pose():
    checker = ParkingSuccessChecker(VehicleConfig())
    state = ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0)
    metrics = checker.evaluate(state, state, collision_free=True)
    assert metrics.front_overlap_ratio > 0.99
    assert metrics.heading_error_rad == 0.0
    assert metrics.success is True
