from common.config import VehicleConfig
from common.types import ArticulatedState
from env.success import ParkingSuccessChecker


def test_success_checker_matches_identical_pose():
    checker = ParkingSuccessChecker(VehicleConfig())
    state = ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0)
    metrics = checker.evaluate(state, state, collision_free=True)
    assert metrics.front_overlap_ratio > 0.99
    assert metrics.heading_error_rad == 0.0
    assert metrics.success is True
