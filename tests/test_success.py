from common.config import VehicleConfig
from common.types import ArticulatedState
from env.success import ParkingSuccessChecker


def test_vehicle_config_defaults_match_ppo_articulated_vehicle():
    config = VehicleConfig()
    assert config.wheel_base == 3.6
    assert config.hitch_offset == 1.8
    assert config.trailer_length == 1.8
    assert config.step_seconds == 0.2
    assert config.integrator_substeps == 80
    assert config.speed_min == -2.5
    assert config.speed_max == 2.5
    assert config.articulation_rate_min == -0.6632251157578452
    assert config.articulation_rate_max == 0.6632251157578452
    assert config.articulation_limit_rad == 0.6283185307179586
    assert config.front_length == 5.0
    assert config.rear_length == 4.4
    assert config.body_width == 3.43


def test_success_checker_matches_identical_pose():
    checker = ParkingSuccessChecker(VehicleConfig())
    state = ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0)
    metrics = checker.evaluate(state, state, collision_free=True)
    assert metrics.front_overlap_ratio > 0.99
    assert metrics.heading_error_rad == 0.0
    assert metrics.success is True
