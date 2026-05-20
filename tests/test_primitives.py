from common.types import ArticulatedState, PrimitiveExecutionContext
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library


def test_parameterized_primitive_rollout_stays_feasible():
    library = build_default_primitive_library()
    executor = ParameterizedPrimitiveExecutor(library)
    state = ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0)
    params = library.dict_to_vector(
        {
            "path_length": 2.0,
            "duration": 1.0,
            "speed_scale": 0.6,
            "omega_scale": 0.5,
            "phi_target": 0.15,
            "smoothness": 0.3,
        }
    )
    rollout = executor.rollout(state, 0, params, PrimitiveExecutionContext(goal_heading=0.2))
    assert rollout.tau > 0
    assert rollout.states[-1].x > rollout.states[0].x
    assert abs(rollout.states[-1].articulation_angle) <= executor.vehicle_config.articulation_limit_rad + 1e-6
