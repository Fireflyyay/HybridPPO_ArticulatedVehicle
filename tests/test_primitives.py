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


def test_reverse_align_near_articulation_limit_unwinds_instead_of_one_step_saturating():
    library = build_default_primitive_library()
    executor = ParameterizedPrimitiveExecutor(library)
    articulation_limit = float(executor.vehicle_config.articulation_limit_rad)
    state = ArticulatedState(x=0.0, y=0.0, front_heading=articulation_limit - 0.01, rear_heading=0.0)
    params = library.dict_to_vector(
        {
            "path_length": 6.0,
            "duration": 5.0,
            "speed_scale": 0.52,
            "phi_target": 0.0,
            "heading_tolerance": 0.19,
            "position_tolerance": 0.44,
            "smoothness": 0.53,
        }
    )

    rollout = executor.rollout(state, 4, params, PrimitiveExecutionContext(goal_heading=1.5))

    assert rollout.tau > 1
    assert float(rollout.controls[0].articulation_rate) * float(state.articulation_angle) < 0.0
    assert abs(float(rollout.states[1].articulation_angle)) < abs(float(state.articulation_angle))


def test_reverse_align_near_articulation_limit_prefers_safe_same_direction_clip():
    library = build_default_primitive_library()
    executor = ParameterizedPrimitiveExecutor(library)
    articulation = float(executor.vehicle_config.articulation_limit_rad) - 0.05
    state = ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=-articulation)
    context = PrimitiveExecutionContext(goal_heading=1.2)
    params = library.dict_to_vector(
        {
            "path_length": 4.0,
            "duration": 4.0,
            "speed_scale": 0.45,
            "phi_target": 0.0,
            "heading_tolerance": 0.2,
            "position_tolerance": 0.4,
            "smoothness": 0.0,
        }
    )
    parameter_map = executor._normalize_parameter_map(4, params)
    raw_control = executor._semantic_control(library.spec(4), state, parameter_map, context)

    rollout = executor.rollout(state, 4, params, context)

    assert float(raw_control.articulation_rate) > 0.0
    assert rollout.tau > 1
    assert float(rollout.controls[0].articulation_rate) > 0.0
    assert abs(float(rollout.states[1].articulation_angle)) < float(executor.vehicle_config.articulation_limit_rad)
