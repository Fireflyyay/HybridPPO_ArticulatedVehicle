import numpy as np

from env.global_guidance import CoarseGlobalGuidance


def test_query_cost_to_go_uses_bilinear_interpolation_for_continuous_progress():
    guidance = CoarseGlobalGuidance(grid_resolution=1.0)
    guidance.cost_to_go_map = np.asarray(
        [
            [4.0, 3.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    guidance.cost_to_go_bounds = (0.0, 1.0, 0.0, 1.0)

    assert np.isclose(guidance.query_cost_to_go(0.5, 0.5), 2.5)