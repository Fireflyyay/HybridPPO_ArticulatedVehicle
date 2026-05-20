import numpy as np

from common.config import PrimitiveExecutorConfig, VehicleConfig
from primitives import build_default_primitive_library
from primitives.proxy_safety import ProxySafetySidecar, build_proxy_safety_sidecar, save_proxy_safety_sidecar



def test_proxy_safety_sidecar_uses_nearest_articulation_bin_and_prefix_scores():
    required_clearance = np.zeros((2, 1, 2, 3, 4), dtype=np.float32)
    required_clearance[0, 0, 0, :, :] = 2.0
    required_clearance[0, 0, 1, :, :] = 8.0
    required_clearance[1, 0, 0, :, :] = 7.0
    required_clearance[1, 0, 1, 0, :] = 2.0
    required_clearance[1, 0, 1, 1:, :] = 9.0

    sidecar = ProxySafetySidecar(
        required_clearance=required_clearance,
        articulation_bin_centers=np.array([-0.4, 0.4], dtype=np.float32),
        parameter_centers=np.zeros((1, 2, 3), dtype=np.float32),
        parameter_scales=np.ones((1, 2, 3), dtype=np.float32),
        nominal_horizons=np.full((1, 2), 3, dtype=np.int64),
        proxy_valid_mask=np.ones((1, 2), dtype=np.bool_),
        lidar_range=10.0,
    )

    lidar = np.full((4,), 0.5, dtype=np.float32)
    negative_query = sidecar.compute_proxy_scores(lidar_observation=lidar, articulation_angle=-0.3, gamma=1.0, eps=1e-4)
    positive_query = sidecar.compute_proxy_scores(lidar_observation=lidar, articulation_angle=0.3, gamma=1.0, eps=1e-4)

    assert negative_query.articulation_bin_index == 0
    assert positive_query.articulation_bin_index == 1
    assert np.allclose(negative_query.prefix_lengths, np.array([[3.0, 0.0]], dtype=np.float32))
    assert np.allclose(negative_query.proxy_scores, np.array([[1.0, 1e-4]], dtype=np.float32))
    assert np.allclose(positive_query.prefix_lengths, np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.allclose(positive_query.proxy_scores, np.array([[1e-4, 1.0 / 3.0]], dtype=np.float32))


def test_proxy_safety_sidecar_zeroes_invalid_proxy_scores():
    sidecar = ProxySafetySidecar(
        required_clearance=np.zeros((1, 1, 2, 2, 2), dtype=np.float32),
        articulation_bin_centers=np.array([0.0], dtype=np.float32),
        parameter_centers=np.zeros((1, 2, 2), dtype=np.float32),
        parameter_scales=np.ones((1, 2, 2), dtype=np.float32),
        nominal_horizons=np.full((1, 2), 2, dtype=np.int64),
        proxy_valid_mask=np.array([[True, False]], dtype=np.bool_),
        lidar_range=8.0,
    )

    query = sidecar.compute_proxy_scores(
        lidar_observation=np.ones((2,), dtype=np.float32),
        articulation_angle=0.0,
        gamma=1.0,
        eps=1e-4,
    )

    assert np.allclose(query.prefix_lengths, np.array([[2.0, 2.0]], dtype=np.float32))
    assert np.allclose(query.proxy_scores, np.array([[1.0, 0.0]], dtype=np.float32))


def test_proxy_safety_builder_smoke():
    library = build_default_primitive_library()
    sidecar = build_proxy_safety_sidecar(
        library=library,
        vehicle_config=VehicleConfig(),
        executor_config=PrimitiveExecutorConfig(max_macro_steps=4),
        lidar_num=4,
        lidar_range=10.0,
        articulation_bin_count=2,
        proxy_resolution=1,
        max_proxies_per_action=1,
    )

    assert sidecar.required_clearance.shape == (2, library.action_dim, 1, 4, 4)
    assert sidecar.proxy_valid_mask.shape == (library.action_dim, 1)
    assert np.all(sidecar.nominal_horizons >= 1)


def test_save_proxy_safety_sidecar_creates_parent_directory(tmp_path):
    sidecar = ProxySafetySidecar(
        required_clearance=np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
        articulation_bin_centers=np.array([0.0], dtype=np.float32),
        parameter_centers=np.zeros((1, 1, 1), dtype=np.float32),
        parameter_scales=np.ones((1, 1, 1), dtype=np.float32),
        nominal_horizons=np.ones((1, 1), dtype=np.int64),
        proxy_valid_mask=np.ones((1, 1), dtype=np.bool_),
        lidar_range=10.0,
    )
    output_path = tmp_path / "nested" / "proxy_safety_sidecar.npz"

    save_proxy_safety_sidecar(str(output_path), sidecar)

    assert output_path.exists()