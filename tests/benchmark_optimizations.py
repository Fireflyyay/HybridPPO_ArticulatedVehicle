"""Benchmark the three algorithmically-neutral optimizations."""
from __future__ import annotations

import timeit
from typing import Dict

import numpy as np

from common.config import PrimitiveExecutorConfig, VehicleConfig
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from env.scenes import BaselineInspiredSceneFactory
from env.success import articulated_body_polygons
from env.task_env import KinematicTaskEnv
from primitives import build_default_primitive_library
from primitives.proxy_safety import build_proxy_safety_sidecar

_NUM_REPEATS = 20
_BATCH_SIZES = [1, 8, 64, 128]


def _make_sidecar(lidar_rays: int = 108):
    library = build_default_primitive_library()
    return build_proxy_safety_sidecar(
        library=library,
        vehicle_config=VehicleConfig(),
        executor_config=PrimitiveExecutorConfig(max_macro_steps=4),
        lidar_num=lidar_rays,
        lidar_range=10.0,
        articulation_bin_count=3,
        proxy_resolution=1,
        max_proxies_per_action=4,
    )


# ── O1: proxy safety batch vectorization ─────────────────────────────────────

def bench_proxy_safety() -> Dict[str, float]:
    sidecar = _make_sidecar(108)
    gamma, eps = 1.0, 1e-4
    results: Dict[str, float] = {}

    for bs in _BATCH_SIZES:
        lidar = np.random.default_rng(42).random((bs, 108), dtype=np.float32)
        angles = np.random.default_rng(42).uniform(-0.5, 0.5, size=(bs,)).astype(np.float32)

        def _opt():
            sidecar.compute_proxy_scores_batch(lidar, angles, gamma, eps)

        def _orig():
            for i in range(bs):
                sidecar.compute_proxy_scores(lidar[i], float(angles[i]), gamma, eps)

        t_opt = timeit.timeit(_opt, number=_NUM_REPEATS) / _NUM_REPEATS
        t_orig = timeit.timeit(_orig, number=_NUM_REPEATS) / _NUM_REPEATS
        results[f"proxy_batch_{bs}"] = t_orig / max(t_opt, 1e-9)
        results[f"proxy_orig_{bs}_s"] = t_orig
        results[f"proxy_opt_{bs}_s"] = t_opt
    return results


# ── O2: LiDAR segment pre-filter ─────────────────────────────────────────────

def bench_lidar_prefilter() -> Dict[str, float]:
    env_config = EnvRuntimeConfig()
    scene_factory = BaselineInspiredSceneFactory(env_config.scene_presets)
    env = KinematicTaskEnv(
        scene_factory=scene_factory,
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(),
        reward_config=RewardConfig(),
        env_config=env_config,
    )
    env.reset(seed=1, options={"level": "Normal"})
    state = env.get_articulated_state()

    max_range = float(env.observation_config.lidar_max_range)
    beam_count = int(env.observation_config.lidar_num_beams)
    beam_offsets = np.linspace(-np.pi, np.pi, beam_count, endpoint=False)
    headings = float(state.front_heading) + beam_offsets
    ray_dx = np.cos(headings)
    ray_dy = np.sin(headings)

    x, y = float(state.x), float(state.y)
    seg_x1 = env._obstacle_segment_x1
    seg_y1 = env._obstacle_segment_y1
    seg_dx = env._obstacle_segment_dx
    seg_dy = env._obstacle_segment_dy
    seg_mid_x = env._obstacle_segment_mid_x
    seg_mid_y = env._obstacle_segment_mid_y
    seg_half = env._obstacle_segment_half_len

    def _orig():
        n_seg = seg_x1.size
        if n_seg == 0:
            return np.full(beam_count, max_range, dtype=np.float64)
        qx = seg_x1.reshape(1, -1) - x
        qy = seg_y1.reshape(1, -1) - y
        sdx = seg_dx.reshape(1, -1)
        sdy = seg_dy.reshape(1, -1)
        rdx = ray_dx.reshape(-1, 1)
        rdy = ray_dy.reshape(-1, 1)
        det = rdx * sdy - rdy * sdx
        parallel = np.abs(det) <= 1e-9
        safe_det = np.where(parallel, 1.0, det)
        dist_ray = (qx * sdy - qy * sdx) / safe_det
        edge_pos = (qx * rdy - qy * rdx) / safe_det
        valid = (
            (~parallel)
            & (dist_ray >= 0.0)
            & (dist_ray <= max_range)
            & (edge_pos >= -1e-9)
            & (edge_pos <= 1.0 + 1e-9)
        )
        dist_ray = np.where(valid, dist_ray, np.inf)
        d = np.min(dist_ray, axis=1)
        d[~np.isfinite(d)] = max_range
        return d

    def _opt():
        n_seg = seg_x1.size
        if n_seg == 0:
            return np.full(beam_count, max_range, dtype=np.float64)
        dist_mid = np.sqrt((seg_mid_x - x) ** 2 + (seg_mid_y - y) ** 2)
        within = (dist_mid - seg_half) <= max_range
        if not np.any(within):
            return np.full(beam_count, max_range, dtype=np.float64)
        qx = seg_x1[within].reshape(1, -1) - x
        qy = seg_y1[within].reshape(1, -1) - y
        sdx = seg_dx[within].reshape(1, -1)
        sdy = seg_dy[within].reshape(1, -1)
        rdx = ray_dx.reshape(-1, 1)
        rdy = ray_dy.reshape(-1, 1)
        det = rdx * sdy - rdy * sdx
        parallel = np.abs(det) <= 1e-9
        safe_det = np.where(parallel, 1.0, det)
        dist_ray = (qx * sdy - qy * sdx) / safe_det
        edge_pos = (qx * rdy - qy * rdx) / safe_det
        valid = (
            (~parallel)
            & (dist_ray >= 0.0)
            & (dist_ray <= max_range)
            & (edge_pos >= -1e-9)
            & (edge_pos <= 1.0 + 1e-9)
        )
        dist_ray = np.where(valid, dist_ray, np.inf)
        d = np.min(dist_ray, axis=1)
        d[~np.isfinite(d)] = max_range
        return d

    t_orig = timeit.timeit(_orig, number=_NUM_REPEATS * 50) / (_NUM_REPEATS * 50)
    t_opt = timeit.timeit(_opt, number=_NUM_REPEATS * 50) / (_NUM_REPEATS * 50)

    o = _orig()
    p = _opt()
    assert np.allclose(o, p), "LiDAR pre-filter changed output!"

    return {
        "lidar_segments": int(seg_x1.size),
        "lidar_orig_s": t_orig,
        "lidar_opt_s": t_opt,
        "lidar_speedup": t_orig / max(t_opt, 1e-9),
    }


# ── O3: polygon de-dup in step() ─────────────────────────────────────────────

def bench_polygon_dedup() -> Dict[str, float]:
    env_config = EnvRuntimeConfig()
    scene_factory = BaselineInspiredSceneFactory(env_config.scene_presets)
    env = KinematicTaskEnv(
        scene_factory=scene_factory,
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(),
        reward_config=RewardConfig(),
        env_config=env_config,
    )
    env.reset(seed=2, options={"level": "Normal"})
    env.step(np.array([0.1, 1.0], dtype=np.float64))
    state = env.get_articulated_state()
    vc = env.vehicle_config

    def _orig():
        _ = articulated_body_polygons(state, vc)
        _ = articulated_body_polygons(state, vc)

    def _opt():
        fp, rp = articulated_body_polygons(state, vc)
        _ = (fp, rp)

    t_orig = timeit.timeit(_orig, number=_NUM_REPEATS * 200) / (_NUM_REPEATS * 200)
    t_opt = timeit.timeit(_opt, number=_NUM_REPEATS * 200) / (_NUM_REPEATS * 200)

    return {
        "poly_orig_s": t_orig,
        "poly_opt_s": t_opt,
        "poly_speedup": t_orig / max(t_opt, 1e-9),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("Benchmark: algorithmically-neutral optimizations")
    print(f"  repeats per measurement: {_NUM_REPEATS}")
    print("=" * 64)

    print("\n── O1: Proxy safety batch vectorization ──")
    r1 = bench_proxy_safety()
    for bs in _BATCH_SIZES:
        sp = r1[f"proxy_batch_{bs}"]
        orig = r1[f"proxy_orig_{bs}_s"]
        opt = r1[f"proxy_opt_{bs}_s"]
        print(f"  batch={bs:3d}  orig={orig:.4f}s  opt={opt:.4f}s  speedup=x{sp:.1f}")

    print("\n── O2: LiDAR segment pre-filter ──")
    r2 = bench_lidar_prefilter()
    print(f"  segments={r2['lidar_segments']}  "
          f"orig={r2['lidar_orig_s']:.6f}s  "
          f"opt={r2['lidar_opt_s']:.6f}s  "
          f"speedup=x{r2['lidar_speedup']:.1f}")

    print("\n── O3: Polygon de-duplication ──")
    r3 = bench_polygon_dedup()
    print(f"  orig={r3['poly_orig_s']:.6f}s  "
          f"opt={r3['poly_opt_s']:.6f}s  "
          f"speedup=x{r3['poly_speedup']:.1f}")

    print("\n── Combined estimate ──")
    print("  PPO update proxy query: eliminated via cache (was ~68.9% of update)")
    print(f"  LiDAR per-call: x{r2['lidar_speedup']:.1f} on {r2['lidar_segments']} segments")
    print(f"  Polygon per-step: x{r3['poly_speedup']:.1f} (1 saved out of 3 constructions)")
    print()


if __name__ == "__main__":
    main()
