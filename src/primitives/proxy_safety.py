from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def primitive_library_signature(spec_payload: Sequence[Mapping[str, object]], parameter_names: Sequence[str], bounds: Mapping[str, Tuple[float, float]]) -> str:
    payload = {
        "specs": [dict(item) for item in spec_payload],
        "parameter_names": list(map(str, parameter_names)),
        "bounds": {str(name): [float(bounds[name][0]), float(bounds[name][1])] for name in parameter_names},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


@dataclass(frozen=True)
class ProxySafetyQueryResult:
    articulation_bin_index: int
    prefix_lengths: np.ndarray
    proxy_scores: np.ndarray


@dataclass(frozen=True)
class ProxySafetySidecar:
    required_clearance: np.ndarray
    articulation_bin_centers: np.ndarray
    parameter_centers: np.ndarray
    parameter_scales: np.ndarray
    nominal_horizons: np.ndarray
    proxy_valid_mask: np.ndarray
    lidar_range: float
    library_signature: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_clearance = np.asarray(self.required_clearance, dtype=np.float32)
        articulation_bin_centers = np.asarray(self.articulation_bin_centers, dtype=np.float32).reshape(-1)
        parameter_centers = np.asarray(self.parameter_centers, dtype=np.float32)
        parameter_scales = np.asarray(self.parameter_scales, dtype=np.float32)
        nominal_horizons = np.asarray(self.nominal_horizons, dtype=np.int64)
        proxy_valid_mask = np.asarray(self.proxy_valid_mask, dtype=np.bool_)
        if required_clearance.ndim != 5:
            raise ValueError("required_clearance must have shape [art_bin, action, proxy, step, ray]")
        if articulation_bin_centers.shape[0] != required_clearance.shape[0]:
            raise ValueError("articulation_bin_centers must match required_clearance art_bin dimension")
        if parameter_centers.shape[:2] != required_clearance.shape[1:3]:
            raise ValueError("parameter_centers must match [action, proxy] dimensions")
        if parameter_scales.shape != parameter_centers.shape:
            raise ValueError("parameter_scales must match parameter_centers shape")
        if nominal_horizons.shape != required_clearance.shape[1:3]:
            raise ValueError("nominal_horizons must match [action, proxy] dimensions")
        if proxy_valid_mask.shape != required_clearance.shape[1:3]:
            raise ValueError("proxy_valid_mask must match [action, proxy] dimensions")
        object.__setattr__(self, "required_clearance", required_clearance)
        object.__setattr__(self, "articulation_bin_centers", articulation_bin_centers)
        object.__setattr__(self, "parameter_centers", parameter_centers)
        object.__setattr__(self, "parameter_scales", parameter_scales)
        object.__setattr__(self, "nominal_horizons", nominal_horizons)
        object.__setattr__(self, "proxy_valid_mask", proxy_valid_mask)
        object.__setattr__(self, "lidar_range", float(self.lidar_range))

    @property
    def num_articulation_bins(self) -> int:
        return int(self.required_clearance.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.required_clearance.shape[1])

    @property
    def num_proxies(self) -> int:
        return int(self.required_clearance.shape[2])

    @property
    def num_steps(self) -> int:
        return int(self.required_clearance.shape[3])

    @property
    def num_rays(self) -> int:
        return int(self.required_clearance.shape[4])

    @property
    def parameter_dim(self) -> int:
        return int(self.parameter_centers.shape[-1])

    def compatible_with(self, action_dim: int, parameter_dim: int, expected_signature: Optional[str] = None) -> bool:
        if int(action_dim) != self.num_actions:
            return False
        if int(parameter_dim) != self.parameter_dim:
            return False
        signature = str(expected_signature or "").strip()
        if not signature:
            return True
        current = str(self.library_signature or "").strip()
        return (not current) or current == signature

    def select_articulation_bin(self, articulation_angle: float) -> int:
        diffs = np.abs(_wrap_angle(float(articulation_angle) - self.articulation_bin_centers))
        return int(np.argmin(diffs))

    def compute_proxy_scores(self, lidar_observation: np.ndarray, articulation_angle: float, gamma: float, eps: float) -> ProxySafetyQueryResult:
        lidar = np.asarray(lidar_observation, dtype=np.float32).reshape(1, -1)
        if lidar.shape[1] != self.num_rays:
            raise ValueError(f"expected lidar observation of length {self.num_rays}, got {lidar.shape[1]}")
        articulation_angles = np.array([float(articulation_angle)], dtype=np.float32)
        proxy_scores, prefix_lengths, bin_indices = self.compute_proxy_scores_batch(
            lidar_observations=lidar, articulation_angles=articulation_angles, gamma=gamma, eps=eps,
        )
        return ProxySafetyQueryResult(
            articulation_bin_index=int(bin_indices[0]),
            prefix_lengths=prefix_lengths[0],
            proxy_scores=proxy_scores[0],
        )

    def compute_proxy_scores_batch(
        self,
        lidar_observations: np.ndarray,
        articulation_angles: np.ndarray,
        gamma: float,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lidar = np.asarray(lidar_observations, dtype=np.float32)
        if lidar.ndim != 2 or lidar.shape[1] != self.num_rays:
            raise ValueError(f"expected lidar observations of shape (batch, {self.num_rays}), got {lidar.shape}")
        batch_size = int(lidar.shape[0])
        angles = np.asarray(articulation_angles, dtype=np.float32).reshape(-1)
        if angles.shape[0] != batch_size:
            raise ValueError(f"articulation_angles size {angles.shape[0]} does not match batch {batch_size}")

        diffs = np.abs(_wrap_angle(angles.reshape(-1, 1) - self.articulation_bin_centers.reshape(1, -1)))
        bin_indices = np.argmin(diffs, axis=1).astype(np.int64)
        dist_obs = np.clip(lidar, 0.0, 1.0) * max(float(self.lidar_range), 1e-6)
        required = self.required_clearance[bin_indices]
        safe_by_ray = required <= dist_obs.reshape(batch_size, 1, 1, 1, -1)
        safe_step = np.all(safe_by_ray, axis=-1)
        step_indices = np.arange(self.num_steps, dtype=np.int64).reshape(1, 1, 1, -1)
        valid_steps = step_indices < np.maximum(self.nominal_horizons, 0).reshape(1, self.num_actions, self.num_proxies, 1)
        safe_step = np.logical_and(safe_step, valid_steps)
        prefix_safe = np.cumprod(safe_step.astype(np.int8), axis=-1)
        prefix_lengths = np.sum(prefix_safe, axis=-1).astype(np.float32)
        horizon = np.maximum(self.nominal_horizons.astype(np.float32), 1.0).reshape(1, self.num_actions, self.num_proxies)
        proxy_scores = np.power(np.clip(prefix_lengths / horizon, 0.0, 1.0), float(gamma)).astype(np.float32)
        proxy_scores = np.clip(proxy_scores, float(eps), 1.0)
        proxy_scores = np.where(self.proxy_valid_mask.reshape(1, self.num_actions, self.num_proxies), proxy_scores, 0.0).astype(np.float32)
        return proxy_scores, prefix_lengths, bin_indices


def save_proxy_safety_sidecar(path: str, sidecar: ProxySafetySidecar) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        required_clearance=np.asarray(sidecar.required_clearance, dtype=np.float32),
        articulation_bin_centers=np.asarray(sidecar.articulation_bin_centers, dtype=np.float32),
        parameter_centers=np.asarray(sidecar.parameter_centers, dtype=np.float32),
        parameter_scales=np.asarray(sidecar.parameter_scales, dtype=np.float32),
        nominal_horizons=np.asarray(sidecar.nominal_horizons, dtype=np.int64),
        proxy_valid_mask=np.asarray(sidecar.proxy_valid_mask, dtype=np.bool_),
        lidar_range=np.asarray(float(sidecar.lidar_range), dtype=np.float32),
        library_signature=np.asarray(str(sidecar.library_signature), dtype=object),
        metadata=np.asarray(dict(sidecar.metadata or {}), dtype=object),
    )


def load_proxy_safety_sidecar(path: str) -> ProxySafetySidecar:
    data = np.load(path, allow_pickle=True)
    metadata = {}
    if "metadata" in data:
        try:
            metadata = data["metadata"].item() or {}
        except Exception:
            metadata = {}
    signature = ""
    if "library_signature" in data:
        try:
            signature = str(data["library_signature"].item())
        except Exception:
            signature = ""
    return ProxySafetySidecar(
        required_clearance=np.asarray(data["required_clearance"], dtype=np.float32),
        articulation_bin_centers=np.asarray(data["articulation_bin_centers"], dtype=np.float32),
        parameter_centers=np.asarray(data["parameter_centers"], dtype=np.float32),
        parameter_scales=np.asarray(data["parameter_scales"], dtype=np.float32),
        nominal_horizons=np.asarray(data["nominal_horizons"], dtype=np.int64),
        proxy_valid_mask=np.asarray(data["proxy_valid_mask"], dtype=np.bool_),
        lidar_range=float(data["lidar_range"]),
        library_signature=signature,
        metadata=metadata,
    )


def build_proxy_parameter_grid(
    library,
    proxy_resolution: int = 2,
    max_proxies_per_action: Optional[int] = None,
    semantic_proxy_resolution: Optional[Mapping[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    proxy_resolution = max(1, int(proxy_resolution))
    parameter_dim = int(library.parameter_dim)
    available_semantics = {str(spec.semantic.value) for spec in library.specs}
    normalized_resolution: Dict[str, int] = {}
    for semantic_name, resolution in dict(semantic_proxy_resolution or {}).items():
        semantic_key = str(semantic_name).strip()
        if semantic_key not in available_semantics:
            raise ValueError(f"unknown semantic proxy resolution override: {semantic_key}")
        normalized_resolution[semantic_key] = max(proxy_resolution, int(resolution))
    per_action_centers = []
    per_action_scales = []
    max_proxy_count = 0
    for spec in library.specs:
        spec_proxy_resolution = int(normalized_resolution.get(str(spec.semantic.value), proxy_resolution))
        active_indices = set(map(int, library.active_indices(spec.primitive_id).tolist()))
        axis_values = []
        axis_scales = []
        for index, name in enumerate(library.parameter_names):
            low, high = library.bounds[str(name)]
            if index in active_indices:
                if spec_proxy_resolution == 1:
                    values = np.array([0.5 * (low + high)], dtype=np.float32)
                else:
                    values = np.linspace(low, high, spec_proxy_resolution, dtype=np.float32)
                scale = max(float(high - low) / max(spec_proxy_resolution - 1, 1), 1e-3)
            else:
                values = np.array([0.5 * (low + high)], dtype=np.float32)
                scale = max(float(high - low), 1e-3)
            axis_values.append(values)
            axis_scales.append(scale)
        centers = np.asarray(list(product(*axis_values)), dtype=np.float32).reshape(-1, parameter_dim)
        scales = np.tile(np.asarray(axis_scales, dtype=np.float32).reshape(1, -1), (centers.shape[0], 1))
        if max_proxies_per_action is not None and centers.shape[0] > int(max_proxies_per_action):
            keep = np.linspace(0, centers.shape[0] - 1, int(max_proxies_per_action), dtype=np.int64)
            centers = centers[keep]
            scales = scales[keep]
        per_action_centers.append(centers)
        per_action_scales.append(scales)
        max_proxy_count = max(max_proxy_count, int(centers.shape[0]))

    parameter_centers = np.zeros((library.action_dim, max_proxy_count, parameter_dim), dtype=np.float32)
    parameter_scales = np.ones((library.action_dim, max_proxy_count, parameter_dim), dtype=np.float32)
    proxy_valid_mask = np.zeros((library.action_dim, max_proxy_count), dtype=np.bool_)
    for action_id, (centers, scales) in enumerate(zip(per_action_centers, per_action_scales)):
        count = int(centers.shape[0])
        parameter_centers[action_id, :count, :] = centers
        parameter_scales[action_id, :count, :] = scales
        proxy_valid_mask[action_id, :count] = True
    return parameter_centers, parameter_scales, proxy_valid_mask


def build_proxy_safety_sidecar(
    library,
    vehicle_config,
    executor_config,
    lidar_num: int,
    lidar_range: float,
    articulation_bin_count: int = 7,
    proxy_resolution: int = 2,
    max_proxies_per_action: Optional[int] = None,
    semantic_proxy_resolution: Optional[Mapping[str, int]] = None,
) -> ProxySafetySidecar:
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    from common.types import ArticulatedState, LowLevelControl, PrimitiveExecutionContext
    from env.success import articulated_body_polygons
    from primitives.library import ParameterizedPrimitiveExecutor, SemanticPrimitive

    lidar_num = max(1, int(lidar_num))
    lidar_range = float(lidar_range)
    articulation_bin_count = max(1, int(articulation_bin_count))
    articulation_limit = float(vehicle_config.articulation_limit_rad)
    articulation_bin_centers = np.linspace(-articulation_limit, articulation_limit, articulation_bin_count, dtype=np.float32)
    parameter_centers, parameter_scales, proxy_valid_mask = build_proxy_parameter_grid(
        library,
        proxy_resolution=proxy_resolution,
        max_proxies_per_action=max_proxies_per_action,
        semantic_proxy_resolution=semantic_proxy_resolution,
    )
    max_steps = int(executor_config.max_macro_steps)
    required_clearance = np.zeros((articulation_bin_count, library.action_dim, parameter_centers.shape[1], max_steps, lidar_num), dtype=np.float32)
    nominal_horizons = np.ones((library.action_dim, parameter_centers.shape[1]), dtype=np.int64)
    executor = ParameterizedPrimitiveExecutor(library, vehicle_config=vehicle_config, executor_config=executor_config)
    ray_angles = np.linspace(-np.pi, np.pi, lidar_num, endpoint=False, dtype=np.float32)
    ray_length = max(lidar_range, vehicle_config.front_length + vehicle_config.rear_length + 10.0)
    origin = Point(0.0, 0.0)

    for art_index, articulation_angle in enumerate(articulation_bin_centers):
        for action_id in range(library.action_dim):
            for proxy_id in range(parameter_centers.shape[1]):
                if not bool(proxy_valid_mask[action_id, proxy_id]):
                    continue
                states = _simulate_proxy_rollout(
                    executor=executor,
                    primitive_id=action_id,
                    parameters=parameter_centers[action_id, proxy_id],
                    articulation_angle=float(articulation_angle),
                )
                nominal_horizon = max(1, len(states) - 1)
                nominal_horizons[action_id, proxy_id] = int(min(nominal_horizon, max_steps))
                prefix_polygons = []
                last_extent = np.zeros((lidar_num,), dtype=np.float32)
                for step_index, state in enumerate(states[:max_steps]):
                    front_poly, rear_poly = articulated_body_polygons(state, vehicle_config)
                    prefix_polygons.extend([front_poly, rear_poly])
                    swept_volume = unary_union(prefix_polygons)
                    if step_index == 0:
                        continue
                    last_extent = _ray_required_clearance(
                        swept_volume=swept_volume,
                        ray_angles=ray_angles,
                        ray_length=ray_length,
                        origin=origin,
                        line_factory=LineString,
                    )
                    required_clearance[art_index, action_id, proxy_id, step_index - 1, :] = last_extent
                if nominal_horizons[action_id, proxy_id] < max_steps:
                    required_clearance[art_index, action_id, proxy_id, nominal_horizons[action_id, proxy_id] :, :] = last_extent

    metadata = {
        "articulation_bin_count": int(articulation_bin_count),
        "proxy_resolution": int(proxy_resolution),
        "semantic_proxy_resolution": {str(key): int(value) for key, value in dict(semantic_proxy_resolution or {}).items()},
        "max_proxies_per_action": None if max_proxies_per_action is None else int(max_proxies_per_action),
        "lidar_num": int(lidar_num),
        "lidar_range": float(lidar_range),
        "index_kind": "full_body_swept_volume_required_clearance",
    }
    return ProxySafetySidecar(
        required_clearance=required_clearance,
        articulation_bin_centers=articulation_bin_centers,
        parameter_centers=parameter_centers,
        parameter_scales=parameter_scales,
        nominal_horizons=nominal_horizons,
        proxy_valid_mask=proxy_valid_mask,
        lidar_range=lidar_range,
        library_signature=getattr(library, "signature", ""),
        metadata=metadata,
    )


def _simulate_proxy_rollout(executor, primitive_id: int, parameters: np.ndarray, articulation_angle: float):
    from common.types import ArticulatedState, LowLevelControl, PrimitiveExecutionContext
    from primitives.library import SemanticPrimitive

    spec = executor.primitive_library.spec(int(primitive_id))
    parameter_map = executor._normalize_parameter_map(int(primitive_id), parameters)
    current_state = ArticulatedState(
        x=0.0,
        y=0.0,
        front_heading=0.0,
        rear_heading=-float(articulation_angle),
        speed=0.0,
        articulation_rate=0.0,
    )
    context = PrimitiveExecutionContext()
    previous_control = LowLevelControl(articulation_rate=0.0, speed=0.0)
    smoothness = float(parameter_map.get("smoothness", 0.0))
    duration_budget = max(executor.vehicle_config.step_seconds, float(parameter_map.get("duration", executor.vehicle_config.step_seconds)))
    max_steps = min(int(executor.executor_config.max_macro_steps), max(1, int(np.ceil(duration_budget / executor.vehicle_config.step_seconds))))
    travelled_distance = 0.0
    elapsed_time = 0.0
    states = [current_state]

    for _ in range(max_steps):
        raw_control = executor._semantic_control(spec, current_state, parameter_map, context)
        control = executor._blend_control(previous_control, raw_control, smoothness)
        next_state = executor.kinematics.step(current_state, control)
        travelled_distance += float(np.hypot(next_state.x - current_state.x, next_state.y - current_state.y))
        elapsed_time += float(executor.vehicle_config.step_seconds)
        states.append(next_state)
        current_state = next_state
        previous_control = control
        if _proxy_termination_reason(executor=executor, spec_semantic=spec.semantic, state=next_state, travelled_distance=travelled_distance, elapsed_time=elapsed_time, parameter_map=parameter_map):
            break
    return states


def _proxy_termination_reason(executor, spec_semantic, state, travelled_distance: float, elapsed_time: float, parameter_map: Mapping[str, float]) -> bool:
    articulation_limit = float(executor.vehicle_config.articulation_limit_rad)
    if abs(float(state.articulation_angle)) >= articulation_limit - 1e-6:
        return True
    if elapsed_time >= float(parameter_map.get("duration", elapsed_time)):
        return True
    if travelled_distance >= float(parameter_map.get("path_length", travelled_distance + 1e-6)):
        return True
    if str(spec_semantic) == str("articulation-recover"):
        guard_limit = max(0.0, articulation_limit - float(executor.executor_config.articulation_guard_margin_rad))
        if abs(float(state.articulation_angle)) <= guard_limit:
            return True
    return False


def _ray_required_clearance(swept_volume, ray_angles: np.ndarray, ray_length: float, origin, line_factory) -> np.ndarray:
    out = np.zeros((ray_angles.shape[0],), dtype=np.float32)
    if swept_volume is None or swept_volume.is_empty:
        return out
    for ray_index, angle in enumerate(ray_angles):
        ray = line_factory(
            [
                (0.0, 0.0),
                (float(ray_length) * np.cos(float(angle)), float(ray_length) * np.sin(float(angle))),
            ]
        )
        intersection = ray.intersection(swept_volume)
        distances = _intersection_distances(origin, intersection)
        if distances:
            out[ray_index] = float(max(distances))
    return out


def _intersection_distances(origin, geometry) -> Sequence[float]:
    if geometry.is_empty:
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Point":
        return [float(origin.distance(geometry))]
    if geom_type == "LineString":
        return [float(np.hypot(x, y)) for x, y in list(geometry.coords)]
    if geom_type == "Polygon":
        return [float(np.hypot(x, y)) for x, y in list(geometry.exterior.coords)]
    if hasattr(geometry, "geoms"):
        distances = []
        for child in geometry.geoms:
            distances.extend(_intersection_distances(origin, child))
        return distances
    return []


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi