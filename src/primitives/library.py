from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from common.config import ParameterBoundsConfig, PrimitiveExecutorConfig, VehicleConfig
from common.types import ArticulatedState, LowLevelControl, PrimitiveExecutionContext, PrimitiveRollout, wrap_to_pi
from env.dynamics import ArticulatedKinematics
from .proxy_safety import ProxySafetySidecar, primitive_library_signature


class SemanticPrimitive(str, Enum):
    FORWARD_LEFT = "forward-left"
    FORWARD_RIGHT = "forward-right"
    REVERSE_LEFT = "reverse-left"
    REVERSE_RIGHT = "reverse-right"
    REVERSE_ALIGN = "reverse-align"
    ARTICULATION_RECOVER = "articulation-recover"
    STRAIGHT_ADJUST = "straight-adjust"
    STOP_CHECK = "stop-check"


@dataclass(frozen=True)
class PrimitiveSpec:
    primitive_id: int
    semantic: SemanticPrimitive
    active_parameters: Tuple[str, ...]
    direction_sign: int = 0
    turn_sign: int = 0


def _default_specs() -> Tuple[PrimitiveSpec, ...]:
    return (
        PrimitiveSpec(0, SemanticPrimitive.FORWARD_LEFT, ("path_length", "duration", "speed_scale", "omega_scale", "phi_target", "smoothness"), direction_sign=1, turn_sign=1),
        PrimitiveSpec(1, SemanticPrimitive.FORWARD_RIGHT, ("path_length", "duration", "speed_scale", "omega_scale", "phi_target", "smoothness"), direction_sign=1, turn_sign=-1),
        PrimitiveSpec(2, SemanticPrimitive.REVERSE_LEFT, ("path_length", "duration", "speed_scale", "omega_scale", "phi_target", "smoothness"), direction_sign=-1, turn_sign=1),
        PrimitiveSpec(3, SemanticPrimitive.REVERSE_RIGHT, ("path_length", "duration", "speed_scale", "omega_scale", "phi_target", "smoothness"), direction_sign=-1, turn_sign=-1),
        PrimitiveSpec(4, SemanticPrimitive.REVERSE_ALIGN, ("path_length", "duration", "speed_scale", "phi_target", "heading_tolerance", "position_tolerance", "smoothness"), direction_sign=-1, turn_sign=0),
        PrimitiveSpec(5, SemanticPrimitive.ARTICULATION_RECOVER, ("duration", "speed_scale", "omega_scale", "phi_target", "smoothness"), direction_sign=0, turn_sign=0),
        PrimitiveSpec(6, SemanticPrimitive.STRAIGHT_ADJUST, ("path_length", "duration", "speed_scale", "heading_tolerance", "position_tolerance", "smoothness"), direction_sign=0, turn_sign=0),
        PrimitiveSpec(7, SemanticPrimitive.STOP_CHECK, ("duration",), direction_sign=0, turn_sign=0),
    )


class ParameterizedPrimitiveLibrary:
    def __init__(self, specs: Optional[Sequence[PrimitiveSpec]] = None, parameter_bounds: Optional[Mapping[str, Tuple[float, float]]] = None, proxy_sidecar: Optional[ProxySafetySidecar] = None) -> None:
        self.specs: Tuple[PrimitiveSpec, ...] = tuple(specs or _default_specs())
        self.specs_by_id = {int(spec.primitive_id): spec for spec in self.specs}
        if len(self.specs_by_id) != len(self.specs):
            raise ValueError("primitive ids must be unique")
        bounds_cfg = ParameterBoundsConfig()
        self.parameter_names = tuple((parameter_bounds or bounds_cfg.bounds).keys())
        self.bounds: Dict[str, Tuple[float, float]] = {str(name): tuple((parameter_bounds or bounds_cfg.bounds)[name]) for name in self.parameter_names}
        self.low = np.asarray([self.bounds[name][0] for name in self.parameter_names], dtype=np.float32)
        self.high = np.asarray([self.bounds[name][1] for name in self.parameter_names], dtype=np.float32)
        self.signature = primitive_library_signature(self._spec_signature_payload(), self.parameter_names, self.bounds)
        self.proxy_sidecar: Optional[ProxySafetySidecar] = None
        if proxy_sidecar is not None:
            self.attach_proxy_sidecar(proxy_sidecar)

    @property
    def action_dim(self) -> int:
        return int(len(self.specs))

    @property
    def parameter_dim(self) -> int:
        return int(len(self.parameter_names))

    def spec(self, primitive_id: Union[int, SemanticPrimitive]) -> PrimitiveSpec:
        if isinstance(primitive_id, SemanticPrimitive):
            for spec in self.specs:
                if spec.semantic == primitive_id:
                    return spec
            raise KeyError(f"unknown primitive semantic: {primitive_id}")
        return self.specs_by_id[int(primitive_id)]

    def active_mask(self, primitive_id: Union[int, SemanticPrimitive]) -> np.ndarray:
        spec = self.spec(primitive_id)
        active = set(spec.active_parameters)
        return np.asarray([name in active for name in self.parameter_names], dtype=np.bool_)

    def active_indices(self, primitive_id: Union[int, SemanticPrimitive]) -> np.ndarray:
        return np.flatnonzero(self.active_mask(primitive_id)).astype(np.int64)

    def clip(self, parameter_vector: Sequence[float]) -> np.ndarray:
        values = np.asarray(parameter_vector, dtype=np.float32).reshape(-1)
        if values.shape[0] != self.parameter_dim:
            raise ValueError(f"expected parameter vector of length {self.parameter_dim}, got {values.shape[0]}")
        return np.clip(values, self.low, self.high)

    def vector_to_dict(self, primitive_id: Union[int, SemanticPrimitive], parameter_vector: Sequence[float]) -> Dict[str, float]:
        values = self.clip(parameter_vector)
        spec = self.spec(primitive_id)
        result: Dict[str, float] = {}
        for idx, name in enumerate(self.parameter_names):
            if name in spec.active_parameters:
                result[name] = float(values[idx])
        return result

    def dict_to_vector(self, values: Mapping[str, float]) -> np.ndarray:
        out = np.asarray([values.get(name, 0.5 * (self.bounds[name][0] + self.bounds[name][1])) for name in self.parameter_names], dtype=np.float32)
        return self.clip(out)

    def attach_proxy_sidecar(self, proxy_sidecar: Optional[ProxySafetySidecar]) -> None:
        if proxy_sidecar is None:
            self.proxy_sidecar = None
            return
        if not proxy_sidecar.compatible_with(action_dim=self.action_dim, parameter_dim=self.parameter_dim, expected_signature=self.signature):
            raise ValueError("proxy safety sidecar is incompatible with this primitive library")
        self.proxy_sidecar = proxy_sidecar

    def _spec_signature_payload(self) -> Tuple[Dict[str, object], ...]:
        return tuple(
            {
                "primitive_id": int(spec.primitive_id),
                "semantic": str(spec.semantic),
                "active_parameters": tuple(map(str, spec.active_parameters)),
                "direction_sign": int(spec.direction_sign),
                "turn_sign": int(spec.turn_sign),
            }
            for spec in self.specs
        )


class ParameterizedPrimitiveExecutor:
    def __init__(self, primitive_library: ParameterizedPrimitiveLibrary, vehicle_config: Optional[VehicleConfig] = None, executor_config: Optional[PrimitiveExecutorConfig] = None) -> None:
        self.primitive_library = primitive_library
        self.vehicle_config = vehicle_config or VehicleConfig()
        self.executor_config = executor_config or PrimitiveExecutorConfig()
        self.kinematics = ArticulatedKinematics(self.vehicle_config)

    def rollout(self, start_state: ArticulatedState, primitive_id: Union[int, SemanticPrimitive], parameters: Union[Sequence[float], Mapping[str, float]], context: Optional[PrimitiveExecutionContext] = None) -> PrimitiveRollout:
        spec = self.primitive_library.spec(primitive_id)
        context = context or PrimitiveExecutionContext()
        parameter_map = self._normalize_parameter_map(spec.primitive_id, parameters)
        states = [start_state]
        controls: List[LowLevelControl] = []
        elapsed_time = 0.0
        travelled_distance = 0.0
        termination_reason = "duration_budget"
        current_state = start_state
        previous_control = LowLevelControl(articulation_rate=0.0, speed=0.0)
        smoothness = float(parameter_map.get("smoothness", 0.0))
        duration_budget = max(self.vehicle_config.step_seconds, float(parameter_map.get("duration", self.vehicle_config.step_seconds)))
        max_steps = min(int(self.executor_config.max_macro_steps), max(1, int(np.ceil(duration_budget / self.vehicle_config.step_seconds))))

        for _ in range(max_steps):
            raw_control = self._semantic_control(spec, current_state, parameter_map, context)
            control = self._blend_control(previous_control, raw_control, smoothness)
            next_state = self.kinematics.step(current_state, control)
            travelled_distance += float(np.hypot(next_state.x - current_state.x, next_state.y - current_state.y))
            elapsed_time += float(self.vehicle_config.step_seconds)
            controls.append(control)
            states.append(next_state)
            termination_reason = self._termination_reason(spec=spec, state=next_state, travelled_distance=travelled_distance, elapsed_time=elapsed_time, parameter_map=parameter_map, context=context)
            current_state = next_state
            previous_control = control
            if termination_reason != "continue":
                break

        if termination_reason == "continue":
            termination_reason = "duration_budget"

        return PrimitiveRollout(
            primitive_id=int(spec.primitive_id),
            parameters=dict(parameter_map),
            states=states,
            controls=controls,
            termination_reason=str(termination_reason),
            travelled_distance=float(travelled_distance),
            elapsed_time=float(elapsed_time),
        )

    def _normalize_parameter_map(self, primitive_id: int, parameters: Union[Sequence[float], Mapping[str, float]]) -> Dict[str, float]:
        values = self.primitive_library.dict_to_vector(parameters) if isinstance(parameters, Mapping) else self.primitive_library.clip(parameters)
        parameter_map = self.primitive_library.vector_to_dict(primitive_id, values)
        parameter_map.setdefault("speed_scale", self.executor_config.max_speed_scale)
        parameter_map.setdefault("omega_scale", 0.5)
        parameter_map.setdefault("phi_target", 0.0)
        parameter_map.setdefault("heading_tolerance", self.executor_config.default_heading_tolerance_rad)
        parameter_map.setdefault("position_tolerance", self.executor_config.default_position_tolerance_m)
        parameter_map.setdefault("path_length", 0.5)
        parameter_map.setdefault("duration", self.vehicle_config.step_seconds)
        parameter_map.setdefault("smoothness", 0.0)
        return parameter_map

    def _semantic_control(self, spec: PrimitiveSpec, state: ArticulatedState, parameter_map: Mapping[str, float], context: PrimitiveExecutionContext) -> LowLevelControl:
        speed_scale = float(np.clip(parameter_map.get("speed_scale", 1.0), self.executor_config.min_speed_scale, self.executor_config.max_speed_scale))
        omega_scale = float(np.clip(parameter_map.get("omega_scale", 0.5), 0.0, 1.0))
        phi_target = float(parameter_map.get("phi_target", 0.0))
        phi_error = wrap_to_pi(phi_target - state.articulation_angle)
        heading_error = self._heading_error(state, context)
        nominal_speed = self.executor_config.nominal_speed * speed_scale
        low_speed = self.executor_config.low_speed * speed_scale
        omega_limit = float(self.vehicle_config.articulation_rate_max)

        if spec.semantic in (SemanticPrimitive.FORWARD_LEFT, SemanticPrimitive.FORWARD_RIGHT, SemanticPrimitive.REVERSE_LEFT, SemanticPrimitive.REVERSE_RIGHT):
            articulation_rate = spec.turn_sign * omega_scale * omega_limit
            articulation_rate += self.executor_config.articulation_controller_gain * phi_error
            speed = spec.direction_sign * nominal_speed
        elif spec.semantic == SemanticPrimitive.REVERSE_ALIGN:
            articulation_rate = self.executor_config.heading_controller_gain * heading_error
            articulation_rate += self.executor_config.articulation_controller_gain * phi_error
            speed = -low_speed
        elif spec.semantic == SemanticPrimitive.STRAIGHT_ADJUST:
            articulation_rate = self.executor_config.heading_controller_gain * heading_error
            articulation_rate += 0.5 * self.executor_config.articulation_controller_gain * (-state.articulation_angle)
            speed = self._choose_straight_adjust_speed(state, context, low_speed)
        elif spec.semantic == SemanticPrimitive.ARTICULATION_RECOVER:
            articulation_rate = self.executor_config.articulation_controller_gain * phi_error
            speed = self._choose_recovery_speed(state, articulation_rate, low_speed)
        elif spec.semantic == SemanticPrimitive.STOP_CHECK:
            articulation_rate = 0.0
            speed = 0.0
        else:
            raise NotImplementedError(f"unsupported primitive semantic: {spec.semantic}")

        articulation_rate = float(np.clip(articulation_rate, self.vehicle_config.articulation_rate_min, self.vehicle_config.articulation_rate_max))
        speed = float(np.clip(speed, self.vehicle_config.speed_min, self.vehicle_config.speed_max))
        return LowLevelControl(articulation_rate=articulation_rate, speed=speed)

    def _blend_control(self, previous: LowLevelControl, current: LowLevelControl, smoothness: float) -> LowLevelControl:
        alpha = float(np.clip(smoothness, 0.0, 0.95))
        return LowLevelControl(
            articulation_rate=alpha * previous.articulation_rate + (1.0 - alpha) * current.articulation_rate,
            speed=alpha * previous.speed + (1.0 - alpha) * current.speed,
        )

    def _termination_reason(self, spec: PrimitiveSpec, state: ArticulatedState, travelled_distance: float, elapsed_time: float, parameter_map: Mapping[str, float], context: PrimitiveExecutionContext) -> str:
        articulation_limit = float(self.vehicle_config.articulation_limit_rad)
        if abs(state.articulation_angle) >= articulation_limit - 1e-6:
            return "articulation_limit"
        if context.collision_checker is not None and bool(context.collision_checker(state)):
            return "collision"
        if elapsed_time >= float(parameter_map.get("duration", elapsed_time)):
            return "duration_budget"
        if travelled_distance >= float(parameter_map.get("path_length", travelled_distance + 1e-6)):
            return "path_length_budget"
        if spec.semantic in (SemanticPrimitive.REVERSE_ALIGN, SemanticPrimitive.STRAIGHT_ADJUST):
            heading_tol = float(parameter_map.get("heading_tolerance", self.executor_config.default_heading_tolerance_rad))
            if abs(self._heading_error(state, context)) <= heading_tol:
                if context.goal_position is None:
                    return "goal_tolerance"
                goal_dist = float(np.hypot(state.x - context.goal_position[0], state.y - context.goal_position[1]))
                if goal_dist <= float(parameter_map.get("position_tolerance", self.executor_config.default_position_tolerance_m)):
                    return "goal_tolerance"
        if spec.semantic == SemanticPrimitive.ARTICULATION_RECOVER:
            guard_limit = max(0.0, articulation_limit - self.executor_config.articulation_guard_margin_rad)
            if abs(state.articulation_angle) <= guard_limit:
                return "articulation_recovered"
        return "continue"

    def _heading_error(self, state: ArticulatedState, context: PrimitiveExecutionContext) -> float:
        if context.goal_heading is None:
            return 0.0
        return wrap_to_pi(float(context.goal_heading) - float(state.front_heading))

    def _choose_straight_adjust_speed(self, state: ArticulatedState, context: PrimitiveExecutionContext, base_speed: float) -> float:
        if context.goal_position is None:
            return base_speed
        dx = float(context.goal_position[0]) - float(state.x)
        dy = float(context.goal_position[1]) - float(state.y)
        projection = dx * np.cos(state.front_heading) + dy * np.sin(state.front_heading)
        return base_speed if projection >= 0.0 else -base_speed

    def _choose_recovery_speed(self, state: ArticulatedState, articulation_rate: float, base_speed: float) -> float:
        best_speed = -base_speed
        best_error = float("inf")
        for candidate_speed in (-base_speed, base_speed):
            candidate_state = self.kinematics.step(state, LowLevelControl(articulation_rate=articulation_rate, speed=candidate_speed))
            candidate_error = abs(candidate_state.articulation_angle)
            if candidate_error < best_error:
                best_error = candidate_error
                best_speed = candidate_speed
        return float(best_speed)


def build_default_primitive_library(proxy_sidecar: Optional[ProxySafetySidecar] = None) -> ParameterizedPrimitiveLibrary:
    return ParameterizedPrimitiveLibrary(specs=_default_specs(), proxy_sidecar=proxy_sidecar)
