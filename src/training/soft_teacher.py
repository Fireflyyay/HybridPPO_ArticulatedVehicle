from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from common.config import VehicleConfig
from common.runtime_config import BASE_OBSERVATION_FEATURE_DIM, GUIDANCE_FEATURE_DIM, ObservationConfig
from common.types import ArticulatedState, PrimitiveExecutionContext, PrimitiveRollout, wrap_to_pi
from env.adapter import UnifiedArticulatedEnvProtocol
from primitives import ParameterizedPrimitiveExecutor, ParameterizedPrimitiveLibrary, build_proxy_parameter_grid


@dataclass(frozen=True)
class TeacherAdvice:
    action_probs: np.ndarray
    parameter_targets: np.ndarray
    weight: float
    diagnostics: Dict[str, float]


class CoarseGuidanceSoftTeacher:
    _EVAL_PREFIX_STEPS = 2
    _MAX_CANDIDATES_PER_ACTION = 6
    _SCORE_TEMPERATURE = 0.35
    _MAX_WEIGHT = 0.65
    _COLLISION_PENALTY = 6.0
    _ARTICULATION_LIMIT_PENALTY = 2.5
    _REVERSE_SWITCH_PENALTY = 0.65

    def __init__(
        self,
        env: UnifiedArticulatedEnvProtocol,
        executor: ParameterizedPrimitiveExecutor,
        primitive_library: ParameterizedPrimitiveLibrary,
        vehicle_config: VehicleConfig,
        observation_config: ObservationConfig,
    ) -> None:
        self.env = env
        self.executor = executor
        self.primitive_library = primitive_library
        self.vehicle_config = vehicle_config
        self.observation_config = observation_config
        candidate_centers, _candidate_scales, candidate_valid_mask = build_proxy_parameter_grid(
            primitive_library,
            proxy_resolution=2,
            max_proxies_per_action=self._MAX_CANDIDATES_PER_ACTION,
        )
        self._candidate_centers = candidate_centers.astype(np.float32)
        self._candidate_valid_mask = candidate_valid_mask.astype(np.bool_)
        self._default_parameter_vector = primitive_library.dict_to_vector({}).astype(np.float32)

    def advise(self, observation: np.ndarray, previous_action_id: Optional[int] = None) -> Optional[TeacherAdvice]:
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        lidar_dim = int(observation.shape[0] - int(BASE_OBSERVATION_FEATURE_DIM) - int(GUIDANCE_FEATURE_DIM))
        if lidar_dim < 0:
            return None

        lidar = observation[:lidar_dim]
        guidance = observation[-int(GUIDANCE_FEATURE_DIM) :]
        info = self.env.current_info()
        teacher_weight, diagnostics = self._teacher_weight(lidar=lidar, guidance=guidance, info=info)
        if teacher_weight <= 0.0:
            return None

        start_state = self.env.get_articulated_state()
        context = self.env.make_primitive_context()
        reference_heading = self._reference_heading(start_state, guidance)
        if reference_heading is None:
            return None

        current_goal_distance = float(self.env.distance_to_goal(state=start_state))
        action_scores = np.full((self.primitive_library.action_dim,), -1e9, dtype=np.float32)
        parameter_targets = np.tile(self._default_parameter_vector.reshape(1, -1), (self.primitive_library.action_dim, 1)).astype(np.float32)

        for action_id in range(self.primitive_library.action_dim):
            best_score = -1e9
            best_parameters = self._default_parameter_vector
            for parameter_vector in self._candidate_vectors_for_action(action_id):
                rollout = self.executor.rollout(
                    start_state=start_state,
                    primitive_id=action_id,
                    parameters=parameter_vector,
                    context=context,
                )
                score = self._score_rollout(
                    start_state=start_state,
                    rollout=rollout,
                    action_id=action_id,
                    previous_action_id=previous_action_id,
                    current_goal_distance=current_goal_distance,
                    reference_heading=reference_heading,
                    context=context,
                )
                if score > best_score:
                    best_score = float(score)
                    best_parameters = np.asarray(parameter_vector, dtype=np.float32).copy()
            action_scores[action_id] = float(best_score)
            parameter_targets[action_id] = best_parameters

        action_probs = self._softmax(action_scores)
        if action_probs is None:
            return None

        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "teacher_weight": float(teacher_weight),
                "teacher_score_max": float(np.max(action_scores)),
                "teacher_score_gap": float(np.max(action_scores) - np.min(action_scores)),
            }
        )
        return TeacherAdvice(
            action_probs=action_probs,
            parameter_targets=parameter_targets,
            weight=float(teacher_weight),
            diagnostics=diagnostics,
        )

    def _candidate_vectors_for_action(self, action_id: int):
        yielded = set()
        default_vector = self._default_parameter_vector.astype(np.float32)
        yielded.add(default_vector.tobytes())
        yield default_vector
        for candidate_id in range(self._candidate_centers.shape[1]):
            if not bool(self._candidate_valid_mask[action_id, candidate_id]):
                continue
            candidate = self.primitive_library.clip(self._candidate_centers[action_id, candidate_id]).astype(np.float32)
            token = candidate.tobytes()
            if token in yielded:
                continue
            yielded.add(token)
            yield candidate

    def _teacher_weight(self, lidar: np.ndarray, guidance: np.ndarray, info: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
        if not bool(info.get("guidance_available", False)):
            return 0.0, {"narrow_score": 0.0, "guidance_confidence": 0.0, "guidance_strength": 0.0}

        guidance_strength = float(np.clip(np.linalg.norm(guidance[:2]), 0.0, 1.0))
        guidance_confidence = float(np.clip(max(float(info.get("guidance_path_confidence", 0.0)), float(guidance[3])), 0.0, 1.0))
        if guidance_strength <= 1e-4 or guidance_confidence <= 1e-4:
            return 0.0, {"narrow_score": 0.0, "guidance_confidence": guidance_confidence, "guidance_strength": guidance_strength}

        corridor_width = info.get("corridor_width")
        if corridor_width is not None:
            narrow_score = float(np.clip((8.0 - float(corridor_width)) / 2.0, 0.0, 1.0))
        else:
            min_clearance = 1.0 if lidar.size == 0 else float(np.clip(np.min(lidar), 0.0, 1.0))
            narrow_score = float(np.clip((0.35 - min_clearance) / 0.20, 0.0, 1.0))

        if str(info.get("level", "")).lower() == "debug":
            narrow_score = 0.0

        teacher_weight = float(np.clip(guidance_confidence * guidance_strength * narrow_score, 0.0, self._MAX_WEIGHT))
        return teacher_weight, {
            "narrow_score": float(narrow_score),
            "guidance_confidence": guidance_confidence,
            "guidance_strength": guidance_strength,
        }

    def _reference_heading(self, start_state: ArticulatedState, guidance: np.ndarray) -> Optional[float]:
        direction = np.asarray(guidance[:2], dtype=np.float32)
        if float(np.linalg.norm(direction)) <= 1e-4:
            return None
        return wrap_to_pi(float(start_state.front_heading) + float(np.arctan2(direction[1], direction[0])))

    def _score_rollout(
        self,
        start_state: ArticulatedState,
        rollout: PrimitiveRollout,
        action_id: int,
        previous_action_id: Optional[int],
        current_goal_distance: float,
        reference_heading: float,
        context: PrimitiveExecutionContext,
    ) -> float:
        prefix_states = rollout.states[1 : min(len(rollout.states), self._EVAL_PREFIX_STEPS + 1)]
        eval_state = prefix_states[-1] if prefix_states else start_state
        dx = float(eval_state.x - start_state.x)
        dy = float(eval_state.y - start_state.y)
        along_track = float(np.cos(reference_heading) * dx + np.sin(reference_heading) * dy)
        cross_track = float(-np.sin(reference_heading) * dx + np.cos(reference_heading) * dy)
        heading_error = abs(wrap_to_pi(float(reference_heading) - float(eval_state.front_heading)))
        goal_progress = float(current_goal_distance - float(self.env.distance_to_goal(state=eval_state)))
        articulation_ratio = max(
            (abs(float(state.articulation_angle)) / max(float(self.vehicle_config.articulation_limit_rad), 1e-6) for state in (prefix_states or [eval_state])),
            default=0.0,
        )
        collision = bool(rollout.termination_reason == "collision")
        if context.collision_checker is not None:
            collision = collision or any(bool(context.collision_checker(state)) for state in prefix_states)

        score = 2.5 * goal_progress
        score += 0.6 * max(along_track, 0.0)
        score -= 1.1 * abs(cross_track)
        score -= 0.5 * heading_error
        score -= 1.2 * max(articulation_ratio - 0.75, 0.0)
        if along_track < -1e-3:
            score -= 0.8
        if collision:
            score -= self._COLLISION_PENALTY
        if rollout.termination_reason == "articulation_limit":
            score -= self._ARTICULATION_LIMIT_PENALTY
        if self._is_invalid_reverse_switch(action_id, previous_action_id):
            score -= self._REVERSE_SWITCH_PENALTY
        return float(score)

    def _is_invalid_reverse_switch(self, action_id: int, previous_action_id: Optional[int]) -> bool:
        if previous_action_id is None:
            return False
        current_direction = int(self.primitive_library.spec(action_id).direction_sign)
        previous_direction = int(self.primitive_library.spec(previous_action_id).direction_sign)
        if current_direction == 0 or previous_direction == 0:
            return False
        return int(np.sign(current_direction)) != int(np.sign(previous_direction))

    def _softmax(self, scores: np.ndarray) -> Optional[np.ndarray]:
        finite_mask = np.isfinite(scores) & (scores > -1e8)
        if not np.any(finite_mask):
            return None
        stabilized = scores[finite_mask] - float(np.max(scores[finite_mask]))
        weights = np.exp(stabilized / max(self._SCORE_TEMPERATURE, 1e-6)).astype(np.float32)
        probs = np.zeros_like(scores, dtype=np.float32)
        probs[finite_mask] = weights / max(float(np.sum(weights)), 1e-6)
        return probs