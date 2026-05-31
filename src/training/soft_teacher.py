from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from common.config import VehicleConfig
from common.runtime_config import BASE_OBSERVATION_FEATURE_DIM, GUIDANCE_FEATURE_DIM, ObservationConfig
from common.types import ArticulatedState, PrimitiveExecutionContext, PrimitiveRollout, wrap_to_pi
from env.adapter import UnifiedArticulatedEnvProtocol
from env.success import ParkingSuccessChecker
from primitives import ParameterizedPrimitiveExecutor, ParameterizedPrimitiveLibrary, build_proxy_parameter_grid


@dataclass(frozen=True)
class TeacherAdvice:
    action_probs: np.ndarray
    parameter_targets: np.ndarray
    weight: float
    diagnostics: Dict[str, object]
    reference_goal_position: Optional[Tuple[float, float]] = None
    reference_goal_heading: Optional[float] = None


class CoarseGuidanceSoftTeacher:
    _MAX_CANDIDATES_PER_ACTION = 6
    _SCORE_TEMPERATURE = 0.35
    _MAX_WEIGHT = 0.65
    _BASE_ACTIVATION = 0.18
    _ARTICULATION_LIMIT_PENALTY = 2.5
    _REVERSE_SWITCH_PENALTY = 0.25
    _STALL_PENALTY = 0.25
    _ROLLBACK_PENALTY = 0.45

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
        self.success_checker = ParkingSuccessChecker(vehicle_config)
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
        start_state = self.env.get_articulated_state()
        current_goal_distance = float(self.env.distance_to_goal(state=start_state))
        teacher_weight, diagnostics = self._teacher_weight(
            lidar=lidar,
            guidance=guidance,
            info=info,
            current_goal_distance=current_goal_distance,
        )
        if teacher_weight <= 0.0:
            return None

        context = self.env.make_primitive_context()
        goal_state = self.env.get_goal_state()
        current_success_metrics = self.success_checker.evaluate(start_state, goal_state, collision_free=True)
        reference_heading = self._reference_heading(
            start_state,
            guidance,
            goal_heading=float(goal_state.front_heading),
            current_goal_distance=current_goal_distance,
        )
        if reference_heading is None:
            return None

        current_topology_cost = self.env.query_cost_to_go(state=start_state)
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
                    current_topology_cost=current_topology_cost,
                    reference_heading=reference_heading,
                    goal_state=goal_state,
                    current_success_metrics=current_success_metrics,
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

    def _teacher_weight(
        self,
        lidar: np.ndarray,
        guidance: np.ndarray,
        info: Dict[str, object],
        current_goal_distance: float,
    ) -> Tuple[float, Dict[str, float]]:
        if not bool(info.get("guidance_available", False)):
            return 0.0, {"narrow_score": 0.0, "guidance_confidence": 0.0, "guidance_strength": 0.0}

        guidance_strength = float(np.clip(np.linalg.norm(guidance[:2]), 0.0, 1.0))
        guidance_confidence = float(np.clip(max(float(info.get("guidance_path_confidence", 0.0)), float(guidance[3])), 0.0, 1.0))
        if guidance_strength <= 1e-4 or guidance_confidence <= 1e-4:
            return 0.0, {"narrow_score": 0.0, "guidance_confidence": guidance_confidence, "guidance_strength": guidance_strength}

        corridor_width = info.get("corridor_width")
        narrow_score = self._narrow_score(lidar=lidar, corridor_width=corridor_width)
        near_goal_factor = float(np.clip(1.0 - current_goal_distance / self._goal_approach_distance(), 0.0, 1.0))
        teacher_activation = float(np.clip(self._BASE_ACTIVATION + 0.55 * narrow_score + 0.25 * near_goal_factor, 0.0, 1.0))

        if str(info.get("level", "")).lower() == "debug":
            narrow_score = 0.0
            teacher_activation = 0.0

        teacher_weight = float(
            np.clip(
                self._MAX_WEIGHT * guidance_confidence * guidance_strength * teacher_activation,
                0.0,
                self._MAX_WEIGHT,
            )
        )
        return teacher_weight, {
            "narrow_score": float(narrow_score),
            "guidance_confidence": guidance_confidence,
            "guidance_strength": guidance_strength,
            "teacher_activation": float(teacher_activation),
            "near_goal_factor": float(near_goal_factor),
            "corridor_width": -1.0 if corridor_width is None else float(corridor_width),
        }

    def _narrow_score(self, lidar: np.ndarray, corridor_width: Optional[object]) -> float:
        if corridor_width is not None:
            vehicle_width = max(float(self.vehicle_config.body_width), 1e-6)
            normalized_clearance = max(float(corridor_width) - vehicle_width, 0.0) / vehicle_width
            return float(np.clip((1.75 - normalized_clearance) / 1.75, 0.0, 1.0))
        min_clearance = 1.0 if lidar.size == 0 else float(np.clip(np.min(lidar), 0.0, 1.0))
        return float(np.clip((0.45 - min_clearance) / 0.25, 0.0, 1.0))

    def _goal_approach_distance(self) -> float:
        return float(max(float(self.vehicle_config.front_length) + float(self.vehicle_config.rear_length), 8.0))

    def _reference_heading(
        self,
        start_state: ArticulatedState,
        guidance: np.ndarray,
        goal_heading: Optional[float] = None,
        current_goal_distance: Optional[float] = None,
    ) -> Optional[float]:
        direction = np.asarray(guidance[:2], dtype=np.float32)
        if float(np.linalg.norm(direction)) <= 1e-4:
            return None if goal_heading is None else float(goal_heading)
        guidance_heading = wrap_to_pi(float(start_state.front_heading) + float(np.arctan2(direction[1], direction[0])))
        if goal_heading is None or current_goal_distance is None:
            return guidance_heading
        near_goal_factor = float(np.clip(1.0 - float(current_goal_distance) / self._goal_approach_distance(), 0.0, 1.0))
        if near_goal_factor <= 1e-4:
            return guidance_heading
        return self._blend_heading(guidance_heading, float(goal_heading), near_goal_factor)

    @staticmethod
    def _blend_heading(primary_heading: float, secondary_heading: float, secondary_weight: float) -> float:
        blend = float(np.clip(secondary_weight, 0.0, 1.0))
        x = float((1.0 - blend) * np.cos(primary_heading) + blend * np.cos(secondary_heading))
        y = float((1.0 - blend) * np.sin(primary_heading) + blend * np.sin(secondary_heading))
        if abs(x) <= 1e-6 and abs(y) <= 1e-6:
            return wrap_to_pi(float(primary_heading))
        return wrap_to_pi(float(np.arctan2(y, x)))

    def _score_rollout(
        self,
        start_state: ArticulatedState,
        rollout: PrimitiveRollout,
        action_id: int,
        previous_action_id: Optional[int],
        current_goal_distance: float,
        current_topology_cost: Optional[float],
        reference_heading: float,
        goal_state: ArticulatedState,
        current_success_metrics,
        context: PrimitiveExecutionContext,
    ) -> float:
        scored_states = rollout.states[1:] if len(rollout.states) > 1 else [start_state]
        eval_state = scored_states[-1]
        goal_heading = float(context.goal_heading) if context.goal_heading is not None else float(reference_heading)
        start_goal_heading_error = abs(wrap_to_pi(goal_heading - float(start_state.front_heading)))
        start_overlap_score = float(0.75 * current_success_metrics.front_overlap_ratio + 0.25 * current_success_metrics.rear_overlap_ratio)
        best_progress = 0.0
        terminal_progress = 0.0
        best_heading_improvement = 0.0
        terminal_heading_improvement = 0.0
        best_overlap_progress = 0.0
        terminal_overlap_progress = 0.0
        best_cross_track = float("inf")
        terminal_cross_track = 0.0
        min_goal_distance = float(current_goal_distance)
        topology_progress_values = []
        reached_success = False

        for state_idx, state in enumerate(scored_states):
            goal_distance = float(self.env.distance_to_goal(state=state))
            goal_progress = float(current_goal_distance - goal_distance)
            min_goal_distance = min(min_goal_distance, goal_distance)
            best_progress = max(best_progress, goal_progress)
            if state_idx == len(scored_states) - 1:
                terminal_progress = goal_progress

            goal_heading_error = abs(wrap_to_pi(goal_heading - float(state.front_heading)))
            heading_improvement = float(start_goal_heading_error - goal_heading_error)
            best_heading_improvement = max(best_heading_improvement, heading_improvement)
            if state_idx == len(scored_states) - 1:
                terminal_heading_improvement = heading_improvement

            state_collision = False if context.collision_checker is None else bool(context.collision_checker(state))
            success_metrics = self.success_checker.evaluate(state, goal_state, collision_free=(not state_collision))
            overlap_score = float(0.75 * success_metrics.front_overlap_ratio + 0.25 * success_metrics.rear_overlap_ratio)
            overlap_progress = float(overlap_score - start_overlap_score)
            best_overlap_progress = max(best_overlap_progress, overlap_progress)
            if state_idx == len(scored_states) - 1:
                terminal_overlap_progress = overlap_progress
            reached_success = reached_success or bool(success_metrics.success)

            dx = float(state.x - start_state.x)
            dy = float(state.y - start_state.y)
            cross_track = abs(float(-np.sin(reference_heading) * dx + np.cos(reference_heading) * dy))
            best_cross_track = min(best_cross_track, cross_track)
            if state_idx == len(scored_states) - 1:
                terminal_cross_track = cross_track

            if current_topology_cost is not None:
                state_topology_cost = self.env.query_cost_to_go(state=state)
                if state_topology_cost is not None:
                    topology_progress_values.append(float(current_topology_cost - state_topology_cost))

        if topology_progress_values:
            best_progress = max(best_progress, max(0.0, max(topology_progress_values)))
            terminal_progress = max(terminal_progress, float(topology_progress_values[-1]))

        articulation_ratio = max(
            (abs(float(state.articulation_angle)) / max(float(self.vehicle_config.articulation_limit_rad), 1e-6) for state in scored_states),
            default=0.0,
        )
        collision = bool(rollout.termination_reason == "collision")
        if context.collision_checker is not None:
            collision = collision or any(bool(context.collision_checker(state)) for state in scored_states)
        if collision:
            return -1e9

        near_goal_factor = float(
            np.clip(
                1.0 - min(float(current_goal_distance), float(min_goal_distance)) / self._goal_approach_distance(),
                0.0,
                1.0,
            )
        )
        final_reference_heading_error = abs(wrap_to_pi(float(reference_heading) - float(eval_state.front_heading)))
        score = 2.4 * best_progress
        score += 1.6 * terminal_progress
        score += (0.45 + 1.00 * near_goal_factor) * best_heading_improvement
        score += (0.25 + 0.75 * near_goal_factor) * terminal_heading_improvement
        score += (0.35 + 1.10 * near_goal_factor) * best_overlap_progress
        score += (0.25 + 1.45 * near_goal_factor) * terminal_overlap_progress
        score += 0.30 * max(float(np.cos(final_reference_heading_error)), 0.0)
        score -= (0.10 + 0.25 * (1.0 - near_goal_factor)) * terminal_cross_track
        if np.isfinite(best_cross_track):
            score -= 0.05 * best_cross_track
        score -= 1.0 * max(articulation_ratio - 0.78, 0.0)
        if rollout.travelled_distance <= 0.10 and current_goal_distance > 0.5 * self._goal_approach_distance():
            score -= self._STALL_PENALTY
        if float(self.env.distance_to_goal(state=eval_state)) > float(current_goal_distance) + 0.75 and best_progress < 0.20:
            score -= self._ROLLBACK_PENALTY
        if reached_success:
            score += 12.0

        if rollout.termination_reason == "articulation_limit":
            score -= self._ARTICULATION_LIMIT_PENALTY
        if self._is_invalid_reverse_switch(action_id, previous_action_id) and best_progress < 0.20 and terminal_heading_improvement < 0.10:
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
