from dataclasses import dataclass

import numpy as np

from common.config import VehicleConfig
from common.types import ArticulatedState, LowLevelControl, wrap_to_pi


@dataclass(frozen=True)
class ArticulatedStepDiagnostics:
    saturated_articulation: bool
    effective_articulation_rate: float


class ArticulatedKinematics:
    """Center-articulated vehicle kinematics."""

    def __init__(self, config: VehicleConfig) -> None:
        self.config = config

    def step(self, state: ArticulatedState, control: LowLevelControl, repeat: int = 1) -> ArticulatedState:
        next_state, _ = self.step_with_diagnostics(state, control, repeat=repeat)
        return next_state

    def step_with_diagnostics(self, state: ArticulatedState, control: LowLevelControl, repeat: int = 1):
        speed = float(np.clip(control.speed, self.config.speed_min, self.config.speed_max))
        articulation_rate = float(np.clip(control.articulation_rate, self.config.articulation_rate_min, self.config.articulation_rate_max))
        x = float(state.x)
        y = float(state.y)
        front_heading = float(state.front_heading)
        rear_heading = float(state.rear_heading)
        effective_rate = articulation_rate
        saturated = False
        dt = float(self.config.step_seconds) / max(1, int(self.config.integrator_substeps))
        l1 = float(self.config.hitch_offset)
        l2 = float(self.config.trailer_length)
        articulation_limit = float(self.config.articulation_limit_rad)

        for _ in range(max(1, int(repeat))):
            for _ in range(max(1, int(self.config.integrator_substeps))):
                phi = wrap_to_pi(front_heading - rear_heading)
                effective_rate = articulation_rate
                if phi >= articulation_limit and articulation_rate > 0.0:
                    effective_rate = 0.0
                    saturated = True
                elif phi <= -articulation_limit and articulation_rate < 0.0:
                    effective_rate = 0.0
                    saturated = True

                denom = l1 * np.cos(phi) + l2
                if abs(float(denom)) < 1e-6:
                    denom = 1e-6 if denom >= 0.0 else -1e-6

                front_heading_dot = (speed * np.sin(phi) + l2 * effective_rate) / denom
                rear_heading_dot = front_heading_dot - effective_rate
                x += speed * np.cos(front_heading) * dt
                y += speed * np.sin(front_heading) * dt
                front_heading = wrap_to_pi(front_heading + front_heading_dot * dt)
                rear_heading = wrap_to_pi(rear_heading + rear_heading_dot * dt)

        return (
            ArticulatedState(x=x, y=y, front_heading=front_heading, rear_heading=rear_heading, speed=speed, articulation_rate=effective_rate),
            ArticulatedStepDiagnostics(saturated_articulation=bool(saturated), effective_articulation_rate=float(effective_rate)),
        )
