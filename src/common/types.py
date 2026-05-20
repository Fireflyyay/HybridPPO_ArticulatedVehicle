from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class ArticulatedState:
    x: float
    y: float
    front_heading: float
    rear_heading: float
    speed: float = 0.0
    articulation_rate: float = 0.0

    @property
    def articulation_angle(self) -> float:
        return wrap_to_pi(float(self.front_heading) - float(self.rear_heading))

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.front_heading, self.rear_heading, self.speed, self.articulation_rate],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LowLevelControl:
    articulation_rate: float
    speed: float

    def as_array(self) -> np.ndarray:
        return np.array([self.articulation_rate, self.speed], dtype=np.float64)


@dataclass(frozen=True)
class MacroAction:
    primitive_id: int
    parameters: np.ndarray


@dataclass(frozen=True)
class PrimitiveExecutionContext:
    goal_position: Optional[Tuple[float, float]] = None
    goal_heading: Optional[float] = None
    collision_checker: Optional[Callable[["ArticulatedState"], bool]] = None
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class PrimitiveRollout:
    primitive_id: int
    parameters: Dict[str, float]
    states: List[ArticulatedState]
    controls: List[LowLevelControl]
    termination_reason: str
    travelled_distance: float
    elapsed_time: float

    @property
    def tau(self) -> int:
        return int(len(self.controls))


@dataclass(frozen=True)
class MacroTransition:
    observation: np.ndarray
    action_id: int
    parameters: np.ndarray
    reward: float
    tau: int
    next_observation: np.ndarray
    done: bool
    log_prob: float
    value: float


@dataclass
class SMDPTargets:
    advantages: np.ndarray
    returns: np.ndarray
    deltas: np.ndarray


ArrayLike = Sequence[float]
