from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np

from common.config import VehicleConfig
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.types import ArticulatedState, PrimitiveExecutionContext
from env.scenes import BaselineInspiredSceneFactory
from env.task_env import KinematicTaskEnv


@runtime_checkable
class UnifiedArticulatedEnvProtocol(Protocol):
    observation_dim: int

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        ...

    def step(self, action: np.ndarray):
        ...

    def build_observation(self) -> np.ndarray:
        ...

    def get_articulated_state(self) -> ArticulatedState:
        ...

    def get_goal_state(self) -> ArticulatedState:
        ...

    def current_info(self) -> Dict[str, object]:
        ...

    def distance_to_goal(self, state: Optional[ArticulatedState] = None) -> float:
        ...

    def make_primitive_context(self) -> PrimitiveExecutionContext:
        ...


class KinematicTaskAdapter:
    def __init__(self, env: KinematicTaskEnv) -> None:
        self.env = env

    @property
    def observation_dim(self) -> int:
        return int(self.env.observation_dim)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def build_observation(self) -> np.ndarray:
        return self.env.build_observation()

    def get_articulated_state(self) -> ArticulatedState:
        return self.env.get_articulated_state()

    def get_goal_state(self) -> ArticulatedState:
        return self.env.get_goal_state()

    def current_info(self) -> Dict[str, object]:
        return self.env.current_info()

    def distance_to_goal(self, state: Optional[ArticulatedState] = None) -> float:
        return float(self.env.distance_to_goal(state))

    def make_primitive_context(self) -> PrimitiveExecutionContext:
        return self.env.make_primitive_context()


def create_env_adapter(
    env_config: EnvRuntimeConfig,
    vehicle_config: VehicleConfig,
    observation_config: ObservationConfig,
    reward_config: RewardConfig,
) -> KinematicTaskAdapter:
    scene_factory = BaselineInspiredSceneFactory(env_config.scene_presets)
    task_env = KinematicTaskEnv(
        scene_factory=scene_factory,
        vehicle_config=vehicle_config,
        observation_config=observation_config,
        reward_config=reward_config,
        env_config=env_config,
    )
    return KinematicTaskAdapter(task_env)