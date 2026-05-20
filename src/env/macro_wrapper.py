from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ..common.types import ArticulatedState, MacroAction, PrimitiveExecutionContext, PrimitiveRollout
from ..primitives.library import ParameterizedPrimitiveExecutor


@runtime_checkable
class LowLevelEnvProtocol(Protocol):
    def step(self, action: np.ndarray):
        ...

    def get_articulated_state(self) -> ArticulatedState:
        ...


class ParameterizedMacroActionWrapper:
    def __init__(self, env: LowLevelEnvProtocol, executor: ParameterizedPrimitiveExecutor, gamma: float = 0.99) -> None:
        self.env = env
        self.executor = executor
        self.gamma = float(gamma)

    def step(self, macro_action: MacroAction, start_state: Optional[ArticulatedState] = None, context: Optional[PrimitiveExecutionContext] = None):
        if start_state is None:
            start_state = self.env.get_articulated_state()
        rollout = self.executor.rollout(start_state=start_state, primitive_id=macro_action.primitive_id, parameters=macro_action.parameters, context=context)
        return self._execute_rollout(macro_action, rollout)

    def _execute_rollout(self, macro_action: MacroAction, rollout: PrimitiveRollout):
        total_reward = 0.0
        last_obs = None
        terminated = False
        truncated = False
        info = {}
        tau = 0

        for index, control in enumerate(rollout.controls):
            step_result = self.env.step(control.as_array())
            if len(step_result) == 5:
                obs, reward, terminated, truncated, step_info = step_result
            elif len(step_result) == 4:
                obs, reward, done, step_info = step_result
                terminated = bool(done)
                truncated = False
            else:
                raise ValueError(f"unexpected env.step result length: {len(step_result)}")
            total_reward += (self.gamma ** index) * float(reward)
            tau = index + 1
            last_obs = obs
            info.update(step_info)
            if terminated or truncated:
                break

        info.update({
            "tau": int(tau),
            "primitive_id": int(macro_action.primitive_id),
            "parameters": np.asarray(macro_action.parameters, dtype=np.float32).copy(),
            "macro_rollout": rollout,
        })
        return last_obs, float(total_reward), bool(terminated), bool(truncated), info
