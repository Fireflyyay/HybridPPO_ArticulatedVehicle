import numpy as np

from common.types import ArticulatedState, MacroAction
from env.macro_wrapper import ParameterizedMacroActionWrapper
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library


class DummyLowLevelEnv:
    def __init__(self):
        self.state = ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0)
        self.rewards = [1.0, 1.0, 1.0, 1.0]
        self.index = 0

    def get_articulated_state(self):
        return self.state

    def step(self, action: np.ndarray):
        reward = self.rewards[self.index]
        self.index += 1
        return np.zeros((4,), dtype=np.float32), reward, False, False, {"action": action.copy()}


def test_macro_wrapper_accumulates_discounted_reward():
    library = build_default_primitive_library()
    executor = ParameterizedPrimitiveExecutor(library)
    env = DummyLowLevelEnv()
    wrapper = ParameterizedMacroActionWrapper(env, executor, gamma=0.9)
    # step_seconds=0.4, duration=0.9 → ceil(0.9/0.4)=3 low-level steps
    params = library.dict_to_vector({"duration": 0.9})
    macro_action = MacroAction(primitive_id=7, parameters=params)
    _, total_reward, _, _, info = wrapper.step(macro_action)
    assert np.isclose(total_reward, 1.0 + 0.9 + 0.9**2)
    assert info["tau"] == 3
