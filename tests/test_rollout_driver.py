import numpy as np

from common.types import ArticulatedState, MacroAction, PrimitiveExecutionContext
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.config import VehicleConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import ActionSelection, HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library
from training.rollout import MacroRolloutDriver


def test_macro_rollout_driver_collects_episode_summary():
    env = create_env_adapter(
        env_config=EnvRuntimeConfig(max_low_level_steps_per_episode=30),
        vehicle_config=VehicleConfig(),
        observation_config=ObservationConfig(lidar_num_beams=8),
        reward_config=RewardConfig(),
    )
    library = build_default_primitive_library()
    agent = HybridPPOAgent(
        config=__import__("common.config", fromlist=["HybridPPOConfig"]).HybridPPOConfig(
            observation_dim=env.observation_dim,
            action_dim=library.action_dim,
            parameter_dim=library.parameter_dim,
            mini_batch_size=2,
            update_epochs=1,
        ),
        primitive_library=library,
    )
    executor = ParameterizedPrimitiveExecutor(library)
    macro_env = ParameterizedMacroActionWrapper(env, executor, gamma=agent.config.gamma)
    driver = MacroRolloutDriver(env, macro_env, agent, max_macro_steps=4)

    summary = driver.collect_episode(level="Debug", seed=3, deterministic=True, store_transition=False)
    assert summary.macro_steps > 0
    assert summary.low_level_steps >= summary.macro_steps
    assert isinstance(summary.total_reward, float)


class _BudgetEnv:
    observation_dim = 4

    def reset(self, seed=None, options=None):
        self._info = {
            "done_reason": "reset",
            "goal_reached": False,
            "success": False,
            "collision": False,
        }
        return np.zeros((self.observation_dim,), dtype=np.float32), dict(self._info)

    def current_info(self):
        return dict(self._info)

    def make_primitive_context(self):
        return PrimitiveExecutionContext()

    def get_articulated_state(self):
        return ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0)

    def build_observation(self):
        return np.zeros((self.observation_dim,), dtype=np.float32)

    def distance_to_goal(self, state=None):
        return 1.0


class _BudgetMacroEnv:
    def step(self, macro_action, start_state=None, context=None):
        return (
            np.zeros((4,), dtype=np.float32),
            0.0,
            False,
            False,
            {"tau": 1, "done_reason": "running", "goal_reached": False, "success": False, "collision": False},
        )


class _BudgetAgent:
    def act(self, observation, deterministic=False, teacher_action_probs=None, teacher_weight=0.0, return_diagnostics=False):
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "selected_action_id": 0,
                "selected_semantic": "forward-left",
                "fallback_triggered": False,
                "all_invalid_fallback": False,
                "hard_valid_ratio": 1.0,
                "soft_score_mean": 1.0,
                "soft_score_min": 1.0,
                "selection_mask_positive_count": 1,
                "selected_matches_raw_argmax": True,
                "selected_matches_semantic_argmax": True,
                "selected_action_hard_valid": True,
                "selected_raw_prob": 0.8,
                "selected_prob": 0.8,
                "selected_soft_score": 1.0,
                "selected_selection_mask": 1.0,
                "selected_proxy_action_mask": 1.0,
                "selected_semantic_score": 1.0,
                "selected_proxy_score_max": 1.0,
                "selected_proxy_prefix_max": 4.0,
            }
        return ActionSelection(
            macro_action=MacroAction(primitive_id=0, parameters=np.zeros((1,), dtype=np.float32)),
            log_prob=0.0,
            value=0.0,
            discrete_log_prob=0.0,
            continuous_log_prob=0.0,
            diagnostics=diagnostics,
        )


def test_macro_rollout_driver_marks_macro_budget_truncation():
    driver = MacroRolloutDriver(
        env=_BudgetEnv(),
        macro_env=_BudgetMacroEnv(),
        agent=_BudgetAgent(),
        max_macro_steps=2,
    )

    summary = driver.collect_episode(level="Debug", seed=1, deterministic=True, store_transition=False)

    assert summary.truncated is True
    assert summary.done_reason == "macro_budget"
    assert summary.last_info["done_reason"] == "macro_budget"
    assert summary.action_diagnostics["steps"] == 2
    assert summary.action_diagnostics["dominant_semantic"] == "forward-left"
    assert summary.action_diagnostics["selected_action_hard_valid_rate"] == 1.0
    assert summary.action_diagnostics["selected_semantic_score_mean"] == 1.0