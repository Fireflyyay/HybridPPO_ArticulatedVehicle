import numpy as np

from common.types import ArticulatedState, MacroAction, PrimitiveExecutionContext
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.config import VehicleConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import ActionSelection, HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library
from training.rollout import MacroRolloutDriver
from training.soft_teacher import TeacherAdvice


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
            {
                "tau": 1,
                "done_reason": "running",
                "goal_reached": False,
                "success": False,
                "collision": False,
                "macro_termination_reason": "duration_budget",
            },
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


class _ReferenceEnv:
    observation_dim = 4

    def __init__(self):
        self._override = None
        self.last_context = None
        self._info = {
            "done_reason": "reset",
            "goal_reached": False,
            "success": False,
            "collision": False,
        }

    def reset(self, seed=None, options=None):
        del seed, options
        self._override = None
        self.last_context = None
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
        goal_position = (9.0, 9.0)
        goal_heading = 0.0
        if self._override is not None:
            goal_position, goal_heading = self._override
        self.last_context = PrimitiveExecutionContext(goal_position=goal_position, goal_heading=goal_heading)
        return self.last_context

    def set_reference_override(self, goal_position, goal_heading: float) -> None:
        self._override = ((float(goal_position[0]), float(goal_position[1])), float(goal_heading))

    def clear_reference_override(self) -> None:
        self._override = None

    def get_articulated_state(self):
        return ArticulatedState(x=0.0, y=0.0, front_heading=0.0, rear_heading=0.0)

    def build_observation(self):
        return np.zeros((self.observation_dim,), dtype=np.float32)

    def distance_to_goal(self, state=None):
        del state
        return 1.0


class _ReferenceMacroEnv:
    def __init__(self):
        self.context = None

    def step(self, macro_action, start_state=None, context=None):
        del macro_action, start_state
        self.context = context
        return (
            np.zeros((4,), dtype=np.float32),
            0.0,
            True,
            False,
            {"tau": 1, "done_reason": "goal_reached", "goal_reached": True, "success": True, "collision": False},
        )


class _ReferenceTeacher:
    def advise(self, observation, previous_action_id=None):
        del observation, previous_action_id
        return TeacherAdvice(
            action_probs=np.asarray([1.0], dtype=np.float32),
            parameter_targets=np.zeros((1, 1), dtype=np.float32),
            weight=0.5,
            diagnostics={"teacher_success": True},
            reference_goal_position=(3.0, 4.0),
            reference_goal_heading=0.75,
        )


class _ArticulationLimitMacroEnv:
    def __init__(self):
        self.calls = 0

    def step(self, macro_action, start_state=None, context=None):
        del macro_action, start_state, context
        self.calls += 1
        reason = "articulation_limit" if self.calls == 1 else "duration_budget"
        tau = 1 if self.calls == 1 else 3
        return (
            np.zeros((4,), dtype=np.float32),
            0.0,
            False,
            False,
            {
                "tau": tau,
                "done_reason": "running",
                "goal_reached": False,
                "success": False,
                "collision": False,
                "macro_termination_reason": reason,
            },
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
    assert summary.action_diagnostics["macro_termination_reason_counts"]["duration_budget"] == 2
    assert summary.action_diagnostics["macro_articulation_limit_rate"] == 0.0


def test_macro_rollout_driver_applies_teacher_reference_override_to_context():
    env = _ReferenceEnv()
    macro_env = _ReferenceMacroEnv()
    driver = MacroRolloutDriver(
        env=env,
        macro_env=macro_env,
        agent=_BudgetAgent(),
        max_macro_steps=1,
        soft_teacher=_ReferenceTeacher(),
    )

    summary = driver.collect_episode(level="Debug", seed=1, deterministic=True, store_transition=False)

    assert summary.success is True
    assert macro_env.context is not None
    assert macro_env.context.goal_position == (3.0, 4.0)
    assert np.isclose(float(macro_env.context.goal_heading), 0.75)


def test_macro_rollout_driver_reports_articulation_limit_macro_diagnostics():
    driver = MacroRolloutDriver(
        env=_BudgetEnv(),
        macro_env=_ArticulationLimitMacroEnv(),
        agent=_BudgetAgent(),
        max_macro_steps=2,
    )

    summary = driver.collect_episode(level="Debug", seed=1, deterministic=True, store_transition=False)

    assert summary.action_diagnostics["macro_termination_reason_counts"]["articulation_limit"] == 1
    assert summary.action_diagnostics["macro_termination_reason_counts"]["duration_budget"] == 1
    assert summary.action_diagnostics["macro_articulation_limit_rate"] == 0.5
    assert summary.action_diagnostics["macro_short_articulation_limit_rate"] == 0.5
    assert summary.action_diagnostics["macro_articulation_limit_tau_mean"] == 1.0
