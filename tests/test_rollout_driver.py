from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.config import VehicleConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
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
            observation_dim=17,
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