from typing import Dict, List

from common.runtime_config import ExperimentConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, ParameterizedPrimitiveLibrary
from training.rollout import EpisodeSummary, MacroRolloutDriver


def _aggregate(results: List[EpisodeSummary]) -> Dict[str, float]:
    count = max(1, len(results))
    return {
        "success_rate": float(sum(1 for item in results if item.success) / count),
        "collision_rate": float(sum(1 for item in results if item.collision) / count),
        "mean_return": float(sum(item.total_reward for item in results) / count),
        "mean_macro_steps": float(sum(item.macro_steps for item in results) / count),
        "mean_low_level_steps": float(sum(item.low_level_steps for item in results) / count),
        "mean_goal_distance": float(sum(item.final_goal_distance for item in results) / count),
    }


class PolicyEvaluator:
    def __init__(self, config: ExperimentConfig, primitive_library: ParameterizedPrimitiveLibrary) -> None:
        self.config = config
        self.primitive_library = primitive_library

    def evaluate(self, agent: HybridPPOAgent, seed_offset: int = 0) -> Dict[str, Dict[str, float]]:
        per_level: Dict[str, Dict[str, float]] = {}
        overall_results: List[EpisodeSummary] = []
        for level_index, level in enumerate(self.config.evaluation.levels):
            env = create_env_adapter(
                env_config=self.config.env,
                vehicle_config=self.config.vehicle,
                observation_config=self.config.observation,
                reward_config=self.config.reward,
            )
            executor = ParameterizedPrimitiveExecutor(
                self.primitive_library,
                vehicle_config=self.config.vehicle,
                executor_config=self.config.primitive_executor,
            )
            macro_env = ParameterizedMacroActionWrapper(env, executor, gamma=self.config.agent.gamma)
            driver = MacroRolloutDriver(
                env=env,
                macro_env=macro_env,
                agent=agent,
                max_macro_steps=self.config.schedule.max_macro_steps_per_episode,
            )
            level_results = [
                driver.collect_episode(
                    level=str(level),
                    seed=int(self.config.seed + seed_offset + level_index * 1000 + episode_idx),
                    deterministic=bool(self.config.evaluation.deterministic),
                    store_transition=False,
                )
                for episode_idx in range(int(self.config.evaluation.episodes_per_level))
            ]
            per_level[str(level)] = _aggregate(level_results)
            overall_results.extend(level_results)
        return {
            "overall": _aggregate(overall_results),
            "levels": per_level,
        }