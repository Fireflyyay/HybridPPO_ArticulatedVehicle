from typing import Dict, Optional

import torch

from common.runtime_config import ExperimentConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library, load_proxy_safety_sidecar
from training.checkpoint import CheckpointManager
from training.evaluator import PolicyEvaluator
from training.logger import TensorBoardLogger
from training.rollout import MacroRolloutDriver


def _resolve_device(device_name: str) -> torch.device:
    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_name))


def _metric_from_path(metrics: Dict[str, object], metric_path: str) -> float:
    cursor = metrics
    for part in str(metric_path).split("/"):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"metric path not found: {metric_path}")
        cursor = cursor[part]
    return float(cursor)


class ExperimentTrainer:
    def __init__(self, config: ExperimentConfig, resume_path: Optional[str] = None) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.logger = TensorBoardLogger(config.logging)
        proxy_sidecar = None
        proxy_sidecar_path = str(config.proxy_safety.sidecar_path).strip()
        if proxy_sidecar_path:
            proxy_sidecar = load_proxy_safety_sidecar(proxy_sidecar_path)
        self.primitive_library = build_default_primitive_library(proxy_sidecar=proxy_sidecar)
        self.agent = HybridPPOAgent(
            config=config.agent.build(
                observation_dim=int(config.observation.observation_dim),
                action_dim=int(self.primitive_library.action_dim),
                parameter_dim=int(self.primitive_library.parameter_dim),
            ),
            primitive_library=self.primitive_library,
            device=self.device,
        )
        self.train_env = create_env_adapter(
            env_config=config.env,
            vehicle_config=config.vehicle,
            observation_config=config.observation,
            reward_config=config.reward,
        )
        self.executor = ParameterizedPrimitiveExecutor(
            self.primitive_library,
            vehicle_config=config.vehicle,
            executor_config=config.primitive_executor,
        )
        self.macro_env = ParameterizedMacroActionWrapper(self.train_env, self.executor, gamma=self.agent.config.gamma)
        self.rollout_driver = MacroRolloutDriver(
            env=self.train_env,
            macro_env=self.macro_env,
            agent=self.agent,
            max_macro_steps=config.schedule.max_macro_steps_per_episode,
        )
        self.evaluator = PolicyEvaluator(config, self.primitive_library)
        self.checkpoints = CheckpointManager(self.logger.run_dir, config.checkpoint, config)
        self.start_episode = 0
        self.update_idx = 0
        self.logger.log_config(config)
        if resume_path is not None:
            metadata = self.checkpoints.load(resume_path, self.agent, map_location=str(self.device))
            self.start_episode = int(metadata["episode_idx"])
            self.update_idx = int(metadata["update_idx"])

    def run(self) -> str:
        pending_episodes = 0
        try:
            for episode_idx in range(int(self.start_episode), int(self.config.schedule.total_episodes)):
                level = self.config.schedule.level_for_episode(episode_idx)
                summary = self.rollout_driver.collect_episode(
                    level=level,
                    seed=int(self.config.seed + episode_idx),
                    deterministic=False,
                    store_transition=True,
                )
                pending_episodes += 1
                self.logger.log_training_episode(
                    episode_idx + 1,
                    {
                        "episode": episode_idx + 1,
                        "level": 0.0 if level == "Debug" else (1.0 if level == "Warmup" else 2.0),
                        "total_reward": summary.total_reward,
                        "macro_steps": summary.macro_steps,
                        "low_level_steps": summary.low_level_steps,
                        "success": float(summary.success),
                        "collision": float(summary.collision),
                        "terminated": float(summary.terminated),
                        "truncated": float(summary.truncated),
                        "final_goal_distance": summary.final_goal_distance,
                    },
                )

                if pending_episodes >= int(self.config.schedule.episodes_per_update) and len(self.agent.buffer) > 0:
                    update_metrics = self.agent.update()
                    self.update_idx += 1
                    pending_episodes = 0
                    self.logger.log_update(episode_idx + 1, update_metrics)

                if int(self.config.evaluation.interval) > 0 and (episode_idx + 1) % int(self.config.evaluation.interval) == 0:
                    eval_metrics = self.evaluator.evaluate(self.agent, seed_offset=(episode_idx + 1) * 100)
                    self.logger.log_evaluation(episode_idx + 1, eval_metrics)
                    metric_value = _metric_from_path(eval_metrics, self.config.checkpoint.best_metric)
                    self.checkpoints.maybe_save_best(
                        self.agent,
                        episode_idx + 1,
                        self.update_idx,
                        metric_value,
                        extra={"evaluation": eval_metrics},
                    )

                if int(self.config.checkpoint.save_interval) > 0 and (episode_idx + 1) % int(self.config.checkpoint.save_interval) == 0:
                    self.checkpoints.save_latest(self.agent, episode_idx + 1, self.update_idx)
                    self.checkpoints.save_periodic(self.agent, episode_idx + 1, self.update_idx)

            if len(self.agent.buffer) > 0:
                update_metrics = self.agent.update()
                self.update_idx += 1
                self.logger.log_update(int(self.config.schedule.total_episodes), update_metrics)

            self.checkpoints.save_latest(self.agent, int(self.config.schedule.total_episodes), self.update_idx)
            return self.logger.run_dir
        finally:
            self.logger.close()