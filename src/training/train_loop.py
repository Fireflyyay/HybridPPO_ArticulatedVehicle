from typing import Dict, Optional, Tuple

import torch

from common.runtime_config import ExperimentConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library, load_proxy_safety_sidecar
from training.checkpoint import CheckpointManager
from training.curriculum import DifficultyAdaptiveSampler, SuccessBandCurriculum
from training.evaluator import PolicyEvaluator
from training.logger import TensorBoardLogger
from training.rollout import MacroRolloutDriver
from training.soft_teacher import CoarseGuidanceSoftTeacher


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


def _auto_checkpoint_metric_path(curriculum: SuccessBandCurriculum) -> str:
    if curriculum.success_band_active() or int(curriculum.level_counts.get(curriculum.target_level, 0)) > 0:
        return f"levels/{curriculum.target_level}/success_rate"
    return f"levels/{curriculum.warmup_level}/success_rate"


class ExperimentTrainer:
    def __init__(self, config: ExperimentConfig, resume_path: Optional[str] = None) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.logger = TensorBoardLogger(config.logging)
        self.curriculum = SuccessBandCurriculum(config.schedule, seed=int(config.seed))
        self.difficulty_sampler: Optional[DifficultyAdaptiveSampler] = None
        if config.schedule.adaptive_sampling_enabled:
            self.difficulty_sampler = DifficultyAdaptiveSampler(config.schedule, seed=int(config.seed))
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
        self.soft_teacher: Optional[CoarseGuidanceSoftTeacher] = None
        if config.teacher_enabled:
            self.soft_teacher = CoarseGuidanceSoftTeacher(
                env=self.train_env,
                executor=self.executor,
                primitive_library=self.primitive_library,
                vehicle_config=config.vehicle,
                observation_config=config.observation,
            )
        self.rollout_driver = MacroRolloutDriver(
            env=self.train_env,
            macro_env=self.macro_env,
            agent=self.agent,
            max_macro_steps=config.schedule.max_macro_steps_per_episode,
            soft_teacher=self.soft_teacher,
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
            extra = dict(metadata.get("extra", {}))
            self.curriculum.load_state_dict(extra.get("curriculum"))
            if self.difficulty_sampler is not None:
                self.difficulty_sampler.load_state_dict(extra.get("difficulty_sampler"))

    def _level_metric(self, level: str) -> float:
        if str(level) == str(self.curriculum.warmup_level):
            return 0.0
        if str(level) == str(self.curriculum.target_level):
            return 1.0
        return -1.0

    def _checkpoint_extra(self, evaluation: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        extra: Dict[str, object] = {
            "curriculum": self.curriculum.state_dict(),
        }
        if self.difficulty_sampler is not None:
            extra["difficulty_sampler"] = self.difficulty_sampler.state_dict()
        if evaluation is not None:
            extra["evaluation"] = evaluation
        return extra

    def _resolve_checkpoint_metric(self, evaluation: Dict[str, object]) -> Tuple[str, float]:
        configured_metric = str(self.config.checkpoint.best_metric).strip()
        candidate_paths = [configured_metric] if configured_metric else ["auto"]
        if candidate_paths[0].lower() == "auto":
            candidate_paths = [
                _auto_checkpoint_metric_path(self.curriculum),
                "overall/success_rate",
            ]
        for metric_path in candidate_paths:
            try:
                return metric_path, _metric_from_path(evaluation, metric_path)
            except KeyError:
                continue
        raise KeyError(f"unable to resolve checkpoint metric from {candidate_paths}")

    def run(self) -> str:
        pending_episodes = 0
        try:
            for episode_idx in range(int(self.start_episode), int(self.config.schedule.total_episodes)):
                reset_options = self.curriculum.reset_options()
                level = str(reset_options["level"])
                if self.difficulty_sampler is not None:
                    bucket = self.difficulty_sampler.choose_bucket(level)
                    reset_options["difficulty_bucket"] = bucket
                summary = self.rollout_driver.collect_episode(
                    level=level,
                    seed=int(self.config.seed + episode_idx),
                    deterministic=False,
                    store_transition=True,
                    reset_options=reset_options,
                )
                pending_episodes += 1
                self.curriculum.record_episode(level, summary.success)
                if self.difficulty_sampler is not None:
                    scene_meta = dict(summary.last_info.get("scene_metadata", {}))
                    heading_diff = float(scene_meta.get("heading_diff_deg", 0.0))
                    actual_bucket = DifficultyAdaptiveSampler.compute_bucket(heading_diff)
                    self.difficulty_sampler.record_episode(level, actual_bucket, summary.success)
                log_payload = {
                    "episode": episode_idx + 1,
                    "level": self._level_metric(level),
                    "total_reward": summary.total_reward,
                    "macro_steps": summary.macro_steps,
                    "low_level_steps": summary.low_level_steps,
                    "success": float(summary.success),
                    "collision": float(summary.collision),
                    "terminated": float(summary.terminated),
                    "truncated": float(summary.truncated),
                    "final_goal_distance": summary.final_goal_distance,
                    "action_diagnostics": summary.action_diagnostics,
                    "curriculum": self.curriculum.metrics(),
                }
                if self.difficulty_sampler is not None:
                    log_payload["difficulty_sampler"] = self.difficulty_sampler.metrics()
                self.logger.log_training_episode(episode_idx + 1, log_payload)

                if pending_episodes >= int(self.config.schedule.episodes_per_update) and len(self.agent.buffer) > 0:
                    update_metrics = self.agent.update()
                    self.update_idx += 1
                    pending_episodes = 0
                    self.logger.log_update(episode_idx + 1, update_metrics)

                if int(self.config.evaluation.interval) > 0 and (episode_idx + 1) % int(self.config.evaluation.interval) == 0:
                    eval_metrics = self.evaluator.evaluate(self.agent, seed_offset=(episode_idx + 1) * 100)
                    metric_path, metric_value = self._resolve_checkpoint_metric(eval_metrics)
                    eval_metrics_for_logging = dict(eval_metrics)
                    eval_metrics_for_logging["checkpoint"] = {"metric_value": float(metric_value)}
                    self.logger.log_evaluation(episode_idx + 1, eval_metrics_for_logging)
                    checkpoint_extra = self._checkpoint_extra(eval_metrics)
                    checkpoint_extra["checkpoint_metric_path"] = str(metric_path)
                    checkpoint_extra["checkpoint_metric_value"] = float(metric_value)
                    self.checkpoints.maybe_save_best(
                        self.agent,
                        episode_idx + 1,
                        self.update_idx,
                        metric_value,
                        extra=checkpoint_extra,
                    )

                if int(self.config.checkpoint.save_interval) > 0 and (episode_idx + 1) % int(self.config.checkpoint.save_interval) == 0:
                    extra = self._checkpoint_extra()
                    self.checkpoints.save_latest(self.agent, episode_idx + 1, self.update_idx, extra=extra)
                    self.checkpoints.save_periodic(self.agent, episode_idx + 1, self.update_idx, extra=extra)

            if len(self.agent.buffer) > 0:
                update_metrics = self.agent.update()
                self.update_idx += 1
                self.logger.log_update(int(self.config.schedule.total_episodes), update_metrics)

            self.checkpoints.save_latest(
                self.agent,
                int(self.config.schedule.total_episodes),
                self.update_idx,
                extra=self._checkpoint_extra(),
            )
            return self.logger.run_dir
        finally:
            self.logger.close()