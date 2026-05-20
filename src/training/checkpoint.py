import os
from typing import Dict, Optional

import torch

from common.runtime_config import CheckpointConfig, ExperimentConfig
from model.agent import HybridPPOAgent


class CheckpointManager:
    def __init__(self, run_dir: str, config: CheckpointConfig, experiment_config: ExperimentConfig) -> None:
        self.config = config
        self.experiment_config = experiment_config
        self.directory = os.path.abspath(os.path.join(run_dir, "checkpoints"))
        os.makedirs(self.directory, exist_ok=True)
        self.best_metric = float("-inf")

    def save_latest(self, agent: HybridPPOAgent, episode_idx: int, update_idx: int, extra: Optional[Dict[str, object]] = None) -> str:
        return self._save(self.config.latest_filename, agent, episode_idx, update_idx, extra=extra)

    def save_periodic(self, agent: HybridPPOAgent, episode_idx: int, update_idx: int, extra: Optional[Dict[str, object]] = None) -> str:
        filename = f"episode_{int(episode_idx):06d}.pt"
        return self._save(filename, agent, episode_idx, update_idx, extra=extra)

    def maybe_save_best(
        self,
        agent: HybridPPOAgent,
        episode_idx: int,
        update_idx: int,
        metric_value: float,
        extra: Optional[Dict[str, object]] = None,
    ) -> Optional[str]:
        if float(metric_value) <= float(self.best_metric):
            return None
        self.best_metric = float(metric_value)
        payload_extra = dict(extra or {})
        payload_extra["best_metric"] = float(self.best_metric)
        return self._save(self.config.best_filename, agent, episode_idx, update_idx, extra=payload_extra)

    def load(self, path: str, agent: HybridPPOAgent, map_location: Optional[str] = None) -> Dict[str, object]:
        checkpoint = torch.load(path, map_location=map_location or "cpu")
        agent.load_checkpoint_state(checkpoint["agent"])
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
        return {
            "episode_idx": int(checkpoint.get("episode_idx", 0)),
            "update_idx": int(checkpoint.get("update_idx", 0)),
            "extra": dict(checkpoint.get("extra", {})),
        }

    def _save(
        self,
        filename: str,
        agent: HybridPPOAgent,
        episode_idx: int,
        update_idx: int,
        extra: Optional[Dict[str, object]] = None,
    ) -> str:
        path = os.path.abspath(os.path.join(self.directory, filename))
        payload = {
            "agent": agent.checkpoint_state(),
            "episode_idx": int(episode_idx),
            "update_idx": int(update_idx),
            "best_metric": float(self.best_metric),
            "config": self.experiment_config.to_dict(),
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
        return path