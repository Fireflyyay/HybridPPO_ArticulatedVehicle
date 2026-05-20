import os
import pprint
import time
from collections.abc import Mapping
from numbers import Number
from typing import Dict

from torch.utils.tensorboard import SummaryWriter

from common.runtime_config import ExperimentConfig, LoggingConfig


def _flatten_scalars(prefix: str, value, target: Dict[str, float]) -> None:
    if isinstance(value, Number):
        target[prefix] = float(value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_prefix = str(key) if prefix == "" else f"{prefix}/{key}"
            _flatten_scalars(child_prefix, item, target)


class TensorBoardLogger:
    def __init__(self, config: LoggingConfig) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_name = str(config.run_name).strip() or f"hybridppo_{timestamp}"
        self.run_dir = os.path.abspath(os.path.join(str(config.log_root), run_name))
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir, flush_secs=int(config.flush_secs))

    def log_config(self, config: ExperimentConfig) -> None:
        self.writer.add_text("config", pprint.pformat(config.to_dict(), sort_dicts=False), 0)

    def log_training_episode(self, episode_idx: int, metrics: Dict[str, float]) -> None:
        flat: Dict[str, float] = {}
        _flatten_scalars("train", metrics, flat)
        for key, value in flat.items():
            self.writer.add_scalar(key, float(value), int(episode_idx))

    def log_update(self, episode_idx: int, metrics: Dict[str, float]) -> None:
        flat: Dict[str, float] = {}
        _flatten_scalars("update", metrics, flat)
        for key, value in flat.items():
            self.writer.add_scalar(key, float(value), int(episode_idx))

    def log_evaluation(self, episode_idx: int, metrics: Dict[str, object]) -> None:
        flat: Dict[str, float] = {}
        _flatten_scalars("eval", metrics, flat)
        for key, value in flat.items():
            self.writer.add_scalar(key, float(value), int(episode_idx))

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()