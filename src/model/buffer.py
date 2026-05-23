from dataclasses import dataclass
from typing import List

import numpy as np

from common.types import MacroTransition


@dataclass(frozen=True)
class SMDPBatch:
    observations: np.ndarray
    action_ids: np.ndarray
    parameters: np.ndarray
    rewards: np.ndarray
    taus: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    teacher_action_probs: np.ndarray
    teacher_parameter_targets: np.ndarray
    teacher_weights: np.ndarray


class SMDPRolloutBuffer:
    def __init__(self, action_dim: int, parameter_dim: int) -> None:
        self._items: List[MacroTransition] = []
        self._action_dim = int(action_dim)
        self._parameter_dim = int(parameter_dim)

    def add(self, transition: MacroTransition) -> None:
        self._items.append(transition)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return int(len(self._items))

    def as_batch(self) -> SMDPBatch:
        if len(self._items) == 0:
            raise ValueError("buffer is empty")
        return SMDPBatch(
            observations=np.stack([item.observation for item in self._items], axis=0).astype(np.float32),
            action_ids=np.asarray([item.action_id for item in self._items], dtype=np.int64),
            parameters=np.stack([item.parameters for item in self._items], axis=0).astype(np.float32),
            rewards=np.asarray([item.reward for item in self._items], dtype=np.float32),
            taus=np.asarray([item.tau for item in self._items], dtype=np.int64),
            next_observations=np.stack([item.next_observation for item in self._items], axis=0).astype(np.float32),
            dones=np.asarray([item.done for item in self._items], dtype=np.float32),
            log_probs=np.asarray([item.log_prob for item in self._items], dtype=np.float32),
            values=np.asarray([item.value for item in self._items], dtype=np.float32),
            teacher_action_probs=np.stack(
                [
                    np.zeros((self._action_dim,), dtype=np.float32)
                    if item.teacher_action_probs is None
                    else np.asarray(item.teacher_action_probs, dtype=np.float32).reshape(self._action_dim)
                    for item in self._items
                ],
                axis=0,
            ).astype(np.float32),
            teacher_parameter_targets=np.stack(
                [
                    np.zeros((self._parameter_dim,), dtype=np.float32)
                    if item.teacher_parameter_target is None
                    else np.asarray(item.teacher_parameter_target, dtype=np.float32).reshape(self._parameter_dim)
                    for item in self._items
                ],
                axis=0,
            ).astype(np.float32),
            teacher_weights=np.asarray([item.teacher_weight for item in self._items], dtype=np.float32),
        )
