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


class SMDPRolloutBuffer:
    def __init__(self) -> None:
        self._items: List[MacroTransition] = []

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
        )
