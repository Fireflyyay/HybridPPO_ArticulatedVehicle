from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from common.runtime_config import TrainingScheduleConfig


class SuccessBandCurriculum:
    def __init__(self, schedule: TrainingScheduleConfig, seed: Optional[int] = None) -> None:
        self.schedule = schedule
        self.warmup_level = str(schedule.warmup_level)
        self.target_level = str(schedule.target_level)
        self.window = max(1, int(schedule.curriculum_recent_window))
        self.level_counts = {
            self.warmup_level: 0,
            self.target_level: 0,
        }
        self.success_history = {
            self.warmup_level: deque(maxlen=self.window),
            self.target_level: deque(maxlen=self.window),
        }
        self.rng = np.random.default_rng(seed)
        self.band_unlocked = False

    def _history(self, level: str) -> Deque[int]:
        level_name = str(level)
        if level_name not in self.success_history:
            raise ValueError(f"unsupported curriculum level: {level_name}")
        return self.success_history[level_name]

    def recent_success_rate(self, level: str) -> float:
        history = self._history(level)
        if len(history) == 0:
            return 0.0
        return float(np.mean(np.asarray(list(history), dtype=np.float32)))

    def warmup_progress(self) -> float:
        return float(self.schedule.warmup_progress_for_index(self.level_counts[self.warmup_level]))

    def _warmup_ready_for_band(self) -> bool:
        if int(self.level_counts[self.warmup_level]) < int(self.schedule.warmup_min_episodes):
            return False
        if float(self.warmup_progress()) < 1.0:
            return False
        return float(self.recent_success_rate(self.warmup_level)) >= float(self.schedule.warmup_mastery_success_rate)

    def success_band_active(self) -> bool:
        return bool(self.band_unlocked)

    def choose_level(self) -> str:
        if not self.success_band_active():
            return self.warmup_level

        target_success_rate = float(self.recent_success_rate(self.target_level))
        low, high = self.schedule.target_success_band
        sample = float(self.rng.random())
        if target_success_rate < float(low):
            if sample < float(self.schedule.warmup_bridge_prob):
                return self.warmup_level
            return self.target_level
        if target_success_rate <= float(high):
            if sample < float(self.schedule.target_focus_prob):
                return self.target_level
            return self.warmup_level
        return self.target_level

    def reset_options(self) -> Dict[str, object]:
        level = str(self.choose_level())
        options: Dict[str, object] = {"level": level}
        if level == self.warmup_level:
            warmup_episode_idx = int(self.level_counts[self.warmup_level])
            options["warmup_episode_idx"] = warmup_episode_idx
            options["warmup_progress"] = float(self.schedule.warmup_progress_for_index(warmup_episode_idx))
        return options

    def record_episode(self, level: str, success: bool) -> None:
        level_name = str(level)
        if level_name not in self.level_counts:
            raise ValueError(f"unsupported curriculum level: {level_name}")
        self.level_counts[level_name] += 1
        self.success_history[level_name].append(1 if bool(success) else 0)
        if not self.band_unlocked and self._warmup_ready_for_band():
            self.band_unlocked = True

    def metrics(self) -> Dict[str, float]:
        return {
            "band_active": 1.0 if self.success_band_active() else 0.0,
            "warmup_progress": float(self.warmup_progress()),
            "warmup_success_rate": float(self.recent_success_rate(self.warmup_level)),
            "target_success_rate": float(self.recent_success_rate(self.target_level)),
            "warmup_episode_count": float(self.level_counts[self.warmup_level]),
            "target_episode_count": float(self.level_counts[self.target_level]),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "level_counts": dict(self.level_counts),
            "success_history": {
                level: list(history)
                for level, history in self.success_history.items()
            },
            "band_unlocked": bool(self.band_unlocked),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        level_counts = state.get("level_counts", {})
        for level in (self.warmup_level, self.target_level):
            self.level_counts[level] = int(level_counts.get(level, 0))

        success_history = state.get("success_history", {})
        for level in (self.warmup_level, self.target_level):
            restored = success_history.get(level, [])
            queue: Deque[int] = deque(maxlen=self.window)
            for item in restored[-self.window :]:
                queue.append(1 if int(item) else 0)
            self.success_history[level] = queue

        self.band_unlocked = bool(state.get("band_unlocked", False))

        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state