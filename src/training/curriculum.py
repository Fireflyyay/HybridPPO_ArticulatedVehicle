from collections import deque
from typing import Deque, Dict, Optional, Tuple

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


class DifficultyAdaptiveSampler:
    BUCKETS: Dict[str, Tuple[float, float]] = {
        "forward-approach": (0.0, 45.0),
        "side-approach": (45.0, 135.0),
        "reverse-approach": (135.0, 180.0),
    }

    def __init__(self, schedule: TrainingScheduleConfig, seed: Optional[int] = None) -> None:
        self._buckets = dict(self.BUCKETS)
        self._min_uniform = int(schedule.adaptive_sampling_min_uniform)
        self._uniform_prob = float(schedule.adaptive_sampling_uniform_prob)
        self._target_success = dict(schedule.adaptive_sampling_target_success)
        self._window = max(1, int(schedule.curriculum_recent_window))
        self._rng = np.random.default_rng(seed)
        self._history: Dict[str, Dict[str, Deque[int]]] = {}
        self._counts: Dict[str, Dict[str, int]] = {}

    def _ensure_level(self, level: str) -> None:
        key = str(level)
        if key not in self._history:
            self._history[key] = {k: deque(maxlen=self._window) for k in self._buckets}
            self._counts[key] = {k: 0 for k in self._buckets}

    def total_episodes(self, level: str) -> int:
        self._ensure_level(level)
        return sum(self._counts[str(level)].values())

    def bucket_success_rate(self, level: str, bucket: str) -> float:
        self._ensure_level(level)
        h = self._history[str(level)].get(str(bucket))
        if h is None or len(h) == 0:
            return 0.0
        return float(np.mean(np.asarray(list(h), dtype=np.float32)))

    def choose_bucket(self, level: str) -> str:
        self._ensure_level(level)
        total = self.total_episodes(level)
        if total < self._min_uniform:
            return self._uniform_choice()
        if self._rng.random() < self._uniform_prob:
            return self._uniform_choice()
        return self._worst_bucket(level)

    def _uniform_choice(self) -> str:
        return str(self._rng.choice(list(self._buckets.keys())))

    def _worst_bucket(self, level: str) -> str:
        target = float(self._target_success.get(str(level), 0.6))
        fail_rates = {
            k: target - self.bucket_success_rate(level, k)
            for k in self._buckets
        }
        return max(fail_rates, key=fail_rates.get)

    def record_episode(self, level: str, bucket: str, success: bool) -> None:
        self._ensure_level(level)
        bk = str(bucket)
        if bk not in self._buckets:
            bk = next(iter(self._buckets))
        self._counts[str(level)][bk] = int(self._counts[str(level)].get(bk, 0)) + 1
        self._history[str(level)][bk].append(1 if bool(success) else 0)

    @classmethod
    def compute_bucket(cls, heading_diff_deg: float) -> str:
        ad = abs(float(heading_diff_deg)) % 360.0
        if ad > 180.0:
            ad = 360.0 - ad
        ad = min(ad, 179.999)
        for name, (lo, hi) in cls.BUCKETS.items():
            if float(lo) <= ad < float(hi):
                return name
        return "reverse-approach"

    @classmethod
    def bucket_heading_range(cls, bucket: str) -> Tuple[float, float]:
        return cls.BUCKETS.get(str(bucket), (0.0, 180.0))

    def metrics(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for level in sorted(self._history.keys()):
            for bucket in sorted(self._buckets.keys()):
                prefix = f"{level}/{bucket}"
                result[f"{prefix}/success_rate"] = self.bucket_success_rate(level, bucket)
                result[f"{prefix}/episode_count"] = float(self._counts.get(level, {}).get(bucket, 0))
        return result

    def state_dict(self) -> Dict[str, object]:
        return {
            "history": {lvl: {bk: list(dq) for bk, dq in buckets.items()} for lvl, buckets in self._history.items()},
            "counts": {lvl: dict(cnt) for lvl, cnt in self._counts.items()},
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        history_data: Dict[str, Dict[str, list]] = state.get("history", {})
        counts_data: Dict[str, Dict[str, int]] = state.get("counts", {})
        for level, bucket_data in history_data.items():
            self._ensure_level(str(level))
            lvl = str(level)
            for bucket, data in bucket_data.items():
                bk = str(bucket)
                if bk not in self._buckets:
                    bk = next(iter(self._buckets))
                dq: Deque[int] = deque(maxlen=self._window)
                for item in data[-self._window :]:
                    dq.append(1 if int(item) else 0)
                self._history[lvl][bk] = dq
        for level, bucket_counts in counts_data.items():
            self._ensure_level(str(level))
            lvl = str(level)
            for bucket, cnt in bucket_counts.items():
                bk = str(bucket)
                if bk not in self._buckets:
                    bk = next(iter(self._buckets))
                self._counts[lvl][bk] = int(cnt)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state