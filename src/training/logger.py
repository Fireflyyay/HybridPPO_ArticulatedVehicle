import os
import json
import pprint
import time
from collections.abc import Mapping
from numbers import Number
from typing import Dict, List, Sequence, Tuple

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
    _CONFIG_JSON_NAME = "config.json"
    _CONFIG_TEXT_NAME = "config.txt"
    _REWARD_SVG_NAME = "reward_curve.svg"

    def __init__(self, config: LoggingConfig) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_name = str(config.run_name).strip() or f"hybridppo_{timestamp}"
        self.run_dir = os.path.abspath(os.path.join(str(config.log_root), run_name))
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir, flush_secs=int(config.flush_secs))
        self._reward_history: List[Tuple[int, float]] = []

    def log_config(self, config: ExperimentConfig) -> None:
        config_dict = config.to_dict()
        config_text = pprint.pformat(config_dict, sort_dicts=False)
        self.writer.add_text("config", config_text, 0)
        self._write_text_file(self._CONFIG_TEXT_NAME, config_text + "\n")
        self._write_text_file(
            self._CONFIG_JSON_NAME,
            json.dumps(config_dict, ensure_ascii=False, indent=2) + "\n",
        )

    def log_training_episode(self, episode_idx: int, metrics: Dict[str, float]) -> None:
        flat: Dict[str, float] = {}
        _flatten_scalars("train", metrics, flat)
        for key, value in flat.items():
            self.writer.add_scalar(key, float(value), int(episode_idx))
        if "total_reward" in metrics:
            self._record_total_reward(int(episode_idx), float(metrics["total_reward"]))
            self._write_reward_curve()

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

    def _record_total_reward(self, episode_idx: int, reward: float) -> None:
        if self._reward_history and int(self._reward_history[-1][0]) == int(episode_idx):
            self._reward_history[-1] = (int(episode_idx), float(reward))
            return
        self._reward_history.append((int(episode_idx), float(reward)))

    def _write_text_file(self, filename: str, content: str) -> None:
        path = os.path.join(self.run_dir, filename)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)

    def _write_reward_curve(self) -> None:
        if len(self._reward_history) == 0:
            return

        canvas_width = 960.0
        canvas_height = 540.0
        margin_left = 72.0
        margin_right = 24.0
        margin_top = 24.0
        margin_bottom = 54.0
        plot_width = canvas_width - margin_left - margin_right
        plot_height = canvas_height - margin_top - margin_bottom

        raw_points = self._downsample_series(self._reward_history, max_points=int(plot_width))
        smooth_points = self._downsample_series(self._moving_average_points(window=100), max_points=int(plot_width))
        reward_values = [point[1] for point in raw_points]
        reward_values.extend(point[1] for point in smooth_points)
        min_reward = min(reward_values)
        max_reward = max(reward_values)
        if abs(max_reward - min_reward) < 1e-6:
            min_reward -= 1.0
            max_reward += 1.0
        reward_pad = 0.08 * (max_reward - min_reward)
        min_reward -= reward_pad
        max_reward += reward_pad

        min_episode = float(raw_points[0][0])
        max_episode = float(raw_points[-1][0])
        if max_episode <= min_episode:
            max_episode = min_episode + 1.0

        grid_lines = []
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = margin_top + (1.0 - ratio) * plot_height
            reward_value = min_reward + ratio * (max_reward - min_reward)
            grid_lines.append(
                f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{margin_left + plot_width:.1f}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1" />'
            )
            grid_lines.append(
                f'<text x="{margin_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{reward_value:.1f}</text>'
            )

        raw_polyline = self._polyline_points(
            raw_points,
            min_episode=min_episode,
            max_episode=max_episode,
            min_reward=min_reward,
            max_reward=max_reward,
            left=margin_left,
            top=margin_top,
            width=plot_width,
            height=plot_height,
        )
        smooth_polyline = self._polyline_points(
            smooth_points,
            min_episode=min_episode,
            max_episode=max_episode,
            min_reward=min_reward,
            max_reward=max_reward,
            left=margin_left,
            top=margin_top,
            width=plot_width,
            height=plot_height,
        )
        latest_episode, latest_reward = self._reward_history[-1]
        latest_avg = self._moving_average_points(window=100)[-1][1]

        svg = "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(canvas_width)}" height="{int(canvas_height)}" viewBox="0 0 {int(canvas_width)} {int(canvas_height)}">',
                '<rect width="100%" height="100%" fill="#ffffff" />',
                '<text x="24" y="34" font-size="20" font-weight="600" fill="#0f172a">Training Reward</text>',
                f'<text x="24" y="56" font-size="12" fill="#475569">episode={latest_episode} latest={latest_reward:.2f} avg100={latest_avg:.2f}</text>',
                *grid_lines,
                f'<line x1="{margin_left:.1f}" y1="{margin_top:.1f}" x2="{margin_left:.1f}" y2="{margin_top + plot_height:.1f}" stroke="#94a3b8" stroke-width="1.5" />',
                f'<line x1="{margin_left:.1f}" y1="{margin_top + plot_height:.1f}" x2="{margin_left + plot_width:.1f}" y2="{margin_top + plot_height:.1f}" stroke="#94a3b8" stroke-width="1.5" />',
                f'<polyline fill="none" stroke="#94a3b8" stroke-width="1.5" points="{raw_polyline}" />',
                f'<polyline fill="none" stroke="#ef4444" stroke-width="2.5" points="{smooth_polyline}" />',
                f'<text x="{margin_left + plot_width:.1f}" y="{margin_top + plot_height + 32:.1f}" text-anchor="end" font-size="12" fill="#475569">Episode</text>',
                '<text x="24" y="90" font-size="12" fill="#94a3b8">raw reward</text>',
                '<text x="120" y="90" font-size="12" fill="#ef4444">moving avg (100)</text>',
                '</svg>',
            ]
        )
        self._write_text_file(self._REWARD_SVG_NAME, svg + "\n")

    def _moving_average_points(self, window: int) -> List[Tuple[int, float]]:
        values: List[Tuple[int, float]] = []
        running_sum = 0.0
        rewards: List[float] = []
        max_window = max(1, int(window))
        for episode_idx, reward in self._reward_history:
            rewards.append(float(reward))
            running_sum += float(reward)
            if len(rewards) > max_window:
                running_sum -= rewards[-max_window - 1]
            count = min(len(rewards), max_window)
            values.append((int(episode_idx), running_sum / float(count)))
        return values

    def _downsample_series(self, points: Sequence[Tuple[int, float]], max_points: int) -> List[Tuple[int, float]]:
        if len(points) <= max_points:
            return [(int(episode_idx), float(value)) for episode_idx, value in points]

        sampled: List[Tuple[int, float]] = []
        bucket_width = float(len(points)) / float(max_points)
        for bucket_idx in range(max_points):
            start = int(bucket_idx * bucket_width)
            stop = max(start + 1, int((bucket_idx + 1) * bucket_width))
            chunk = points[start:stop]
            if len(chunk) == 0:
                continue
            mean_episode = int(round(sum(point[0] for point in chunk) / float(len(chunk))))
            mean_value = sum(point[1] for point in chunk) / float(len(chunk))
            sampled.append((mean_episode, mean_value))
        if sampled[-1][0] != int(points[-1][0]):
            sampled[-1] = (int(points[-1][0]), float(points[-1][1]))
        return sampled

    def _polyline_points(
        self,
        points: Sequence[Tuple[int, float]],
        min_episode: float,
        max_episode: float,
        min_reward: float,
        max_reward: float,
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> str:
        coords = []
        reward_span = max(max_reward - min_reward, 1e-6)
        episode_span = max(max_episode - min_episode, 1e-6)
        for episode_idx, reward in points:
            x = left + (float(episode_idx) - min_episode) / episode_span * width
            y = top + (max_reward - float(reward)) / reward_span * height
            coords.append(f"{x:.2f},{y:.2f}")
        return " ".join(coords)