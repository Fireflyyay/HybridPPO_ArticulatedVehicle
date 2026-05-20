from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Tuple

import numpy as np

from .config import HybridPPOConfig, PrimitiveExecutorConfig, VehicleConfig


@dataclass(frozen=True)
class ObservationConfig:
    lidar_num_beams: int = 108
    lidar_max_range: float = 30.0
    goal_distance_scale: float = 70.0

    @property
    def observation_dim(self) -> int:
        return int(self.lidar_num_beams + 9)


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 10.0
    heading_weight: float = 0.5
    overlap_weight: float = 10.0
    step_penalty: float = -0.05
    success_reward: float = 25.0
    collision_penalty: float = -25.0
    out_of_bounds_penalty: float = -25.0
    timeout_penalty: float = -15.0


@dataclass(frozen=True)
class SceneLevelConfig:
    world_min: float = -40.0
    world_max: float = 40.0
    grid_width: int = 80
    grid_height: int = 80
    block_size: float = 1.0
    boundary_margin: float = 8.0
    pair_distance_range: Tuple[float, float] = (12.0, 48.0)
    pair_heading_diff_range_deg: Tuple[float, float] = (0.0, 180.0)
    heading_jitter_rad: float = float(np.deg2rad(15.0))
    corridor_width_range: Tuple[int, int] = (6, 8)
    main_corridor_count: int = 1
    branch_count_range: Tuple[int, int] = (0, 0)
    segment_length_range: Tuple[int, int] = (8, 16)
    turn_probability: float = 0.25
    parking_bay_count_range: Tuple[int, int] = (1, 1)
    parking_bay_length_range: Tuple[int, int] = (6, 8)
    parking_bay_depth_range: Tuple[int, int] = (8, 12)
    parking_head_wall_clearance: float = 1.0


def build_scene_presets() -> Dict[str, SceneLevelConfig]:
    return {
        "Debug": SceneLevelConfig(
            boundary_margin=8.0,
            pair_distance_range=(12.0, 48.0),
            corridor_width_range=(80, 80),
            parking_bay_count_range=(0, 0),
        ),
        "Warmup": SceneLevelConfig(
            boundary_margin=2.0,
            pair_distance_range=(6.0, 30.0),
            pair_heading_diff_range_deg=(0.0, 120.0),
            corridor_width_range=(6, 6),
            branch_count_range=(0, 0),
            parking_bay_count_range=(1, 1),
            parking_bay_length_range=(6, 8),
            parking_bay_depth_range=(12, 14),
            parking_head_wall_clearance=1.0,
        ),
        "Normal": SceneLevelConfig(
            boundary_margin=2.0,
            pair_distance_range=(10.0, 60.0),
            pair_heading_diff_range_deg=(0.0, 180.0),
            corridor_width_range=(8, 8),
            main_corridor_count=3,
            branch_count_range=(3, 5),
            segment_length_range=(8, 16),
            turn_probability=0.28,
            parking_bay_count_range=(2, 3),
            parking_bay_length_range=(7, 8),
            parking_bay_depth_range=(8, 8),
            parking_head_wall_clearance=1.0,
        ),
    }


@dataclass(frozen=True)
class EnvRuntimeConfig:
    default_level: str = "Normal"
    max_low_level_steps_per_episode: int = 500
    scene_presets: Mapping[str, SceneLevelConfig] = field(default_factory=build_scene_presets)


@dataclass(frozen=True)
class HybridPPOHyperConfig:
    hidden_dim: int = 256
    action_embedding_dim: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef_discrete: float = 0.01
    entropy_coef_continuous: float = 0.001
    max_grad_norm: float = 0.5
    mini_batch_size: int = 1024
    update_epochs: int = 10
    std_floor: float = 0.05
    soft_mask_enabled: bool = False
    soft_mask_gamma: float = 1.0
    soft_mask_eps: float = 1e-4
    soft_mask_logit_scale: float = 1.0
    soft_mask_floor: float = 0.2
    soft_mask_temperature: float = 1.0
    soft_mask_fallback_bonus: float = 1.5
    continuous_safety_temperature: float = 1.0
    safety_loss_coef: float = 0.0

    def build(self, observation_dim: int, action_dim: int, parameter_dim: int) -> HybridPPOConfig:
        return HybridPPOConfig(
            observation_dim=int(observation_dim),
            action_dim=int(action_dim),
            parameter_dim=int(parameter_dim),
            hidden_dim=int(self.hidden_dim),
            action_embedding_dim=int(self.action_embedding_dim),
            actor_lr=float(self.actor_lr),
            critic_lr=float(self.critic_lr),
            gamma=float(self.gamma),
            gae_lambda=float(self.gae_lambda),
            clip_epsilon=float(self.clip_epsilon),
            value_coef=float(self.value_coef),
            entropy_coef_discrete=float(self.entropy_coef_discrete),
            entropy_coef_continuous=float(self.entropy_coef_continuous),
            max_grad_norm=float(self.max_grad_norm),
            mini_batch_size=int(self.mini_batch_size),
            update_epochs=int(self.update_epochs),
            std_floor=float(self.std_floor),
            soft_mask_enabled=bool(self.soft_mask_enabled),
            soft_mask_gamma=float(self.soft_mask_gamma),
            soft_mask_eps=float(self.soft_mask_eps),
            soft_mask_logit_scale=float(self.soft_mask_logit_scale),
            soft_mask_floor=float(self.soft_mask_floor),
            soft_mask_temperature=float(self.soft_mask_temperature),
            soft_mask_fallback_bonus=float(self.soft_mask_fallback_bonus),
            continuous_safety_temperature=float(self.continuous_safety_temperature),
            safety_loss_coef=float(self.safety_loss_coef),
        )


@dataclass(frozen=True)
class ProxySafetyConfig:
    sidecar_path: str = ""


@dataclass(frozen=True)
class TrainingScheduleConfig:
    total_episodes: int = 10000
    episodes_per_update: int = 128
    max_macro_steps_per_episode: int = 64
    debug_phase_episodes: int = 1000
    warmup_level: str = "Warmup"
    warmup_episodes: int = 2000
    default_train_level: str = "Normal"
    debug_level: str = "Debug"

    def level_for_episode(self, episode_idx: int) -> str:
        if int(episode_idx) < int(self.debug_phase_episodes):
            return str(self.debug_level)
        if int(episode_idx) < int(self.debug_phase_episodes) + int(self.warmup_episodes):
            return str(self.warmup_level)
        return str(self.default_train_level)


@dataclass(frozen=True)
class LoggingConfig:
    log_root: str = "runs"
    run_name: str = ""
    flush_secs: int = 5


@dataclass(frozen=True)
class CheckpointConfig:
    save_interval: int = 100
    latest_filename: str = "latest.pt"
    best_filename: str = "best.pt"
    best_metric: str = "overall/success_rate"


@dataclass(frozen=True)
class EvaluationConfig:
    interval: int = 100
    episodes_per_level: int = 5
    levels: Tuple[str, ...] = ("Debug", "Warmup", "Normal")
    deterministic: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    primitive_executor: PrimitiveExecutorConfig = field(default_factory=PrimitiveExecutorConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvRuntimeConfig = field(default_factory=EnvRuntimeConfig)
    agent: HybridPPOHyperConfig = field(default_factory=HybridPPOHyperConfig)
    proxy_safety: ProxySafetyConfig = field(default_factory=ProxySafetyConfig)
    schedule: TrainingScheduleConfig = field(default_factory=TrainingScheduleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)