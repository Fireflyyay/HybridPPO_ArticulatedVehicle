from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .config import HybridPPOConfig, PrimitiveExecutorConfig, VehicleConfig


BASE_OBSERVATION_FEATURE_DIM = 9
GUIDANCE_FEATURE_DIM = 4


@dataclass(frozen=True)
class ObservationConfig:
    lidar_num_beams: int = 108
    lidar_max_range: float = 30.0
    goal_distance_scale: float = 70.0

    @property
    def observation_dim(self) -> int:
        return int(self.lidar_num_beams + BASE_OBSERVATION_FEATURE_DIM + GUIDANCE_FEATURE_DIM)


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 8.0
    distance_weight: float = 2.5
    heading_weight: float = 2.0
    overlap_weight: float = 14.0
    step_penalty: float = -0.05
    success_reward: float = 35.0
    collision_penalty: float = -25.0
    out_of_bounds_penalty: float = -25.0
    timeout_penalty: float = -25.0
    topology_sigma: float = 8.0
    near_goal_radius: float = 10.0
    reverse_penalty_coef: float = -0.2


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
    warmup_corridor_max_width: float = 6.0
    warmup_turn_count_range: Tuple[int, int] = (1, 2)

    def warmup_corridor_min_width(self) -> float:
        return float(min(self.corridor_width_range[0], self.corridor_width_range[1]))

    def resolve_warmup_corridor_width(self, progress: Optional[float] = None) -> float:
        min_width = float(self.warmup_corridor_min_width())
        max_width = max(min_width, float(self.warmup_corridor_max_width))
        if progress is None:
            return min_width
        clamped_progress = float(np.clip(float(progress), 0.0, 1.0))
        return float(max(min_width, max_width - (max_width - min_width) * clamped_progress))


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
            warmup_corridor_max_width=16.0,
            warmup_turn_count_range=(1, 2),
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
    entropy_coef_discrete: float = 0.020
    entropy_coef_continuous: float = 0.008
    max_grad_norm: float = 0.5
    mini_batch_size: int = 1024
    update_epochs: int = 10
    std_floor: float = 0.08
    soft_mask_enabled: bool = False
    soft_mask_gamma: float = 1.0
    soft_mask_eps: float = 1e-4
    soft_mask_logit_scale: float = 0.8
    soft_mask_floor: float = 0.1
    soft_mask_temperature: float = 1.0
    soft_mask_fallback_bonus: float = 1.5
    continuous_safety_temperature: float = 1.0
    safety_loss_coef: float = 0.0
    invalid_loss_coef: float = 0.01

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
            invalid_loss_coef=float(self.invalid_loss_coef),
        )


@dataclass(frozen=True)
class ProxySafetyConfig:
    sidecar_path: str = "data/proxy_safety_sidecar.npz"


@dataclass(frozen=True)
class TrainingScheduleConfig:
    total_episodes: int = 35000
    episodes_per_update: int = 128
    max_macro_steps_per_episode: int = 96
    warmup_level: str = "Warmup"
    target_level: str = "Normal"
    warmup_corridor_convergence_episodes: int = 4000
    curriculum_recent_window: int = 100
    warmup_min_episodes: int = 2500
    warmup_mastery_success_rate: float = 0.25
    target_success_band: Tuple[float, float] = (0.25, 0.60)
    target_focus_prob: float = 0.9
    warmup_bridge_prob: float = 0.5
    adaptive_sampling_enabled: bool = True
    adaptive_sampling_min_uniform: int = 200
    adaptive_sampling_uniform_prob: float = 0.5
    adaptive_sampling_target_success: Dict[str, float] = field(default_factory=lambda: {"Warmup": 0.7, "Normal": 0.5})

    def warmup_progress_for_index(self, warmup_episode_idx: int) -> float:
        index = max(0, int(warmup_episode_idx))
        convergence = max(1, int(self.warmup_corridor_convergence_episodes))
        return float(min(1.0, index / convergence))


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
    best_metric: str = "auto"


@dataclass(frozen=True)
class EvaluationConfig:
    interval: int = 100
    episodes_per_level: int = 10
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
    teacher_enabled: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)