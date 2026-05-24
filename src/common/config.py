from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class VehicleConfig:
    wheel_base: float = 3.6
    hitch_offset: float = 1.8
    trailer_length: float = 1.8
    step_seconds: float = 0.4
    integrator_substeps: int = 5
    speed_min: float = -2.5
    speed_max: float = 2.5
    articulation_rate_min: float = -0.6632251157578452
    articulation_rate_max: float = 0.6632251157578452
    articulation_limit_rad: float = 0.6283185307179586
    front_length: float = 5.0
    rear_length: float = 4.4
    body_width: float = 3.43


@dataclass(frozen=True)
class PrimitiveExecutorConfig:
    nominal_speed: float = 1.0
    low_speed: float = 0.45
    max_macro_steps: int = 32
    articulation_guard_margin_rad: float = 0.07
    heading_controller_gain: float = 1.2
    articulation_controller_gain: float = 1.5
    default_heading_tolerance_rad: float = 0.12
    default_position_tolerance_m: float = 0.35
    min_speed_scale: float = 0.15
    max_speed_scale: float = 1.0


@dataclass(frozen=True)
class HybridPPOConfig:
    observation_dim: int
    action_dim: int
    parameter_dim: int
    hidden_dim: int = 256
    action_embedding_dim: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef_discrete: float = 0.01
    entropy_coef_continuous: float = 0.001
    max_grad_norm: float = 0.5
    mini_batch_size: int = 64
    update_epochs: int = 10
    std_floor: float = 0.05
    soft_mask_enabled: bool = True
    soft_mask_gamma: float = 1.0
    soft_mask_eps: float = 1e-4
    soft_mask_logit_scale: float = 1.0
    soft_mask_floor: float = 0.2
    soft_mask_temperature: float = 1.0
    soft_mask_fallback_bonus: float = 1.5
    continuous_safety_temperature: float = 1.0
    safety_loss_coef: float = 0.0


@dataclass(frozen=True)
class SuccessCriteriaConfig:
    heading_threshold_rad: float = 0.2617993877991494
    front_overlap_threshold: float = 0.7
    require_collision_free: bool = True


@dataclass(frozen=True)
class ParameterBoundsConfig:
    bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "path_length": (0.3, 8.0),
            "duration": (0.2, 3.2),
            "speed_scale": (0.15, 1.0),
            "omega_scale": (0.05, 1.0),
            "phi_target": (-0.55, 0.55),
            "heading_tolerance": (0.03, 0.35),
            "position_tolerance": (0.08, 0.8),
            "smoothness": (0.0, 0.95),
        }
    )
