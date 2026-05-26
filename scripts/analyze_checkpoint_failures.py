import argparse
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from common.config import PrimitiveExecutorConfig, VehicleConfig
from common.runtime_config import (
    CheckpointConfig,
    EnvRuntimeConfig,
    EvaluationConfig,
    ExperimentConfig,
    HybridPPOHyperConfig,
    LoggingConfig,
    ObservationConfig,
    ProxySafetyConfig,
    RewardConfig,
    SceneLevelConfig,
    TrainingScheduleConfig,
)
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library, load_proxy_safety_sidecar
from training.rollout import MacroRolloutDriver


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze checkpoint failures by replaying sampled scenes.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing config/events/checkpoints.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to <run-dir>/checkpoints/best.pt.")
    parser.add_argument("--event-file", type=str, default=None, help="Optional tensorboard event file path.")
    parser.add_argument("--debug-episodes", type=int, default=60)
    parser.add_argument("--warmup-episodes", type=int, default=80)
    parser.add_argument("--normal-episodes", type=int, default=160)
    parser.add_argument("--warmup-progress", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seed-offset", type=int, default=10000)
    parser.add_argument("--max-failure-examples", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy instead of deterministic evaluation.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the JSON report.")
    parser.add_argument("--disable-soft-mask", action="store_true", help="Disable soft mask during analysis replay.")
    parser.add_argument("--soft-mask-logit-scale", type=float, default=None, help="Override soft mask logit scale.")
    parser.add_argument("--soft-mask-floor", type=float, default=None, help="Override soft mask floor.")
    parser.add_argument("--soft-mask-gamma", type=float, default=None, help="Override soft mask proxy gamma.")
    parser.add_argument("--soft-mask-fallback-bonus", type=float, default=None, help="Override soft mask fallback bias bonus.")
    parser.add_argument("--proxy-sidecar", type=str, default=None, help="Override proxy safety sidecar path.")
    parser.add_argument("--disable-proxy-sidecar", action="store_true", help="Disable proxy safety sidecar loading.")
    return parser


def _build_config(stored: Dict[str, Any]) -> ExperimentConfig:
    scene_presets = {
        str(name): SceneLevelConfig(**dict(values))
        for name, values in dict(stored["env"]["scene_presets"]).items()
    }
    return ExperimentConfig(
        seed=int(stored["seed"]),
        device=str(stored["device"]),
        vehicle=VehicleConfig(**dict(stored["vehicle"])),
        primitive_executor=PrimitiveExecutorConfig(**dict(stored["primitive_executor"])),
        observation=ObservationConfig(**dict(stored["observation"])),
        reward=RewardConfig(**dict(stored["reward"])),
        env=EnvRuntimeConfig(
            default_level=str(stored["env"]["default_level"]),
            max_low_level_steps_per_episode=int(stored["env"]["max_low_level_steps_per_episode"]),
            scene_presets=scene_presets,
        ),
        agent=HybridPPOHyperConfig(**dict(stored["agent"])),
        proxy_safety=ProxySafetyConfig(**dict(stored["proxy_safety"])),
        schedule=TrainingScheduleConfig(**dict(stored["schedule"])),
        logging=LoggingConfig(**dict(stored["logging"])),
        checkpoint=CheckpointConfig(**dict(stored["checkpoint"])),
        evaluation=EvaluationConfig(**dict(stored["evaluation"])),
        teacher_enabled=bool(stored.get("teacher_enabled", False)),
    )


def _compact_evaluation(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, Any] = {}
    for key in ("overall", "levels", "checkpoint"):
        if key in value:
            result[key] = value[key]
    return result


def _scalar_snapshot(scalars: Sequence[Any]) -> Dict[str, float]:
    values = [item.value for item in scalars]
    steps = [item.step for item in scalars]
    best_index = int(np.argmax(values))
    return {
        "last_step": int(steps[-1]),
        "last_value": float(values[-1]),
        "best_step": int(steps[best_index]),
        "best_value": float(values[best_index]),
    }


def _event_summary(event_path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if event_path is None or not event_path.is_file():
        return {}
    accumulator = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    interesting_tags = (
        "train/total_reward",
        "train/success",
        "train/curriculum/warmup_progress",
        "train/curriculum/warmup_success_rate",
        "train/curriculum/target_success_rate",
        "eval/overall/success_rate",
        "eval/levels/Debug/success_rate",
        "eval/levels/Warmup/success_rate",
        "eval/levels/Normal/success_rate",
        "eval/levels/Normal/mean_goal_distance",
        "eval/checkpoint/metric_value",
    )
    return {
        tag: _scalar_snapshot(accumulator.Scalars(tag))
        for tag in interesting_tags
        if tag in scalar_tags
    }


def _resolve_event_path(run_dir: Optional[Path], event_file: Optional[str]) -> Optional[Path]:
    if event_file:
        path = Path(event_file).expanduser().resolve()
        return path if path.is_file() else None
    if run_dir is None or not run_dir.is_dir():
        return None
    matches = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _resolve_checkpoint_path(run_dir: Optional[Path], checkpoint: Optional[str]) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    if run_dir is None:
        raise ValueError("either --run-dir or --checkpoint must be provided")
    path = (run_dir / "checkpoints" / "best.pt").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _resolve_run_dir(run_dir: Optional[str], checkpoint_path: Path) -> Path:
    if run_dir:
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"run directory not found: {path}")
        return path
    return checkpoint_path.parent.parent.resolve()


def _resolve_proxy_sidecar(config: ExperimentConfig, repo_root: Path):
    configured = Path(str(config.proxy_safety.sidecar_path)).expanduser()
    if configured.is_file():
        return load_proxy_safety_sidecar(str(configured))
    repo_local = repo_root / "data" / "proxy_safety_sidecar.npz"
    if repo_local.is_file():
        return load_proxy_safety_sidecar(str(repo_local))
    return None


def _apply_config_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    agent = config.agent
    if args.disable_soft_mask:
        agent = replace(agent, soft_mask_enabled=False)
    if args.soft_mask_logit_scale is not None:
        agent = replace(agent, soft_mask_logit_scale=float(args.soft_mask_logit_scale))
    if args.soft_mask_floor is not None:
        agent = replace(agent, soft_mask_floor=float(args.soft_mask_floor))
    if args.soft_mask_gamma is not None:
        agent = replace(agent, soft_mask_gamma=float(args.soft_mask_gamma))
    if args.soft_mask_fallback_bonus is not None:
        agent = replace(agent, soft_mask_fallback_bonus=float(args.soft_mask_fallback_bonus))

    proxy_safety = config.proxy_safety
    if args.disable_proxy_sidecar:
        proxy_safety = replace(proxy_safety, sidecar_path="")
    elif args.proxy_sidecar is not None:
        proxy_safety = replace(proxy_safety, sidecar_path=str(Path(args.proxy_sidecar).expanduser().resolve()))

    return replace(config, agent=agent, proxy_safety=proxy_safety)


def _torch_load_checkpoint(path: Path) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.load(str(path), map_location="cpu")


def _load_agent(payload: Dict[str, Any], config: ExperimentConfig, repo_root: Path) -> Tuple[HybridPPOAgent, Any]:
    proxy_sidecar = _resolve_proxy_sidecar(config, repo_root)
    primitive_library = build_default_primitive_library(proxy_sidecar=proxy_sidecar)
    agent = HybridPPOAgent(
        config=config.agent.build(
            observation_dim=int(config.observation.observation_dim),
            action_dim=int(primitive_library.action_dim),
            parameter_dim=int(primitive_library.parameter_dim),
        ),
        primitive_library=primitive_library,
        device="cpu",
    )
    agent.load_checkpoint_state(payload["agent"])
    agent.policy.eval()
    agent.value_net.eval()
    return agent, primitive_library


def _proxy_sidecar_summary(primitive_library: Any) -> Dict[str, object]:
    sidecar = getattr(primitive_library, "proxy_sidecar", None)
    if sidecar is None:
        return {"status": "disabled"}
    metadata = dict(getattr(sidecar, "metadata", {}) or {})
    return {
        "status": "loaded",
        "path_metadata": metadata,
        "num_actions": int(sidecar.num_actions),
        "num_proxies": int(sidecar.num_proxies),
        "num_steps": int(sidecar.num_steps),
        "num_rays": int(sidecar.num_rays),
        "articulation_bin_count": int(sidecar.num_articulation_bins),
        "parameter_dim": int(sidecar.parameter_dim),
        "lidar_range": float(sidecar.lidar_range),
        "proxy_resolution": metadata.get("proxy_resolution"),
        "max_proxies_per_action": metadata.get("max_proxies_per_action"),
    }


def _hard_examples(failures: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        failures,
        key=lambda item: (
            item["done_reason"] != "timeout",
            -float(item["final_goal_distance"]),
            -float(item["heading_error_rad"]),
        ),
    )
    return ordered[: max(0, int(limit))]


def _scene_buckets(failures: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], int] = defaultdict(int)
    for item in failures:
        key = (
            item.get("scene_type"),
            item.get("corridor_width"),
            item.get("turn_count"),
            item.get("free_shape_count"),
            item.get("valid_candidate_count"),
        )
        buckets[key] += 1
    ranked = sorted(buckets.items(), key=lambda entry: (-entry[1], str(entry[0])))
    return [
        {
            "scene_type": key[0],
            "corridor_width": key[1],
            "turn_count": key[2],
            "free_shape_count": key[3],
            "valid_candidate_count": key[4],
            "count": int(count),
        }
        for key, count in ranked[: max(0, int(limit))]
    ]


def _sorted_counter(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda entry: (-entry[1], str(entry[0])))}


def _episode_action_summary(step_records: List[Dict[str, object]]) -> Dict[str, object]:
    primitive_counts: Counter = Counter()
    semantic_counts: Counter = Counter()
    fallback_trigger_count = 0
    fallback_selected_count = 0
    selected_matches_raw_argmax_count = 0
    selected_matches_fallback_argmax_count = 0
    mask_degenerate_count = 0
    selection_probs: List[float] = []
    selection_masks: List[float] = []
    semantic_scores: List[float] = []
    proxy_scores: List[float] = []
    prefix_lengths: List[float] = []
    soft_scores: List[float] = []
    hard_valid_ratios: List[float] = []
    all_invalid_fallback_count = 0
    selected_action_hard_valid_count = 0

    for step in step_records:
        primitive_key = f"{int(step['primitive_id'])}:{str(step['semantic'])}"
        primitive_counts[primitive_key] += 1
        semantic_counts[str(step["semantic"])] += 1
        fallback_trigger_count += int(bool(step.get("fallback_triggered", False)))
        fallback_selected_count += int(bool(step.get("selected_matches_fallback_argmax", False)) and bool(step.get("fallback_triggered", False)))
        selected_matches_raw_argmax_count += int(bool(step.get("selected_matches_raw_argmax", False)))
        selected_matches_fallback_argmax_count += int(bool(step.get("selected_matches_fallback_argmax", False)))
        mask_degenerate_count += int(int(step.get("selection_mask_positive_count", 0)) == 0)
        all_invalid_fallback_count += int(bool(step.get("all_invalid_fallback", False)))
        selected_action_hard_valid_count += int(bool(step.get("selected_action_hard_valid", False)))
        selection_probs.append(float(step.get("selected_prob", 0.0)))
        if step.get("selected_selection_mask") is not None:
            selection_masks.append(float(step["selected_selection_mask"]))
        if step.get("selected_semantic_score") is not None:
            semantic_scores.append(float(step["selected_semantic_score"]))
        if step.get("selected_soft_score") is not None:
            soft_scores.append(float(step["selected_soft_score"]))
        if step.get("hard_valid_ratio") is not None:
            hard_valid_ratios.append(float(step["hard_valid_ratio"]))
        if step.get("selected_proxy_score_max") is not None:
            proxy_scores.append(float(step["selected_proxy_score_max"]))
        if step.get("selected_proxy_prefix_max") is not None:
            prefix_lengths.append(float(step["selected_proxy_prefix_max"]))

    steps = len(step_records)
    dominant_semantic = None
    if semantic_counts:
        dominant_semantic = semantic_counts.most_common(1)[0][0]

    return {
        "steps": int(steps),
        "dominant_semantic": dominant_semantic,
        "primitive_counts": _sorted_counter(primitive_counts),
        "semantic_counts": _sorted_counter(semantic_counts),
        "fallback_trigger_count": int(fallback_trigger_count),
        "fallback_selected_count": int(fallback_selected_count),
        "selected_matches_raw_argmax_count": int(selected_matches_raw_argmax_count),
        "selected_matches_fallback_argmax_count": int(selected_matches_fallback_argmax_count),
        "mask_degenerate_count": int(mask_degenerate_count),
        "all_invalid_fallback_count": int(all_invalid_fallback_count),
        "selected_action_hard_valid_count": int(selected_action_hard_valid_count),
        "selected_prob_mean": None if not selection_probs else float(mean(selection_probs)),
        "selected_selection_mask_mean": None if not selection_masks else float(mean(selection_masks)),
        "selected_selection_mask_min": None if not selection_masks else float(min(selection_masks)),
        "selected_semantic_score_mean": None if not semantic_scores else float(mean(semantic_scores)),
        "selected_semantic_score_min": None if not semantic_scores else float(min(semantic_scores)),
        "selected_soft_score_mean": None if not soft_scores else float(mean(soft_scores)),
        "selected_soft_score_min": None if not soft_scores else float(min(soft_scores)),
        "hard_valid_ratio_mean": None if not hard_valid_ratios else float(mean(hard_valid_ratios)),
        "hard_valid_ratio_min": None if not hard_valid_ratios else float(min(hard_valid_ratios)),
        "selected_proxy_score_mean": None if not proxy_scores else float(mean(proxy_scores)),
        "selected_proxy_score_min": None if not proxy_scores else float(min(proxy_scores)),
        "selected_proxy_prefix_mean": None if not prefix_lengths else float(mean(prefix_lengths)),
    }


def _aggregate_action_summaries(rows: List[Dict[str, Any]]) -> Dict[str, object]:
    primitive_counts: Counter = Counter()
    semantic_counts: Counter = Counter()
    steps = 0
    fallback_trigger_count = 0
    fallback_selected_count = 0
    selected_matches_raw_argmax_count = 0
    selected_matches_fallback_argmax_count = 0
    mask_degenerate_count = 0
    all_invalid_fallback_count = 0
    selected_action_hard_valid_count = 0
    selection_probs: List[float] = []
    selection_masks: List[float] = []
    semantic_scores: List[float] = []
    soft_scores: List[float] = []
    hard_valid_ratios: List[float] = []
    proxy_scores: List[float] = []
    prefix_lengths: List[float] = []

    for row in rows:
        summary = dict(row.get("action_summary", {}))
        row_steps = int(summary.get("steps", 0))
        steps += row_steps
        primitive_counts.update({str(key): int(value) for key, value in dict(summary.get("primitive_counts", {})).items()})
        semantic_counts.update({str(key): int(value) for key, value in dict(summary.get("semantic_counts", {})).items()})
        fallback_trigger_count += int(summary.get("fallback_trigger_count", 0))
        fallback_selected_count += int(summary.get("fallback_selected_count", 0))
        selected_matches_raw_argmax_count += int(summary.get("selected_matches_raw_argmax_count", 0))
        selected_matches_fallback_argmax_count += int(summary.get("selected_matches_fallback_argmax_count", 0))
        mask_degenerate_count += int(summary.get("mask_degenerate_count", 0))
        all_invalid_fallback_count += int(summary.get("all_invalid_fallback_count", 0))
        selected_action_hard_valid_count += int(summary.get("selected_action_hard_valid_count", 0))
        if summary.get("selected_prob_mean") is not None and row_steps > 0:
            selection_probs.extend([float(summary["selected_prob_mean"])] * row_steps)
        if summary.get("selected_selection_mask_mean") is not None and row_steps > 0:
            selection_masks.extend([float(summary["selected_selection_mask_mean"])] * row_steps)
        if summary.get("selected_semantic_score_mean") is not None and row_steps > 0:
            semantic_scores.extend([float(summary["selected_semantic_score_mean"])] * row_steps)
        if summary.get("selected_soft_score_mean") is not None and row_steps > 0:
            soft_scores.extend([float(summary["selected_soft_score_mean"])] * row_steps)
        if summary.get("hard_valid_ratio_mean") is not None and row_steps > 0:
            hard_valid_ratios.extend([float(summary["hard_valid_ratio_mean"])] * row_steps)
        if summary.get("selected_proxy_score_mean") is not None and row_steps > 0:
            proxy_scores.extend([float(summary["selected_proxy_score_mean"])] * row_steps)
        if summary.get("selected_proxy_prefix_mean") is not None and row_steps > 0:
            prefix_lengths.extend([float(summary["selected_proxy_prefix_mean"])] * row_steps)

    dominant_semantic = None
    if semantic_counts:
        dominant_semantic = semantic_counts.most_common(1)[0][0]

    if steps == 0:
        return {
            "episodes": int(len(rows)),
            "steps": 0,
            "dominant_semantic": None,
            "primitive_counts": {},
            "semantic_counts": {},
            "fallback_trigger_rate": 0.0,
            "fallback_selected_rate_overall": 0.0,
            "fallback_selected_rate_when_triggered": 0.0,
            "selected_matches_raw_argmax_rate": 0.0,
            "selected_matches_fallback_argmax_rate": 0.0,
            "mask_degenerate_rate": 0.0,
            "all_invalid_fallback_rate": 0.0,
            "selected_action_hard_valid_rate": 0.0,
            "selected_stop_check_rate": 0.0,
            "selected_articulation_recover_rate": 0.0,
            "selected_straight_adjust_rate": 0.0,
            "selected_selection_mask_mean": None,
            "selected_selection_mask_min": None,
            "selected_semantic_score_mean": None,
            "selected_semantic_score_min": None,
            "selected_soft_score_mean": None,
            "selected_soft_score_min": None,
            "hard_valid_ratio_mean": None,
            "hard_valid_ratio_min": None,
            "selected_proxy_score_mean": None,
            "selected_proxy_score_min": None,
            "selected_proxy_prefix_mean": None,
            "selected_prob_mean": None,
        }

    return {
        "episodes": int(len(rows)),
        "steps": int(steps),
        "dominant_semantic": dominant_semantic,
        "primitive_counts": _sorted_counter(primitive_counts),
        "semantic_counts": _sorted_counter(semantic_counts),
        "fallback_trigger_rate": float(fallback_trigger_count / steps),
        "fallback_selected_rate_overall": float(fallback_selected_count / steps),
        "fallback_selected_rate_when_triggered": float(fallback_selected_count / max(1, fallback_trigger_count)),
        "selected_matches_raw_argmax_rate": float(selected_matches_raw_argmax_count / steps),
        "selected_matches_fallback_argmax_rate": float(selected_matches_fallback_argmax_count / steps),
        "mask_degenerate_rate": float(mask_degenerate_count / steps),
        "all_invalid_fallback_rate": float(all_invalid_fallback_count / steps),
        "selected_action_hard_valid_rate": float(selected_action_hard_valid_count / steps),
        "selected_stop_check_rate": float(semantic_counts.get("stop-check", 0) / steps),
        "selected_articulation_recover_rate": float(semantic_counts.get("articulation-recover", 0) / steps),
        "selected_straight_adjust_rate": float(semantic_counts.get("straight-adjust", 0) / steps),
        "selected_selection_mask_mean": None if not selection_masks else float(mean(selection_masks)),
        "selected_selection_mask_min": None if not selection_masks else float(min(selection_masks)),
        "selected_semantic_score_mean": None if not semantic_scores else float(mean(semantic_scores)),
        "selected_semantic_score_min": None if not semantic_scores else float(min(semantic_scores)),
        "selected_soft_score_mean": None if not soft_scores else float(mean(soft_scores)),
        "selected_soft_score_min": None if not soft_scores else float(min(soft_scores)),
        "hard_valid_ratio_mean": None if not hard_valid_ratios else float(mean(hard_valid_ratios)),
        "hard_valid_ratio_min": None if not hard_valid_ratios else float(min(hard_valid_ratios)),
        "selected_proxy_score_mean": None if not proxy_scores else float(mean(proxy_scores)),
        "selected_proxy_score_min": None if not proxy_scores else float(min(proxy_scores)),
        "selected_proxy_prefix_mean": None if not prefix_lengths else float(mean(prefix_lengths)),
        "selected_prob_mean": None if not selection_probs else float(mean(selection_probs)),
    }


def _reason_breakdown(failures: List[Dict[str, Any]], episodes: int) -> Dict[str, Dict[str, float]]:
    reason_counts = Counter(item["done_reason"] for item in failures)
    breakdown: Dict[str, Dict[str, float]] = {}
    for reason, count in sorted(reason_counts.items(), key=lambda entry: (-entry[1], entry[0])):
        subset = [item for item in failures if item["done_reason"] == reason]
        breakdown[str(reason)] = {
            "count": int(count),
            "rate": float(count / max(1, episodes)),
            "mean_final_goal_distance": float(mean(item["final_goal_distance"] for item in subset)),
            "mean_heading_error_rad": float(mean(item["heading_error_rad"] for item in subset)),
            "mean_front_overlap_ratio": float(mean(item["front_overlap_ratio"] for item in subset)),
            "mean_macro_steps": float(mean(item["macro_steps"] for item in subset)),
            "collision_rate_within_reason": float(mean(float(item["collision"]) for item in subset)),
        }
    return breakdown


def _reset_options(level: str, warmup_progress: Optional[float]) -> Optional[Dict[str, object]]:
    if str(level) != "Warmup" or warmup_progress is None:
        return None
    return {"level": str(level), "warmup_progress": float(warmup_progress)}


def _collect_case_rows(
    config: ExperimentConfig,
    agent: HybridPPOAgent,
    primitive_library: Any,
    *,
    label: str,
    level: str,
    episodes: int,
    deterministic: bool,
    warmup_progress: Optional[float],
    seed_offset: int,
) -> List[Dict[str, Any]]:
    env = create_env_adapter(
        env_config=config.env,
        vehicle_config=config.vehicle,
        observation_config=config.observation,
        reward_config=config.reward,
    )
    executor = ParameterizedPrimitiveExecutor(
        primitive_library,
        vehicle_config=config.vehicle,
        executor_config=config.primitive_executor,
    )
    macro_env = ParameterizedMacroActionWrapper(env, executor, gamma=config.agent.gamma)

    rows: List[Dict[str, Any]] = []
    for episode_idx in range(max(0, int(episodes))):
        reset_options = _reset_options(level, warmup_progress)
        seed = int(config.seed + seed_offset + episode_idx)
        options = {"level": str(level)}
        if reset_options is not None:
            options.update(dict(reset_options))
            options["level"] = str(level)

        observation, _ = env.reset(seed=seed, options=options)
        observation = np.asarray(observation, dtype=np.float32)
        total_reward = 0.0
        macro_steps = 0
        low_level_steps = 0
        last_info: Dict[str, object] = env.current_info()
        terminated = False
        truncated = False
        previous_action_id: Optional[int] = None
        step_records: List[Dict[str, object]] = []

        while macro_steps < int(config.schedule.max_macro_steps_per_episode) and not (terminated or truncated):
            context = env.make_primitive_context()
            selection = agent.act(
                observation,
                deterministic=bool(deterministic),
                return_diagnostics=True,
            )
            next_observation, reward, terminated, truncated, step_info = macro_env.step(
                selection.macro_action,
                start_state=env.get_articulated_state(),
                context=context,
            )
            if next_observation is None:
                next_observation = env.build_observation()
            next_observation = np.asarray(next_observation, dtype=np.float32)
            tau = int(step_info.get("tau", 1))
            macro_steps += 1
            low_level_steps += tau
            forced_truncation = bool((not terminated) and (not truncated) and macro_steps >= int(config.schedule.max_macro_steps_per_episode))
            if forced_truncation:
                truncated = True
                step_info = dict(step_info)
                step_info["done_reason"] = "macro_budget"
                step_info["truncated"] = True
                step_info["done"] = True

            diagnostics = dict(selection.diagnostics or {})
            diagnostics.update(
                {
                    "primitive_id": int(selection.macro_action.primitive_id),
                    "semantic": str(primitive_library.spec(int(selection.macro_action.primitive_id)).semantic.value),
                    "tau": int(tau),
                    "reward": float(reward),
                    "step_index": int(macro_steps),
                    "step_done_reason": str(step_info.get("done_reason", "running")),
                }
            )
            step_records.append(diagnostics)

            total_reward += float(reward)
            observation = next_observation
            last_info = dict(step_info)
            previous_action_id = int(selection.macro_action.primitive_id)

        scene_metadata = dict(last_info.get("scene_metadata", {}))
        success = bool(last_info.get("goal_reached", False) or last_info.get("success", False))
        rows.append(
            {
                "label": str(label),
                "level": str(level),
                "seed": int(seed),
                "success": bool(success),
                "done_reason": str(last_info.get("done_reason", "running")),
                "collision": bool(last_info.get("collision", False)),
                "macro_steps": int(macro_steps),
                "low_level_steps": int(low_level_steps),
                "total_reward": float(total_reward),
                "final_goal_distance": float(env.distance_to_goal()),
                "goal_reached": bool(last_info.get("goal_reached", False)),
                "guidance_available": bool(last_info.get("guidance_available", False)),
                "guidance_path_confidence": float(last_info.get("guidance_path_confidence", 0.0)),
                "front_overlap_ratio": float(last_info.get("front_overlap_ratio", 0.0)),
                "rear_overlap_ratio": float(last_info.get("rear_overlap_ratio", 0.0)),
                "heading_error_rad": float(last_info.get("heading_error_rad", 0.0)),
                "scene_type": scene_metadata.get("scene_type"),
                "corridor_width": scene_metadata.get("corridor_width"),
                "corridor_min_width": scene_metadata.get("corridor_min_width"),
                "warmup_progress": scene_metadata.get("warmup_progress"),
                "turn_count": scene_metadata.get("turn_count"),
                "free_shape_count": scene_metadata.get("free_shape_count"),
                "valid_candidate_count": scene_metadata.get("valid_candidate_count"),
                "action_summary": _episode_action_summary(step_records),
            }
        )
    return rows


def _summarize_case(rows: List[Dict[str, Any]], max_failure_examples: int) -> Dict[str, Any]:
    if len(rows) == 0:
        return {
            "episodes": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "guidance_missing_rate": 0.0,
            "mean_goal_distance": 0.0,
            "failure_reason_breakdown": {},
            "top_failure_scene_buckets": [],
            "failure_examples": [],
        }

    failures = [row for row in rows if not row["success"]]
    episodes = len(rows)
    successes = episodes - len(failures)
    return {
        "label": rows[0]["label"],
        "level": rows[0]["level"],
        "episodes": int(episodes),
        "successes": int(successes),
        "failures": int(len(failures)),
        "success_rate": float(successes / max(1, episodes)),
        "collision_rate": float(mean(float(row["collision"]) for row in rows)),
        "guidance_missing_rate": float(mean(0.0 if row["guidance_available"] else 1.0 for row in rows)),
        "mean_goal_distance": float(mean(row["final_goal_distance"] for row in rows)),
        "failure_reason_breakdown": _reason_breakdown(failures, episodes),
        "action_diagnostics": {
            "overall": _aggregate_action_summaries(rows),
            "failures": _aggregate_action_summaries(failures),
            "macro_budget_failures": _aggregate_action_summaries([row for row in failures if row["done_reason"] == "macro_budget"]),
            "collision_failures": _aggregate_action_summaries([row for row in failures if row["done_reason"] == "collision"]),
        },
        "top_failure_scene_buckets": _scene_buckets(failures, limit=6),
        "failure_examples": _hard_examples(failures, limit=max_failure_examples),
    }


def _build_cases(args: argparse.Namespace) -> Iterable[Tuple[str, str, int, Optional[float], int]]:
    current_seed_offset = int(args.seed_offset)
    if int(args.debug_episodes) > 0:
        yield "Debug", "Debug", int(args.debug_episodes), None, current_seed_offset
        current_seed_offset += 10_000
    for progress in args.warmup_progress:
        if int(args.warmup_episodes) <= 0:
            break
        label = f"Warmup(p={float(progress):.2f})"
        yield label, "Warmup", int(args.warmup_episodes), float(progress), current_seed_offset
        current_seed_offset += 10_000
    if int(args.normal_episodes) > 0:
        yield "Normal", "Normal", int(args.normal_episodes), None, current_seed_offset


def _print_human_report(report: Dict[str, Any]) -> None:
    checkpoint_meta = report["checkpoint_meta"]
    print("=== CHECKPOINT META ===")
    print(json.dumps(checkpoint_meta, ensure_ascii=False, indent=2))
    print("=== CHECKPOINT STORED EVAL ===")
    print(json.dumps(report["checkpoint_stored_eval"], ensure_ascii=False, indent=2))
    print("=== EVENT SUMMARY ===")
    print(json.dumps(report["event_summary"], ensure_ascii=False, indent=2))
    print("=== EXTENDED DETERMINISTIC EVAL ===")
    for case in report["extended_eval"]:
        print(
            f"{case['label']}: success={case['successes']}/{case['episodes']} "
            f"({case['success_rate']:.3f}) fail={case['failures']} collision={case['collision_rate']:.3f} "
            f"guidance_missing={case['guidance_missing_rate']:.3f} goal_dist={case['mean_goal_distance']:.3f}"
        )
        print(json.dumps(case["failure_reason_breakdown"], ensure_ascii=False, indent=2))
        print(json.dumps(case["action_diagnostics"], ensure_ascii=False, indent=2))


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(
        None if args.run_dir is None else Path(args.run_dir).expanduser().resolve(),
        args.checkpoint,
    )
    run_dir = _resolve_run_dir(args.run_dir, checkpoint_path)
    event_path = _resolve_event_path(run_dir, args.event_file)
    payload = _torch_load_checkpoint(checkpoint_path)
    config = _apply_config_overrides(_build_config(dict(payload["config"])), args)
    repo_root = Path(__file__).resolve().parents[1]
    agent, primitive_library = _load_agent(payload, config, repo_root)
    extra = dict(payload.get("extra", {}))

    report = {
        "checkpoint_meta": {
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "event_path": None if event_path is None else str(event_path),
            "episode_idx": int(payload.get("episode_idx", -1)),
            "update_idx": int(payload.get("update_idx", -1)),
            "best_metric": float(payload.get("best_metric", float("nan"))),
            "checkpoint_metric_path": extra.get("checkpoint_metric_path"),
            "checkpoint_metric_value": extra.get("checkpoint_metric_value"),
            "curriculum_band_unlocked": dict(extra.get("curriculum", {})).get("band_unlocked"),
            "curriculum_level_counts": dict(extra.get("curriculum", {})).get("level_counts"),
            "deterministic": not bool(args.stochastic),
            "analysis_config": {
                "soft_mask_enabled": bool(config.agent.soft_mask_enabled),
                "soft_mask_gamma": float(config.agent.soft_mask_gamma),
                "soft_mask_logit_scale": float(config.agent.soft_mask_logit_scale),
                "soft_mask_floor": float(config.agent.soft_mask_floor),
                "soft_mask_fallback_bonus": float(config.agent.soft_mask_fallback_bonus),
                "proxy_sidecar_path": str(config.proxy_safety.sidecar_path),
            },
            "proxy_sidecar": _proxy_sidecar_summary(primitive_library),
        },
        "checkpoint_stored_eval": _compact_evaluation(dict(extra.get("evaluation", {}))),
        "event_summary": _event_summary(event_path),
        "extended_eval": [],
    }

    for label, level, episodes, warmup_progress, seed_offset in _build_cases(args):
        rows = _collect_case_rows(
            config=config,
            agent=agent,
            primitive_library=primitive_library,
            label=label,
            level=level,
            episodes=episodes,
            deterministic=not bool(args.stochastic),
            warmup_progress=warmup_progress,
            seed_offset=seed_offset,
        )
        report["extended_eval"].append(
            _summarize_case(rows, max_failure_examples=int(args.max_failure_examples))
        )

    return report


def main() -> None:
    args = build_argument_parser().parse_args()
    report = build_report(args)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    _print_human_report(report)


if __name__ == "__main__":
    main()