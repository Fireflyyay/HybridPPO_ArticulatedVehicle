import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "hybridppo_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join("/tmp", "hybridppo_cache"))

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as PolygonPatch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from common.runtime_config import ExperimentConfig
from common.types import ArticulatedState, MacroAction, wrap_to_pi
from env.adapter import create_env_adapter
from env.success import articulated_body_polygons
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library, load_proxy_safety_sidecar


@dataclass(frozen=True)
class MacroStepTrace:
    primitive_id: int
    primitive_name: str
    tau: int
    reward: float
    termination_reason: str
    parameters: Dict[str, float]


@dataclass(frozen=True)
class PathTrace:
    level: str
    seed: int
    states: Tuple[ArticulatedState, ...]
    guidance_path: Optional[np.ndarray]
    macro_steps: Tuple[MacroStepTrace, ...]
    total_reward: float
    success: bool
    collision: bool
    terminated: bool
    truncated: bool
    done_reason: str
    final_goal_distance: float
    scene: object
    last_info: Dict[str, object]
    escape_targets: Tuple[Tuple[float, float], ...] = ()
    reference_targets: Tuple[Tuple[float, float, float], ...] = ()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize policy-planned articulated vehicle paths and save PNGs under img/."
    )
    parser.add_argument(
        "--level",
        choices=("Debug", "Warmup", "Normal"),
        default=ExperimentConfig().env.default_level,
        help="Scene difficulty to sample.",
    )
    parser.add_argument("--num-paths", type=int, default=3, help="Number of paths to render.")
    parser.add_argument("--seed", type=int, default=ExperimentConfig().seed, help="Base scene seed.")
    parser.add_argument(
        "--warmup-progress",
        type=float,
        default=None,
        help="Optional Warmup curriculum progress in [0, 1]. Only used for Warmup scenes.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to latest config-compatible runs/**/checkpoints/best.pt.",
    )
    parser.add_argument("--runs-dir", type=str, default=ExperimentConfig().logging.log_root, help="Runs directory.")
    parser.add_argument("--output-dir", type=str, default="img", help="Directory for output PNGs.")
    parser.add_argument("--prefix", type=str, default="policy_path", help="Output filename prefix.")
    parser.add_argument(
        "--max-macro-steps",
        type=int,
        default=ExperimentConfig().schedule.max_macro_steps_per_episode,
        help="Macro action budget per path.",
    )
    parser.add_argument("--vehicle-samples", type=int, default=5, help="Number of intermediate vehicle poses to draw.")
    parser.add_argument("--dpi", type=int, default=200, help="Saved image DPI.")
    parser.add_argument(
        "--device",
        type=str,
        default=ExperimentConfig().device,
        help="Torch device: auto, cpu, cuda, etc.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample stochastic policy actions instead of deterministic actions.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only save successful plans. By default failed plans are saved too.",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Allow fallback to the newest best.pt if no config-compatible checkpoint is found.",
    )
    parser.add_argument(
        "--enable-phase-reference",
        action="store_true",
        default=False,
        help="Enable phase-based reference target in the environment observation.",
    )
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_name))


def _torch_load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.load(str(path), map_location=map_location)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _values_equal(left: object, right: object, atol: float = 1e-9) -> bool:
    if _is_number(left) and _is_number(right):
        return bool(abs(float(left) - float(right)) <= atol)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(map(str, left.keys()))
        right_keys = set(map(str, right.keys()))
        if left_keys != right_keys:
            return False
        return all(_values_equal(left[key], right[key], atol=atol) for key in left.keys())
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_equal(l_item, r_item, atol=atol) for l_item, r_item in zip(left, right))
    return left == right


def _contains_current_config(stored: object, current: object, atol: float = 1e-9) -> bool:
    if isinstance(stored, Mapping) and isinstance(current, Mapping):
        for key, current_value in current.items():
            if key not in stored:
                return False
            if not _contains_current_config(stored[key], current_value, atol=atol):
                return False
        return True
    if isinstance(stored, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(stored) != len(current):
            return False
        return all(_contains_current_config(s_item, c_item, atol=atol) for s_item, c_item in zip(stored, current))
    return _values_equal(stored, current, atol=atol)


def _checkpoint_matches_current_config(checkpoint: Mapping[str, object], config: ExperimentConfig) -> bool:
    stored = checkpoint.get("config")
    if not isinstance(stored, Mapping):
        return False
    current = config.to_dict()
    keys = (
        "vehicle",
        "primitive_executor",
        "observation",
        "reward",
        "env",
        "agent",
    )
    return all(_contains_current_config(stored.get(key), current.get(key)) for key in keys)


def _find_best_checkpoint(
    runs_dir: Path,
    config: ExperimentConfig,
    best_filename: str,
    allow_config_mismatch: bool,
) -> Path:
    candidates = sorted(
        runs_dir.rglob(f"checkpoints/{best_filename}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no {best_filename} found under {runs_dir}")

    newest_any = candidates[0]
    for candidate in candidates:
        try:
            checkpoint = _torch_load_checkpoint(candidate)
        except Exception:
            continue
        if _checkpoint_matches_current_config(checkpoint, config):
            return candidate

    if allow_config_mismatch:
        return newest_any
    raise FileNotFoundError(
        f"no config-compatible {best_filename} found under {runs_dir}; "
        "pass --checkpoint explicitly or use --allow-config-mismatch"
    )


def _resolve_proxy_sidecar_path(config: ExperimentConfig) -> Optional[Path]:
    configured = Path(str(config.proxy_safety.sidecar_path)).expanduser()
    if configured.is_file():
        return configured
    repo_local = Path(REPO_ROOT) / "data" / "proxy_safety_sidecar.npz"
    if repo_local.is_file():
        return repo_local
    return None


def _build_agent(config: ExperimentConfig, checkpoint_path: Path, device: torch.device) -> Tuple[HybridPPOAgent, object]:
    sidecar_path = _resolve_proxy_sidecar_path(config)
    proxy_sidecar = load_proxy_safety_sidecar(str(sidecar_path)) if sidecar_path is not None else None
    primitive_library = build_default_primitive_library(proxy_sidecar=proxy_sidecar)
    agent = HybridPPOAgent(
        config=config.agent.build(
            observation_dim=int(config.observation.observation_dim),
            action_dim=int(primitive_library.action_dim),
            parameter_dim=int(primitive_library.parameter_dim),
        ),
        primitive_library=primitive_library,
        device=device,
    )
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=str(device))
    agent.load_checkpoint_state(checkpoint["agent"])
    agent.policy.eval()
    agent.value_net.eval()
    return agent, primitive_library


def _reset_options(level: str, warmup_progress: Optional[float]) -> Dict[str, object]:
    options: Dict[str, object] = {"level": str(level)}
    if str(level) == "Warmup" and warmup_progress is not None:
        options["warmup_progress"] = float(np.clip(float(warmup_progress), 0.0, 1.0))
    return options


def _simulate_path(
    config: ExperimentConfig,
    agent: HybridPPOAgent,
    primitive_library,
    level: str,
    seed: int,
    max_macro_steps: int,
    deterministic: bool,
    warmup_progress: Optional[float],
) -> PathTrace:
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

    observation, _ = env.reset(seed=int(seed), options=_reset_options(level, warmup_progress))
    task_env = env.env
    scene = task_env._scene
    guidance_path = None
    if task_env._global_guidance.path_points_world is not None:
        guidance_path = np.asarray(task_env._global_guidance.path_points_world, dtype=np.float64).copy()

    states: List[ArticulatedState] = [env.get_articulated_state()]
    macro_steps: List[MacroStepTrace] = []
    reference_targets: List[Tuple[float, float, float]] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    last_info: Dict[str, object] = env.current_info()

    with torch.no_grad():
        for _ in range(int(max_macro_steps)):
            if terminated or truncated:
                break
            selection = agent.act(np.asarray(observation, dtype=np.float32), deterministic=deterministic)
            action = MacroAction(
                primitive_id=int(selection.macro_action.primitive_id),
                parameters=np.asarray(selection.macro_action.parameters, dtype=np.float32).copy(),
            )
            context = env.make_primitive_context()
            _capture_reference_target(reference_targets, task_env)
            rollout = executor.rollout(
                start_state=env.get_articulated_state(),
                primitive_id=action.primitive_id,
                parameters=action.parameters,
                context=context,
            )
            macro_reward = 0.0
            executed_tau = 0
            next_observation = None
            for step_index, control in enumerate(rollout.controls):
                next_observation, reward, terminated, truncated, step_info = env.step(control.as_array())
                macro_reward += (float(agent.config.gamma) ** step_index) * float(reward)
                total_reward += float(reward)
                executed_tau += 1
                last_info = dict(step_info)
                states.append(env.get_articulated_state())
                if terminated or truncated:
                    break

            if next_observation is None:
                next_observation = env.build_observation()
            observation = np.asarray(next_observation, dtype=np.float32)
            spec = primitive_library.spec(action.primitive_id)
            macro_steps.append(
                MacroStepTrace(
                    primitive_id=int(action.primitive_id),
                    primitive_name=str(spec.semantic.value),
                    tau=int(executed_tau),
                    reward=float(macro_reward),
                    termination_reason=str(rollout.termination_reason),
                    parameters=primitive_library.vector_to_dict(action.primitive_id, action.parameters),
                )
            )

        if not (terminated or truncated) and len(macro_steps) >= int(max_macro_steps):
            truncated = True
            last_info = dict(last_info)
            last_info["done"] = True
            last_info["truncated"] = True
            last_info["done_reason"] = "macro_budget"

    success = bool(last_info.get("goal_reached", False) or last_info.get("success", False))
    return PathTrace(
        level=str(level),
        seed=int(seed),
        states=tuple(states),
        guidance_path=guidance_path,
        macro_steps=tuple(macro_steps),
        total_reward=float(total_reward),
        success=success,
        collision=bool(last_info.get("collision", False)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        done_reason=str(last_info.get("done_reason", "running")),
        final_goal_distance=float(env.distance_to_goal()),
        scene=scene,
        last_info=dict(last_info),
        reference_targets=tuple(reference_targets),
    )


def _polygon_patch(
    points: Sequence[Tuple[float, float]],
    facecolor: str,
    edgecolor: str,
    alpha: float,
    linewidth: float = 1.0,
    linestyle: str = "-",
    label: str = "",
) -> PolygonPatch:
    return PolygonPatch(
        list(points),
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


def _add_shapely_polygon(ax: Axes, polygon, facecolor: str, edgecolor: str, alpha: float) -> None:
    if polygon.is_empty:
        return
    ax.add_patch(
        _polygon_patch(
            [(float(x), float(y)) for x, y in polygon.exterior.coords],
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
        )
    )
    for interior in polygon.interiors:
        ax.add_patch(
            _polygon_patch(
                [(float(x), float(y)) for x, y in interior.coords],
                facecolor=ax.get_facecolor(),
                edgecolor="none",
                alpha=1.0,
            )
        )


def _add_vehicle(
    ax: Axes,
    state: ArticulatedState,
    vehicle_config,
    front_colors: Tuple[str, str],
    rear_colors: Tuple[str, str],
    alpha: float,
    label: str = "",
    linestyle: str = "-",
) -> None:
    front_poly, rear_poly = articulated_body_polygons(state, vehicle_config)
    ax.add_patch(
        _polygon_patch(
            [(float(x), float(y)) for x, y in front_poly.exterior.coords],
            facecolor=front_colors[0],
            edgecolor=front_colors[1],
            alpha=alpha,
            linewidth=1.2,
            linestyle=linestyle,
            label=label,
        )
    )
    ax.add_patch(
        _polygon_patch(
            [(float(x), float(y)) for x, y in rear_poly.exterior.coords],
            facecolor=rear_colors[0],
            edgecolor=rear_colors[1],
            alpha=max(0.0, alpha - 0.08),
            linewidth=1.2,
            linestyle=linestyle,
        )
    )


def _intermediate_indices(count: int, sample_count: int) -> List[int]:
    if count <= 2 or sample_count <= 0:
        return []
    candidates = np.linspace(1, count - 2, min(sample_count, count - 2), dtype=np.int64)
    return sorted(set(int(item) for item in candidates.tolist()))


def _capture_reference_target(
    reference_targets: List[Tuple[float, float, float]],
    task_env,
) -> None:
    ref_pos = getattr(task_env, '_active_reference_goal_position', None)
    ref_heading = getattr(task_env, '_active_reference_goal_heading', None)
    if ref_pos is None or ref_heading is None:
        return
    current = (float(ref_pos[0]), float(ref_pos[1]), float(ref_heading))
    if reference_targets:
        prev = reference_targets[-1]
        if (abs(current[0] - prev[0]) < 0.25 and abs(current[1] - prev[1]) < 0.25
                and abs(wrap_to_pi(current[2] - prev[2])) < np.deg2rad(8.0)):
            return
    reference_targets.append(current)


def _axis_limits(trace: PathTrace, margin: float = 8.0) -> Tuple[float, float, float, float]:
    scene = trace.scene
    xmin, xmax, ymin, ymax = scene.world_bounds
    xs = [float(state.x) for state in trace.states] + [float(scene.start_state.x), float(scene.goal_state.x)]
    ys = [float(state.y) for state in trace.states] + [float(scene.start_state.y), float(scene.goal_state.y)]
    if trace.guidance_path is not None and len(trace.guidance_path) > 0:
        xs.extend(trace.guidance_path[:, 0].astype(float).tolist())
        ys.extend(trace.guidance_path[:, 1].astype(float).tolist())
    view_xmin = max(float(xmin), min(xs) - margin)
    view_xmax = min(float(xmax), max(xs) + margin)
    view_ymin = max(float(ymin), min(ys) - margin)
    view_ymax = min(float(ymax), max(ys) + margin)
    if view_xmax - view_xmin < 12.0:
        center = 0.5 * (view_xmin + view_xmax)
        view_xmin, view_xmax = center - 6.0, center + 6.0
    if view_ymax - view_ymin < 12.0:
        center = 0.5 * (view_ymin + view_ymax)
        view_ymin, view_ymax = center - 6.0, center + 6.0
    return view_xmin, view_xmax, view_ymin, view_ymax


def _render_trace(
    trace: PathTrace,
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_path: Path,
    vehicle_samples: int,
    dpi: int,
) -> None:
    figure, ax = plt.subplots(figsize=(9.0, 9.0))
    scene = trace.scene
    ax.set_facecolor("#f7f6f1")
    for obstacle in scene.obstacles:
        _add_shapely_polygon(ax, obstacle, facecolor="#46484d", edgecolor="#25272b", alpha=0.95)

    if trace.guidance_path is not None and len(trace.guidance_path) >= 2:
        ax.plot(
            trace.guidance_path[:, 0],
            trace.guidance_path[:, 1],
            color="#4d80d8",
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            label="Coarse route",
        )

    route_xy = np.asarray([(float(state.x), float(state.y)) for state in trace.states], dtype=np.float64)
    route_color = "#2a9d8f" if trace.success else "#d1495b"
    if len(route_xy) >= 2:
        ax.plot(route_xy[:, 0], route_xy[:, 1], color=route_color, linewidth=2.6, alpha=0.95, label="Policy path")
        ax.scatter(route_xy[:, 0], route_xy[:, 1], s=8, color=route_color, alpha=0.55)

    _add_vehicle(
        ax,
        scene.start_state,
        config.vehicle,
        front_colors=("#6cc8c1", "#18736d"),
        rear_colors=("#9be1dc", "#18736d"),
        alpha=0.9,
        label="Start parking pose",
    )
    _add_vehicle(
        ax,
        scene.goal_state,
        config.vehicle,
        front_colors=("#f2a65a", "#9b4d0d"),
        rear_colors=("#f7c58f", "#9b4d0d"),
        alpha=0.88,
        label="Goal parking pose",
    )

    for index in _intermediate_indices(len(trace.states), int(vehicle_samples)):
        _add_vehicle(
            ax,
            trace.states[index],
            config.vehicle,
            front_colors=("#c9c46a", "#6b671d"),
            rear_colors=("#ddd987", "#6b671d"),
            alpha=0.34,
            linestyle="-",
        )

    if trace.states:
        _add_vehicle(
            ax,
            trace.states[-1],
            config.vehicle,
            front_colors=("#ffffff", route_color),
            rear_colors=("#ffffff", route_color),
            alpha=0.62,
            linestyle="--",
            label="Final pose",
        )

    if trace.reference_targets:
        ref_targets = np.asarray(trace.reference_targets, dtype=np.float64)
        goal_pos = np.array([float(trace.scene.goal_state.x), float(trace.scene.goal_state.y)], dtype=np.float64)
        arrow_length = 2.5
        for idx in range(len(ref_targets)):
            rx, ry, rhead = float(ref_targets[idx, 0]), float(ref_targets[idx, 1]), float(ref_targets[idx, 2])
            is_goal = bool(np.hypot(rx - goal_pos[0], ry - goal_pos[1]) < 1.0)
            color = "#2a9d8f" if is_goal else "#4d80d8"
            label = "Goal-phase ref" if (is_goal and idx == 0) else ("Guidance-phase ref" if (not is_goal and idx == 0) else "")
            ax.plot(rx, ry, "o", color=color, markersize=5, alpha=0.85, zorder=5, label=label or None)
            dx = arrow_length * np.cos(rhead)
            dy = arrow_length * np.sin(rhead)
            ax.arrow(rx, ry, dx, dy, head_width=1.2, head_length=0.8, fc=color, ec=color, alpha=0.65, linewidth=1.0, zorder=5)
        if len(ref_targets) >= 2:
            ax.plot(ref_targets[:, 0], ref_targets[:, 1], color="#4d80d8", linewidth=0.8, linestyle=":", alpha=0.4)

    if trace.escape_targets:
        esc_positions = np.asarray(trace.escape_targets, dtype=np.float64)
        ax.scatter(
            esc_positions[:, 0], esc_positions[:, 1],
            marker="D", s=48, facecolor="#ffb347", edgecolor="#b35e00",
            linewidth=1.2, alpha=0.92, zorder=5, label="Escape sub-goal",
        )
        if len(esc_positions) >= 2:
            ax.plot(
                esc_positions[:, 0], esc_positions[:, 1],
                color="#ffb347", linewidth=1.0, linestyle=":", alpha=0.55,
            )

    view_xmin, view_xmax, view_ymin, view_ymax = _axis_limits(trace)
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    status = "SUCCESS" if trace.success else f"FAILED: {trace.done_reason}"
    title = (
        f"{trace.level} | seed={trace.seed} | {status}\n"
        f"macro={len(trace.macro_steps)} low={max(0, len(trace.states) - 1)} "
        f"distance={trace.final_goal_distance:.2f}m return={trace.total_reward:.2f} | "
        f"{checkpoint_path.parent.parent.name}/{checkpoint_path.name}"
    )
    ax.set_title(title, fontsize=11)
    handles: List[object] = [
        Patch(facecolor="#46484d", edgecolor="#25272b", label="Obstacle"),
        Line2D([0], [0], color="#4d80d8", linestyle="--", linewidth=1.8, label="Coarse route"),
        Line2D([0], [0], color=route_color, linewidth=2.6, label="Policy path"),
        Patch(facecolor="#6cc8c1", edgecolor="#18736d", label="Start parking pose"),
        Patch(facecolor="#f2a65a", edgecolor="#9b4d0d", label="Goal parking pose"),
        Patch(facecolor="#c9c46a", edgecolor="#6b671d", alpha=0.45, label="Intermediate vehicle"),
        Patch(facecolor="#ffffff", edgecolor=route_color, label="Final pose"),
    ]
    if trace.escape_targets:
        handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="#ffb347", markeredgecolor="#b35e00", markersize=8, label="Escape sub-goal"))
    if trace.reference_targets:
        handles.append(Line2D([0], [0], marker="o", color="#4d80d8", markersize=6, linestyle="None", label="Guidance-phase ref"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _filename(prefix: str, trace: PathTrace, index: int) -> str:
    status = "success" if trace.success else f"failed_{trace.done_reason}"
    safe_status = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in status.lower())
    return f"{prefix}_{index:03d}_{trace.level.lower()}_seed{trace.seed}_{safe_status}.png"


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig()
    if bool(args.enable_phase_reference):
        from dataclasses import replace
        config = replace(config, env=replace(config.env, phase_reference_enabled=True))
    device = _resolve_device(str(args.device))
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else _find_best_checkpoint(
        Path(args.runs_dir).expanduser(),
        config,
        best_filename=str(config.checkpoint.best_filename),
        allow_config_mismatch=bool(args.allow_config_mismatch),
    )
    checkpoint_path = checkpoint_path.resolve()
    agent, primitive_library = _build_agent(config, checkpoint_path, device=device)
    output_dir = Path(args.output_dir).expanduser()
    deterministic = not bool(args.stochastic)
    saved_paths: List[Path] = []
    skipped = 0

    for index in range(int(args.num_paths)):
        seed = int(args.seed) + index
        trace = _simulate_path(
            config=config,
            agent=agent,
            primitive_library=primitive_library,
            level=str(args.level),
            seed=seed,
            max_macro_steps=int(args.max_macro_steps),
            deterministic=deterministic,
            warmup_progress=args.warmup_progress,
        )
        if bool(args.success_only) and not trace.success:
            skipped += 1
            continue
        output_path = output_dir / _filename(str(args.prefix), trace, index)
        _render_trace(
            trace=trace,
            config=config,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            vehicle_samples=int(args.vehicle_samples),
            dpi=int(args.dpi),
        )
        saved_paths.append(output_path)
        print(
            f"{output_path} | success={trace.success} reason={trace.done_reason} "
            f"macro={len(trace.macro_steps)} low={max(0, len(trace.states) - 1)} "
            f"distance={trace.final_goal_distance:.2f}"
        )

    if skipped:
        print(f"skipped {skipped} failed path(s) because --success-only was set")
    if not saved_paths:
        raise RuntimeError("no path images were saved")


if __name__ == "__main__":
    main()
