#!/usr/bin/env python3
"""完整训练链路速度对比 benchmark。

依次测试:
  1. 场景生成 (Debug / Warmup / Normal 各 20 次)
  2. 单集 rollout (各 4 集, 含 Soft Mask ON/OFF)
  3. PPO update (1 次完整 update)

输出 Markdown 格式对比报告。
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── path ────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from common.runtime_config import (
    ExperimentConfig,
    HybridPPOHyperConfig,
    TrainingScheduleConfig,
    build_scene_presets,
)
from common.config import VehicleConfig
from env.adapter import create_env_adapter
from env.scenes import BaselineInspiredSceneFactory as OldSceneFactory


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════

def _fmt_ms(vals: List[float]) -> str:
    if not vals:
        return "—"
    arr = np.array(vals) * 1000
    return f"{np.mean(arr):.2f}ms  (min={np.min(arr):.2f}, p99={np.percentile(arr, 99):.2f})"


def _fmt_s(vals: List[float]) -> str:
    if not vals:
        return "—"
    arr = np.array(vals)
    return f"{np.mean(arr):.3f}s  (sum={np.sum(arr):.3f})"


def _fmt_count(vals: List[float]) -> str:
    if not vals:
        return "—"
    arr = np.array(vals)
    return f"μ={np.mean(arr):.1f}  (min={np.min(arr):.0f}, max={np.max(arr):.0f})"


def time_it(n: int, func, *args, **kwargs):
    """运行 func n 次, 返回耗时列表 (秒)."""
    times = []
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return times


# ══════════════════════════════════════════════════════════════════════
# 场景生成 benchmark
# ══════════════════════════════════════════════════════════════════════

def benchmark_scene_gen(
    factory,
    presets: dict,
    vehicle_config: VehicleConfig,
    level: str,
    trials: int = 20,
    warmup_options: Optional[dict] = None,
):
    """对某一 level 的场景生成做 trials 次, 返回:
    {label: [seconds]} 字典, 以及成功/失败次数.
    """
    rng = np.random.default_rng(42)
    results: Dict[str, List[float]] = defaultdict(list)
    failures = 0

    for i in range(trials):
        seed = int(42 + i)
        rng = np.random.default_rng(seed)
        try:
            t0 = time.perf_counter()
            scene = factory.generate(level, rng, options=warmup_options)
            elapsed = time.perf_counter() - t0
            results["total"].append(elapsed)

            # obstacle 段数
            seg_count = 0
            for poly in scene.obstacles:
                seg_count += len(poly.exterior.coords) - 1
                for interior in poly.interiors:
                    seg_count += len(interior.coords) - 1
            results["obstacle_segments"].append(float(seg_count))

            # free ratio (如果 metadata 里有)
            free_ratio = scene.metadata.get("free_ratio")
            if free_ratio is not None:
                results["free_ratio"].append(float(free_ratio))

        except RuntimeError:
            failures += 1
            results["total"].append(float("nan"))

    return dict(results), failures


def benchmark_all_scene_gen(factory, presets, vehicle_config, trials: int = 20):
    """对所有 level 做场景生成 benchmark."""
    report: Dict[str, dict] = {}
    for level in ["Debug", "Warmup", "Normal"]:
        if level not in presets:
            continue
        opts = None
        if level == "Warmup":
            opts = {"warmup_progress": 0.5}
        data, failures = benchmark_scene_gen(
            factory, presets, vehicle_config, level, trials=trials, warmup_options=opts
        )
        data["failures"] = [float(failures)]
        report[level] = data
    return report


# ══════════════════════════════════════════════════════════════════════
# Episode rollout benchmark
# ══════════════════════════════════════════════════════════════════════

def benchmark_episodes(config: ExperimentConfig, episodes: int = 8):
    """跑 episodes 集, 返回各指标统计."""
    from model.agent import HybridPPOAgent
    from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library, load_proxy_safety_sidecar
    from env.macro_wrapper import ParameterizedMacroActionWrapper
    from training.rollout import MacroRolloutDriver

    proxy = None
    if str(config.proxy_safety.sidecar_path).strip():
        proxy = load_proxy_safety_sidecar(str(config.proxy_safety.sidecar_path))
    lib = build_default_primitive_library(proxy_sidecar=proxy)
    env = create_env_adapter(config.env, config.vehicle, config.observation, config.reward)
    agent = HybridPPOAgent(
        config=config.agent.build(
            observation_dim=int(config.observation.observation_dim),
            action_dim=int(lib.action_dim),
            parameter_dim=int(lib.parameter_dim),
        ),
        primitive_library=lib,
        device="cpu",
    )
    executor = ParameterizedPrimitiveExecutor(lib, vehicle_config=config.vehicle, executor_config=config.primitive_executor)
    macro_env = ParameterizedMacroActionWrapper(env, executor, gamma=agent.config.gamma)
    rollout = MacroRolloutDriver(
        env=env, macro_env=macro_env, agent=agent,
        max_macro_steps=config.schedule.max_macro_steps_per_episode,
        soft_teacher=None,
    )

    stats: Dict[str, List[float]] = defaultdict(list)
    for ep in range(episodes):
        level = "Warmup" if ep < episodes // 2 else "Normal"
        t0 = time.perf_counter()
        summary = rollout.collect_episode(
            level=level, seed=int(config.seed + ep),
            deterministic=False, store_transition=False,
            reset_options={"level": level, "warmup_progress": 0.5} if level == "Warmup" else None,
        )
        elapsed = time.perf_counter() - t0
        stats["episode_time_s"].append(elapsed)
        stats["macro_steps"].append(float(summary.macro_steps))
        stats["low_level_steps"].append(float(summary.low_level_steps))
        stats["success"].append(float(summary.success))
        stats["collision"].append(float(summary.collision))
        stats["reward"].append(float(summary.total_reward))
        stats["goal_distance"].append(float(summary.final_goal_distance))

    return dict(stats)


# ══════════════════════════════════════════════════════════════════════
# PPO update benchmark
# ══════════════════════════════════════════════════════════════════════

def benchmark_ppo_update(config: ExperimentConfig, n_transitions: int = 512):
    """填充 buffer 后跑一次完整 PPO update."""
    from model.agent import HybridPPOAgent
    from primitives import build_default_primitive_library, load_proxy_safety_sidecar
    from common.types import MacroTransition
    from model.distributions import AffineBeta

    proxy = None
    if str(config.proxy_safety.sidecar_path).strip():
        proxy = load_proxy_safety_sidecar(str(config.proxy_safety.sidecar_path))
    lib = build_default_primitive_library(proxy_sidecar=proxy)
    agent = HybridPPOAgent(
        config=config.agent.build(
            observation_dim=int(config.observation.observation_dim),
            action_dim=int(lib.action_dim),
            parameter_dim=int(lib.parameter_dim),
        ),
        primitive_library=lib,
        device="cpu",
    )

    obs_dim = int(config.observation.observation_dim)

    # 填充假 transitions
    for i in range(n_transitions):
        agent.store_transition(
            MacroTransition(
                observation=np.random.randn(obs_dim).astype(np.float32),
                action_id=int(np.random.randint(0, lib.action_dim)),
                parameters=np.random.randn(lib.parameter_dim).astype(np.float32) * 0.5,
                reward=float(np.random.randn() * 0.1),
                tau=int(np.random.randint(1, 16)),
                next_observation=np.random.randn(obs_dim).astype(np.float32),
                done=bool(i == n_transitions - 1 or np.random.random() < 0.05),
                log_prob=float(np.random.randn() * 0.5 - 1.0),
                value=float(np.random.randn() * 0.5),
                teacher_action_probs=np.ones(lib.action_dim, dtype=np.float32) / lib.action_dim,
                teacher_parameter_target=np.random.randn(lib.parameter_dim).astype(np.float32) * 0.3,
                teacher_weight=0.0,
                proxy_scores=np.ones(lib.action_dim, dtype=np.float32) * 0.5,
                proxy_prefix_lengths=np.ones(lib.action_dim, dtype=np.float32) * 8.0,
            )
        )

    t0 = time.perf_counter()
    metrics = agent.update()
    elapsed = time.perf_counter() - t0

    return {"update_time_s": elapsed, "metrics": metrics}


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def build_config(soft_mask: bool = True) -> ExperimentConfig:
    config = ExperimentConfig()
    config = replace(config, agent=replace(config.agent, soft_mask_enabled=soft_mask))
    return config


def run_full_benchmark(label: str, soft_mask: bool) -> dict:
    """运行完整 benchmark, 返回结果 dict."""
    config = build_config(soft_mask=soft_mask)
    presets = build_scene_presets()
    vehicle = VehicleConfig()

    print(f"\n{'='*65}")
    print(f"  [{label}]  soft_mask={'ON' if soft_mask else 'OFF'}")
    print(f"{'='*65}")

    # ── 1. 场景生成 ──
    print("  [1/4] 场景生成 benchmark ...")
    factory = OldSceneFactory(presets, vehicle_config=vehicle)
    scene_report = benchmark_all_scene_gen(factory, presets, vehicle, trials=20)

    # ── 2. Episode rollout ──
    print("  [2/4] Episode rollout benchmark (8 集) ...")
    ep_stats = benchmark_episodes(config, episodes=8)

    # ── 3. PPO update ──
    print("  [3/4] PPO update benchmark ...")
    update_stats = benchmark_ppo_update(config, n_transitions=512)

    # ── 4. Low-level micro ──
    print("  [4/4] Low-level micro benchmark ...")
    ll_stats = benchmark_low_level_micro(config)

    return {
        "label": label,
        "soft_mask": soft_mask,
        "scene_gen": scene_report,
        "episodes": ep_stats,
        "update": update_stats,
        "low_level": ll_stats,
    }


def benchmark_low_level_micro(config: ExperimentConfig) -> dict:
    """精细化 low-level 子步骤耗时."""
    env = create_env_adapter(config.env, config.vehicle, config.observation, config.reward)
    task_env = env.env
    task_env.reset(seed=42, options={"level": "Warmup"})
    state = task_env.get_articulated_state()

    from common.types import LowLevelControl
    ctrl = LowLevelControl(articulation_rate=0.0, speed=0.5)

    results = {}
    n = 200

    # kinematics
    t = time_it(n, lambda s=state, c=ctrl: task_env.kinematics.step(s, c))
    results["kinematics_ms"] = np.mean(t) * 1000 / n

    # lidar
    state2 = task_env.get_articulated_state()
    t = time_it(n, lambda s=state2: task_env._lidar_observation(s))
    results["lidar_ms"] = np.mean(t) * 1000

    # collision
    t = time_it(n, lambda s=state2: task_env._intersects_obstacles(s))
    results["collision_ms"] = np.mean(t) * 1000

    # build_obs
    t = time_it(n, lambda: task_env.build_observation())
    results["build_obs_ms"] = np.mean(t) * 1000

    # env.step (full)
    task_env.reset(seed=42, options={"level": "Warmup"})
    action = np.array([0.0, 0.5], dtype=np.float32)
    t = time_it(n, lambda a=action: task_env.step(a))
    results["env_step_full_ms"] = np.mean(t) * 1000

    return results


# ══════════════════════════════════════════════════════════════════════
# 报告渲染
# ══════════════════════════════════════════════════════════════════════

def render_markdown_report(results: list) -> str:
    """将多个 benchmark 结果渲染为 Markdown 对比表格."""
    lines = []
    lines.append("# 训练链路速度对比报告")
    lines.append(f"  **时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Low-level micro ──
    lines.append("## 1. Low-Level Step 子步骤耗时 (单次)")
    lines.append("")
    lines.append("| 子步骤 | " + " | ".join(r["label"] for r in results) + " |")
    lines.append("|--------|" + "|".join("--------|" for _ in results) + "|")
    for key, display in [
        ("kinematics_ms", "kinematics"),
        ("lidar_ms", "LiDAR 射线投射"),
        ("collision_ms", "碰撞检测"),
        ("build_obs_ms", "build_obs (含LiDAR)"),
        ("env_step_full_ms", "env.step 完整"),
    ]:
        vals = [f"{r['low_level'].get(key, 0):.3f}ms" for r in results]
        lines.append(f"| {display} | " + " | ".join(vals) + " |")
    lines.append("")

    # ── 场景生成 ──
    lines.append("## 2. 场景生成耗时 (20 次均值)")
    lines.append("")
    for level in ["Debug", "Warmup", "Normal"]:
        lines.append(f"### {level}")
        lines.append("")
        lines.append("| 指标 | " + " | ".join(r["label"] for r in results) + " |")
        lines.append("|------|" + "|".join("------|" for _ in results) + "|")
        for key, display in [
            ("total", "生成耗时 (ms)"),
            ("obstacle_segments", "障碍段数"),
            ("free_ratio", "free_ratio"),
            ("failures", "失败次数 (/20)"),
        ]:
            vals = []
            for r in results:
                sg = r["scene_gen"].get(level, {})
                arr = sg.get(key, [])
                if key == "failures":
                    vals.append(f"{int(arr[0]) if arr else 0}")
                elif key == "free_ratio":
                    valid = [v for v in arr if not math.isnan(v)]
                    vals.append(f"{np.mean(valid):.3f}" if valid else "—")
                else:
                    valid = [v for v in arr if not math.isnan(v)]
                    if key == "total":
                        vals.append(_fmt_ms(valid) if valid else "—")
                    elif key == "obstacle_segments":
                        vals.append(_fmt_count(valid) if valid else "—")
                    else:
                        vals.append(f"{np.mean(valid):.1f}" if valid else "—")
            lines.append(f"| {display} | " + " | ".join(vals) + " |")
        lines.append("")

    # ── Episode ──
    lines.append("## 3. Episode Rollout 统计 (8 集)")
    lines.append("")
    lines.append("| 指标 | " + " | ".join(r["label"] for r in results) + " |")
    lines.append("|------|" + "|".join("------|" for _ in results) + "|")
    for key, display, fmt_fn in [
        ("episode_time_s", "单集耗时", _fmt_s),
        ("macro_steps", "Macro 步数/集", _fmt_count),
        ("low_level_steps", "Low-level 步数/集", _fmt_count),
        ("success", "成功率", lambda v: f"{np.mean(v)*100:.0f}%" if v else "—"),
        ("collision", "碰撞率", lambda v: f"{np.mean(v)*100:.0f}%" if v else "—"),
        ("reward", "平均 Reward", lambda v: f"{np.mean(v):.1f}" if v else "—"),
    ]:
        vals = []
        for r in results:
            arr = r["episodes"].get(key, [])
            vals.append(fmt_fn(arr) if arr else "—")
        lines.append(f"| {display} | " + " | ".join(vals) + " |")
    lines.append("")

    # ── PPO Update ──
    lines.append("## 4. PPO Update 耗时 (512 transitions)")
    lines.append("")
    lines.append("| 指标 | " + " | ".join(r["label"] for r in results) + " |")
    lines.append("|------|" + "|".join("------|" for _ in results) + "|")
    for key, display in [
        ("update_time_s", "Update 耗时 (s)"),
    ]:
        vals = [f"{r['update'].get(key, 0):.3f}s" for r in results]
        lines.append(f"| {display} | " + " | ".join(vals) + " |")

    # metrics
    lines.append("| 指标 | " + " | ".join(r["label"] for r in results) + " |")
    lines.append("|------|" + "|".join("------|" for _ in results) + "|")
    metric_keys = list(results[0]["update"]["metrics"].keys())
    for mk in metric_keys[:10]:
        vals = [f"{r['update']['metrics'].get(mk, 0):.4f}" for r in results]
        lines.append(f"| {mk} | " + " | ".join(vals) + " |")
    lines.append("")

    # ── 汇总 ──
    lines.append("## 5. 汇总估算")
    lines.append("")
    lines.append("| 指标 | " + " | ".join(r["label"] for r in results) + " |")
    lines.append("|------|" + "|".join("------|" for _ in results) + "|")
    for r in results:
        ep_time = np.mean(r["episodes"].get("episode_time_s", [0]))
        macro_steps = np.mean(r["episodes"].get("macro_steps", [0]))
        ll_steps = np.mean(r["episodes"].get("low_level_steps", [0]))
        reset_time = 0.05  # ~50ms
        ppo_time = r["update"].get("update_time_s", 0)
        total_per_ep = ep_time / (128 / 8)  # amortize PPO over episodes_per_update
        lines.append(
            f"| {r['label']} | "
            f"reset={reset_time*1000:.0f}ms/ep | "
            f"macro={macro_steps:.0f}步, ll={ll_steps:.0f}步 | "
            f"单集={ep_time:.3f}s | "
            f"PPO更新={ppo_time:.3f}s |"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full training pipeline benchmark")
    parser.add_argument("--mode", choices=["all", "soft-on", "soft-off"], default="all")
    parser.add_argument("--output", type=str, default="", help="Save report to file")
    args = parser.parse_args()

    results = []

    if args.mode in ("all", "soft-on"):
        results.append(run_full_benchmark("SoftMask ON", soft_mask=True))

    if args.mode in ("all", "soft-off"):
        results.append(run_full_benchmark("SoftMask OFF", soft_mask=False))

    report = render_markdown_report(results)
    print("\n" + report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
