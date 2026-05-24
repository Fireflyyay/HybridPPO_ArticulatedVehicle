#!/usr/bin/env python3
"""Old vs New 场景生成 + 训练链路速度对比。

对比维度:
  1. 场景生成: 各 level 的 reset 耗时、障碍段数、重试次数
  2. Episode rollout: 平均 macro 步数、low-level 步数、单集耗时
  3. Low-level micro: kinematics / LiDAR / collision / build_obs 每一步耗时
"""

from __future__ import annotations

import gc
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from common.runtime_config import (
    ExperimentConfig,
    build_scene_presets,
    SceneLevelConfig,
)
from common.config import VehicleConfig
from common.types import ArticulatedState, LowLevelControl
from env.adapter import create_env_adapter
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library, load_proxy_safety_sidecar
from env.macro_wrapper import ParameterizedMacroActionWrapper
from training.rollout import MacroRolloutDriver


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _ms(vals: List[float]) -> str:
    a = np.array([v for v in vals if not math.isnan(v)])
    if len(a) == 0:
        return "—"
    return f"{np.mean(a)*1000:.1f}ms (σ={np.std(a)*1000:.1f})"


def _count(vals: List[float]) -> str:
    a = np.array([v for v in vals if not math.isnan(v)])
    if len(a) == 0:
        return "—"
    return f"μ={np.mean(a):.1f} (min={np.min(a):.0f} max={np.max(a):.0f})"


def _pct(vals: List[float]) -> str:
    a = np.array([v for v in vals if not math.isnan(v)])
    if len(a) == 0:
        return "—"
    return f"{np.mean(a)*100:.0f}%"


# ═══════════════════════════════════════════════════════════════════════
# 1. Scene generation benchmark
# ═══════════════════════════════════════════════════════════════════════

def bench_scene_gen(factory, presets, vehicle, trials: int = 30):
    """返回 {level: {metric: [values]}}."""
    results = {}
    for level in ["Debug", "Warmup", "Normal"]:
        cfg = presets.get(level, SceneLevelConfig())
        metrics: Dict[str, List[float]] = defaultdict(list)
        failures = 0
        for i in range(trials):
            rng = np.random.default_rng(42 + i)
            opts = {"warmup_progress": 0.5} if level == "Warmup" else None
            gc.collect()
            t0 = time.perf_counter()
            try:
                scene = factory.generate(level, rng, options=opts)
            except RuntimeError:
                failures += 1
                metrics["time_s"].append(float("nan"))
                continue
            elapsed = time.perf_counter() - t0
            metrics["time_s"].append(elapsed)

            # 障碍段数
            segs = 0
            for p in scene.obstacles:
                segs += len(p.exterior.coords) - 1
                for interior in p.interiors:
                    segs += len(interior.coords) - 1
            metrics["obstacle_segments"].append(float(segs))

            # polygon 数
            metrics["polygon_count"].append(float(len(scene.obstacles)))

            # free_ratio (如果 metadata 包含)
            fr = scene.metadata.get("free_ratio")
            if fr is not None:
                metrics["free_ratio"].append(float(fr))

            # corridor_width
            cw = scene.metadata.get("corridor_width")
            if cw is not None:
                metrics["corridor_width"].append(float(cw))

        metrics["failures"] = [float(failures)]
        results[level] = dict(metrics)
    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. Episode rollout benchmark
# ═══════════════════════════════════════════════════════════════════════

def bench_episodes(config: ExperimentConfig, soft_mask: bool, episodes: int = 10):
    config = replace(config, agent=replace(config.agent, soft_mask_enabled=soft_mask))
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
    roller = MacroRolloutDriver(env=env, macro_env=macro_env, agent=agent, max_macro_steps=64, soft_teacher=None)

    stats: Dict[str, List[float]] = defaultdict(list)
    for ep in range(episodes):
        level = "Warmup" if ep < episodes // 2 else "Normal"
        opts = {"level": level}
        if level == "Warmup":
            opts["warmup_progress"] = 0.5
        gc.collect()
        t0 = time.perf_counter()
        s = roller.collect_episode(level=level, seed=int(config.seed + ep), deterministic=False, store_transition=False, reset_options=opts)
        elapsed = time.perf_counter() - t0
        stats["ep_time_s"].append(elapsed)
        stats["macro_steps"].append(float(s.macro_steps))
        stats["low_level_steps"].append(float(s.low_level_steps))
        stats["success"].append(float(s.success))
        stats["collision"].append(float(s.collision))
        stats["reward"].append(float(s.total_reward))

    return dict(stats)


# ═══════════════════════════════════════════════════════════════════════
# 3. Low-level micro
# ═══════════════════════════════════════════════════════════════════════

def bench_low_level(config: ExperimentConfig, n: int = 300):
    env = create_env_adapter(config.env, config.vehicle, config.observation, config.reward)
    te = env.env
    te.reset(seed=42, options={"level": "Warmup"})
    state = te.get_articulated_state()
    ctrl = LowLevelControl(articulation_rate=0.0, speed=0.5)

    out = {}

    # kinematics
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        state = te.kinematics.step(state, ctrl)
        t.append(time.perf_counter() - t0)
    out["kinematics_us"] = np.mean(t) * 1e6

    # lidar
    s2 = te.get_articulated_state()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        te._lidar_observation(s2)
        t.append(time.perf_counter() - t0)
    out["lidar_us"] = np.mean(t) * 1e6

    # collision
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        te._intersects_obstacles(s2)
        t.append(time.perf_counter() - t0)
    out["collision_us"] = np.mean(t) * 1e6

    # build_obs
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        te.build_observation()
        t.append(time.perf_counter() - t0)
    out["build_obs_us"] = np.mean(t) * 1e6

    # env.step
    te.reset(seed=42, options={"level": "Warmup"})
    action = np.array([0.0, 0.5], dtype=np.float32)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        te.step(action)
        t.append(time.perf_counter() - t0)
    out["env_step_us"] = np.mean(t) * 1e6

    return out


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

def print_report(old_data: dict, new_data: dict = None):
    """Print Markdown comparison table. new_data 为 None 时只输出 old."""
    multi = new_data is not None
    labels = ["OLD (当前代码)"] if not multi else ["OLD (当前代码)", "NEW (commit)"]
    datasets = [old_data] if not multi else [old_data, new_data]

    print("\n" + "=" * 75)
    print("  训练链路速度对比报告")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # ── Low-level ──
    print("\n## 1. Low-Level Step 子步骤耗时\n")
    header = "| 子步骤 |" + "|".join(f" {l} |" for l in labels) + ""
    sep = "|--------|" + "|".join("--------|" for _ in labels) + ""
    print(header)
    print(sep)
    for key, name in [
        ("kinematics_us", "kinematics"),
        ("lidar_us", "LiDAR (108 beams)"),
        ("collision_us", "碰撞检测"),
        ("build_obs_us", "build_obs (含LiDAR)"),
        ("env_step_us", "env.step (完整)"),
    ]:
        vals = []
        for d in datasets:
            v = d["low_level"].get(key, 0)
            vals.append(f"**{v:.1f}μs**" if v > 0 else "—")
        print(f"| {name} |" + "|".join(f" {v} |" for v in vals) + "")

    # ── Scene gen ──
    print("\n## 2. 场景生成 (各 level 30 次)\n")
    for level in ["Debug", "Warmup", "Normal"]:
        print(f"### {level}\n")
        print(header.replace("子步骤", "指标"))
        print(sep)
        for key, name, fmt in [
            ("time_s", "生成耗时", _ms),
            ("obstacle_segments", "障碍段数", _count),
            ("polygon_count", "障碍多边形数", _count),
            ("free_ratio", "free_ratio", _count),
            ("failures", "失败/30", lambda v: f"{int(v[0]) if v else 0}"),
        ]:
            vals = []
            for d in datasets:
                sg = d["scene"].get(level, {})
                arr = sg.get(key, [])
                v = fmt(arr) if arr else "—"
                # highlight significant differences
                vals.append(v)
            print(f"| {name} |" + "|".join(f" {v} |" for v in vals) + "")
        print("")

    # ── Episode ──
    print("\n## 3. Episode Rollout (各 10 集, SoftMask ON)\n")
    print(header.replace("子步骤", "指标"))
    print(sep)
    for key, name, fmt in [
        ("ep_time_s", "单集耗时", lambda v: _ms([x*1000 for x in v])),  # hack for ms display
        ("macro_steps", "Macro steps/集", _count),
        ("low_level_steps", "Low-level steps/集", _count),
        ("success", "成功率", _pct),
        ("collision", "碰撞率", _pct),
        ("reward", "平均 Reward", lambda v: f"{np.mean(v):.1f}" if v else "—"),
    ]:
        vals = []
        for d in datasets:
            arr = d["episodes_soft_on"].get(key, [])
            vals.append(fmt(arr) if arr else "—")
        print(f"| {name} |" + "|".join(f" {v} |" for v in vals) + "")

    # ── 汇总估算 ──
    if multi:
        print("\n## 4. 速度对比汇总\n")
        for d, label in zip(datasets, labels):
            ep = d["episodes_soft_on"]
            ep_time = np.mean([v for v in ep["ep_time_s"] if not math.isnan(v)])
            macro = np.mean(ep["macro_steps"])
            ll = np.mean(ep["low_level_steps"])
            reset = np.mean([v for v in d["scene"]["Warmup"]["time_s"] if not math.isnan(v)])
            print(f"  **{label}**: reset={reset*1000:.0f}ms | macro={macro:.0f}步 | ll={ll:.0f}步 | 单集={ep_time:.3f}s")

        old_ep = np.mean([v for v in old_data["episodes_soft_on"]["ep_time_s"] if not math.isnan(v)])
        new_ep = np.mean([v for v in new_data["episodes_soft_on"]["ep_time_s"] if not math.isnan(v)])
        speedup = old_ep / new_ep if new_ep > 0 else 1.0
        print(f"\n  **NEW 比 OLD 快 {speedup:.2f}×** (按单集平均耗时)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def collect_all(import_name: str, soft_mask: bool = True) -> dict:
    """Collect all benchmark data for one scene generator version."""
    # Dynamically import the scene factory
    if import_name == "env.scenes":
        from env.scenes import BaselineInspiredSceneFactory as Factory
    else:
        import importlib
        mod = importlib.import_module(import_name)
        Factory = mod.BaselineInspiredSceneFactory

    config = ExperimentConfig()
    presets = build_scene_presets()
    vehicle = VehicleConfig()

    print(f"\n  >>> Benchmarking scene gen: {import_name} ...")
    factory = Factory(presets, vehicle_config=vehicle)
    scene_data = bench_scene_gen(factory, presets, vehicle, trials=30)

    print(f"  >>> Benchmarking episodes (soft_mask=ON) ...")
    ep_on = bench_episodes(config, soft_mask=True, episodes=10)

    print(f"  >>> Benchmarking low-level micro ...")
    ll = bench_low_level(config, n=300)

    return {"scene": scene_data, "episodes_soft_on": ep_on, "low_level": ll}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["old", "new", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("old", "both"):
        print("\n" + "=" * 60)
        print("  PHASE 1: OLD scene generation (当前代码)")
        print("=" * 60)
        old_data = collect_all("env.scenes", soft_mask=True)

    if args.mode in ("new", "both"):
        print("\n" + "=" * 60)
        print("  PHASE 2: NEW scene generation (commit 76aa860)")
        print("=" * 60)
        # Temporarily swap scenes.py
        scenes_path = os.path.join(_PROJECT_ROOT, "src", "env", "scenes.py")
        backup_path = scenes_path + ".backup_bench"
        new_scenes_path = os.path.join(_PROJECT_ROOT, "src", "env", "scenes_new.py")

        if not os.path.exists(new_scenes_path):
            print("  ERROR: scenes_new.py not found. Please save the commit version as scenes_new.py first.")
            print("  Run: cp src/env/scenes.py src/env/scenes_new.py  # then manually apply commit changes")
            sys.exit(1)

        # Swap
        shutil.copy(scenes_path, backup_path)
        shutil.copy(new_scenes_path, scenes_path)

        # Invalidate import cache
        for mod_name in list(sys.modules.keys()):
            if "env.scenes" in mod_name:
                del sys.modules[mod_name]

        try:
            new_data = collect_all("env.scenes", soft_mask=True)
        finally:
            # Restore
            shutil.copy(backup_path, scenes_path)
            os.remove(backup_path)

    if args.mode == "both":
        print_report(old_data, new_data)
    elif args.mode == "old":
        print_report(old_data)
    else:
        print_report(new_data)


if __name__ == "__main__":
    main()
