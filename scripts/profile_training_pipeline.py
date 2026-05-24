#!/usr/bin/env python3
"""训练链路耗时 profile 脚本。

使用方法:
    cd /home/cyberbus/Public/HybridPPO_ArticulatedVehicle
    python scripts/profile_training_pipeline.py --episodes 8 --seed 42

可选开关:
    --no-soft-mask   关闭 soft mask (proxy safety)
    --level WARMUP   只测 Warmup 场景
    --level NORMAL   只测 Normal 场景
"""

from __future__ import annotations

import argparse
import time
import sys
import os
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

# 确保项目 src 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.runtime_config import ExperimentConfig
from env.adapter import create_env_adapter
from env.macro_wrapper import ParameterizedMacroActionWrapper
from model.agent import HybridPPOAgent
from primitives import ParameterizedPrimitiveExecutor, build_default_primitive_library, load_proxy_safety_sidecar
from training.rollout import MacroRolloutDriver


# ═══════════════════════════════════════════════════════════════════════════
# 精细计时工具
# ═══════════════════════════════════════════════════════════════════════════

class Timer:
    """微秒级计时器，支持嵌套标签。"""

    def __init__(self):
        self._records: Dict[str, List[float]] = defaultdict(list)
        self._stack: List[Tuple[str, float]] = []

    def start(self, label: str):
        self._stack.append((label, time.perf_counter()))

    def stop(self):
        label, t0 = self._stack.pop()
        elapsed = time.perf_counter() - t0
        self._records[label].append(elapsed)
        return elapsed

    def summary(self, sort_by: str = "total") -> str:
        lines = []
        total_all = sum(sum(v) for v in self._records.values())
        lines.append(f"{'Label':<45} {'calls':>7}  {'total(s)':>10}  {'mean(ms)':>10}  {'%':>7}")
        lines.append("-" * 85)

        items = []
        for label, vals in self._records.items():
            total = sum(vals)
            mean_ms = (total / len(vals)) * 1000 if vals else 0
            pct = (total / total_all * 100) if total_all > 0 else 0
            items.append((label, len(vals), total, mean_ms, pct))

        if sort_by == "total":
            items.sort(key=lambda x: -x[2])
        elif sort_by == "mean":
            items.sort(key=lambda x: -x[3])
        elif sort_by == "calls":
            items.sort(key=lambda x: -x[1])

        for label, calls, total, mean_ms, pct in items:
            lines.append(f"  {label:<43} {calls:>7d}  {total:>10.4f}  {mean_ms:>10.3f}  {pct:>6.1f}%")

        lines.append("-" * 85)
        lines.append(f"  {'TOTAL':<43} {'':>7}  {total_all:>10.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 带计时的包装器
# ═══════════════════════════════════════════════════════════════════════════

class TimedEnvAdapter:
    """包装 env adapter，对 reset / step / build_observation 计时。"""

    def __init__(self, env, timer: Timer):
        self._env = env
        self._timer = timer

    @property
    def observation_dim(self) -> int:
        return self._env.observation_dim

    def reset(self, seed=None, options=None):
        self._timer.start("env.reset")
        result = self._env.reset(seed=seed, options=options)
        self._timer.stop()
        return result

    def step(self, action):
        self._timer.start("env.step_total")
        result = self._env.step(action)
        self._timer.stop()
        return result

    def build_observation(self):
        self._timer.start("env.build_obs")
        result = self._env.build_observation()
        self._timer.stop()
        return result

    def get_articulated_state(self):
        return self._env.get_articulated_state()

    def get_goal_state(self):
        return self._env.get_goal_state()

    def current_info(self):
        return self._env.current_info()

    def distance_to_goal(self, state=None):
        return self._env.distance_to_goal(state)

    def make_primitive_context(self):
        self._timer.start("env.make_context")
        result = self._env.make_primitive_context()
        self._timer.stop()
        return result


class TimedMacroWrapper:
    """包装 macro wrapper，对 macro_env.step 计时。"""

    def __init__(self, macro_env, timer: Timer):
        self._macro_env = macro_env
        self._timer = timer
        self.env = macro_env.env

    def step(self, macro_action, start_state=None, context=None):
        self._timer.start("macro.step")
        result = self._macro_env.step(macro_action, start_state=start_state, context=context)
        self._timer.stop()
        return result


class TimedAgent:
    """包装 agent，对 act / store_transition 计时。"""

    def __init__(self, agent: HybridPPOAgent, timer: Timer):
        self._agent = agent
        self._timer = timer
        self.config = agent.config
        self.buffer = agent.buffer
        self.policy = agent.policy
        self.value_net = agent.value_net
        self.primitive_library = agent.primitive_library
        self.device = agent.device

    def act(self, observation, deterministic=False, teacher_action_probs=None, teacher_weight=0.0):
        self._timer.start("agent.act")
        result = self._agent.act(
            observation, deterministic=deterministic,
            teacher_action_probs=teacher_action_probs,
            teacher_weight=teacher_weight,
        )
        self._timer.stop()
        return result

    def store_transition(self, transition):
        self._timer.start("agent.store")
        self._agent.store_transition(transition)
        self._timer.stop()

    def estimate_value(self, observation):
        return self._agent.estimate_value(observation)

    def update(self):
        return self._agent.update()

    def checkpoint_state(self):
        return self._agent.checkpoint_state()

    def load_checkpoint_state(self, state):
        return self._agent.load_checkpoint_state(state)


# ═══════════════════════════════════════════════════════════════════════════
# 对 low-level step 的精细 profile（单独跑一集）
# ═══════════════════════════════════════════════════════════════════════════

def profile_low_level_steps(env, num_steps: int = 200):
    """精细测量每个 low-level step 的各子模块耗时。"""
    env.reset(seed=42, options={"level": "Warmup"})

    timers = {
        "ll_kinematics": [],
        "ll_lidar": [],
        "ll_collision": [],
        "ll_reward": [],
        "ll_obs_build": [],
        "ll_total": [],
    }

    for _ in range(num_steps):
        action = np.array([0.0, 0.5], dtype=np.float32)

        t0 = time.perf_counter()
        result = env.step(action)
        timers["ll_total"].append(time.perf_counter() - t0)

    return timers


# ═══════════════════════════════════════════════════════════════════════════
# 主 profile 逻辑
# ═══════════════════════════════════════════════════════════════════════════

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    if args.no_soft_mask:
        config = replace(config, agent=replace(config.agent, soft_mask_enabled=False))
    if args.no_proxy:
        config = replace(config, proxy_safety=replace(config.proxy_safety, sidecar_path=""))
    return config


def run_profile(args: argparse.Namespace):
    config = build_config(args)
    timer = Timer()

    # ── 初始化 ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  Training Pipeline Profiler")
    print("=" * 70)
    print(f"  device         : {config.device}")
    print(f"  soft_mask      : {config.agent.soft_mask_enabled}")
    print(f"  proxy_sidecar  : {'loaded' if config.proxy_safety.sidecar_path else 'NONE'}")
    print(f"  teacher        : {config.teacher_enabled}")
    print(f"  episodes       : {args.episodes}")
    print(f"  level          : {args.level}")
    print("-" * 70)

    timer.start("00_init_total")

    timer.start("01_load_proxy_sidecar")
    proxy_sidecar = None
    proxy_sidecar_path = str(config.proxy_safety.sidecar_path).strip()
    if proxy_sidecar_path:
        proxy_sidecar = load_proxy_safety_sidecar(proxy_sidecar_path)
    timer.stop()

    timer.start("02_build_primitive_library")
    primitive_library = build_default_primitive_library(proxy_sidecar=proxy_sidecar)
    timer.stop()

    timer.start("03_create_agent")
    agent = HybridPPOAgent(
        config=config.agent.build(
            observation_dim=int(config.observation.observation_dim),
            action_dim=int(primitive_library.action_dim),
            parameter_dim=int(primitive_library.parameter_dim),
        ),
        primitive_library=primitive_library,
        device="cpu",
    )
    timer.stop()

    timer.start("04_create_env")
    raw_env = create_env_adapter(
        env_config=config.env,
        vehicle_config=config.vehicle,
        observation_config=config.observation,
        reward_config=config.reward,
    )
    timed_env = TimedEnvAdapter(raw_env, timer)
    timer.stop()

    timer.start("05_create_executor")
    executor = ParameterizedPrimitiveExecutor(
        primitive_library,
        vehicle_config=config.vehicle,
        executor_config=config.primitive_executor,
    )
    timer.stop()

    timer.start("06_create_macro_env")
    raw_macro = ParameterizedMacroActionWrapper(timed_env, executor, gamma=agent.config.gamma)
    macro_env = TimedMacroWrapper(raw_macro, timer)
    timer.stop()

    timer.start("07_create_rollout_driver")
    timed_agent = TimedAgent(agent, timer)
    rollout = MacroRolloutDriver(
        env=timed_env,
        macro_env=macro_env,
        agent=timed_agent,
        max_macro_steps=config.schedule.max_macro_steps_per_episode,
        soft_teacher=None,
    )
    timer.stop()

    timer.stop()  # 00_init_total

    # ── 运行 episodes ───────────────────────────────────────────────
    timer.start("08_episodes_total")
    level = args.level if args.level else "Warmup"

    for ep in range(args.episodes):
        timer.start(f"ep_{ep}")
        summary = rollout.collect_episode(
            level=level,
            seed=int(config.seed + ep),
            deterministic=False,
            store_transition=False,
        )
        timer.stop()
        print(f"  ep {ep:>3d} | macro={summary.macro_steps:>3d}  "
              f"ll={summary.low_level_steps:>4d}  "
              f"reward={summary.total_reward:>7.1f}  "
              f"done={summary.done_reason:<12s}  "
              f"dist={summary.final_goal_distance:.2f}")

    timer.stop()  # 08_episodes_total

    # ── 输出报告 ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  TIME BREAKDOWN (by total time)")
    print("=" * 70)
    print(timer.summary(sort_by="total"))

    # ── 低层级补充分析 ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  LOW-LEVEL STEP BREAKDOWN (separate instrumented run)")
    print("=" * 70)
    ll_timers = profile_low_level_steps(raw_env, num_steps=args.ll_steps)
    for key in ["ll_total", "ll_kinematics", "ll_lidar", "ll_collision", "ll_reward", "ll_obs_build"]:
        vals = ll_timers.get(key, [])
        if vals:
            print(f"  {key:<20}  n={len(vals):>4d}  total={sum(vals):.4f}s  "
                  f"mean={np.mean(vals)*1000:.3f}ms  "
                  f"median={np.median(vals)*1000:.3f}ms  "
                  f"p99={np.percentile(vals,99)*1000:.3f}ms")

    # ── 场景障碍段数统计 ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  OBSTACLE SEGMENT COUNT (per scene type)")
    print("=" * 70)
    for lvl in ["Debug", "Warmup", "Normal"]:
        for _ in range(5):
            try:
                raw_env.reset(seed=args.seed, options={"level": lvl})
                info = raw_env.current_info()
                seg_count = 0
                # Access the underlying env's obstacle segments
                task_env = raw_env.env  # KinematicTaskAdapter wraps KinematicTaskEnv directly
                seg_count = len(task_env._obstacle_segment_x1)
                corridor_w = info.get("corridor_width", "?")
                scene_type = info.get("scene_type", info.get("level", "?"))
                print(f"  {lvl:<8}  corridor_width={corridor_w}  obstacle_segments={seg_count}")
                break
            except RuntimeError:
                continue

    return timer


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Profile training pipeline latency")
    parser.add_argument("--episodes", type=int, default=4, help="Number of episodes to run")
    parser.add_argument("--ll-steps", type=int, default=100, help="Low-level steps for fine-grained profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=str, default="", choices=["", "Debug", "Warmup", "Normal"])
    parser.add_argument("--no-soft-mask", action="store_true", help="Disable soft mask (proxy safety)")
    parser.add_argument("--no-proxy", action="store_true", help="Don't load proxy safety sidecar at all")
    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
