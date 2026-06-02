#!/usr/bin/env python3
"""Diagnose soft mask action filtering quality — extended edition.

Targets score distribution, STOP_CHECK handling, false-kill / missed-invalid
source location, and verified region coverage.

Run:
    PYTHONPATH=src python scripts/diagnose_soft_mask.py \
        --sidecar data/proxy_safety_sidecar.npz \
        --scenes 5 --grid-spacing 8.0 \
        --output results/soft_mask_diag.json \
        --report results/soft_mask_diag.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_SEMANTIC_LABELS: Dict[int, str] = {
    0: "FORWARD_LEFT",
    1: "FORWARD_RIGHT",
    2: "REVERSE_LEFT",
    3: "REVERSE_RIGHT",
    4: "REVERSE_ALIGN",
    5: "ARTICULATION_RECOVER",
    6: "STRAIGHT_ADJUST",
    7: "STOP_CHECK",
}

_SCORE_BUCKETS = [
    (0.0, 0.01, "approx_zero"),
    (0.01, 0.1, "0.01-0.10"),
    (0.1, 0.3, "0.10-0.30"),
    (0.3, 0.7, "0.30-0.70"),
    (0.7, 0.999, "0.70-1.00"),
    (0.999, 1.001, "approx_one"),
]

_VEHICLE_HALF_WIDTH = 3.43 / 2.0

_DEFAULT_MIDPOINT_PARAMS: Dict[str, float] = {
    "path_length": 2.0,
    "duration": 1.5,
    "speed_scale": 0.5,
    "omega_scale": 0.3,
    "phi_target": 0.0,
    "heading_tolerance": 0.15,
    "position_tolerance": 0.4,
    "smoothness": 0.3,
}


def _ensure_imports() -> None:
    _src = os.path.join(os.path.dirname(__file__), "..", "src")
    if _src not in sys.path:
        sys.path.insert(0, os.path.abspath(_src))


_ensure_imports()
from common.config import PrimitiveExecutorConfig, VehicleConfig
from common.runtime_config import EnvRuntimeConfig, ObservationConfig, RewardConfig
from common.types import ArticulatedState, PrimitiveExecutionContext, wrap_to_pi
from env.adapter import create_env_adapter
from env.success import articulated_body_polygons
from primitives import (
    ParameterizedPrimitiveExecutor,
    ParameterizedPrimitiveLibrary,
    SemanticPrimitive,
    build_default_primitive_library,
    load_proxy_safety_sidecar,
    ProxySafetySidecar,
)
from shapely.geometry import Point, LineString


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionDetail:
    action_id: int
    action_label: str
    oracle_valid: bool
    oracle_collision_step: Optional[int] = None
    oracle_collision_body: str = ""
    proxy_score: float = 0.0
    proxy_has_safe: bool = False
    proxy_prefix_max: float = 0.0
    proxy_horizon: int = 1
    state_gate_allowed: bool = True
    state_gate_reason: str = ""


@dataclass
class PositionResult:
    scene_idx: int
    x: float
    y: float
    front_heading: float
    rear_heading: float
    articulation_angle: float
    region: str
    pose_relation: str
    min_clearance_norm: float = 1.0
    min_clearance_m: float = 30.0
    goal_distance_norm: float = 0.0
    relative_heading_rad: float = 0.0
    actions: List[ActionDetail] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Region classification — improved
# ---------------------------------------------------------------------------

def _sampled_free_grid(
    obstacle_union, world_bounds: Tuple[float, float, float, float], spacing: float
) -> List[Tuple[float, float]]:
    """Sample free-space points on a regular grid, checking multiple orientations."""
    xmin, xmax, ymin, ymax = world_bounds
    cfg = VehicleConfig()
    pts: List[Tuple[float, float]] = []
    xs = np.arange(xmin + 3.0, xmax - 3.0, spacing)
    ys = np.arange(ymin + 3.0, ymax - 3.0, spacing)
    test_headings = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    for x in xs:
        for y in ys:
            collision_free = False
            for fh in test_headings:
                test_state = ArticulatedState(x=float(x), y=float(y), front_heading=fh, rear_heading=fh, speed=0.0, articulation_rate=0.0)
                front_poly, rear_poly = articulated_body_polygons(test_state, cfg)
                collides = False
                if obstacle_union is not None:
                    if front_poly.intersects(obstacle_union) or rear_poly.intersects(obstacle_union):
                        collides = True
                wb = box_impl(world_bounds)
                if not wb.covers(front_poly) or not wb.covers(rear_poly):
                    collides = True
                if not collides:
                    collision_free = True
                    break
            if collision_free:
                pts.append((float(x), float(y)))
    return pts


def box_impl(bounds: Tuple[float, float, float, float]):
    from shapely.geometry import box as _box
    return _box(float(bounds[0]), float(bounds[2]), float(bounds[1]), float(bounds[3]))


def _targeted_positions_from_scene(task_env, obstacle_union) -> List[Tuple[float, float, str]]:
    """Generate positions near scene start/goal and corridor center for coverage."""
    scene = task_env._scene
    pts: List[Tuple[float, float, str]] = []
    xmin, xmax, ymin, ymax = scene.world_bounds

    # Corridor centerline at various y
    main_x = float(scene.start_state.x)
    for y_frac in np.linspace(0.15, 0.85, 6):
        y = float(ymin + (ymax - ymin) * y_frac)
        offs = [-3.0, 0.0, 3.0]
        for ox in offs:
            px, py = main_x + ox, y
            test = ArticulatedState(x=px, y=py, front_heading=np.pi/2, rear_heading=np.pi/2, speed=0.0, articulation_rate=0.0)
            if not task_env.predict_collision(test):
                pts.append((px, py, "centerline"))

    # Near start state
    sx, sy = float(scene.start_state.x), float(scene.start_state.y)
    for dx, dy in [(0,0), (2,0), (-2,0), (0,2), (0,-2), (3,1), (-3,1), (0,4), (0,-4)]:
        px, py = sx+dx, sy+dy
        if xmin+3 < px < xmax-3 and ymin+3 < py < ymax-3:
            test = ArticulatedState(x=px, y=py, front_heading=np.pi/2, rear_heading=np.pi/2, speed=0.0, articulation_rate=0.0)
            if not task_env.predict_collision(test):
                pts.append((px, py, "near_start"))

    # Near goal state
    gx, gy = float(scene.goal_state.x), float(scene.goal_state.y)
    for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1), (2,0), (-2,0), (0,2), (0,-2), (3,0), (-3,0)]:
        px, py = gx+dx, gy+dy
        if xmin+3 < px < xmax-3 and ymin+3 < py < ymax-3:
            test = ArticulatedState(x=px, y=py, front_heading=np.pi/2, rear_heading=np.pi/2, speed=0.0, articulation_rate=0.0)
            if not task_env.predict_collision(test):
                pts.append((px, py, "near_goal"))

    return pts
    from shapely.geometry import box as _box
    return _box(float(bounds[0]), float(bounds[2]), float(bounds[1]), float(bounds[3]))


def _section_depths(
    cx: float, cy: float, obstacle_union, max_dist: float = 30.0
) -> List[float]:
    """Return free-space depth in 8 directions (every 45°)."""
    depths = []
    for k in range(8):
        angle = k * np.pi / 4.0
        ray = LineString([
            (cx, cy),
            (cx + np.cos(angle) * max_dist, cy + np.sin(angle) * max_dist),
        ])
        if obstacle_union is not None and not obstacle_union.is_empty:
            inter = ray.intersection(obstacle_union)
            if not inter.is_empty:
                d = Point(cx, cy).distance(inter)
                depths.append(float(d))
                continue
        depths.append(float(max_dist))
    return depths


def _region_from_depths(depths: List[float]) -> str:
    """Classify a point into a region based on 8-directional free-space depths.

    Heuristics tuned for Normal scenes (main corridor + branches + bays):
    - bay_inside: walls on 6+ sides, at most 1 wide direction (the bay mouth)
    - bay_exit: 2-3 wide directions with at least 1 narrow pair adjacent to wide
    - corridor_straight: 2 opposite wide directions, rest narrow
    - corridor_corner: 3+ wide directions or 2 adjacent wide + 1-2 more
    - endpoint_redirection: exactly 1 wide direction
    """
    narrow_mask = [d < 5.5 for d in depths]
    wide_mask = [d > 10.0 for d in depths]
    narrow_count = sum(narrow_mask)
    wide_count = sum(wide_mask)
    mid_count = len(depths) - narrow_count - wide_count

    # Extreme tight space → bay inside
    if narrow_count >= 6 and wide_count <= 1:
        return "bay_inside"

    # Exactly one exit direction → endpoint or bay
    if wide_count == 1:
        wi = next(i for i, w in enumerate(wide_mask) if w)
        narrow_opposite = narrow_mask[(wi + 4) % 8]
        if narrow_opposite and narrow_count >= 4:
            return "bay_inside"
        return "endpoint_redirection"

    if wide_count == 2:
        wi = [i for i, w in enumerate(wide_mask) if w]
        opposite = abs(wi[0] - wi[1]) == 4
        if opposite:
            if narrow_count >= 4:
                return "corridor_straight"
            else:
                return "bay_exit"
        else:
            # Two adjacent wide → corner or bay exit
            if narrow_count >= 3:
                return "corridor_corner"
            return "bay_exit"

    if wide_count == 3:
        # Check if 2 of them are opposite (corridor crossing junction)
        wi = [i for i, w in enumerate(wide_mask) if w]
        has_opp = any(abs(wi[a] - wi[b]) == 4 for a in range(3) for b in range(a + 1, 3))
        if has_opp:
            return "corridor_corner"
        return "bay_exit"

    if wide_count >= 4:
        return "corridor_corner"

    if wide_count == 0:
        if narrow_count >= 7:
            return "bay_inside"
        if narrow_count <= 3:
            return "corridor_corner"
        return "bay_inside"

    return "unknown"


def classify_position_improved(
    x: float, y: float, obstacle_union, world_bounds,
) -> str:
    """Improved region classifier using 8-directional depth analysis."""
    return _region_from_depths(_section_depths(x, y, obstacle_union))


def classify_pose_relation(front_heading: float, articulation_angle: float) -> str:
    abs_phi = abs(articulation_angle)
    if abs_phi > 0.45:
        return "high_articulation"
    heading_mod = wrap_to_pi(front_heading)
    if abs(heading_mod) < np.deg2rad(25.0) or abs(abs(heading_mod) - np.pi) < np.deg2rad(25.0):
        return "aligned"
    if abs(abs(heading_mod) - np.pi / 2) < np.deg2rad(25.0):
        return "perpendicular"
    return "opposite"


# ---------------------------------------------------------------------------
# Oracle collision checking — extended
# ---------------------------------------------------------------------------

def _make_executor() -> ParameterizedPrimitiveExecutor:
    return ParameterizedPrimitiveExecutor(
        build_default_primitive_library(),
        vehicle_config=VehicleConfig(),
        executor_config=PrimitiveExecutorConfig(max_macro_steps=32),
    )


def oracle_collision_with_details(
    task_env,
    state: ArticulatedState,
    action_id: int,
    executor: ParameterizedPrimitiveExecutor,
    param_sets: Optional[List[Dict[str, float]]] = None,
) -> Tuple[bool, Optional[int], str]:
    """Return (valid, collision_step, body_part) for the safest parameter set."""
    if param_sets is None:
        param_sets = _default_oracle_param_sets(action_id)
    context = PrimitiveExecutionContext(
        goal_position=None, goal_heading=None,
        collision_checker=task_env.predict_collision if task_env is not None else None,
    )
    best_result = (False, None, "")
    for params in param_sets:
        try:
            rollout = executor.rollout(
                start_state=state, primitive_id=action_id,
                parameters=params, context=context,
            )
        except Exception:
            continue
        collided = False
        coll_step = None
        coll_body = ""
        for step_idx, step_state in enumerate(rollout.states[1:], start=1):
            if task_env is not None and task_env.predict_collision(step_state):
                collided = True
                coll_step = step_idx
                front_poly, rear_poly = articulated_body_polygons(step_state, task_env.vehicle_config)
                obs_union = task_env._obstacle_union
                if obs_union is not None:
                    if front_poly.intersects(obs_union):
                        coll_body = "front"
                    elif rear_poly.intersects(obs_union):
                        coll_body = "rear"
                    else:
                        coll_body = "out_of_bounds"
                break
        if not collided:
            return True, None, ""
        if best_result[1] is None or (coll_step is not None and coll_step > (best_result[1] or 0)):
            best_result = (False, coll_step, coll_body)
    return best_result


def _default_oracle_param_sets(action_id: int) -> List[Dict[str, float]]:
    base = dict(_DEFAULT_MIDPOINT_PARAMS)
    sets = [base]
    label = _SEMANTIC_LABELS.get(action_id, "")
    if "FORWARD" in label:
        sets.append({**base, "speed_scale": 0.3, "path_length": 1.5})
        sets.append({**base, "speed_scale": 0.7, "path_length": 3.5})
    elif "REVERSE" in label:
        sets.append({**base, "speed_scale": 0.3, "path_length": 1.5})
        sets.append({**base, "speed_scale": 0.7, "path_length": 3.5})
    elif "STRAIGHT" in label:
        sets.append({**base, "speed_scale": 0.3, "path_length": 1.0})
    return sets


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _build_obs_for_state(task_env, state: ArticulatedState, goal_state: ArticulatedState) -> np.ndarray:
    saved_state = task_env._state
    saved_goal = task_env._goal_state
    try:
        task_env._state = state
        task_env._goal_state = goal_state
        return task_env.build_observation()
    finally:
        task_env._state = saved_state
        task_env._goal_state = saved_goal


def _extract_features(obs: np.ndarray, lidar_beams: int) -> Dict[str, float]:
    """Extract ego features from observation (same layout as _state_gate_valid_mask)."""
    offset = lidar_beams
    if obs.shape[0] < offset + 9:
        return {}
    f = obs[offset : offset + 9]
    goal_dist = float(f[0])
    rel_angle = float(np.arctan2(f[2], f[1]))
    rel_heading = float(np.arctan2(f[4], f[3]))
    articulation = float(np.arctan2(f[6], f[5]))
    min_clearance = float(np.min(obs[:offset])) if offset > 0 else 1.0
    return {
        "goal_distance_norm": goal_dist,
        "relative_angle_rad": rel_angle,
        "relative_heading_rad": rel_heading,
        "articulation_rad": articulation,
        "min_clearance_norm": min_clearance,
    }


def _state_gate_for_action(action_id: int, features: Dict[str, float]) -> Tuple[bool, str]:
    """Replicate HybridPPOAgent._state_gate_valid_mask logic."""
    if action_id == 5:  # ARTICULATION_RECOVER
        if abs(features["articulation_rad"]) < 0.41887902047863906:
            return False, "phi_below_recover_threshold"
    if action_id == 7:  # STOP_CHECK
        reasons = []
        if features["goal_distance_norm"] > 0.08:
            reasons.append("goal_too_far")
        if abs(features["relative_heading_rad"]) > 0.20943951023931953:
            reasons.append("heading_misaligned")
        if features["min_clearance_norm"] > 0.10:
            reasons.append("clearance_too_high")
        if reasons:
            return False, "+".join(reasons)
    return True, ""


# ---------------------------------------------------------------------------
# Proxy safety query
# ---------------------------------------------------------------------------

def query_proxy_mask(
    sidecar: ProxySafetySidecar, lidar: np.ndarray, articulation_angle: float,
    gamma: float = 1.0, eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (action_scores, has_safe, prefix_lengths, horizons)."""
    result = sidecar.compute_proxy_scores(
        lidar_observation=lidar.astype(np.float32),
        articulation_angle=float(articulation_angle),
        gamma=gamma, eps=eps,
    )
    action_scores = np.max(result.proxy_scores, axis=-1).astype(np.float32)
    has_safe = np.max(result.prefix_lengths, axis=-1) > 0.0
    prefix_max = np.max(result.prefix_lengths, axis=-1).astype(np.float32)
    horizon = sidecar.nominal_horizons[:, 0].astype(np.float32)
    return action_scores, has_safe, prefix_max, horizon


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------

def score_distribution_analysis(all_details: List[ActionDetail], stop_only: bool = False):
    """Return counts per score bucket."""
    buckets = {name: 0 for _, _, name in _SCORE_BUCKETS}
    total = 0
    for det in all_details:
        if stop_only and det.action_id != 7:
            continue
        total += 1
        score = det.proxy_score
        placed = False
        for lo, hi, name in _SCORE_BUCKETS:
            if score >= lo and (score < hi or (hi == 1.001 and score <= hi)):
                buckets[name] += 1
                placed = True
                break
        if not placed:
            buckets["approx_zero"] += 1
    return buckets, total


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_headings() -> List[float]:
    return [
        0.0, np.pi / 2, np.pi, -np.pi / 2,
        np.deg2rad(15), np.deg2rad(45), np.deg2rad(75),
        np.deg2rad(105), np.deg2rad(135), np.deg2rad(165),
        np.deg2rad(-15), np.deg2rad(-45),
    ]


def sample_articulation_angles(limit: float = 0.6283) -> List[float]:
    return [0.0, 0.10, -0.10, 0.25, -0.25, 0.40, -0.40, limit * 0.92, -limit * 0.92]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_results: List[PositionResult]) -> Dict:
    """Return full metrics including score distribution and regional breakdowns."""
    records = []
    for pr in all_results:
        for ad in pr.actions:
            records.append({
                "scene": pr.scene_idx, "x": pr.x, "y": pr.y,
                "front_heading": pr.front_heading,
                "articulation_angle": pr.articulation_angle,
                "region": pr.region, "pose_relation": pr.pose_relation,
                "min_clearance_norm": pr.min_clearance_norm,
                "min_clearance_m": pr.min_clearance_m,
                "action_id": ad.action_id, "action_label": ad.action_label,
                "oracle_valid": ad.oracle_valid,
                "oracle_collision_step": ad.oracle_collision_step,
                "oracle_collision_body": ad.oracle_collision_body,
                "proxy_score": float(ad.proxy_score),
                "proxy_has_safe": bool(ad.proxy_has_safe),
                "proxy_prefix_max": float(ad.proxy_prefix_max),
                "proxy_horizon": int(ad.proxy_horizon),
                "state_gate_allowed": ad.state_gate_allowed,
                "state_gate_reason": ad.state_gate_reason,
            })

    total = len(records)

    invalid = [r for r in records if not r["oracle_valid"]]
    valid = [r for r in records if r["oracle_valid"]]

    def _fraction(subset, cond_fn):
        if not subset:
            return 0.0
        return sum(1 for r in subset if cond_fn(r)) / len(subset)

    def _killed(r):
        return r["proxy_score"] < 0.01 or not r["proxy_has_safe"]

    # Score bucket counts
    bucket_counts: Dict[str, int] = {}
    bucket_counts_valid: Dict[str, int] = {}
    for r in records:
        s = r["proxy_score"]
        for lo, hi, name in _SCORE_BUCKETS:
            if (s >= lo and s < hi) or (hi >= 1.0 and s >= lo and s <= 1.01):
                bucket_counts[name] = bucket_counts.get(name, 0) + 1
                if r["oracle_valid"]:
                    bucket_counts_valid[name] = bucket_counts_valid.get(name, 0) + 1
                break

    # False kills detail
    false_kills = []
    for r in valid:
        if _killed(r):
            false_kills.append(r)

    missed_invalids = []
    for r in invalid:
        if r["proxy_score"] > 0.7 and r["proxy_has_safe"]:
            missed_invalids.append(r)

    # Region x action breakdown
    region_action: Dict[str, Dict[str, Dict]] = {}
    for r in records:
        ra_key = r["region"]
        a_key = r["action_label"]
        region_action.setdefault(ra_key, {}).setdefault(a_key, {"total": 0, "invalid": 0, "fk": 0, "mi": 0})
        region_action[ra_key][a_key]["total"] += 1
        if not r["oracle_valid"]:
            region_action[ra_key][a_key]["invalid"] += 1
        else:
            if _killed(r):
                region_action[ra_key][a_key]["fk"] += 1
        if not r["oracle_valid"] and r["proxy_score"] > 0.7 and r["proxy_has_safe"]:
            region_action[ra_key][a_key]["mi"] += 1

    # STOP_CHECK detailed analysis
    stop_records = [r for r in records if r["action_id"] == 7]
    stop_valid = sum(1 for r in stop_records if r["oracle_valid"])
    stop_sg_blocked = sum(1 for r in stop_records if not r["state_gate_allowed"])
    stop_fk = sum(1 for r in stop_records if r["oracle_valid"] and _killed(r))
    stop_fk_by_proxy = sum(1 for r in stop_records if r["oracle_valid"] and _killed(r) and r["state_gate_allowed"])
    stop_fk_by_sg = sum(1 for r in stop_records if r["oracle_valid"] and not r["state_gate_allowed"])

    region_counts: Dict[str, int] = {}
    for r in records:
        region_counts[r["region"]] = region_counts.get(r["region"], 0) + 1

    return {
        "total_records": total,
        "total_invalid": len(invalid),
        "total_valid": len(valid),
        "invalid_filtered": sum(1 for r in invalid if _killed(r)),
        "valid_false_killed": sum(1 for r in valid if _killed(r)),
        "invalid_filter_rate": round(_fraction(invalid, _killed), 4),
        "valid_false_kill_rate": round(_fraction(valid, _killed), 4),
        "score_buckets": bucket_counts,
        "score_buckets_valid_only": bucket_counts_valid,
        "region_counts": region_counts,
        "region_action_breakdown": region_action,
        "stop_check_analysis": {
            "total": len(stop_records),
            "oracle_valid": stop_valid,
            "state_gate_blocked": stop_sg_blocked,
            "false_kill_total": stop_fk,
            "false_kill_by_proxy": stop_fk_by_proxy,
            "false_kill_by_state_gate": stop_fk_by_sg,
        },
        "false_kill_samples": false_kills[:80],
        "missed_invalid_samples": missed_invalids[:80],
        "records": records,
    }


def generate_markdown_report(metrics: Dict, output_path: str) -> None:
    L: List[str] = []
    L.append("# Soft Mask Diagnostic Report — Extended Edition")
    L.append("")
    L.append(f"**Total records**: {metrics['total_records']}")
    L.append(f"**Invalid / Valid**: {metrics['total_invalid']} / {metrics['total_valid']}")
    L.append("")

    L.append("## 1. Score Distribution (binary degeneration check)")
    L.append("")
    L.append("| Bucket | All Count | Valid Only Count |")
    L.append("|---|---:|")
    for _, _, name in _SCORE_BUCKETS:
        all_n = metrics["score_buckets"].get(name, 0)
        val_n = metrics["score_buckets_valid_only"].get(name, 0)
        L.append(f"| {name} | {all_n} | {val_n} |")
    L.append("")

    L.append("## 2. Region Coverage")
    L.append("")
    L.append("| Region | Sample Count |")
    L.append("|---|---|")
    for region, count in sorted(metrics["region_counts"].items()):
        L.append(f"| {region} | {count} |")
    L.append("")

    L.append("## 3. STOP_CHECK Special Analysis")
    L.append("")
    sa = metrics["stop_check_analysis"]
    L.append(f"- Total STOP_CHECK records: {sa['total']}")
    L.append(f"- Oracle valid: {sa['oracle_valid']}")
    L.append(f"- Blocked by state-gate: {sa['state_gate_blocked']}")
    L.append(f"- False kill total: {sa['false_kill_total']}")
    L.append(f"  - False kill by **proxy safety** (state-gate allowed): {sa['false_kill_by_proxy']}")
    L.append(f"  - False kill by **state-gate** (proxy irrelevant): {sa['false_kill_by_state_gate']}")
    L.append("")

    L.append("## 4. Region × Action Breakdown")
    L.append("")
    for region, actions in sorted(metrics["region_action_breakdown"].items()):
        L.append(f"### {region}")
        L.append("| Action | Total | Invalid | False Kill | Missed Invalid | FK Rate | MI Rate |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for action, stats in sorted(actions.items()):
            total = stats["total"]
            inv = stats["invalid"]
            fk = stats["fk"]
            mi = stats["mi"]
            valid = total - inv
            fk_rate = fk / max(valid, 1)
            mi_rate = mi / max(inv, 1)
            L.append(f"| {action} | {total} | {inv} | {fk} | {mi} | {fk_rate:.3f} | {mi_rate:.3f} |")
        L.append("")

    L.append("## 5. False Kill Source Location")
    L.append("")
    L.append("Top false-killed samples (oracle-valid, proxy-killed):")
    L.append("")
    L.append("| Action | Region | Pose | Phi | Clearance(m) | Score | SafePrefix |")
    L.append("|---|---:|---:|---:|---:|---|")
    for fk in metrics["false_kill_samples"][:25]:
        L.append(
            f"| {fk['action_label']} | {fk['region']} | {fk['pose_relation']} "
            f"| {fk['articulation_angle']:.3f} | {fk['min_clearance_m']:.1f} "
            f"| {fk['proxy_score']:.4f} | {fk['proxy_prefix_max']:.0f}/{fk['proxy_horizon']} |"
        )
    L.append("")

    L.append("## 6. Missed Invalid Source Location")
    L.append("")
    L.append("Top missed-invalid samples (oracle-invalid, proxy-score=1.0):")
    L.append("")
    L.append("| Action | Region | Pose | CollStep | Body | Score | Clearance(m) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for mi in metrics["missed_invalid_samples"][:25]:
        L.append(
            f"| {mi['action_label']} | {mi['region']} | {mi['pose_relation']} "
            f"| {mi['oracle_collision_step'] or '?'} | {mi['oracle_collision_body']} "
            f"| {mi['proxy_score']:.4f} | {mi['min_clearance_m']:.1f} |"
        )
    L.append("")

    L.append("## 7. Key Findings")
    L.append("")

    bucket_all = metrics["score_buckets"]
    approx_zero = bucket_all.get("approx_zero", 0)
    approx_one = bucket_all.get("approx_one", 0)
    mid_buckets = sum(v for k, v in bucket_all.items() if k not in ("approx_zero", "approx_one"))
    total = sum(bucket_all.values())
    binaryness = (approx_zero + approx_one) / max(total, 1)
    L.append(f"### Score Distribution")
    L.append(f"- **Binary degeneration**: {binaryness:.1%} of scores are at extremes (approx_zero or approx_one). Only {mid_buckets} records ({mid_buckets/max(total,1):.1%}) have intermediate scores.")
    if binaryness > 0.95:
        L.append("- **Verdict**: The soft mask has effectively **degenerated to a hard mask**. "
                   "The `soft_mask_floor` and `soft_mask_logit_scale` hyperparameters have near-zero effect "
                   "because scores are already binary.")
    else:
        L.append("- Intermediate scores exist, soft mask retains some graded behavior.")
    L.append("")

    region_cov = metrics["region_counts"]
    missing = [r for r in ["bay_exit", "bay_inside", "corridor_straight", "corridor_corner", "endpoint_redirection"] if r not in region_cov or region_cov[r] == 0]
    if missing:
        L.append(f"### Region Coverage Gap: {missing}")
    else:
        L.append("### All regions have sample coverage.")
    L.append("")

    L.append("### How to Run")
    L.append("```bash")
    L.append("PYTHONPATH=src python scripts/diagnose_soft_mask.py \\")
    L.append("    --sidecar data/proxy_safety_sidecar.npz \\")
    L.append("    --scenes 5 --grid-spacing 8.0 \\")
    L.append("    --output results/soft_mask_diag.json \\")
    L.append("    --report results/soft_mask_diag.md")
    L.append("```")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose soft mask quality — extended")
    parser.add_argument("--sidecar", type=str, default="data/proxy_safety_sidecar.npz")
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--grid-spacing", type=float, default=8.0,
                        help="Free-space grid sampling spacing in meters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/soft_mask_diag.json")
    parser.add_argument("--report", type=str, default="results/soft_mask_diag.md")
    parser.add_argument("--lidar-beams", type=int, default=108)
    parser.add_argument("--max-oracle-steps", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading sidecar: {args.sidecar}")
    sidecar = load_proxy_safety_sidecar(args.sidecar)
    print(f"  num_rays={sidecar.num_rays}, actions={sidecar.num_actions}")
    meta = sidecar.metadata
    print(f"  lidar_model_version={meta.get('lidar_model_version', 'MISSING!')}")

    vehicle_cfg = VehicleConfig()
    obs_cfg = ObservationConfig(lidar_num_beams=args.lidar_beams)
    env_cfg = EnvRuntimeConfig(max_low_level_steps_per_episode=500)
    reward_cfg = RewardConfig()
    executor_cfg = PrimitiveExecutorConfig(max_macro_steps=args.max_oracle_steps)

    executor = ParameterizedPrimitiveExecutor(
        build_default_primitive_library(),
        vehicle_config=vehicle_cfg,
        executor_config=executor_cfg,
    )

    headings = sample_headings()
    art_angles = sample_articulation_angles(limit=float(vehicle_cfg.articulation_limit_rad))

    all_results: List[PositionResult] = []

    for scene_idx in range(args.scenes):
        seed = int(args.seed + scene_idx * 100)
        env = create_env_adapter(env_cfg, vehicle_cfg, obs_cfg, reward_cfg)
        _, _ = env.reset(seed=seed, options={"level": "Normal"})
        task_env = env.env
        scene = task_env._scene
        goal_state = task_env._goal_state
        obstacle_union = task_env._obstacle_union

        positions = _sampled_free_grid(
            obstacle_union, scene.world_bounds, float(args.grid_spacing),
        )
        targeted = _targeted_positions_from_scene(task_env, obstacle_union)
        all_positions = list(positions) + [(px, py) for px, py, _src in targeted]
        # Deduplicate
        seen = set()
        unique = []
        for p in all_positions:
            key = (round(p[0], 2), round(p[1], 2))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        all_positions = unique
        print(f"\nScene {scene_idx+1}/{args.scenes} seed={seed}: {len(positions)} grid + {len(targeted)} targeted → {len(all_positions)} unique")

        for px, py in all_positions:
            region = classify_position_improved(px, py, obstacle_union, scene.world_bounds)
            for fh in headings:
                for art in art_angles:
                    rh = wrap_to_pi(fh - art)
                    state = ArticulatedState(
                        x=float(px), y=float(py),
                        front_heading=float(fh), rear_heading=float(rh),
                        speed=0.0, articulation_rate=0.0,
                    )
                    if task_env.predict_collision(state):
                        continue

                    pose_rel = classify_pose_relation(fh, art)
                    try:
                        obs = _build_obs_for_state(task_env, state, goal_state)
                    except Exception:
                        continue
                    feat = _extract_features(obs, args.lidar_beams)
                    if not feat:
                        continue

                    lidar_slice = obs[:args.lidar_beams].astype(np.float32)
                    art_angle_val = float(state.articulation_angle)

                    proxy_scores, proxy_has_safe, prefix_max, horizon = query_proxy_mask(
                        sidecar, lidar_slice, art_angle_val,
                    )

                    presult = PositionResult(
                        scene_idx=scene_idx, x=float(px), y=float(py),
                        front_heading=float(fh), rear_heading=float(rh),
                        articulation_angle=float(art),
                        region=region, pose_relation=pose_rel,
                        min_clearance_norm=feat["min_clearance_norm"],
                        min_clearance_m=feat["min_clearance_norm"] * float(obs_cfg.lidar_max_range),
                        goal_distance_norm=feat["goal_distance_norm"],
                        relative_heading_rad=feat["relative_heading_rad"],
                    )

                    for aid in range(8):
                        oracle_valid, coll_step, coll_body = oracle_collision_with_details(
                            task_env, state, aid, executor,
                        )
                        sg_allowed, sg_reason = _state_gate_for_action(aid, feat)
                        presult.actions.append(ActionDetail(
                            action_id=aid,
                            action_label=_SEMANTIC_LABELS[aid],
                            oracle_valid=oracle_valid,
                            oracle_collision_step=coll_step,
                            oracle_collision_body=coll_body,
                            proxy_score=float(proxy_scores[aid]),
                            proxy_has_safe=bool(proxy_has_safe[aid]),
                            proxy_prefix_max=float(prefix_max[aid]),
                            proxy_horizon=int(horizon[aid]),
                            state_gate_allowed=sg_allowed,
                            state_gate_reason=sg_reason,
                        ))

                    all_results.append(presult)

        print(f"  Accumulated {len(all_results)} position-poses")

    print(f"\nTotal: {sum(len(r.actions) for r in all_results)} action records")

    metrics = compute_metrics(all_results)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(f"JSON → {args.output}")

    generate_markdown_report(metrics, args.report)
    print(f"Report → {args.report}")

    bucket_all = metrics["score_buckets"]
    total = sum(bucket_all.values())
    print(f"\n=== Score Distribution ===")
    for _, _, name in _SCORE_BUCKETS:
        n = bucket_all.get(name, 0)
        print(f"  {name}: {n} ({n/max(total,1):.1%})")
    print(f"  → binary ratio: {(bucket_all.get('approx_zero',0)+bucket_all.get('approx_one',0))/max(total,1):.1%}")
    print(f"\n=== Region Coverage ===")
    for region, count in sorted(metrics["region_counts"].items()):
        print(f"  {region}: {count}")
    print(f"\n=== STOP_CHECK ===")
    sa = metrics["stop_check_analysis"]
    print(f"  total={sa['total']}, valid={sa['oracle_valid']}, sg_blocked={sa['state_gate_blocked']}, fk={sa['false_kill_total']}")
    print(f"  fk_by_proxy={sa['false_kill_by_proxy']}, fk_by_state_gate={sa['false_kill_by_state_gate']}")


if __name__ == "__main__":
    main()
