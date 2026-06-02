# AGENTS.md — HybridPPO Articulated Vehicle

## Commands

```bash
# Run all tests (PYTHONPATH=src is set automatically via pyproject.toml)
pytest

# Run a single test file
pytest tests/test_task_env.py

# Run a single test by name
pytest tests/test_task_env.py -k test_reverse_penalty

# Training (must be run from repo root)
PYTHONPATH=src python train.py --episodes 200 --episodes-per-update 8

# Generate proxy safety sidecar (only if enabling soft mask)
PYTHONPATH=src python example/build_proxy_safety_sidecar.py \
    --output data/proxy_safety_sidecar.npz \
    --lidar-beams 108 --lidar-range 30
```

## Architecture

- **Entry**: `train.py` → `src/training/cli.py` → `src/training/train_loop.py`
- **Package root**: `src/` — all imports use `common.`, `env.`, `model.`, etc. (no `src.` prefix)
- **SMDP**: macro-actions with variable duration τ; GAE discount = γ^τ
- **Action space**: 8 discrete semantic primitives × 8 continuous Beta-distributed parameters
- **Observation**: 108 LiDAR beams + 9 state features + 4 guidance features = 121 dims
- **3 scene levels**: Debug (open world), Warmup (single corridor + bay), Normal (multi-corridor block mixing plant with branches and bays)
- **Parking success**: front-heading error ≤ 15°, front polygon overlap ≥ 70%, collision-free

## Config

All configs are **frozen dataclasses** in `src/common/runtime_config.py`. To modify them, use `dataclasses.replace()` — never attribute assignment.

- `VehicleConfig` in `src/common/config.py` — vehicle geometry, speed limits, articulation limits
- `TrainingScheduleConfig` — episode counts, curriculum parameters, **adaptive sampling** toggle
- `RewardConfig` — all reward weights (incl. `heading_weight`, `reverse_penalty_coef`, `near_goal_radius` gating)
- `HybridPPOHyperConfig` — PPO, entropy, soft mask, teacher parameters

CLI arguments in `cli.py` override config fields via `replace()`.

## Key subsystems

| File | Purpose |
|------|---------|
| `src/primitives/library.py` | 8 parameterized primitives (forward/reverse-left/right, reverse-align, articulation-recover, straight-adjust, stop-check) |
| `src/env/scenes.py` | Scene generation (Debug/Warmup/block-mixing). Accepts `options` dict for `heading_diff_range_deg` override and `warmup_progress`. |
| `src/env/global_guidance.py` | A* path planning + backward Dijkstra cost-to-go map for topology-based reward shaping |
| `src/env/success.py` | Parking success: heading + front-overlap (rear overlap tracked but NOT required) |
| `src/training/curriculum.py` | `SuccessBandCurriculum` (Warmup→Normal level switching) + `DifficultyAdaptiveSampler` (per-heading-diff bucket selection, disabled by default) |
| `src/model/agent/hybrid_ppo_agent.py` | Full agent with soft mask, proxy safety, teacher guidance |
| `src/training/soft_teacher.py` | Behavioral cloning teacher (disabled by default, `teacher_enabled=False`) |

## Quirks and gotchas

- **`reverse_penalty_coef = -0.2`**: Despite the name, the default value *tolerates* backward motion (gives a small positive bonus when moving away from goal). This causes the test `test_topology_progress_only_rewards_new_frontier_advance` to fail — it is a **pre-existing** test failure, not a regression.
- **Heading and overlap rewards are gated** by `near_goal_factor` (only active within 10m of goal) — see `task_env.py:257-259`.
- **Soft mask is off by default** (`soft_mask_enabled=False`). Enabling it requires a pre-built sidecar file matching the current LiDAR + vehicle config.
- **Teacher is off by default** (`teacher_enabled=False`). Even when teacher code is loaded, `--disable-teacher` is the CLI default.
- **DifficultyAdaptiveSampler** is off by default (`adaptive_sampling_enabled=False`). Enable it in config to sample scenes by heading-diff difficulty buckets (forward/side/reverse approach), with per-level target success rates.
- When modifying scene generation, always add `heading_diff_deg` to metadata dict for downstream tracking.
- **Module imports**: Tests run with `pythonpath = ["src"]` in pyproject.toml. Training needs `PYTHONPATH=src`. The `src/` directory is NOT a package itself — don't use `src.` prefix in imports.

## Testing

- 71 tests total; 1 pre-existing failure (see above)
- `tests/test_primitives.py` and `tests/test_success.py` are the fastest smoke tests after vehicle param changes
- `tests/test_task_env.py` covers reward computation, topology progress, reverse penalty
- Tests use small LiDAR (8 beams) and short episodes for speed
