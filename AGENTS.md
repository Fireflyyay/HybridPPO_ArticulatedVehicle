# AGENTS.md

## Environment

- **Conda env**: `HOPE` (required by repo convention — see `.github/instructions/repo-grounding.instructions.md`)
- **PYTHONPATH**: always include `src`, e.g. `PYTHONPATH=src conda run -n HOPE python ...`
- **Install**: `pip install -e .[dev]`

## Commands

```bash
# Run all tests (pyproject.toml sets pythonpath = ["src"])
pytest

# Run a single test file
pytest tests/test_agent.py

# Minimal training smoke test
PYTHONPATH=src conda run -n HOPE python train.py --episodes 200 --episodes-per-update 8

# Generate proxy safety sidecar (required before enabling --enable-soft-mask)
PYTHONPATH=src conda run -n HOPE python example/build_proxy_safety_sidecar.py \
  --output data/proxy_safety_sidecar.npz \
  --lidar-beams 108 --lidar-range 30 --articulation-bins 7 --proxy-resolution 2
```

- `tests/benchmark_optimizations.py` is a performance benchmark, not a unit test.
- No CI, lint, or typecheck config exists in this repo.

## Architecture (high signal)

This is a **hierarchical RL** system for center-articulated vehicle navigation:

1. **HybridPPOAgent** → discrete primitive ID (1 of 8) + continuous parameters (8-dim AffineBeta)
2. **ParameterizedPrimitiveLibrary** → maps ID+params to a `PrimitiveRollout` (low-level steering/speed sequence)
3. **ParameterizedMacroActionWrapper** → steps the low-level `KinematicTaskEnv` through the rollout
4. **SMDP**: variable-length macro steps; GAE targets use `gamma^tau` discounting per transition

**Key modules** (only what an agent might miss):
- `src/common/config.py` — `VehicleConfig` is the **single source of truth** for vehicle geometry. Changing it affects env, success checker, and primitives. If you change it, regenerate the proxy safety sidecar.
- `src/common/runtime_config.py` — `ExperimentConfig` holds all defaults; CLI flags in `src/training/cli.py` overlay onto it.
- `src/primitives/library.py` — 8 semantic primitives with action-dependent active parameter subsets (not all primitives use all 8 parameters).
- `src/env/global_guidance.py` — A* path + Dijkstra cost-to-go; provides 4-dim guidance hint appended to the 121-dim observation.

## Safety / Soft Mask

- The **ProxySafetySidecar** must be **pre-generated offline** before `--enable-soft-mask` works. Parameters (`--lidar-beams`, `--lidar-range`) must match training config exactly.
- If you modify `VehicleConfig`, `PrimitiveExecutorConfig`, or primitive parameter bounds, **regenerate the sidecar** or safety scores will be inaccurate.

## Curriculum

Three scene levels: `Debug` (empty), `Warmup` (corridor, narrowing from 16m→6m), `Normal` (complex block-mixing).
- `--warmup-min-episodes` gates progression from Warmup to Normal via success band.

## Phase Reference Target

`--enable-phase-reference` switches the heading-reward reference from the final goal heading to the local guidance path tangent when far from the goal. The reference automatically shifts back to the final goal heading when all three docking conditions are met: path progress > 90%, goal distance < near_goal_radius × 1.5, and path tangent aligns with goal heading within 30°.
- This is orthogonal to `--disable-escape-reference` (which handles narrow-passage / near-start overrides).
- Visualization: `example/visualize_policy_paths.py` draws guidance-phase reference points as blue circles with heading arrows, and goal-phase reference points as green circles.

## Evidence rules (from `.github/copilot-instructions.md`)

- Do **not** invent files, functions, classes, config keys, or CLI flags.
- Preserve PPO + primitive + macro-wrapper architecture unless explicitly instructed otherwise.
- Do not silently change action-space semantics, reward scale, kinematics, or action mask behavior.

## Additional instruction files

- `.github/copilot-instructions.md` — architecture description and evidence rules
- `.github/instructions/repo-grounding.instructions.md` — coding rules for `src/**/*.py`
