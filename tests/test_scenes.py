import numpy as np

from common.config import VehicleConfig
from common.runtime_config import SceneLevelConfig, TrainingScheduleConfig
from env.scenes import BaselineInspiredSceneFactory


def test_scene_level_config_clamps_warmup_corridor_width_to_minimum():
    config = SceneLevelConfig(corridor_width_range=(6, 6), warmup_corridor_max_width=12.0)

    assert np.isclose(config.resolve_warmup_corridor_width(0.0), 12.0)
    assert np.isclose(config.resolve_warmup_corridor_width(0.5), 9.0)
    assert np.isclose(config.resolve_warmup_corridor_width(1.0), 6.0)
    assert np.isclose(config.resolve_warmup_corridor_width(1.5), 6.0)


def test_warmup_scene_uses_curriculum_progress_for_corridor_width():
    config = SceneLevelConfig(corridor_width_range=(6, 6), warmup_corridor_max_width=12.0)
    factory = BaselineInspiredSceneFactory({"Warmup": config}, vehicle_config=VehicleConfig())

    wide_scene = factory.generate("Warmup", np.random.default_rng(7), options={"warmup_progress": 0.0})
    narrow_scene = factory.generate("Warmup", np.random.default_rng(7), options={"warmup_progress": 1.0})

    assert np.isclose(float(wide_scene.metadata["corridor_width"]), 12.0)
    assert np.isclose(float(narrow_scene.metadata["corridor_width"]), 6.0)
    assert float(wide_scene.metadata["corridor_width"]) > float(narrow_scene.metadata["corridor_width"])


def test_warmup_scene_uses_block_mixing_grid_variant_without_legacy_metadata():
    factory = BaselineInspiredSceneFactory({"Warmup": SceneLevelConfig()}, vehicle_config=VehicleConfig())

    scene = factory.generate("Warmup", np.random.default_rng(7), options={"warmup_progress": 0.25})

    assert scene.metadata["scene_type"] == "block_mixing_plant"
    assert scene.metadata["scene_variant"] == "warmup_curriculum"
    assert scene.metadata["corridor_generation_mode"] == "polyline_warmup"
    assert int(scene.metadata["parking_bay_count"]) == 1
    assert 0.0 < float(scene.metadata["free_ratio"]) < 1.0
    assert "aligned_to" not in scene.metadata


def test_normal_scene_uses_constructive_grid_metadata():
    factory = BaselineInspiredSceneFactory({"Normal": SceneLevelConfig()}, vehicle_config=VehicleConfig())

    scene = factory.generate("Normal", np.random.default_rng(7))

    assert scene.metadata["scene_type"] == "block_mixing_plant"
    assert scene.metadata["scene_variant"] == "constructive_grid"
    assert scene.metadata["corridor_generation_mode"] == "constructive_attachment"
    assert int(scene.metadata["parking_bay_count"]) >= 1
    assert int(scene.metadata["valid_start_candidate_count"]) >= 1
    assert int(scene.metadata["valid_goal_candidate_count"]) >= 1
    assert 0.0 < float(scene.metadata["free_ratio"]) < 1.0
    assert len(scene.obstacles) > 0


def test_training_schedule_emits_clamped_warmup_progress():
    schedule = TrainingScheduleConfig(
        debug_phase_episodes=2,
        warmup_episodes=8,
        warmup_corridor_convergence_episodes=4,
    )

    assert schedule.reset_options_for_episode(0) == {"level": "Debug"}
    assert schedule.reset_options_for_episode(2) == {
        "level": "Warmup",
        "warmup_episode_idx": 0,
        "warmup_progress": 0.0,
    }
    assert schedule.reset_options_for_episode(4) == {
        "level": "Warmup",
        "warmup_episode_idx": 2,
        "warmup_progress": 0.5,
    }
    assert schedule.reset_options_for_episode(8) == {
        "level": "Warmup",
        "warmup_episode_idx": 6,
        "warmup_progress": 1.0,
    }