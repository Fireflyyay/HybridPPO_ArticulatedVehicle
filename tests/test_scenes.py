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


def test_warmup_scene_uses_warmup_bay_metadata():
    factory = BaselineInspiredSceneFactory({"Warmup": SceneLevelConfig()}, vehicle_config=VehicleConfig())

    scene = factory.generate("Warmup", np.random.default_rng(7), options={"warmup_progress": 0.25})

    assert scene.metadata["scene_type"] == "warmup_centerline"
    assert scene.metadata["aligned_to"] == "ppo_articulated_vehicle"
    assert float(scene.metadata["corridor_width"]) > 0.0
    assert "warmup_progress" in scene.metadata


def test_normal_scene_uses_block_mixing_metadata():
    factory = BaselineInspiredSceneFactory({"Normal": SceneLevelConfig()}, vehicle_config=VehicleConfig())

    scene = factory.generate("Normal", np.random.default_rng(7))

    assert scene.metadata["scene_type"] == "block_mixing_plant"
    assert scene.metadata["aligned_to"] == "ppo_articulated_vehicle"
    assert float(scene.metadata["corridor_width"]) > 0.0
    assert int(scene.metadata["free_shape_count"]) >= 1
    assert int(scene.metadata["valid_candidate_count"]) >= 2
    assert len(scene.obstacles) > 0


def test_training_schedule_clamps_warmup_progress_from_warmup_index():
    schedule = TrainingScheduleConfig(
        warmup_corridor_convergence_episodes=4,
    )

    assert np.isclose(schedule.warmup_progress_for_index(0), 0.0)
    assert np.isclose(schedule.warmup_progress_for_index(2), 0.5)
    assert np.isclose(schedule.warmup_progress_for_index(4), 1.0)
    assert np.isclose(schedule.warmup_progress_for_index(6), 1.0)