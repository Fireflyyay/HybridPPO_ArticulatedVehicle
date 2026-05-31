import numpy as np

from common.runtime_config import TrainingScheduleConfig
from training.curriculum import SuccessBandCurriculum


def test_success_band_waits_for_narrow_warmup_mastery():
    schedule = TrainingScheduleConfig(
        warmup_min_episodes=6,
        warmup_corridor_convergence_episodes=4,
        curriculum_recent_window=4,
        warmup_mastery_success_rate=0.75,
    )
    curriculum = SuccessBandCurriculum(schedule, seed=7)

    expected_progress = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    for index, progress in enumerate(expected_progress):
        options = curriculum.reset_options()
        assert options["level"] == "Warmup"
        assert np.isclose(float(options["warmup_progress"]), progress)
        assert options["level"] != "Debug"
        curriculum.record_episode("Warmup", success=True)
        assert curriculum.success_band_active() == (index >= 5)


def test_warmup_progress_is_damped_when_success_lags():
    schedule = TrainingScheduleConfig(
        warmup_min_episodes=8,
        warmup_corridor_convergence_episodes=4,
        curriculum_recent_window=4,
        warmup_mastery_success_rate=0.5,
    )
    curriculum = SuccessBandCurriculum(schedule, seed=13)

    for _ in range(4):
        curriculum.record_episode("Warmup", success=False)

    options = curriculum.reset_options()

    assert options["level"] == "Warmup"
    assert np.isclose(float(options["warmup_progress"]), 0.5)
    assert np.isclose(curriculum.warmup_progress(), 0.5)


def test_warmup_reset_options_include_configurable_bay_exit_task():
    exit_curriculum = SuccessBandCurriculum(TrainingScheduleConfig(warmup_bay_exit_prob=1.0), seed=13)
    entry_curriculum = SuccessBandCurriculum(TrainingScheduleConfig(warmup_bay_exit_prob=0.0), seed=13)

    assert exit_curriculum.reset_options()["warmup_task"] == "BayExit"
    assert entry_curriculum.reset_options()["warmup_task"] == "BayEntry"


def test_success_band_bridges_back_to_warmup_when_target_is_below_band():
    schedule = TrainingScheduleConfig(
        warmup_min_episodes=4,
        warmup_corridor_convergence_episodes=2,
        curriculum_recent_window=4,
        warmup_mastery_success_rate=0.75,
        target_success_band=(0.25, 0.75),
        target_focus_prob=1.0,
        warmup_bridge_prob=1.0,
    )
    curriculum = SuccessBandCurriculum(schedule, seed=3)

    for _ in range(4):
        curriculum.record_episode("Warmup", success=True)

    assert curriculum.success_band_active()
    assert curriculum.reset_options()["level"] == "Warmup"


def test_success_band_focuses_target_level_when_target_is_in_band():
    schedule = TrainingScheduleConfig(
        warmup_min_episodes=4,
        warmup_corridor_convergence_episodes=2,
        curriculum_recent_window=4,
        warmup_mastery_success_rate=0.75,
        target_success_band=(0.25, 0.75),
        target_focus_prob=1.0,
        warmup_bridge_prob=1.0,
    )
    curriculum = SuccessBandCurriculum(schedule, seed=11)

    for _ in range(4):
        curriculum.record_episode("Warmup", success=True)

    curriculum.record_episode("Normal", success=True)
    curriculum.record_episode("Normal", success=False)
    curriculum.record_episode("Normal", success=True)
    curriculum.record_episode("Normal", success=False)

    assert np.isclose(curriculum.recent_success_rate("Normal"), 0.5)
    assert curriculum.reset_options()["level"] == "Normal"


def test_success_band_state_roundtrip_preserves_recent_history():
    schedule = TrainingScheduleConfig(
        warmup_min_episodes=4,
        warmup_corridor_convergence_episodes=2,
        curriculum_recent_window=3,
    )
    curriculum = SuccessBandCurriculum(schedule, seed=5)
    curriculum.record_episode("Warmup", success=False)
    curriculum.record_episode("Warmup", success=True)
    curriculum.record_episode("Warmup", success=True)
    curriculum.record_episode("Normal", success=False)
    curriculum.record_episode("Normal", success=True)

    restored = SuccessBandCurriculum(schedule, seed=99)
    restored.load_state_dict(curriculum.state_dict())

    assert restored.level_counts == curriculum.level_counts
    assert np.isclose(restored.recent_success_rate("Warmup"), curriculum.recent_success_rate("Warmup"))
    assert np.isclose(restored.recent_success_rate("Normal"), curriculum.recent_success_rate("Normal"))
