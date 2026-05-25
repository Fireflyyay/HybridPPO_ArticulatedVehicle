from common.runtime_config import TrainingScheduleConfig
from training.curriculum import SuccessBandCurriculum
from training.train_loop import _auto_checkpoint_metric_path


def test_auto_checkpoint_metric_tracks_curriculum_stage():
    schedule = TrainingScheduleConfig(
        warmup_min_episodes=2,
        warmup_corridor_convergence_episodes=1,
        curriculum_recent_window=2,
        warmup_mastery_success_rate=1.0,
    )
    curriculum = SuccessBandCurriculum(schedule, seed=5)

    assert _auto_checkpoint_metric_path(curriculum) == "levels/Warmup/success_rate"

    curriculum.record_episode("Warmup", success=True)
    curriculum.record_episode("Warmup", success=True)

    assert curriculum.success_band_active()
    assert _auto_checkpoint_metric_path(curriculum) == "levels/Normal/success_rate"