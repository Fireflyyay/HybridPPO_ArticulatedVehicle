from common.runtime_config import CheckpointConfig, EvaluationConfig, ExperimentConfig, LoggingConfig
from training.cli import build_argument_parser, build_experiment_config
from training.train_loop import ExperimentTrainer


def test_build_experiment_config_honors_disable_teacher_flag():
    parser = build_argument_parser()
    args = parser.parse_args(["--disable-teacher"])

    config = build_experiment_config(args)

    assert config.teacher_enabled is False


def test_trainer_skips_teacher_construction_when_disabled(tmp_path):
    config = ExperimentConfig(
        teacher_enabled=False,
        logging=LoggingConfig(log_root=str(tmp_path), run_name="disable-teacher"),
        evaluation=EvaluationConfig(interval=0, episodes_per_level=1),
        checkpoint=CheckpointConfig(save_interval=0),
    )

    trainer = ExperimentTrainer(config)
    try:
        assert trainer.soft_teacher is None
    finally:
        trainer.logger.close()