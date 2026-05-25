import json

from common.runtime_config import ExperimentConfig, LoggingConfig
from training.logger import TensorBoardLogger


def test_logger_writes_effective_config_snapshot_and_reward_curve(tmp_path):
    logger = TensorBoardLogger(LoggingConfig(log_root=str(tmp_path), run_name="unit_test_run", flush_secs=1))
    try:
        config = ExperimentConfig()
        logger.log_config(config)
        logger.log_training_episode(1, {"total_reward": -5.0, "success": 0.0})
        logger.log_training_episode(2, {"total_reward": 3.0, "success": 1.0})
    finally:
        logger.close()

    config_json_path = tmp_path / "unit_test_run" / "config.json"
    config_txt_path = tmp_path / "unit_test_run" / "config.txt"
    reward_svg_path = tmp_path / "unit_test_run" / "reward_curve.svg"

    assert config_json_path.exists()
    assert config_txt_path.exists()
    assert reward_svg_path.exists()

    config_payload = json.loads(config_json_path.read_text(encoding="utf-8"))
    assert config_payload["teacher_enabled"] is False
    assert "schedule" in config_payload

    reward_svg = reward_svg_path.read_text(encoding="utf-8")
    assert "<svg" in reward_svg
    assert "Training Reward" in reward_svg
    assert "avg100" in reward_svg