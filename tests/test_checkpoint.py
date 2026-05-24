from common.runtime_config import CheckpointConfig, ExperimentConfig
from model.agent import HybridPPOAgent
from primitives import build_default_primitive_library
from training.checkpoint import CheckpointManager


def test_checkpoint_manager_round_trip(tmp_path):
    library = build_default_primitive_library()
    config = ExperimentConfig()
    agent = HybridPPOAgent(
        config=config.agent.build(
            observation_dim=config.observation.observation_dim,
            action_dim=library.action_dim,
            parameter_dim=library.parameter_dim,
        ),
        primitive_library=library,
    )
    manager = CheckpointManager(str(tmp_path), CheckpointConfig(save_interval=1), config)
    path = manager.save_latest(
        agent,
        episode_idx=4,
        update_idx=2,
        extra={
            "curriculum": {
                "band_unlocked": True,
                "level_counts": {"Warmup": 4, "Normal": 1},
            }
        },
    )
    meta = manager.load(path, agent)
    assert meta["episode_idx"] == 4
    assert meta["update_idx"] == 2
    assert meta["extra"]["curriculum"]["band_unlocked"] is True