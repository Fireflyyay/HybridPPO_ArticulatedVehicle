import numpy as np

from common.config import HybridPPOConfig
from common.types import MacroTransition
from model.agent import HybridPPOAgent
from primitives import build_default_primitive_library


def test_hybrid_agent_action_and_update_smoke():
    library = build_default_primitive_library()
    config = HybridPPOConfig(
        observation_dim=10,
        action_dim=library.action_dim,
        parameter_dim=library.parameter_dim,
        mini_batch_size=2,
        update_epochs=2,
    )
    agent = HybridPPOAgent(config, library)

    selection = agent.act(np.zeros((10,), dtype=np.float32))
    assert 0 <= selection.macro_action.primitive_id < library.action_dim
    assert selection.macro_action.parameters.shape == (library.parameter_dim,)
    assert np.all(selection.macro_action.parameters >= library.low - 1e-6)
    assert np.all(selection.macro_action.parameters <= library.high + 1e-6)

    for index in range(4):
        obs = np.linspace(0.0, 1.0, 10, dtype=np.float32) + index * 0.05
        sample = agent.act(obs)
        agent.store_transition(
            MacroTransition(
                observation=obs,
                action_id=sample.macro_action.primitive_id,
                parameters=sample.macro_action.parameters,
                reward=1.0 - 0.1 * index,
                tau=1 + (index % 2),
                next_observation=obs + 0.1,
                done=bool(index == 3),
                log_prob=sample.log_prob,
                value=sample.value,
            )
        )

    metrics = agent.update()
    assert "actor_loss" in metrics
    assert "value_loss" in metrics
