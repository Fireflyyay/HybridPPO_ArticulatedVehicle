import numpy as np

from ..common.types import SMDPTargets


def accumulate_macro_reward(step_rewards, gamma: float) -> float:
    total = 0.0
    for index, reward in enumerate(step_rewards):
        total += float(gamma) ** int(index) * float(reward)
    return float(total)


def compute_smdp_targets(rewards: np.ndarray, taus: np.ndarray, dones: np.ndarray, values: np.ndarray, next_values: np.ndarray, gamma: float, gae_lambda: float) -> SMDPTargets:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    taus = np.asarray(taus, dtype=np.int64).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    next_values = np.asarray(next_values, dtype=np.float32).reshape(-1)
    discounts = np.power(float(gamma), taus.astype(np.float32), dtype=np.float32)
    deltas = rewards + discounts * (1.0 - dones) * next_values - values
    advantages = np.zeros_like(deltas, dtype=np.float32)
    gae = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        gae = float(deltas[index]) + float(discounts[index]) * float(gae_lambda) * (1.0 - float(dones[index])) * gae
        advantages[index] = gae
    returns = advantages + values
    return SMDPTargets(advantages=advantages, returns=returns, deltas=deltas)
