import numpy as np

from hybridppo_articulated_vehicle.smdp import accumulate_macro_reward, compute_smdp_targets


def test_accumulate_macro_reward_uses_low_level_discount():
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.9
    assert np.isclose(accumulate_macro_reward(rewards, gamma), 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)


def test_compute_smdp_targets_respects_tau():
    rewards = np.array([1.0, 0.5], dtype=np.float32)
    taus = np.array([2, 1], dtype=np.int64)
    dones = np.array([0.0, 1.0], dtype=np.float32)
    values = np.array([0.2, 0.1], dtype=np.float32)
    next_values = np.array([0.4, 0.0], dtype=np.float32)
    gamma = 0.9
    lam = 0.95

    targets = compute_smdp_targets(rewards, taus, dones, values, next_values, gamma, lam)
    delta_1 = 0.5 - 0.1
    delta_0 = 1.0 + gamma**2 * 0.4 - 0.2
    expected_adv_1 = delta_1
    expected_adv_0 = delta_0 + gamma**2 * lam * expected_adv_1

    assert np.isclose(targets.deltas[1], delta_1)
    assert np.isclose(targets.deltas[0], delta_0)
    assert np.isclose(targets.advantages[1], expected_adv_1)
    assert np.isclose(targets.advantages[0], expected_adv_0)
