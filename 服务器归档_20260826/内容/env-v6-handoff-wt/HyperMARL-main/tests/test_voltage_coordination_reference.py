import numpy as np

from baselines.MAPPO.voltage_coordination_reference import adjust_action


def test_reference_is_noop_above_soft_voltage_threshold():
    action = np.zeros((4, 7), dtype=np.float32)
    action[:, 1] = 0.4
    np.testing.assert_allclose(
        adjust_action(action, {"voltage_min_pu": 0.98}), action
    )


def test_reference_cancels_charge_and_spares_low_soc_agent():
    action = np.zeros((4, 7), dtype=np.float32)
    action[:, 1] = 0.5
    out = adjust_action(
        action,
        {"voltage_min_pu": 0.94},
        soc=np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
        pcc_power=np.array([2500, 1000, 3000, 2200], dtype=np.float32),
        bat_power=np.full(4, 1000, dtype=np.float32),
    )
    assert np.all(out[:, 1] <= 0.0)
    assert out[0, 1] > out[2, 1]
    assert np.all(out <= 1.0) and np.all(out >= -1.0)


def test_reference_preserves_shape_and_leaves_hydrogen_order_unchanged():
    action = np.zeros((4, 7), dtype=np.float32)
    out = adjust_action(
        action,
        {"voltage_min_pu": 0.93},
        h2_level=np.array([20, 200, 20, 200], dtype=np.float32),
        pending_h2=np.zeros(4, dtype=np.float32),
    )
    assert out.shape == action.shape
    np.testing.assert_allclose(out[:, 5], action[:, 5])
