"""
Tests for the GlucoRL reward calculator.

Verifies that the shaped reward function produces the correct per-step
reward for each glucose zone (in-range, mild hypo, severe hypo, mild
hyper, severe hyper) and for the overdose penalty.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.reward_calculator import calculate_step_reward


class TestRewardInRange:
    """Glucose in target range [70, 180] should yield +1.0."""

    def test_glucose_120(self):
        r = calculate_step_reward(120.0, 115.0, 0.0, 110.0, 0.0)
        assert r.step_total == 1.0
        assert r.tir_contribution == 1.0
        assert r.hypo_penalty == 0.0
        assert r.hyper_penalty == 0.0
        assert r.overdose_penalty == 0.0

    def test_glucose_150(self):
        r = calculate_step_reward(150.0, 140.0, 0.0, 130.0, 0.0)
        assert r.step_total == 1.0

    def test_edge_70(self):
        r = calculate_step_reward(70.0, 72.0, 0.0, 75.0, 0.0)
        assert r.step_total == 1.0

    def test_edge_180(self):
        r = calculate_step_reward(180.0, 175.0, 0.0, 170.0, 0.0)
        assert r.step_total == 1.0


class TestRewardMildHypo:
    """Glucose in [54, 70) should yield -1.0."""

    def test_glucose_65(self):
        r = calculate_step_reward(65.0, 72.0, 0.0, 80.0, 0.0)
        assert r.step_total == -1.0
        assert r.tir_contribution == 0.0
        assert r.hypo_penalty == -1.0
        assert r.hyper_penalty == 0.0

    def test_glucose_55(self):
        r = calculate_step_reward(55.0, 60.0, 0.0, 65.0, 0.0)
        assert r.step_total == -1.0

    def test_edge_54(self):
        r = calculate_step_reward(54.0, 58.0, 0.0, 62.0, 0.0)
        assert r.step_total == -1.0


class TestRewardSevereHypo:
    """Glucose < 54 should yield -3.0."""

    def test_glucose_50(self):
        r = calculate_step_reward(50.0, 58.0, 0.0, 65.0, 0.0)
        assert r.step_total == -3.0
        assert r.tir_contribution == 0.0
        assert r.hypo_penalty == -3.0
        assert r.hyper_penalty == 0.0

    def test_glucose_30(self):
        r = calculate_step_reward(30.0, 45.0, 0.0, 55.0, 0.0)
        assert r.step_total == -3.0

    def test_glucose_10(self):
        r = calculate_step_reward(10.0, 20.0, 0.0, 30.0, 0.0)
        assert r.step_total == -3.0


class TestRewardMildHyper:
    """Glucose in (180, 250] should yield -0.5."""

    def test_glucose_200(self):
        r = calculate_step_reward(200.0, 185.0, 0.0, 175.0, 0.0)
        assert r.step_total == -0.5
        assert r.tir_contribution == 0.0
        assert r.hypo_penalty == 0.0
        assert r.hyper_penalty == -0.5

    def test_glucose_250(self):
        r = calculate_step_reward(250.0, 240.0, 0.0, 230.0, 0.0)
        assert r.step_total == -0.5

    def test_glucose_181(self):
        r = calculate_step_reward(181.0, 178.0, 0.0, 175.0, 0.0)
        assert r.step_total == -0.5


class TestRewardSevereHyper:
    """Glucose > 250 should yield -1.5."""

    def test_glucose_300(self):
        r = calculate_step_reward(300.0, 260.0, 0.0, 220.0, 0.0)
        assert r.step_total == -1.5
        assert r.tir_contribution == 0.0
        assert r.hypo_penalty == 0.0
        assert r.hyper_penalty == -1.5

    def test_glucose_400(self):
        r = calculate_step_reward(400.0, 350.0, 0.0, 300.0, 0.0)
        assert r.step_total == -1.5

    def test_glucose_251(self):
        r = calculate_step_reward(251.0, 248.0, 0.0, 245.0, 0.0)
        assert r.step_total == -1.5


class TestRewardOverdose:
    """Glucose < 54 with a large bolus 2 steps ago should yield -6.0."""

    def test_overdose(self):
        r = calculate_step_reward(40.0, 55.0, 0.0, 120.0, 10.0)
        assert r.step_total == -6.0
        assert r.hypo_penalty == -3.0
        assert r.overdose_penalty == -3.0

    def test_no_overdose_small_bolus(self):
        """Bolus <= 5.0 should not trigger overdose penalty."""
        r = calculate_step_reward(40.0, 55.0, 0.0, 120.0, 4.0)
        assert r.overdose_penalty == 0.0
        assert r.step_total == -3.0

    def test_no_overdose_glucose_above_54(self):
        """Glucose >= 54 should not trigger overdose penalty even with large bolus."""
        r = calculate_step_reward(55.0, 60.0, 0.0, 120.0, 10.0)
        assert r.overdose_penalty == 0.0
        assert r.step_total == -1.0


class TestRewardNonBinary:
    """Verify reward produces meaningfully different values across zones."""

    def test_variance_across_zones(self):
        zones = [
            (120.0, 1.0),    # in range
            (65.0, -1.0),    # mild hypo
            (50.0, -3.0),    # severe hypo
            (200.0, -0.5),   # mild hyper
            (300.0, -1.5),   # severe hyper
        ]
        rewards = set()
        for glucose, expected in zones:
            r = calculate_step_reward(glucose, 120.0, 0.0, 120.0, 0.0)
            assert r.step_total == expected, (
                f"glucose={glucose}: expected {expected}, got {r.step_total}"
            )
            rewards.add(r.step_total)

        assert len(rewards) == 5, "Reward should produce 5 distinct values across zones"
