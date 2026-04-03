"""
Tests for the GlucoRL environment.

Verifies that:
  - reset() returns a plausible initial observation
  - step() advances the simulation correctly
  - Borderline actions do not crash
  - Episode terminates after 480 steps
  - Emergency termination on consecutive severe hypoglycemia
  - state() returns consistent history lengths
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.glucorl_environment import GlucoRLEnvironment
from models import GlucoAction, GlucoObservation, GlucoState


# ── Reset ────────────────────────────────────────────────────────────


class TestReset:
    """reset() should return a valid initial observation."""

    def test_returns_observation(self):
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert isinstance(obs, GlucoObservation)

    def test_glucose_plausible_range(self):
        """Initial glucose should be in physiologically plausible range."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert 40.0 <= obs.glucose_mg_dl <= 400.0, (
            f"Initial glucose {obs.glucose_mg_dl} outside plausible range"
        )

    def test_step_zero(self):
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert obs.step == 0

    def test_done_false(self):
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert obs.done is False

    def test_trend_stable_initially(self):
        """With only one reading, trend should be stable."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert obs.glucose_trend == "stable"

    def test_task_2_patient(self):
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=2, seed=42)
        assert obs.patient_id == "adult#001"

    def test_task_3_patient_hidden(self):
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=3, seed=42)
        assert obs.patient_id is None

    def test_reset_clears_state(self):
        """Reset should clear all episode state."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(10):
            env.step(action)

        # Reset should start fresh
        obs = env.reset(task_id=1, seed=42)
        assert obs.step == 0
        state = env.state
        assert state.step == 0
        assert len(state.glucose_history) == 1  # Only initial reading


# ── Step ─────────────────────────────────────────────────────────────


class TestStep:
    """step() should advance the simulation correctly."""

    def test_step_advances_counter(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        obs = env.step(action)
        assert obs.step == 1

    def test_multiple_steps_advance(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for i in range(10):
            obs = env.step(action)
        assert obs.step == 10

    def test_returns_reward(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        obs = env.step(action)
        assert obs.reward is not None
        assert isinstance(obs.reward, (int, float))

    def test_glucose_changes(self):
        """Glucose should change over multiple steps (simulation is running)."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=2, seed=42)
        initial = obs.glucose_mg_dl
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        # Run past a meal to ensure glucose changes
        for _ in range(150):
            obs = env.step(action)
        assert obs.glucose_mg_dl != initial, "Glucose should change after meals"

    def test_borderline_action_zero_basal(self):
        """basal_rate=0.0 should not crash."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=0.0, bolus_dose=0.0)
        obs = env.step(action)
        assert isinstance(obs, GlucoObservation)

    def test_borderline_action_max_bolus(self):
        """bolus_dose=20.0 should not crash."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=20.0)
        obs = env.step(action)
        assert isinstance(obs, GlucoObservation)

    def test_borderline_action_max_basal(self):
        """basal_rate=5.0 should not crash."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=5.0, bolus_dose=0.0)
        obs = env.step(action)
        assert isinstance(obs, GlucoObservation)

    def test_extreme_combined_action(self):
        """Max basal + max bolus should not crash."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=5.0, bolus_dose=20.0)
        obs = env.step(action)
        assert isinstance(obs, GlucoObservation)

    def test_time_of_day_advances(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        obs = env.step(action)
        assert obs.time_of_day_hours > 0.0


# ── Episode termination ──────────────────────────────────────────────


class TestTermination:
    """Episode should terminate correctly."""

    def test_done_after_480_steps(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        done = False
        for step in range(480):
            obs = env.step(action)
            if obs.done:
                done = True
                break
        assert done, "Episode should end after 480 steps"
        assert obs.step == 480

    def test_done_stays_done(self):
        """After done=True, subsequent steps should still return done=True."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(480):
            obs = env.step(action)
        assert obs.done is True
        # Step again after done
        obs2 = env.step(action)
        assert obs2.done is True

    def test_emergency_termination_severe_hypo(self):
        """5 consecutive severe hypo events should terminate the episode."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        # Massive insulin to crash glucose
        action = GlucoAction(basal_rate=5.0, bolus_dose=20.0)
        terminated_early = False
        for step in range(480):
            obs = env.step(action)
            if obs.done and step < 479:
                terminated_early = True
                break
        assert terminated_early, (
            "Massive insulin should cause early termination via severe hypo"
        )


# ── State ────────────────────────────────────────────────────────────


class TestState:
    """state property should return consistent data."""

    def test_returns_gluco_state(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        state = env.state
        assert isinstance(state, GlucoState)

    def test_glucose_history_length(self):
        """glucose_history should have initial + N step readings."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(25):
            env.step(action)
        state = env.state
        # 1 initial + 25 steps = 26
        assert len(state.glucose_history) == 26, (
            f"Expected 26, got {len(state.glucose_history)}"
        )

    def test_reward_history_length(self):
        """reward_history should have exactly N entries after N steps."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(25):
            env.step(action)
        state = env.state
        assert len(state.reward_history) == 25

    def test_step_count_matches(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(37):
            env.step(action)
        state = env.state
        assert state.step == 37
        assert state.step_count == 37

    def test_tir_computed(self):
        """TIR should reflect actual in-range readings."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(100):
            env.step(action)
        state = env.state
        # Task 1 with reasonable basal should have high TIR
        assert state.tir_current > 0.5, (
            f"TIR should be >50% with constant basal on Task 1, got {state.tir_current}"
        )

    def test_episode_id_set(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42, episode_id="test-ep-001")
        state = env.state
        assert state.episode_id == "test-ep-001"

    def test_task_id_set(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=2, seed=42)
        state = env.state
        assert state.task_id == 2


# ── Meal announcements ───────────────────────────────────────────────


class TestMealAnnouncements:
    """Meal announcements should only appear in Task 2."""

    def test_task_1_no_announcements(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(100):
            obs = env.step(action)
            assert obs.meal_announced is False

    def test_task_2_announces_before_meal(self):
        """Meal should be announced in the 10 steps before step 100."""
        env = GlucoRLEnvironment()
        env.reset(task_id=2, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        announced = False
        for step in range(100):
            obs = env.step(action)
            if obs.meal_announced:
                announced = True
                assert obs.meal_grams_announced == 50.0
                assert 90 <= obs.step <= 100
        assert announced, "Meal should be announced before step 100"

    def test_task_3_no_announcements(self):
        env = GlucoRLEnvironment()
        env.reset(task_id=3, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(150):
            obs = env.step(action)
            assert obs.meal_announced is False
            if obs.done:
                break
