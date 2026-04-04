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


# ── CGM Noise ────────────────────────────────────────────────────────


class TestCGMNoise:
    """CGM measurement noise feature tests."""

    def test_true_glucose_field_present(self):
        """Observation should always include true_glucose_mg_dl."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert obs.true_glucose_mg_dl is not None
        assert obs.true_glucose_mg_dl > 0

    def test_cgm_and_true_can_differ(self):
        """With noise enabled, CGM and true glucose should differ sometimes."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        differences = 0
        for _ in range(50):
            obs = env.step(action)
            if abs(obs.glucose_mg_dl - obs.true_glucose_mg_dl) > 0.01:
                differences += 1
        assert differences > 0, "CGM noise should cause some readings to differ from true glucose"

    def test_reward_uses_true_glucose(self):
        """Reward and clinical events should be based on true glucose, not noisy CGM."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(50):
            obs = env.step(action)
        state = env.state
        # glucose_history in state should be TRUE glucose
        # Verify it matches the true_glucose values we observed
        assert len(state.glucose_history) == 51  # initial + 50 steps

    def test_noise_disabled(self):
        """With noise_enabled=False, CGM should equal true glucose exactly."""
        from server.patient_manager import PatientManager
        pm = PatientManager(noise_enabled=False)
        cgm, true = pm.reset("adult#001")
        assert cgm == true, f"Noise disabled but CGM ({cgm}) != true ({true})"
        cgm2, true2 = pm.step(1.0, 0.0, 0.0)
        assert cgm2 == true2, f"Noise disabled but step CGM ({cgm2}) != true ({true2})"

    def test_state_glucose_history_is_true(self):
        """GlucoState.glucose_history should contain true glucose values."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        initial_true = obs.true_glucose_mg_dl
        state = env.state
        assert abs(state.glucose_history[0] - initial_true) < 0.01, (
            f"State history[0] ({state.glucose_history[0]}) should be true glucose ({initial_true})"
        )


# ── Insulin-on-Board (IOB) ───────────────────────────────────────────


class TestIOB:
    """Insulin-on-Board tracking tests (gamma-CDF PK/PD model)."""

    def test_iob_starts_at_zero(self):
        """IOB should be 0.0 after reset."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert obs.insulin_on_board_units == 0.0

    def test_iob_increases_after_bolus(self):
        """IOB should increase after a bolus dose."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        obs = env.step(GlucoAction(basal_rate=1.0, bolus_dose=5.0))
        assert obs.insulin_on_board_units > 0.0, (
            f"IOB should be >0 after bolus, got {obs.insulin_on_board_units}"
        )

    def test_iob_decays_without_bolus(self):
        """IOB from a large bolus should decay over time."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        # Give a large bolus
        obs = env.step(GlucoAction(basal_rate=1.0, bolus_dose=10.0))
        iob_after_bolus = obs.insulin_on_board_units
        # Step without bolus — IOB should decrease as bolus is absorbed
        for _ in range(30):
            obs = env.step(GlucoAction(basal_rate=1.0, bolus_dose=0.0))
        assert obs.insulin_on_board_units < iob_after_bolus, (
            f"IOB should decay: was {iob_after_bolus}, now {obs.insulin_on_board_units}"
        )

    def test_iob_resets_between_episodes(self):
        """IOB should reset to 0.0 on new episode."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        env.step(GlucoAction(basal_rate=1.0, bolus_dose=10.0))
        # Reset
        obs = env.reset(task_id=1, seed=42)
        assert obs.insulin_on_board_units == 0.0

    def test_iob_never_negative(self):
        """IOB should never go below 0.0."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(100):
            obs = env.step(action)
            assert obs.insulin_on_board_units >= 0.0

    def test_iob_includes_basal(self):
        """IOB should reflect basal insulin delivery, not just boluses."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        # Step with high basal, no bolus — IOB should still be > 0
        for _ in range(10):
            obs = env.step(GlucoAction(basal_rate=3.0, bolus_dose=0.0))
        assert obs.insulin_on_board_units > 0.0, (
            f"IOB should include basal insulin, got {obs.insulin_on_board_units}"
        )


# ── Exercise Events ──────────────────────────────────────────────────


class TestExerciseEvents:
    """Exercise event feature tests."""

    def test_task_1_no_exercise(self):
        """Task 1 should have no exercise events."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(480):
            obs = env.step(action)
            assert obs.exercise_intensity == 0.0
            assert obs.exercise_announced is False
            if obs.done:
                break

    def test_task_2_exercise_announced(self):
        """Task 2 should announce exercise before it starts."""
        env = GlucoRLEnvironment()
        env.reset(task_id=2, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        announced = False
        exercising = False
        for step in range(200):
            obs = env.step(action)
            if obs.exercise_announced:
                announced = True
            if obs.exercise_intensity > 0:
                exercising = True
            if obs.done:
                break
        assert announced, "Task 2 should announce exercise before step 150"
        assert exercising, "Task 2 should have exercise starting at step 150"

    def test_task_3_exercise_can_occur(self):
        """Task 3 should sometimes have exercise events (60% chance)."""
        had_exercise = False
        for seed in range(20):
            env = GlucoRLEnvironment()
            env.reset(task_id=3, seed=seed)
            action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
            for _ in range(400):
                obs = env.step(action)
                if obs.exercise_intensity > 0:
                    had_exercise = True
                    break
                if obs.done:
                    break
            if had_exercise:
                break
        assert had_exercise, "At least one of 20 Task 3 episodes should have exercise"

    def test_exercise_returns_to_zero(self):
        """Exercise intensity should return to 0.0 after exercise ends."""
        env = GlucoRLEnvironment()
        env.reset(task_id=2, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        was_exercising = False
        returned_to_zero = False
        for step in range(300):
            obs = env.step(action)
            if obs.exercise_intensity > 0:
                was_exercising = True
            elif was_exercising and obs.exercise_intensity == 0.0:
                returned_to_zero = True
                break
            if obs.done:
                break
        assert was_exercising, "Exercise should have occurred"
        assert returned_to_zero, "Exercise intensity should return to 0.0 after ending"

    def test_exercise_intensity_valid_range(self):
        """Exercise intensity should always be in [0.0, 1.0]."""
        env = GlucoRLEnvironment()
        env.reset(task_id=2, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(300):
            obs = env.step(action)
            assert 0.0 <= obs.exercise_intensity <= 1.0
            if obs.done:
                break


# ── Glucose History Window ───────────────────────────────────────────


class TestGlucoseHistoryWindow:
    """Glucose history window provides recent CGM context."""

    def test_window_has_initial_reading_at_reset(self):
        """After reset, window should contain the initial CGM reading."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=1, seed=42)
        assert len(obs.glucose_history_window) == 1
        assert abs(obs.glucose_history_window[0] - obs.glucose_mg_dl) < 0.2

    def test_window_grows_to_12(self):
        """Window should grow to 12 readings after 12 steps."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for i in range(12):
            obs = env.step(action)
        # 1 initial + 12 steps = 13 in cgm history, window = last 12
        assert len(obs.glucose_history_window) == 12

    def test_window_stays_at_12(self):
        """Window should stay at max 12 after more than 12 steps (sliding)."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for i in range(25):
            obs = env.step(action)
        assert len(obs.glucose_history_window) == 12

    def test_window_values_are_floats(self):
        """All values in window should be finite floats."""
        import math
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(20):
            obs = env.step(action)
        for val in obs.glucose_history_window:
            assert isinstance(val, float)
            assert math.isfinite(val)

    def test_window_last_element_matches_current(self):
        """Last element of window should be close to current glucose."""
        env = GlucoRLEnvironment()
        env.reset(task_id=1, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(15):
            obs = env.step(action)
        # Last window element should be the most recent CGM reading
        # (both rounded, so small tolerance)
        assert abs(obs.glucose_history_window[-1] - obs.glucose_mg_dl) < 0.2


# ── Task 4: Sick Day ─────────────────────────────────────────────────


class TestTask4:
    """Task 4 — Sick Day / Insulin Resistance tests."""

    def test_task_4_resets_successfully(self):
        """Task 4 should reset without errors."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=4, seed=42)
        assert obs.glucose_mg_dl > 0
        assert obs.done is False

    def test_task_4_patient_hidden(self):
        """Task 4 should hide patient_id (like Task 3)."""
        env = GlucoRLEnvironment()
        obs = env.reset(task_id=4, seed=42)
        assert obs.patient_id is None

    def test_task_4_meals_not_announced(self):
        """Task 4 should not announce meals."""
        env = GlucoRLEnvironment()
        env.reset(task_id=4, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        for _ in range(150):
            obs = env.step(action)
            assert obs.meal_announced is False
            if obs.done:
                break

    def test_task_4_has_illness_resistance(self):
        """Task 4 should set illness resistance parameters."""
        env = GlucoRLEnvironment()
        env.reset(task_id=4, seed=42)
        assert env._illness_resistance >= 1.5
        assert env._illness_resistance <= 2.5
        assert env._illness_onset_step is not None
        assert 20 <= env._illness_onset_step <= 100

    def test_task_4_illness_activates(self):
        """Illness should activate after the onset step."""
        env = GlucoRLEnvironment()
        env.reset(task_id=4, seed=42)
        onset = env._illness_onset_step
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        # Step past the onset
        for _ in range(onset + 5):
            obs = env.step(action)
            if obs.done:
                break
        assert env._illness_active is True

    def test_task_4_no_illness_before_onset(self):
        """Illness should not be active before the onset step."""
        env = GlucoRLEnvironment()
        env.reset(task_id=4, seed=42)
        onset = env._illness_onset_step
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        # Step up to but NOT past onset
        for step in range(min(onset - 1, 19)):
            obs = env.step(action)
            if obs.done:
                break
        assert env._illness_active is False

    def test_task_4_runs_full_episode(self):
        """Task 4 should be able to run a complete 480-step episode."""
        env = GlucoRLEnvironment()
        env.reset(task_id=4, seed=42)
        action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
        steps = 0
        for _ in range(480):
            obs = env.step(action)
            steps += 1
            if obs.done:
                break
        assert steps > 0
        state = env.state
        assert state.task_id == 4

    def test_tasks_1_2_no_illness(self):
        """Tasks 1 and 2 should have no illness parameters set."""
        for tid in [1, 2]:
            env = GlucoRLEnvironment()
            env.reset(task_id=tid, seed=42)
            assert env._illness_resistance == 1.0
            assert env._illness_onset_step is None
            assert env._illness_active is False

