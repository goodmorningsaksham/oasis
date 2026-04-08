"""
Tests for the GlucoRL task graders.

Verifies that all graders:
  - Return a float in [0.0, 1.0]
  - Are deterministic (same input → same output)
  - Score correctly for known glucose histories
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import GlucoState
from server.graders import score_task_1, score_task_2, score_task_3, score_task_4, grade, grade_detailed


def _make_state(
    glucose_history: list[float],
    task_id: int = 1,
    severe_hypo_events: int = 0,
    hypo_events: int = 0,
    hyper_events: int = 0,
) -> GlucoState:
    """Helper to build a GlucoState with a known glucose history."""
    return GlucoState(
        task_id=task_id,
        patient_name="adult#001",
        step=len(glucose_history) - 1,
        done=True,
        glucose_history=glucose_history,
        reward_history=[0.0] * (len(glucose_history) - 1),
        tir_current=0.0,
        hypo_events=hypo_events,
        severe_hypo_events=severe_hypo_events,
        hyper_events=hyper_events,
        episode_reward_total=0.0,
    )


# ── Grader output range ──────────────────────────────────────────────


class TestGraderRange:
    """All graders must return float in [0.0, 1.0]."""

    def test_task_1_range(self):
        state = _make_state([140.0] + [150.0] * 480)
        score = score_task_1(state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_task_2_range(self):
        state = _make_state([140.0] + [150.0] * 480)
        score = score_task_2(state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_task_3_range(self):
        state = _make_state([140.0] + [150.0] * 480, task_id=3)
        score = score_task_3(state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_task_4_range(self):
        state = _make_state([140.0] + [150.0] * 480, task_id=4)
        score = score_task_4(state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_history(self):
        """Empty history (only initial reading) should return SCORE_MIN, not crash."""
        state = _make_state([140.0])
        assert score_task_1(state) == 0.01
        assert score_task_2(state) == 0.01
        assert score_task_3(state) == 0.01
        assert score_task_4(state) == 0.01

    def test_grade_dispatch(self):
        """grade() dispatches correctly to per-task graders."""
        state = _make_state([140.0] + [150.0] * 480)
        for task_id in [1, 2, 3, 4]:
            score = grade(task_id, state)
            assert 0.0 <= score <= 1.0

    def test_grade_invalid_task(self):
        """grade() raises ValueError for unknown task_id."""
        state = _make_state([140.0] + [150.0] * 480)
        try:
            grade(99, state)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── Determinism ──────────────────────────────────────────────────────


class TestGraderDeterminism:
    """Same glucose history must always produce the same score."""

    def test_task_1_deterministic(self):
        history = [140.0] + [120.0 + (i % 30) for i in range(480)]
        state = _make_state(history)
        scores = [score_task_1(state) for _ in range(5)]
        assert all(s == scores[0] for s in scores), f"Non-deterministic: {scores}"

    def test_task_2_deterministic(self):
        history = [140.0] + [120.0 + (i % 50) for i in range(480)]
        state = _make_state(history)
        scores = [score_task_2(state) for _ in range(5)]
        assert all(s == scores[0] for s in scores), f"Non-deterministic: {scores}"

    def test_task_3_deterministic(self):
        history = [140.0] + [130.0 + (i % 40) for i in range(480)]
        state = _make_state(history, task_id=3)
        scores = [score_task_3(state) for _ in range(5)]
        assert all(s == scores[0] for s in scores), f"Non-deterministic: {scores}"


# ── Task 1 scoring correctness ──────────────────────────────────────


class TestTask1Scoring:
    """Task 1 grader scoring logic."""

    def test_all_in_range_high_score(self):
        """All glucose in [70, 180] should score > 0.9."""
        history = [140.0] + [120.0] * 480
        state = _make_state(history, severe_hypo_events=0)
        score = score_task_1(state)
        assert score > 0.9, f"All in-range should score >0.9, got {score}"

    def test_perfect_tir_no_hypo_gets_bonus(self):
        """100% TIR with no severe hypo should score 0.999 (clamped from 1.05)."""
        history = [140.0] + [140.0] * 480
        state = _make_state(history, severe_hypo_events=0)
        score = score_task_1(state)
        assert score == 0.99

    def test_severe_hypo_penalty(self):
        """10 severe hypo events should score < 0.2."""
        history = [140.0] + [45.0] * 50 + [120.0] * 430
        state = _make_state(history, severe_hypo_events=10)
        score = score_task_1(state)
        assert score < 0.2, f"10 severe hypo should score <0.2, got {score}"

    def test_all_hypo_scores_low(self):
        """All glucose below range should score SCORE_MIN."""
        history = [60.0] + [50.0] * 480
        state = _make_state(history, severe_hypo_events=480)
        score = score_task_1(state)
        assert score == 0.01

    def test_half_in_range(self):
        """50% in range should give roughly 0.5 (plus/minus penalty)."""
        history = [140.0] + [120.0] * 240 + [200.0] * 240
        state = _make_state(history, severe_hypo_events=0)
        score = score_task_1(state)
        assert 0.4 <= score <= 0.6, f"50% TIR should be ~0.5, got {score}"


# ── Task 2 scoring correctness ──────────────────────────────────────


class TestTask2Scoring:
    """Task 2 grader includes post-meal spike penalties."""

    def test_no_spikes_high_score(self):
        """All glucose in range with no meal spikes should score well."""
        history = [140.0] + [130.0] * 480
        state = _make_state(history, severe_hypo_events=0)
        score = score_task_2(state)
        assert score > 0.9, f"Perfect control should score >0.9, got {score}"

    def test_post_meal_severe_spikes_penalty(self):
        """Glucose > 250 after meals should penalise by 0.15 per meal."""
        # Build history with spikes at meal steps (100, 200, 320)
        history = [140.0] + [130.0] * 480
        # Inject spikes: steps 100-160 = 260 mg/dL
        for i in range(100, 160):
            if i < len(history):
                history[i] = 260.0
        for i in range(200, 260):
            if i < len(history):
                history[i] = 260.0
        for i in range(320, 380):
            if i < len(history):
                history[i] = 260.0

        state = _make_state(history, severe_hypo_events=0, hyper_events=180)
        score = score_task_2(state)
        # 3 severe spikes = -0.45, so score should be noticeably reduced
        assert score < 0.8, f"3 severe spikes should reduce score, got {score}"

    def test_hypo_penalty_caps(self):
        """Severe hypo penalty caps at 0.3."""
        history = [140.0] + [130.0] * 480
        state = _make_state(history, severe_hypo_events=10)
        score = score_task_2(state)
        # TIR ~1.0 - hypo_penalty 0.3 = ~0.7
        assert 0.6 <= score <= 0.75, f"Expected ~0.7 with capped hypo penalty, got {score}"


# ── Task 3 scoring correctness ──────────────────────────────────────


class TestTask3Scoring:
    """Task 3 grader uses TIR minus severe hypo penalty."""

    def test_perfect_episode(self):
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=3, severe_hypo_events=0)
        score = score_task_3(state)
        assert score == 0.99

    def test_severe_hypo_reduces_score(self):
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=3, severe_hypo_events=3)
        score = score_task_3(state)
        # TIR 1.0 - 3*0.15 = 0.55
        assert abs(score - 0.55) < 0.01, f"Expected ~0.55, got {score}"

    def test_floor_at_zero(self):
        """Score should not go below SCORE_MIN."""
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=3, severe_hypo_events=20)
        score = score_task_3(state)
        assert score == 0.01


# ── Detailed grader breakdown ────────────────────────────────────────


class TestDetailedGrader:
    """grade_detailed() returns full score decomposition."""

    def test_returns_dict_with_required_keys(self):
        history = [140.0] + [120.0] * 480
        state = _make_state(history)
        result = grade_detailed(1, state)
        required_keys = [
            "total", "tir_score", "tir_readings", "total_readings",
            "hypo_penalty", "post_meal_penalties", "bonus",
            "components", "clinical_summary",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_total_matches_grade(self):
        """grade_detailed total should exactly match grade() output."""
        history = [140.0] + [120.0] * 240 + [200.0] * 240
        state = _make_state(history, severe_hypo_events=0)
        for task_id in [1, 2, 3, 4]:
            simple = grade(task_id, state)
            detailed = grade_detailed(task_id, state)
            assert abs(simple - detailed["total"]) < 1e-9, (
                f"Task {task_id}: grade()={simple} != grade_detailed()={detailed['total']}"
            )

    def test_all_values_are_finite(self):
        """All numeric values in the result should be finite floats or ints."""
        import math
        history = [140.0] + [150.0] * 480
        state = _make_state(history)
        for task_id in [1, 2, 3, 4]:
            result = grade_detailed(task_id, state)
            assert math.isfinite(result["total"])
            assert math.isfinite(result["tir_score"])
            assert isinstance(result["tir_readings"], int)
            assert isinstance(result["total_readings"], int)
            assert math.isfinite(result["hypo_penalty"])

    def test_task_1_components(self):
        """Task 1 detailed result should include no_hypo_bonus."""
        history = [140.0] + [120.0] * 480
        state = _make_state(history, severe_hypo_events=0)
        result = grade_detailed(1, state)
        assert result["bonus"] == 0.05
        assert "no_hypo_bonus" in result["components"]
        assert result["tir_score"] == 1.0

    def test_task_2_post_meal_penalties(self):
        """Task 2 detailed result should include per-meal spike info."""
        history = [140.0] + [130.0] * 480
        # Inject spike at meal step 100
        for i in range(100, 160):
            if i < len(history):
                history[i] = 260.0
        state = _make_state(history)
        result = grade_detailed(2, state)
        assert "step_100" in result["post_meal_penalties"]
        meal_100 = result["post_meal_penalties"]["step_100"]
        assert meal_100["penalty"] == 0.15  # peak > 250
        assert meal_100["peak"] == 260.0

    def test_task_3_detailed(self):
        """Task 3 detailed result should include severe_hypo_penalty component."""
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=3, severe_hypo_events=2)
        result = grade_detailed(3, state)
        assert abs(result["hypo_penalty"] - 0.3) < 1e-9  # 2 * 0.15
        assert "severe_hypo_penalty" in result["components"]

    def test_empty_history(self):
        """Empty history should return SCORE_MIN breakdown."""
        state = _make_state([140.0])
        result = grade_detailed(1, state)
        assert result["total"] == 0.01
        assert result["total_readings"] == 0

    def test_invalid_task_raises(self):
        """Invalid task_id should raise ValueError."""
        state = _make_state([140.0] + [120.0] * 480)
        try:
            grade_detailed(99, state)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── Task 4 scoring correctness ──────────────────────────────────────


class TestTask4Scoring:
    """Task 4 grader includes severe hyper penalty for illness-driven spikes."""

    def test_perfect_control(self):
        """All glucose in range should score high."""
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=4, severe_hypo_events=0)
        score = score_task_4(state)
        assert score == 0.99

    def test_severe_hyper_penalty(self):
        """Many steps above 300 mg/dL should reduce score significantly."""
        # 200 steps at 310 (severe hyper) + 280 steps at 120 (in range)
        history = [140.0] + [310.0] * 200 + [120.0] * 280
        state = _make_state(history, task_id=4, severe_hypo_events=0, hyper_events=200)
        score = score_task_4(state)
        # TIR = 280/480 = 0.583, severe_hyper_penalty = min(0.4, 200/480*2) = 0.4
        assert score < 0.3, f"Severe hyper should heavily penalise, got {score}"

    def test_hypo_penalty(self):
        """Severe hypo events should also penalise."""
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=4, severe_hypo_events=3)
        score = score_task_4(state)
        # TIR 1.0 - 3*0.15 = 0.55
        assert abs(score - 0.55) < 0.01, f"Expected ~0.55, got {score}"

    def test_combined_penalties(self):
        """Both hypo and hyper penalties should stack."""
        history = [140.0] + [310.0] * 100 + [120.0] * 380
        state = _make_state(history, task_id=4, severe_hypo_events=2, hyper_events=100)
        score = score_task_4(state)
        assert score < 0.6, f"Combined penalties should reduce score, got {score}"

    def test_floor_at_zero(self):
        """Score should not go below SCORE_MIN."""
        history = [140.0] + [310.0] * 480
        state = _make_state(history, task_id=4, severe_hypo_events=10, hyper_events=480)
        score = score_task_4(state)
        assert score == 0.01

    def test_detailed_has_severe_hyper(self):
        """grade_detailed for Task 4 should include severe_hyper components."""
        history = [140.0] + [310.0] * 100 + [120.0] * 380
        state = _make_state(history, task_id=4)
        result = grade_detailed(4, state)
        assert "severe_hyper_penalty" in result["components"]
        assert "severe_hyper_steps" in result["components"]
        assert result["components"]["severe_hyper_steps"] == 100