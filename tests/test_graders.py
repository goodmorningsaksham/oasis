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
from server.graders import score_task_1, score_task_2, score_task_3, grade


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

    def test_empty_history(self):
        """Empty history (only initial reading) should return 0.0, not crash."""
        state = _make_state([140.0])
        assert score_task_1(state) == 0.0
        assert score_task_2(state) == 0.0
        assert score_task_3(state) == 0.0

    def test_grade_dispatch(self):
        """grade() dispatches correctly to per-task graders."""
        state = _make_state([140.0] + [150.0] * 480)
        for task_id in [1, 2, 3]:
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
        """100% TIR with no severe hypo should score 1.0 (TIR=1.0 + 0.05 bonus, capped)."""
        history = [140.0] + [140.0] * 480
        state = _make_state(history, severe_hypo_events=0)
        score = score_task_1(state)
        assert score == 1.0

    def test_severe_hypo_penalty(self):
        """10 severe hypo events should score < 0.2."""
        history = [140.0] + [45.0] * 50 + [120.0] * 430
        state = _make_state(history, severe_hypo_events=10)
        score = score_task_1(state)
        assert score < 0.2, f"10 severe hypo should score <0.2, got {score}"

    def test_all_hypo_scores_low(self):
        """All glucose below range should score near 0."""
        history = [60.0] + [50.0] * 480
        state = _make_state(history, severe_hypo_events=480)
        score = score_task_1(state)
        assert score == 0.0

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
        assert score == 1.0

    def test_severe_hypo_reduces_score(self):
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=3, severe_hypo_events=3)
        score = score_task_3(state)
        # TIR 1.0 - 3*0.15 = 0.55
        assert abs(score - 0.55) < 0.01, f"Expected ~0.55, got {score}"

    def test_floor_at_zero(self):
        """Score should not go below 0.0."""
        history = [140.0] + [120.0] * 480
        state = _make_state(history, task_id=3, severe_hypo_events=20)
        score = score_task_3(state)
        assert score == 0.0
