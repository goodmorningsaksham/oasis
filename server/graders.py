"""
Task graders for the GlucoRL environment.

Each grader takes a completed episode state and returns a deterministic
score in [0.0, 1.0]. The same glucose history always produces the same score.
"""

import logging
import random
from typing import Optional

from models import GlucoState
from server.constants import (
    GLUCOSE_TARGET_LOW,
    GLUCOSE_TARGET_HIGH,
    GLUCOSE_SEVERE_HYPO,
    MEAL_SCHEDULE,
    ALL_PATIENT_NAMES,
    TASK3_EVAL_PATIENTS,
)

logger = logging.getLogger(__name__)


def score_task_1(state: GlucoState) -> float:
    """
    Task 1 grader — Basal Rate Control.

    Score is based on Time-in-Range (TIR) with a bonus for no severe hypo
    and a penalty for each severe hypo event.

    Args:
        state: Completed episode state.

    Returns:
        Score in [0.0, 1.0].
    """
    glucose_history = state.glucose_history[1:]  # exclude initial reading
    if not glucose_history:
        return 0.0

    total = len(glucose_history)
    in_range = sum(
        1 for g in glucose_history
        if GLUCOSE_TARGET_LOW <= g <= GLUCOSE_TARGET_HIGH
    )
    tir = in_range / total

    score = tir

    # Bonus: no severe hypo events
    if state.severe_hypo_events == 0:
        score += 0.05

    # Penalty: subtract 0.1 for each severe hypo event
    score -= state.severe_hypo_events * 0.1

    return max(0.0, min(1.0, score))


def score_task_2(state: GlucoState) -> float:
    """
    Task 2 grader — Meal Bolus Timing.

    Score is TIR minus penalties for post-meal glucose spikes and
    severe hypoglycemia events.

    Args:
        state: Completed episode state.

    Returns:
        Score in [0.0, 1.0].
    """
    glucose_history = state.glucose_history[1:]  # exclude initial reading
    if not glucose_history:
        return 0.0

    total = len(glucose_history)
    in_range = sum(
        1 for g in glucose_history
        if GLUCOSE_TARGET_LOW <= g <= GLUCOSE_TARGET_HIGH
    )
    tir = in_range / total

    # Post-meal spike penalties
    # glucose_history[0] corresponds to step 1, so meal at step N is index N-1
    # But we use the full history (including initial) for indexing meals
    full_history = state.glucose_history
    post_meal_spike_penalty = 0.0

    for meal_step, _cho in MEAL_SCHEDULE.items():
        # Look at glucose from meal_step to meal_step + 60 (3 hours post-meal)
        start_idx = meal_step
        end_idx = min(meal_step + 60, len(full_history))
        if start_idx >= len(full_history):
            continue
        window = full_history[start_idx:end_idx]
        if not window:
            continue
        peak = max(window)
        if peak > 250:
            post_meal_spike_penalty += 0.15
        elif peak > 200:
            post_meal_spike_penalty += 0.08
        elif peak > 180:
            post_meal_spike_penalty += 0.03

    # Hypo penalty
    hypo_penalty = min(0.3, state.severe_hypo_events * 0.1)

    score = tir - post_meal_spike_penalty - hypo_penalty
    return max(0.0, min(1.0, score))


def score_task_3_single(state: GlucoState) -> float:
    """
    Score a single Task 3 episode.

    Args:
        state: Completed episode state.

    Returns:
        Per-episode score in [0.0, 1.0].
    """
    glucose_history = state.glucose_history[1:]
    if not glucose_history:
        return 0.0

    total = len(glucose_history)
    in_range = sum(
        1 for g in glucose_history
        if GLUCOSE_TARGET_LOW <= g <= GLUCOSE_TARGET_HIGH
    )
    tir = in_range / total

    score = tir - (state.severe_hypo_events * 0.15)
    return max(0.0, score)


def score_task_3(state: GlucoState) -> float:
    """
    Task 3 grader — Cross-Patient Generalisation.

    This is a convenience function that scores a single completed episode.
    The full Task 3 evaluation runs 5 episodes with different patients
    (using TASK3_EVAL_PATIENTS) and averages the per-episode scores.

    For a single-episode call, this returns the per-episode score.

    Args:
        state: Completed episode state.

    Returns:
        Score in [0.0, 1.0].
    """
    return score_task_3_single(state)


# Grader dispatch by task ID
GRADERS = {
    1: score_task_1,
    2: score_task_2,
    3: score_task_3,
}


def grade(task_id: int, state: GlucoState) -> float:
    """
    Score a completed episode for the given task.

    Args:
        task_id: Task number (1, 2, or 3).
        state: Completed episode state.

    Returns:
        Score in [0.0, 1.0].

    Raises:
        ValueError: If task_id is not 1, 2, or 3.
    """
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")
    return grader(state)
