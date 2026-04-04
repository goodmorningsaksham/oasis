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


def score_task_4(state: GlucoState) -> float:
    """
    Task 4 grader — Sick Day Management.

    Harder than Task 3: agent must detect and adapt to unknown insulin
    resistance from simulated illness. Illness makes hyperglycemia more
    likely, so severe hyper (>300 mg/dL) is penalised more heavily.

    Expected scores: constant_basal ~0.05-0.15, PID ~0.10-0.20, good RL ~0.45+

    Args:
        state: Completed episode state.

    Returns:
        Score in [0.0, 1.0].
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

    # Illness makes hyper more likely — penalise severe hyper (>300) harder
    severe_hyper_steps = sum(1 for g in glucose_history if g > 300.0)
    severe_hyper_penalty = min(0.4, severe_hyper_steps / total * 2)

    score = tir - (state.severe_hypo_events * 0.15) - severe_hyper_penalty
    return max(0.0, min(1.0, score))


# Grader dispatch by task ID
GRADERS = {
    1: score_task_1,
    2: score_task_2,
    3: score_task_3,
    4: score_task_4,
}


def grade(task_id: int, state: GlucoState) -> float:
    """
    Score a completed episode for the given task.

    Args:
        task_id: Task number (1, 2, 3, or 4).
        state: Completed episode state.

    Returns:
        Score in [0.0, 1.0].

    Raises:
        ValueError: If task_id is not 1, 2, 3, or 4.
    """
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, 3, or 4.")
    return grader(state)


def grade_detailed(task_id: int, state: GlucoState) -> dict:
    """
    Return full decomposed score breakdown for a completed episode.

    Uses identical logic to grade() but exposes all intermediate components
    for research and debugging.

    Args:
        task_id: Task number (1, 2, 3, or 4).
        state: Completed episode state.

    Returns:
        Dict with keys:
            total: float             — final clamped score [0.0, 1.0]
            tir_score: float         — raw Time-in-Range fraction
            tir_readings: int        — number of steps in target range
            total_readings: int      — total steps scored
            hypo_penalty: float      — penalty applied for severe hypo events
            post_meal_penalties: dict — per-meal spike penalties (Task 2 only)
            bonus: float             — any bonus applied (e.g. Task 1 no-hypo bonus)
            components: dict         — all additive components before clamping
            clinical_summary: dict   — hypo_events, hyper_events, severe_hypo_events

    Raises:
        ValueError: If task_id is not 1, 2, 3, or 4.
    """
    if task_id not in (1, 2, 3, 4):
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, 3, or 4.")

    glucose_history = state.glucose_history[1:]  # exclude initial reading
    if not glucose_history:
        return {
            "total": 0.0,
            "tir_score": 0.0,
            "tir_readings": 0,
            "total_readings": 0,
            "hypo_penalty": 0.0,
            "post_meal_penalties": {},
            "bonus": 0.0,
            "components": {},
            "clinical_summary": {
                "hypo_events": state.hypo_events,
                "hyper_events": state.hyper_events,
                "severe_hypo_events": state.severe_hypo_events,
            },
        }

    total_readings = len(glucose_history)
    tir_readings = sum(
        1 for g in glucose_history
        if GLUCOSE_TARGET_LOW <= g <= GLUCOSE_TARGET_HIGH
    )
    tir_score = tir_readings / total_readings

    clinical_summary = {
        "hypo_events": state.hypo_events,
        "hyper_events": state.hyper_events,
        "severe_hypo_events": state.severe_hypo_events,
    }

    if task_id == 1:
        bonus = 0.05 if state.severe_hypo_events == 0 else 0.0
        severe_hypo_penalty = state.severe_hypo_events * 0.1
        raw = tir_score + bonus - severe_hypo_penalty
        total = max(0.0, min(1.0, raw))
        return {
            "total": total,
            "tir_score": tir_score,
            "tir_readings": tir_readings,
            "total_readings": total_readings,
            "hypo_penalty": severe_hypo_penalty,
            "post_meal_penalties": {},
            "bonus": bonus,
            "components": {
                "tir": tir_score,
                "no_hypo_bonus": bonus,
                "severe_hypo_penalty": -severe_hypo_penalty,
            },
            "clinical_summary": clinical_summary,
        }

    elif task_id == 2:
        full_history = state.glucose_history
        post_meal_penalties = {}
        post_meal_total = 0.0

        for meal_step, _cho in MEAL_SCHEDULE.items():
            start_idx = meal_step
            end_idx = min(meal_step + 60, len(full_history))
            if start_idx >= len(full_history):
                post_meal_penalties[f"step_{meal_step}"] = {
                    "peak": None, "penalty": 0.0,
                }
                continue
            window = full_history[start_idx:end_idx]
            if not window:
                post_meal_penalties[f"step_{meal_step}"] = {
                    "peak": None, "penalty": 0.0,
                }
                continue
            peak = max(window)
            if peak > 250:
                penalty = 0.15
            elif peak > 200:
                penalty = 0.08
            elif peak > 180:
                penalty = 0.03
            else:
                penalty = 0.0
            post_meal_penalties[f"step_{meal_step}"] = {
                "peak": round(peak, 1), "penalty": penalty,
            }
            post_meal_total += penalty

        hypo_penalty = min(0.3, state.severe_hypo_events * 0.1)
        raw = tir_score - post_meal_total - hypo_penalty
        total = max(0.0, min(1.0, raw))

        return {
            "total": total,
            "tir_score": tir_score,
            "tir_readings": tir_readings,
            "total_readings": total_readings,
            "hypo_penalty": hypo_penalty,
            "post_meal_penalties": post_meal_penalties,
            "bonus": 0.0,
            "components": {
                "tir": tir_score,
                "post_meal_spike_penalty": -post_meal_total,
                "severe_hypo_penalty": -hypo_penalty,
            },
            "clinical_summary": clinical_summary,
        }

    elif task_id == 3:
        hypo_penalty = state.severe_hypo_events * 0.15
        raw = tir_score - hypo_penalty
        total = max(0.0, raw)

        return {
            "total": total,
            "tir_score": tir_score,
            "tir_readings": tir_readings,
            "total_readings": total_readings,
            "hypo_penalty": hypo_penalty,
            "post_meal_penalties": {},
            "bonus": 0.0,
            "components": {
                "tir": tir_score,
                "severe_hypo_penalty": -hypo_penalty,
            },
            "clinical_summary": clinical_summary,
        }

    else:  # task_id == 4
        severe_hyper_steps = sum(1 for g in glucose_history if g > 300.0)
        severe_hyper_penalty = min(0.4, severe_hyper_steps / total_readings * 2)
        hypo_penalty = state.severe_hypo_events * 0.15
        raw = tir_score - hypo_penalty - severe_hyper_penalty
        total = max(0.0, min(1.0, raw))

        return {
            "total": total,
            "tir_score": tir_score,
            "tir_readings": tir_readings,
            "total_readings": total_readings,
            "hypo_penalty": hypo_penalty,
            "post_meal_penalties": {},
            "bonus": 0.0,
            "components": {
                "tir": tir_score,
                "severe_hypo_penalty": -hypo_penalty,
                "severe_hyper_penalty": -severe_hyper_penalty,
                "severe_hyper_steps": severe_hyper_steps,
            },
            "clinical_summary": clinical_summary,
        }
