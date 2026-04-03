"""
Reward calculator for the GlucoRL environment.

Provides a shaped, per-step reward signal based on current glucose level.
Rewards being in the 70-180 mg/dL target range, penalises hypoglycemia
(asymmetrically heavier — hypo is more acutely dangerous) and hyperglycemia,
and includes an overdose penalty for bolus-induced crashes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import GlucoReward
from server.constants import (
    GLUCOSE_TARGET_LOW,
    GLUCOSE_TARGET_HIGH,
    GLUCOSE_SEVERE_HYPO,
    GLUCOSE_SEVERE_HYPER,
)


def calculate_step_reward(
    glucose: float,
    prev_glucose: float,
    bolus_given: float,
    glucose_2_steps_ago: float,
    bolus_2_steps_ago: float,
) -> GlucoReward:
    """
    Calculate the decomposed reward for a single environment step.

    Args:
        glucose: Current blood glucose reading in mg/dL.
        prev_glucose: Glucose reading from the previous step.
        bolus_given: Bolus dose administered at the current step.
        glucose_2_steps_ago: Glucose reading from two steps ago.
        bolus_2_steps_ago: Bolus dose administered two steps ago.

    Returns:
        GlucoReward with individual components and step total.

    Reward design:
        +1.0  if glucose in [70, 180] mg/dL   (target range)
        -1.0  if glucose in [54, 70)           (mild hypoglycemia)
        -3.0  if glucose < 54                  (severe hypoglycemia)
        -0.5  if glucose in (180, 250]         (mild hyperglycemia)
        -1.5  if glucose > 250                 (severe hyperglycemia)
        -3.0  overdose penalty if glucose < 54 and bolus > 5 two steps ago
    """

    # 1. Time-in-Range contribution
    if GLUCOSE_TARGET_LOW <= glucose <= GLUCOSE_TARGET_HIGH:
        tir_contribution = 1.0
    else:
        tir_contribution = 0.0

    # 2. Hypoglycemia penalty (asymmetric — hypo is more dangerous)
    if glucose < GLUCOSE_SEVERE_HYPO:
        hypo_penalty = -3.0      # severe hypoglycemia — life threatening
    elif glucose < GLUCOSE_TARGET_LOW:
        hypo_penalty = -1.0      # mild hypoglycemia
    else:
        hypo_penalty = 0.0

    # 3. Hyperglycemia penalty
    if glucose > GLUCOSE_SEVERE_HYPER:
        hyper_penalty = -1.5     # severe hyperglycemia
    elif glucose > GLUCOSE_TARGET_HIGH:
        hyper_penalty = -0.5     # mild hyperglycemia
    else:
        hyper_penalty = 0.0

    # 4. Overdose penalty — punishes bolus that caused a crash
    overdose_penalty = 0.0
    if glucose < GLUCOSE_SEVERE_HYPO and bolus_2_steps_ago > 5.0:
        overdose_penalty = -3.0  # dangerous overdose pattern

    step_total = tir_contribution + hypo_penalty + hyper_penalty + overdose_penalty

    return GlucoReward(
        tir_contribution=tir_contribution,
        hypo_penalty=hypo_penalty,
        hyper_penalty=hyper_penalty,
        overdose_penalty=overdose_penalty,
        step_total=step_total,
    )
