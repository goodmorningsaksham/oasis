"""
Data models for the GlucoRL Environment.

Defines Action, Observation, State for insulin dosing RL training.
All models inherit from OpenEnv base types for spec compliance.
"""

from typing import Optional, Literal

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


class GlucoAction(Action):
    """
    Insulin dosing action taken by the agent each step (every 3 minutes).

    basal_rate: continuous background insulin in units/hr (0.0 to 5.0)
    bolus_dose: correction/meal insulin in total units (0.0 to 20.0)
    """
    basal_rate: float = Field(
        default=1.0, ge=0.0, le=5.0,
        description="Basal insulin rate in units/hr",
    )
    bolus_dose: float = Field(
        default=0.0, ge=0.0, le=20.0,
        description="Bolus insulin dose in units",
    )


class GlucoObservation(Observation):
    """
    What the agent observes at each step.

    Inherits from OpenEnv Observation which provides:
      - done: bool (whether episode has terminated)
      - reward: float | None (reward signal from last action)
      - metadata: dict (additional metadata)
    """
    glucose_mg_dl: float = Field(
        description="Current CGM glucose reading in mg/dL",
    )
    glucose_trend: Literal[
        "rapidly_falling", "falling", "stable", "rising", "rapidly_rising"
    ] = Field(
        description="CGM trend arrow based on rate of change",
    )
    meal_announced: bool = Field(
        default=False,
        description="True if a meal is coming in the next 30 minutes (Task 2 only)",
    )
    meal_grams_announced: float = Field(
        default=0.0,
        description="Carbohydrate grams in the announced upcoming meal",
    )
    time_of_day_hours: float = Field(
        description="Current time in simulated day (0.0 to 24.0 hours)",
    )
    step: int = Field(
        description="Current step number (0 to 479)",
    )
    patient_id: Optional[str] = Field(
        default=None,
        description="Patient identifier (None in Task 3 to force generalisation)",
    )
    last_action_basal: float = Field(
        default=1.0,
        description="Basal rate from previous step",
    )
    last_action_bolus: float = Field(
        default=0.0,
        description="Bolus dose from previous step",
    )


class GlucoReward(BaseModel):
    """
    Decomposed reward signal for the current step.
    """
    tir_contribution: float = Field(
        description="Reward for being in target range 70-180 mg/dL: +1.0 if in range",
    )
    hypo_penalty: float = Field(
        description="Penalty for hypoglycemia: -1.0 if <70, -3.0 if <54 mg/dL",
    )
    hyper_penalty: float = Field(
        description="Penalty for hyperglycemia: -0.5 if >180, -1.5 if >250 mg/dL",
    )
    overdose_penalty: float = Field(
        default=0.0,
        description="Penalty of -3.0 if glucose crashes below 54 within 2 steps of a bolus",
    )
    step_total: float = Field(
        description="Total reward for this step (sum of components)",
    )


class GlucoState(State):
    """
    Full environment state returned by the state property.

    Inherits from OpenEnv State which provides:
      - episode_id: Optional[str]
      - step_count: int (>= 0)
    """
    task_id: int = Field(
        description="Current task: 1, 2, or 3",
    )
    patient_name: str = Field(
        description="simglucose patient identifier",
    )
    step: int = Field(
        default=0,
        description="Current step in episode",
    )
    done: bool = Field(
        default=False,
        description="Whether episode has ended",
    )
    glucose_history: list[float] = Field(
        default_factory=list,
        description="Full glucose reading history for this episode",
    )
    reward_history: list[float] = Field(
        default_factory=list,
        description="Step reward history for this episode",
    )
    tir_current: float = Field(
        default=0.0,
        description="Current Time-in-Range percentage (0.0 to 1.0)",
    )
    hypo_events: int = Field(
        default=0,
        description="Number of hypoglycemia steps so far",
    )
    severe_hypo_events: int = Field(
        default=0,
        description="Steps below 54 mg/dL so far",
    )
    hyper_events: int = Field(
        default=0,
        description="Number of hyperglycemia steps so far",
    )
    episode_reward_total: float = Field(
        default=0.0,
        description="Cumulative reward for the episode",
    )
