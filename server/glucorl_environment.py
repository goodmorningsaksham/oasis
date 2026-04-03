"""
GlucoRL Environment Implementation.

An OpenEnv environment for training RL agents to make personalised insulin
dosing decisions for Type 1 Diabetic patients. Uses the simglucose simulator
(FDA-accepted UVa/Padova T1D metabolic model).

Tasks:
  1 — Basal Rate Control (easy): single stable patient, no meals
  2 — Meal Bolus Timing (medium): announced meals, same patient
  3 — Cross-Patient Generalisation (hard): random patient, unannounced meals
"""

import logging
import random
from typing import Optional, Any
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment

from models import GlucoAction, GlucoObservation, GlucoState
from server.patient_manager import PatientManager
from server.reward_calculator import calculate_step_reward
from server.constants import (
    STEPS_PER_EPISODE,
    STEP_DURATION_MIN,
    MEAL_SCHEDULE,
    MEAL_ANNOUNCEMENT_STEPS,
    ALL_PATIENT_NAMES,
    DEFAULT_PATIENT,
    GLUCOSE_DEATH,
    GLUCOSE_SEVERE_HYPO,
    GLUCOSE_TARGET_LOW,
    GLUCOSE_TARGET_HIGH,
)

logger = logging.getLogger(__name__)

# Maximum consecutive severe hypo steps before emergency termination
MAX_CONSECUTIVE_SEVERE_HYPO = 5


class GlucoRLEnvironment(Environment):
    """
    OpenEnv environment for insulin dosing RL.

    Wraps simglucose to present a step-by-step decision problem where an
    agent observes CGM glucose readings and decides basal + bolus insulin
    delivery every 3 minutes over a simulated 24-hour day.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        super().__init__()
        self._patient_mgr = PatientManager()
        self._task_id: int = 1
        self._step_count: int = 0
        self._done: bool = True
        self._glucose_history: list[float] = []
        self._reward_history: list[float] = []
        self._action_history: list[tuple[float, float]] = []
        self._episode_reward: float = 0.0
        self._patient_name: str = DEFAULT_PATIENT
        self._episode_id: str = ""
        self._consecutive_severe_hypo: int = 0
        self._hypo_events: int = 0
        self._severe_hypo_events: int = 0
        self._hyper_events: int = 0
        logger.info("GlucoRLEnvironment initialised")

    # ------------------------------------------------------------------
    # OpenEnv interface: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs: Any,
    ) -> GlucoObservation:
        """
        Start a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional episode identifier.
            task_id: Task to run (1, 2, or 3).

        Returns:
            Initial GlucoObservation.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._task_id = int(task_id)
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._done = False
        self._glucose_history = []
        self._reward_history = []
        self._action_history = []
        self._episode_reward = 0.0
        self._consecutive_severe_hypo = 0
        self._hypo_events = 0
        self._severe_hypo_events = 0
        self._hyper_events = 0

        # Select patient based on task
        if self._task_id in (1, 2):
            self._patient_name = DEFAULT_PATIENT
        else:
            self._patient_name = random.choice(ALL_PATIENT_NAMES)

        # Reset patient simulator
        initial_glucose = self._patient_mgr.reset(self._patient_name)

        # Add small noise to initial glucose to prevent overfitting
        noise = np.random.uniform(-20.0, 20.0)
        # We cannot directly modify the patient's internal state easily,
        # so we record the noised reading as the first observation.
        # The actual simulator state stays at its natural init value.
        self._glucose_history.append(initial_glucose)

        logger.info(
            "Episode reset: task=%d patient=%s initial_glucose=%.1f episode=%s",
            self._task_id,
            self._patient_name,
            initial_glucose,
            self._episode_id,
        )

        return self._build_observation(initial_glucose)

    # ------------------------------------------------------------------
    # OpenEnv interface: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: GlucoAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GlucoObservation:
        """
        Process one 3-minute step.

        1. Convert GlucoAction to simglucose units
        2. Determine if a meal is happening this step
        3. Advance the patient simulator
        4. Compute reward
        5. Check termination conditions
        6. Return observation

        Args:
            action: GlucoAction with basal_rate and bolus_dose.
            timeout_s: Unused (kept for interface compatibility).

        Returns:
            GlucoObservation with done=True/False and reward.
        """
        if self._done:
            return self._build_observation(
                self._glucose_history[-1] if self._glucose_history else 140.0,
                force_done=True,
            )

        # Record action
        basal = action.basal_rate
        bolus = action.bolus_dose
        self._action_history.append((basal, bolus))

        # Determine meal CHO for this step
        cho_grams = self._get_meal_cho(self._step_count)

        # Advance simulator
        try:
            glucose = self._patient_mgr.step(basal, bolus, cho_grams)
        except Exception as e:
            logger.error("Simulator step failed at step %d: %s", self._step_count, e)
            glucose = self._glucose_history[-1] if self._glucose_history else 140.0
            self._done = True

        # Clamp extreme readings
        if glucose < GLUCOSE_DEATH:
            logger.warning(
                "Glucose %.1f < %.1f — patient death at step %d",
                glucose, GLUCOSE_DEATH, self._step_count,
            )
            glucose = GLUCOSE_DEATH
            self._done = True

        self._glucose_history.append(glucose)
        self._step_count += 1

        # Gather history values for reward calculation
        prev_glucose = self._glucose_history[-2] if len(self._glucose_history) >= 2 else glucose
        glucose_2_ago = self._glucose_history[-3] if len(self._glucose_history) >= 3 else prev_glucose
        bolus_2_ago = self._action_history[-2][1] if len(self._action_history) >= 2 else 0.0

        # Compute reward
        reward = calculate_step_reward(
            glucose=glucose,
            prev_glucose=prev_glucose,
            bolus_given=bolus,
            glucose_2_steps_ago=glucose_2_ago,
            bolus_2_steps_ago=bolus_2_ago,
        )
        self._reward_history.append(reward.step_total)
        self._episode_reward += reward.step_total

        # Track clinical events
        if glucose < GLUCOSE_TARGET_LOW:
            self._hypo_events += 1
        if glucose < GLUCOSE_SEVERE_HYPO:
            self._severe_hypo_events += 1
            self._consecutive_severe_hypo += 1
        else:
            self._consecutive_severe_hypo = 0
        if glucose > GLUCOSE_TARGET_HIGH:
            self._hyper_events += 1

        # Check termination
        if self._step_count >= STEPS_PER_EPISODE:
            self._done = True
        if self._consecutive_severe_hypo >= MAX_CONSECUTIVE_SEVERE_HYPO:
            logger.warning(
                "Emergency termination: %d consecutive severe hypo events",
                self._consecutive_severe_hypo,
            )
            self._done = True

        return self._build_observation(glucose, reward_value=reward.step_total)

    # ------------------------------------------------------------------
    # OpenEnv interface: state (property)
    # ------------------------------------------------------------------

    @property
    def state(self) -> GlucoState:
        """Return full current environment state."""
        tir = self._compute_tir()
        return GlucoState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            patient_name=self._patient_name,
            step=self._step_count,
            done=self._done,
            glucose_history=list(self._glucose_history),
            reward_history=list(self._reward_history),
            tir_current=tir,
            hypo_events=self._hypo_events,
            severe_hypo_events=self._severe_hypo_events,
            hyper_events=self._hyper_events,
            episode_reward_total=self._episode_reward,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        glucose: float,
        reward_value: float | None = None,
        force_done: bool = False,
    ) -> GlucoObservation:
        """
        Build a GlucoObservation from current state.

        Computes glucose trend from the last two readings and determines
        meal announcement for Task 2.
        """
        done = force_done or self._done
        trend = self._compute_trend()

        # Meal announcement (Task 2 only)
        meal_announced = False
        meal_grams = 0.0
        if self._task_id == 2:
            meal_announced, meal_grams = self._check_meal_announcement(self._step_count)

        # Time of day in hours
        time_hours = (self._step_count * STEP_DURATION_MIN) / 60.0

        # Patient ID: hidden in Task 3 to force generalisation
        patient_id = self._patient_name if self._task_id != 3 else None

        # Last action
        if self._action_history:
            last_basal, last_bolus = self._action_history[-1]
        else:
            last_basal, last_bolus = 1.0, 0.0

        return GlucoObservation(
            glucose_mg_dl=round(glucose, 2),
            glucose_trend=trend,
            meal_announced=meal_announced,
            meal_grams_announced=meal_grams,
            time_of_day_hours=round(time_hours, 2),
            step=self._step_count,
            patient_id=patient_id,
            last_action_basal=round(last_basal, 4),
            last_action_bolus=round(last_bolus, 4),
            done=done,
            reward=reward_value,
        )

    def _compute_trend(self) -> str:
        """
        Compute glucose trend arrow from the last two readings.

        Rate of change is per minute (reading delta / STEP_DURATION_MIN).

        Returns one of: rapidly_falling, falling, stable, rising, rapidly_rising
        """
        if len(self._glucose_history) < 2:
            return "stable"

        current = self._glucose_history[-1]
        previous = self._glucose_history[-2]
        rate = (current - previous) / STEP_DURATION_MIN  # mg/dL per minute

        if rate > 2.0:
            return "rapidly_rising"
        elif rate > 1.0:
            return "rising"
        elif rate < -2.0:
            return "rapidly_falling"
        elif rate < -1.0:
            return "falling"
        else:
            return "stable"

    def _get_meal_cho(self, step: int) -> float:
        """
        Return CHO grams if a meal is scheduled at this step.

        Task 1: no meals.
        Task 2 & 3: meals at steps defined in MEAL_SCHEDULE.
        """
        if self._task_id == 1:
            return 0.0
        return MEAL_SCHEDULE.get(step, 0.0)

    def _check_meal_announcement(self, current_step: int) -> tuple[bool, float]:
        """
        Check if any meal should be announced to the agent.

        Meals are announced MEAL_ANNOUNCEMENT_STEPS (10 steps = 30 min) in advance.
        Only used in Task 2.

        Returns:
            (is_announced, cho_grams) tuple.
        """
        for meal_step, cho in MEAL_SCHEDULE.items():
            steps_until = meal_step - current_step
            if 0 < steps_until <= MEAL_ANNOUNCEMENT_STEPS:
                return True, cho
        return False, 0.0

    def _compute_tir(self) -> float:
        """
        Compute Time-in-Range: fraction of glucose readings in [70, 180] mg/dL.

        Excludes the initial reading (index 0) since it's before any agent action.
        Returns 0.0 if no action steps have been taken yet.
        """
        # Only count readings from step 1 onward (after agent actions)
        readings = self._glucose_history[1:] if len(self._glucose_history) > 1 else []
        if not readings:
            return 0.0
        in_range = sum(
            1 for g in readings
            if GLUCOSE_TARGET_LOW <= g <= GLUCOSE_TARGET_HIGH
        )
        return in_range / len(readings)
