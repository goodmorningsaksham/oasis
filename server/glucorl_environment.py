"""
OASIS Environment Implementation.

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
from collections import deque
from typing import Optional, Any
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment

from models import GlucoAction, GlucoObservation, GlucoState
from server.patient_manager import PatientManager
from server.reward_calculator import calculate_step_reward
from server.reward_calculator import grade
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
    IOB_T_PEAK_MIN,
    IOB_T_END_MIN,
    IOB_HISTORY_STEPS,
    EXERCISE_INTENSITY_LEVELS,
    EXERCISE_DURATION_STEPS,
    EXERCISE_SENSITIVITY_MULTIPLIER,
    EXERCISE_SCHEDULE_TASK2,
    EXERCISE_ANNOUNCEMENT_STEPS,
    ILLNESS_RESISTANCE_MIN,
    ILLNESS_RESISTANCE_MAX,
    ILLNESS_ONSET_STEP_MIN,
    ILLNESS_ONSET_STEP_MAX,
)

logger = logging.getLogger(__name__)

# Maximum consecutive severe hypo steps before emergency termination
MAX_CONSECUTIVE_SEVERE_HYPO = 5


def _precompute_iob_curve() -> np.ndarray:
    """Precompute the gamma-CDF insulin absorption curve at module load time.

    Returns array of shape (IOB_HISTORY_STEPS,) where each value F_k[i]
    is the cumulative fraction of insulin absorbed by time offset i.
    IOB for a dose is then (1 - F_k[time_since_dose]) * dose.
    """
    from scipy.stats import gamma as gamma_dist
    shape_k = 2
    scale_theta = IOB_T_PEAK_MIN / (shape_k - 1)  # = 55 for rapid-acting
    time_points = np.linspace(0, IOB_T_END_MIN, IOB_HISTORY_STEPS)
    F_k = gamma_dist.cdf(time_points, a=shape_k, scale=scale_theta)
    return F_k


# Precomputed once at import — shape (160,)
_IOB_ABSORPTION_CURVE = _precompute_iob_curve()


class GlucoRLEnvironment(Environment):
    """
    OpenEnv environment for insulin dosing RL.

    Wraps simglucose to present a step-by-step decision problem where an
    agent observes CGM glucose readings and decides basal + bolus insulin
    delivery every 3 minutes over a simulated 24-hour day.

    The agent observes noisy CGM readings (σ=10 mg/dL per ISO 15197).
    Rewards and grader scores are computed on true subcutaneous glucose (Gsub).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        super().__init__()
        self._patient_mgr = PatientManager(noise_enabled=True)
        self._task_id: int = 1
        self._step_count: int = 0
        self._done: bool = True
        # True glucose history (Gsub) — used by graders, reward, TIR, state
        self._glucose_history: list[float] = []
        # Noisy CGM glucose history — used by agent observation and trend
        self._cgm_glucose_history: list[float] = []
        self._reward_history: list[float] = []
        self._action_history: list[tuple[float, float]] = []
        self._episode_reward: float = 0.0
        self._patient_name: str = DEFAULT_PATIENT
        self._episode_id: str = ""
        self._consecutive_severe_hypo: int = 0
        self._hypo_events: int = 0
        self._severe_hypo_events: int = 0
        self._hyper_events: int = 0
        self._insulin_history: deque = deque(
            [0.0] * IOB_HISTORY_STEPS, maxlen=IOB_HISTORY_STEPS
        )
        self._current_exercise_intensity: float = 0.0
        self._exercise_steps_remaining: int = 0
        self._exercise_schedule: dict[int, float] = {}  # {step: intensity}
        self._exercise_duration_map: dict[int, int] = {}  # {step: duration_steps}
        self._hypo_start_step: int | None = None   # Step when hypo began
        self._hyper_start_step: int | None = None   # Step when hyper began
        self._illness_resistance: float = 1.0       # Insulin resistance multiplier (1.0 = no illness)
        self._illness_onset_step: int | None = None  # Step when illness begins
        self._illness_active: bool = False
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
        self._cgm_glucose_history = []
        self._reward_history = []
        self._action_history = []
        self._episode_reward = 0.0
        self._consecutive_severe_hypo = 0
        self._hypo_events = 0
        self._severe_hypo_events = 0
        self._hyper_events = 0
        self._insulin_history = deque(
            [0.0] * IOB_HISTORY_STEPS, maxlen=IOB_HISTORY_STEPS
        )
        self._current_exercise_intensity = 0.0
        self._exercise_steps_remaining = 0
        self._hypo_start_step = None
        self._hyper_start_step = None
        self._illness_resistance = 1.0
        self._illness_onset_step = None
        self._illness_active = False

        # Build exercise schedule based on task
        if self._task_id == 1:
            # Task 1: no exercise
            self._exercise_schedule = {}
            self._exercise_duration_map = {}
        elif self._task_id == 2:
            # Task 2: fixed schedule, announced
            self._exercise_schedule = dict(EXERCISE_SCHEDULE_TASK2)
            self._exercise_duration_map = {
                step: 20 for step in EXERCISE_SCHEDULE_TASK2
            }
        else:
            # Task 3 & 4: random exercise event (unannounced)
            self._exercise_schedule = {}
            self._exercise_duration_map = {}
            if random.random() < 0.6:
                ex_step = random.randint(60, 350)
                ex_intensity = random.choice(EXERCISE_INTENSITY_LEVELS)
                ex_duration = random.choice(EXERCISE_DURATION_STEPS)
                self._exercise_schedule[ex_step] = ex_intensity
                self._exercise_duration_map[ex_step] = ex_duration

        # Task 4: generate illness (insulin resistance) parameters
        if self._task_id == 4:
            self._illness_resistance = random.uniform(
                ILLNESS_RESISTANCE_MIN, ILLNESS_RESISTANCE_MAX
            )
            self._illness_onset_step = random.randint(
                ILLNESS_ONSET_STEP_MIN, ILLNESS_ONSET_STEP_MAX
            )

        if self._task_id in (1, 2):
            self._patient_name = DEFAULT_PATIENT
        else:
            # Task 3 & 4: random patient
            self._patient_name = random.choice(ALL_PATIENT_NAMES)

        # Reset patient simulator — returns (cgm_glucose, true_glucose)
        cgm_glucose, true_glucose = self._patient_mgr.reset(self._patient_name)

        self._glucose_history.append(true_glucose)
        self._cgm_glucose_history.append(cgm_glucose)

        logger.info(
            "Episode reset: task=%d patient=%s true_glucose=%.1f cgm=%.1f episode=%s",
            self._task_id,
            self._patient_name,
            true_glucose,
            cgm_glucose,
            self._episode_id,
        )

        return self._build_observation(cgm_glucose, true_glucose)

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
        4. Compute reward (on TRUE glucose, not noisy CGM)
        5. Check termination conditions (on TRUE glucose)
        6. Return observation (agent sees noisy CGM glucose)

        Args:
            action: GlucoAction with basal_rate and bolus_dose.
            timeout_s: Unused (kept for interface compatibility).

        Returns:
            GlucoObservation with done=True/False and reward.
        """
        if self._done:
            last_true = self._glucose_history[-1] if self._glucose_history else 140.0
            last_cgm = self._cgm_glucose_history[-1] if self._cgm_glucose_history else 140.0
            return self._build_observation(
                last_cgm, last_true, force_done=True,
            )

        # Record action
        basal = action.basal_rate
        bolus = action.bolus_dose
        self._action_history.append((basal, bolus))

        # Track insulin delivered this step for PK/PD IOB model
        # Basal: convert U/hr to U per 3-min step; Bolus: total units
        insulin_this_step = (basal * STEP_DURATION_MIN / 60.0) + bolus
        self._insulin_history.append(insulin_this_step)

        # Update exercise state
        if self._step_count in self._exercise_schedule:
            self._current_exercise_intensity = self._exercise_schedule[self._step_count]
            self._exercise_steps_remaining = self._exercise_duration_map.get(
                self._step_count, 20
            )
        if self._exercise_steps_remaining > 0:
            self._exercise_steps_remaining -= 1
            if self._exercise_steps_remaining <= 0:
                self._current_exercise_intensity = 0.0

        # Compute insulin sensitivity multiplier from exercise
        sensitivity = EXERCISE_SENSITIVITY_MULTIPLIER.get(
            self._current_exercise_intensity, 1.0
        ) if self._current_exercise_intensity > 0 else 1.0

        # Apply illness resistance (Task 4): reduces insulin effectiveness
        if (self._task_id == 4
                and self._illness_onset_step is not None
                and self._step_count >= self._illness_onset_step):
            self._illness_active = True
            sensitivity *= (1.0 / self._illness_resistance)

        # Determine meal CHO for this step
        cho_grams = self._get_meal_cho(self._step_count)

        # Advance simulator — returns (cgm_glucose, true_glucose)
        try:
            cgm_glucose, true_glucose = self._patient_mgr.step(
                basal, bolus, cho_grams,
                insulin_sensitivity_multiplier=sensitivity,
            )
        except Exception as e:
            logger.error("Simulator step failed at step %d: %s", self._step_count, e)
            true_glucose = self._glucose_history[-1] if self._glucose_history else 140.0
            cgm_glucose = true_glucose
            self._done = True

        # Clamp extreme readings — check TRUE glucose for patient safety
        if true_glucose < GLUCOSE_DEATH:
            logger.warning(
                "Glucose %.1f < %.1f — patient death at step %d",
                true_glucose, GLUCOSE_DEATH, self._step_count,
            )
            true_glucose = GLUCOSE_DEATH
            cgm_glucose = GLUCOSE_DEATH
            self._done = True

        self._glucose_history.append(true_glucose)
        self._cgm_glucose_history.append(cgm_glucose)
        self._step_count += 1

        # Gather history values for reward — uses TRUE glucose
        prev_true = self._glucose_history[-2] if len(self._glucose_history) >= 2 else true_glucose
        true_2_ago = self._glucose_history[-3] if len(self._glucose_history) >= 3 else prev_true
        bolus_2_ago = self._action_history[-2][1] if len(self._action_history) >= 2 else 0.0

        # Compute recovery context for reward bonus
        steps_since_hypo = None
        if self._hypo_start_step is not None:
            steps_since_hypo = self._step_count - self._hypo_start_step
        steps_since_hyper = None
        if self._hyper_start_step is not None:
            steps_since_hyper = self._step_count - self._hyper_start_step

        # Compute reward on TRUE glucose (not noisy CGM)
        reward = calculate_step_reward(
            glucose=true_glucose,
            prev_glucose=prev_true,
            bolus_given=bolus,
            glucose_2_steps_ago=true_2_ago,
            bolus_2_steps_ago=bolus_2_ago,
            steps_since_hypo_start=steps_since_hypo,
            steps_since_hyper_start=steps_since_hyper,
        )
        self._reward_history.append(reward.step_total)
        self._episode_reward += reward.step_total

        # Track clinical events on TRUE glucose
        if true_glucose < GLUCOSE_TARGET_LOW:
            self._hypo_events += 1
            if self._hypo_start_step is None:
                self._hypo_start_step = self._step_count
        else:
            self._hypo_start_step = None

        if true_glucose < GLUCOSE_SEVERE_HYPO:
            self._severe_hypo_events += 1
            self._consecutive_severe_hypo += 1
        else:
            self._consecutive_severe_hypo = 0

        if true_glucose > GLUCOSE_TARGET_HIGH:
            self._hyper_events += 1
            if self._hyper_start_step is None:
                self._hyper_start_step = self._step_count
        else:
            self._hyper_start_step = None

        # Check termination
        if self._step_count >= STEPS_PER_EPISODE:
            self._done = True
        if self._consecutive_severe_hypo >= MAX_CONSECUTIVE_SEVERE_HYPO:
            logger.warning(
                "Emergency termination: %d consecutive severe hypo events",
                self._consecutive_severe_hypo,
            )
            self._done = True

        if self._done:
            final_score = grade(self._task_id, self.state)
            return self._build_observation(
                cgm_glucose, true_glucose, reward_value=final_score,
            )
        return self._build_observation(
            cgm_glucose, true_glucose, reward_value=reward.step_total,
        )

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
        cgm_glucose: float,
        true_glucose: float,
        reward_value: float | None = None,
        force_done: bool = False,
    ) -> GlucoObservation:
        """
        Build a GlucoObservation from current state.

        The agent sees noisy CGM glucose. Trend is computed from CGM history
        (matching real CGM device behaviour). true_glucose_mg_dl is always
        provided for research and debugging.
        """
        done = force_done or self._done
        trend = self._compute_trend()

        # Meal announcement (Task 2 only)
        meal_announced = False
        meal_grams = 0.0
        if self._task_id == 2:
            meal_announced, meal_grams = self._check_meal_announcement(self._step_count)

        # Exercise announcement (Task 2 only)
        exercise_announced = False
        if self._task_id == 2:
            exercise_announced = self._check_exercise_announcement(self._step_count)

        # Time of day in hours
        time_hours = (self._step_count * STEP_DURATION_MIN) / 60.0

        # Patient ID: hidden in Task 3 & 4 to force generalisation
        patient_id = self._patient_name if self._task_id in (1, 2) else None

        # Last action
        if self._action_history:
            last_basal, last_bolus = self._action_history[-1]
        else:
            last_basal, last_bolus = 1.0, 0.0

        # Glucose history window: last 12 CGM readings (36 min of context)
        window = self._cgm_glucose_history[-12:] if self._cgm_glucose_history else []
        window = [round(g, 1) for g in window]

        return GlucoObservation(
            glucose_mg_dl=round(cgm_glucose, 2),
            glucose_trend=trend,
            meal_announced=meal_announced,
            meal_grams_announced=meal_grams,
            time_of_day_hours=round(time_hours, 2),
            step=self._step_count,
            patient_id=patient_id,
            last_action_basal=round(last_basal, 4),
            last_action_bolus=round(last_bolus, 4),
            true_glucose_mg_dl=round(true_glucose, 2),
            insulin_on_board_units=self._compute_iob(),
            exercise_intensity=self._current_exercise_intensity,
            exercise_announced=exercise_announced,
            glucose_history_window=window,
            illness_active=False,  # Never exposed to agent — debug only
            done=done,
            reward=reward_value,
        )

    def _compute_trend(self) -> str:
        """
        Compute glucose trend arrow from the last two CGM readings.

        Uses noisy CGM history (matching real CGM device behaviour).
        Rate of change is per minute (reading delta / STEP_DURATION_MIN).

        Returns one of: rapidly_falling, falling, stable, rising, rapidly_rising
        """
        if len(self._cgm_glucose_history) < 2:
            return "stable"

        current = self._cgm_glucose_history[-1]
        previous = self._cgm_glucose_history[-2]
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

    def _check_exercise_announcement(self, current_step: int) -> bool:
        """
        Check if an exercise event should be announced to the agent.

        Exercise is announced EXERCISE_ANNOUNCEMENT_STEPS (10 steps = 30 min)
        in advance. Only used in Task 2.

        Returns:
            True if exercise is upcoming within the announcement window.
        """
        for ex_step in self._exercise_schedule:
            steps_until = ex_step - current_step
            if 0 < steps_until <= EXERCISE_ANNOUNCEMENT_STEPS:
                return True
        return False

    def _compute_iob(self) -> float:
        """Compute insulin-on-board using PK/PD gamma-CDF absorption model.

        IOB = sum of all past insulin doses, each weighted by the fraction
        NOT YET absorbed at that time offset. A dose injected 5 minutes ago
        has most of its insulin still "on board" (high weight). A dose
        injected 4 hours ago has almost none remaining (low weight).

        Returns:
            Insulin-on-board in units, rounded to 4 decimal places.
        """
        # Reverse history so index 0 = most recent dose
        history_reversed = np.array(list(self._insulin_history))[::-1]
        # (1 - F_k) = fraction of insulin NOT YET absorbed at each offset
        remaining_fraction = 1.0 - _IOB_ABSORPTION_CURVE
        iob = float(np.sum(history_reversed * remaining_fraction))
        return max(0.0, round(iob, 4))

    def _compute_tir(self) -> float:
        """
        Compute Time-in-Range: fraction of TRUE glucose readings in [70, 180] mg/dL.

        Uses true glucose (not noisy CGM) for accurate clinical assessment.
        Excludes the initial reading (index 0) since it's before any agent action.
        Returns 0.0 if no action steps have been taken yet.
        """
        readings = self._glucose_history[1:] if len(self._glucose_history) > 1 else []
        if not readings:
            return 0.0
        in_range = sum(
            1 for g in readings
            if GLUCOSE_TARGET_LOW <= g <= GLUCOSE_TARGET_HIGH
        )
        return in_range / len(readings)
