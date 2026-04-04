"""
Patient manager for the GlucoRL environment.

Wraps simglucose's T1DPatient with:
  - Unit conversion (U/hr -> U/min for basal, total units -> U/min for bolus)
  - 3-minute environment step (3 x 1-minute patient mini-steps)
  - Meal injection via CHO field on the patient action
  - Optional CGM measurement noise (σ=10 mg/dL per ISO 15197)
  - Safe exception handling for extreme glucose values
"""

import logging
import numpy as np
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.env import Action as SimAction

from server.constants import STEP_DURATION_MIN, GLUCOSE_DEATH

logger = logging.getLogger(__name__)

# CGM noise standard deviation (mg/dL) — matches ISO 15197 accuracy spec
CGM_NOISE_STD = 10.0
# Clamp range for noisy CGM readings
CGM_MIN = 20.0
CGM_MAX = 600.0


class PatientManager:
    """
    Manages a simglucose T1DPatient instance.

    Handles initialisation, stepping with unit conversion, meal injection,
    and glucose reading. Each call to step() advances the simulation by
    STEP_DURATION_MIN minutes (3 minutes = 3 patient mini-steps).

    Args:
        noise_enabled: If True, add Gaussian noise to CGM readings.
            The true glucose (Gsub) is always available separately.
    """

    def __init__(self, noise_enabled: bool = True):
        self._patient: T1DPatient | None = None
        self._name: str = ""
        self.noise_enabled: bool = noise_enabled

    def reset(self, patient_name: str) -> tuple[float, float]:
        """
        Create a fresh patient and return the initial glucose readings.

        Args:
            patient_name: simglucose patient identifier, e.g. 'adult#001'.

        Returns:
            Tuple of (cgm_glucose, true_glucose) in mg/dL.
            cgm_glucose has optional noise applied; true_glucose is raw Gsub.
        """
        self._name = patient_name
        self._patient = T1DPatient.withName(patient_name)
        true_glucose = float(self._patient.observation.Gsub)
        cgm_glucose = self._apply_noise(true_glucose)
        logger.info(
            "Patient %s reset — true glucose: %.1f mg/dL, CGM: %.1f mg/dL",
            patient_name, true_glucose, cgm_glucose,
        )
        return cgm_glucose, true_glucose

    def step(
        self,
        basal_rate_uhr: float,
        bolus_dose_units: float,
        cho_grams: float = 0.0,
        insulin_sensitivity_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        """
        Advance the patient simulation by one environment step (3 minutes).

        Performs STEP_DURATION_MIN patient mini-steps (each 1 minute).
        Meals (CHO) are injected on the first mini-step only.

        Args:
            basal_rate_uhr: Basal insulin rate in units/hr (from GlucoAction).
            bolus_dose_units: Bolus insulin in total units (from GlucoAction).
            cho_grams: Carbohydrate grams to inject this step (0.0 if no meal).
            insulin_sensitivity_multiplier: Factor applied to effective insulin
                to simulate exercise (>1.0) or insulin resistance (<1.0).
                Default 1.0 = no modification.

        Returns:
            Tuple of (cgm_glucose, true_glucose) in mg/dL.

        Raises:
            RuntimeError: If patient has not been reset.
        """
        if self._patient is None:
            raise RuntimeError("Patient not initialised — call reset() first")

        # Convert units:
        #   basal: U/hr -> U/min
        #   bolus: total units spread over 3-minute step -> U/min
        basal_umin = basal_rate_uhr / 60.0
        bolus_umin = bolus_dose_units / float(STEP_DURATION_MIN)
        insulin_umin = (basal_umin + bolus_umin) * insulin_sensitivity_multiplier

        try:
            for mini in range(STEP_DURATION_MIN):
                # Inject CHO only on the first mini-step
                cho = cho_grams if mini == 0 else 0.0
                patient_action = SimAction(insulin=insulin_umin, CHO=cho)
                self._patient.step(patient_action)
        except Exception as e:
            logger.error("Patient step failed: %s", e)
            raise

        true_glucose = float(self._patient.observation.Gsub)
        cgm_glucose = self._apply_noise(true_glucose)
        return cgm_glucose, true_glucose

    def _apply_noise(self, true_glucose: float) -> float:
        """
        Apply CGM measurement noise to a true glucose reading.

        Adds Gaussian noise with σ=10 mg/dL and clamps to [20, 600].

        Args:
            true_glucose: Raw subcutaneous glucose (Gsub) in mg/dL.

        Returns:
            Noisy CGM reading (or true_glucose if noise is disabled).
        """
        if not self.noise_enabled:
            return true_glucose
        noise = np.random.normal(0.0, CGM_NOISE_STD)
        cgm = true_glucose + noise
        return float(max(CGM_MIN, min(CGM_MAX, cgm)))

    def get_glucose(self) -> tuple[float, float]:
        """
        Return the current glucose readings without advancing simulation.

        Returns:
            Tuple of (cgm_glucose, true_glucose) in mg/dL.
        """
        if self._patient is None:
            raise RuntimeError("Patient not initialised — call reset() first")
        true_glucose = float(self._patient.observation.Gsub)
        cgm_glucose = self._apply_noise(true_glucose)
        return cgm_glucose, true_glucose

    @property
    def name(self) -> str:
        """Return the patient identifier."""
        return self._name

    @property
    def time_minutes(self) -> float:
        """Return the current simulation time in minutes."""
        if self._patient is None:
            return 0.0
        return float(self._patient.t)
