"""
Patient manager for the GlucoRL environment.

Wraps simglucose's T1DPatient with:
  - Unit conversion (U/hr -> U/min for basal, total units -> U/min for bolus)
  - 3-minute environment step (3 x 1-minute patient mini-steps)
  - Meal injection via CHO field on the patient action
  - Safe exception handling for extreme glucose values
"""

import logging
import numpy as np
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.env import Action as SimAction

from server.constants import STEP_DURATION_MIN, GLUCOSE_DEATH

logger = logging.getLogger(__name__)


class PatientManager:
    """
    Manages a simglucose T1DPatient instance.

    Handles initialisation, stepping with unit conversion, meal injection,
    and glucose reading. Each call to step() advances the simulation by
    STEP_DURATION_MIN minutes (3 minutes = 3 patient mini-steps).
    """

    def __init__(self):
        self._patient: T1DPatient | None = None
        self._name: str = ""

    def reset(self, patient_name: str) -> float:
        """
        Create a fresh patient and return the initial glucose reading.

        Args:
            patient_name: simglucose patient identifier, e.g. 'adult#001'.

        Returns:
            Initial blood glucose in mg/dL.
        """
        self._name = patient_name
        self._patient = T1DPatient.withName(patient_name)
        glucose = float(self._patient.observation.Gsub)
        logger.info(
            "Patient %s reset — initial glucose: %.1f mg/dL", patient_name, glucose
        )
        return glucose

    def step(
        self,
        basal_rate_uhr: float,
        bolus_dose_units: float,
        cho_grams: float = 0.0,
    ) -> float:
        """
        Advance the patient simulation by one environment step (3 minutes).

        Performs STEP_DURATION_MIN patient mini-steps (each 1 minute).
        Meals (CHO) are injected on the first mini-step only.

        Args:
            basal_rate_uhr: Basal insulin rate in units/hr (from GlucoAction).
            bolus_dose_units: Bolus insulin in total units (from GlucoAction).
            cho_grams: Carbohydrate grams to inject this step (0.0 if no meal).

        Returns:
            Blood glucose reading in mg/dL after the step.

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
        insulin_umin = basal_umin + bolus_umin

        try:
            for mini in range(STEP_DURATION_MIN):
                # Inject CHO only on the first mini-step
                cho = cho_grams if mini == 0 else 0.0
                patient_action = SimAction(insulin=insulin_umin, CHO=cho)
                self._patient.step(patient_action)
        except Exception as e:
            logger.error("Patient step failed: %s", e)
            raise

        glucose = float(self._patient.observation.Gsub)
        return glucose

    def get_glucose(self) -> float:
        """Return the current glucose reading without advancing simulation."""
        if self._patient is None:
            raise RuntimeError("Patient not initialised — call reset() first")
        return float(self._patient.observation.Gsub)

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
