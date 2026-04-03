"""
PID controller baseline for blood glucose management.

A simple Proportional-Integral-Derivative controller targeting 120 mg/dL
(centre of the safe 70–180 range). This mirrors the control strategy used
by commercial insulin pumps and serves as a benchmark that RL agents
should eventually outperform.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import GlucoAction


class PIDController:
    """
    PID controller for insulin dosing.

    Adjusts basal insulin rate proportionally to the error between current
    glucose and target, with integral and derivative terms for steady-state
    accuracy and overshoot damping. Adds a simple correction bolus when
    glucose is very high.

    Args:
        target_glucose: Desired glucose level in mg/dL (default 120).
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
    """

    def __init__(
        self,
        target_glucose: float = 120.0,
        kp: float = 0.02,
        ki: float = 0.0005,
        kd: float = 0.01,
    ):
        self.target = target_glucose
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        """Reset the integral and derivative state for a new episode."""
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, glucose: float) -> GlucoAction:
        """
        Compute insulin action given current glucose reading.

        Args:
            glucose: Current blood glucose in mg/dL.

        Returns:
            GlucoAction with basal_rate and bolus_dose.
        """
        error = glucose - self.target

        # Integral with anti-windup clamping
        self.integral += error
        self.integral = max(-500.0, min(500.0, self.integral))

        # Reset integral when glucose crosses below target to prevent
        # accumulated error from driving dangerous over-delivery
        if glucose < self.target:
            self.integral = min(self.integral, 0.0)

        derivative = error - self.prev_error
        self.prev_error = error

        # PID-adjusted basal rate (baseline ~1.0 U/hr)
        basal_adjustment = (
            self.kp * error + self.ki * self.integral + self.kd * derivative
        )
        basal_rate = max(0.0, min(5.0, 1.0 + basal_adjustment))

        # Simple correction bolus when glucose is very high
        if glucose > 200.0:
            bolus_dose = max(0.0, (glucose - 200.0) / 50.0)
        else:
            bolus_dose = 0.0

        # Safety: if glucose is dropping low, cut insulin aggressively
        if glucose < 90.0:
            basal_rate = max(0.0, basal_rate * 0.3)
            bolus_dose = 0.0
        if glucose < 70.0:
            basal_rate = 0.0
            bolus_dose = 0.0

        return GlucoAction(
            basal_rate=round(basal_rate, 4),
            bolus_dose=round(min(bolus_dose, 20.0), 4),
        )
