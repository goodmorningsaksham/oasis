"""
Shared constants for the GlucoRL environment.

Central place for glucose thresholds, episode configuration, meal schedule,
and patient identifiers. Imported by environment, reward_calculator, graders,
and pid_controller.
"""

import random

# ---------------------------------------------------------------------------
# Glucose thresholds (mg/dL)
# ---------------------------------------------------------------------------
GLUCOSE_TARGET_LOW = 70.0       # Lower bound of safe range
GLUCOSE_TARGET_HIGH = 180.0     # Upper bound of safe range
GLUCOSE_SEVERE_HYPO = 54.0      # Severe hypoglycemia — life-threatening
GLUCOSE_SEVERE_HYPER = 250.0    # Severe hyperglycemia — organ damage risk
GLUCOSE_DEATH = 10.0            # Simulation termination — patient death

# ---------------------------------------------------------------------------
# Episode configuration
# ---------------------------------------------------------------------------
STEPS_PER_EPISODE = 480         # 480 steps × 3 min = 24 hours
STEP_DURATION_MIN = 3           # Minutes per simulation step

# ---------------------------------------------------------------------------
# Meal schedule: {step_number: carbohydrate_grams}
# ---------------------------------------------------------------------------
MEAL_SCHEDULE = {
    100: 50.0,   # Breakfast: step 100 = 5:00 hours in, 50g CHO
    200: 70.0,   # Lunch:     step 200 = 10:00 hours in, 70g CHO
    320: 80.0,   # Dinner:    step 320 = 16:00 hours in, 80g CHO
}
MEAL_ANNOUNCEMENT_STEPS = 10    # Announce meal this many steps in advance (30 min)

# ---------------------------------------------------------------------------
# All available patient names in simglucose (30 total)
# ---------------------------------------------------------------------------
ALL_PATIENT_NAMES = (
    [f"adolescent#00{i}" for i in range(1, 10)]
    + ["adolescent#010"]
    + [f"adult#00{i}" for i in range(1, 10)]
    + ["adult#010"]
    + [f"child#00{i}" for i in range(1, 10)]
    + ["child#010"]
)

# Default patient for Task 1 and Task 2
DEFAULT_PATIENT = "adult#001"

# ---------------------------------------------------------------------------
# Task 3 deterministic patient sample (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
TASK3_EVAL_PATIENTS = random.Random(42).sample(ALL_PATIENT_NAMES, 5)
