"""
OASIS Inference Script
========================
Runs a language model agent through all OASIS tasks.

Required environment variables:
    HF_TOKEN       HuggingFace token (NO default — must be set)

Optional environment variables:
    API_BASE_URL   LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier (default: meta-llama/Llama-3.1-8B-Instruct)
    OASIS_ENV_URL  Environment server URL (default: http://localhost:8000)
                   Also accepts GLUCORL_ENV_URL for backward compatibility
"""

import os
import json
import re
import time

from openai import OpenAI
from client import GlucoEnv
from models import GlucoAction

# ---------------------------------------------------------------------------
# Environment variables — defaults per submission guide
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "oasis"
MAX_STEPS = 480
INFERENCE_STEP_INTERVAL = 5
TEMPERATURE = 0.1
MAX_TOKENS = 100
FALLBACK_ACTION = GlucoAction(basal_rate=1.0, bolus_dose=0.0)

# Validator requires scores strictly in (0, 1) AND rewards are 2-decimal formatted
# Normalize raw rewards (-6.0 to +1.5) into (0.01, 0.99) to preserve signal
SCORE_MIN = 0.01
SCORE_MAX = 0.99
RAW_REWARD_MIN = -6.0   # worst: severe hypo (-3.0) + overdose (-3.0)
RAW_REWARD_MAX = 1.5    # best: in-range (+1.0) + recovery bonus (+0.5)

TASK_NAMES = {
    1: "basal_rate_control",
    2: "meal_bolus_timing",
    3: "cross_patient_generalisation",
}

SYSTEM_PROMPT = """You are an AI insulin dosing system for a Type 1 Diabetic patient.
At each step you observe:
- Current blood glucose in mg/dL (target range: 70-180)
- Glucose trend (rapidly_falling/falling/stable/rising/rapidly_rising)
- Whether a meal is coming soon and how many carbohydrates
- Current time of day

You must respond with ONLY a JSON object like this:
{"basal_rate": 1.2, "bolus_dose": 0.0}

Rules:
- basal_rate: 0.0 to 5.0 units/hr (continuous background insulin)
- bolus_dose: 0.0 to 20.0 units (meal/correction insulin, use 0 if no meal)
- If glucose is falling or low, reduce basal_rate and set bolus_dose to 0
- If glucose is rising high, increase basal_rate slightly
- If a meal is announced, give a bolus_dose proportional to meal_grams / 10
- Never give bolus when glucose is below 120 mg/dL
Do not include any explanation. Respond with only the JSON."""


# ---------------------------------------------------------------------------
# Reward normalization — validator requires strictly (0, 1)
# ---------------------------------------------------------------------------

def normalize_reward(reward: float) -> float:
    """Normalize raw step reward into (0.01, 0.99) range.

    Maps the full reward range linearly so the validator sees values
    strictly between 0 and 1, while preserving relative ordering.

    Examples:
        +1.00 (in range)     → 0.92
        +1.50 (in range+recovery) → 0.99
        -0.50 (mild hyper)   → 0.73
        -1.00 (mild hypo)    → 0.66
        -1.50 (severe hyper) → 0.60
        -3.00 (severe hypo)  → 0.40
        -6.00 (hypo+overdose)→ 0.01
    """
    normalized = SCORE_MIN + (reward - RAW_REWARD_MIN) / (RAW_REWARD_MAX - RAW_REWARD_MIN) * (SCORE_MAX - SCORE_MIN)
    return max(SCORE_MIN, min(SCORE_MAX, round(normalized, 4)))


# ---------------------------------------------------------------------------
# Strict stdout logging — [START] / [STEP] / [END]
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def build_user_prompt(obs, step: int) -> str:
    """Format the current observation into a prompt for the LLM."""
    return (
        f"Step: {step}\n"
        f"Glucose: {obs.glucose_mg_dl:.1f} mg/dL\n"
        f"Trend: {obs.glucose_trend}\n"
        f"Time: {obs.time_of_day_hours:.1f} hours\n"
        f"Meal announced: {obs.meal_announced}\n"
        f"Meal carbs: {obs.meal_grams_announced:.0f}g\n"
        f"Last basal: {obs.last_action_basal:.2f} u/hr\n"
        f"Last bolus: {obs.last_action_bolus:.2f} u\n"
        f"\n"
        f'Respond with JSON only: {{"basal_rate": X, "bolus_dose": Y}}'
    )


def parse_action(response_text: str) -> GlucoAction:
    """Extract a GlucoAction from the LLM's text response."""
    if not response_text:
        return FALLBACK_ACTION
    try:
        match = re.search(r'\{[^}]+\}', response_text)
        if match:
            data = json.loads(match.group(0))
            return GlucoAction(
                basal_rate=float(data.get('basal_rate', 1.0)),
                bolus_dose=float(data.get('bolus_dose', 0.0)),
            )
    except Exception:
        pass
    return FALLBACK_ACTION


def action_to_str(action: GlucoAction) -> str:
    """Format action as a compact string for [STEP] logging."""
    return f"basal={action.basal_rate:.2f}_bolus={action.bolus_dose:.2f}"


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(client_openai: OpenAI, env_url: str, task_id: int):
    """Run a single task episode with strict [START]/[STEP]/[END] logging."""
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")

    log_start(task_name, BENCHMARK, MODEL_NAME)

    rewards = []
    steps_completed = 0
    action = FALLBACK_ACTION
    llm_failed = False

    try:
        with GlucoEnv(base_url=env_url) as env:
            result = env.reset(task_id=task_id)
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                error_msg = None

                # Agent decides every INFERENCE_STEP_INTERVAL steps
                if step % INFERENCE_STEP_INTERVAL == 0 and not llm_failed:
                    user_prompt = build_user_prompt(obs, step)
                    try:
                        completion = client_openai.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                        )
                        response_text = completion.choices[0].message.content or ""
                        action = parse_action(response_text)
                    except Exception as exc:
                        error_msg = str(exc)[:100]
                        action = FALLBACK_ACTION
                        llm_failed = True

                result = env.step(action)
                obs = result.observation
                step_reward = result.reward if result.reward is not None else 0.0

                # Normalize reward to (0.01, 0.99) for validator
                clamped = normalize_reward(step_reward)
                rewards.append(clamped)
                steps_completed = step

                log_step(step, action_to_str(action), clamped, result.done, error_msg)

            # Compute episode score: mean of normalized rewards, clamped to (0.01, 0.99)
            if rewards:
                score = sum(rewards) / len(rewards)
                score = max(SCORE_MIN, min(SCORE_MAX, score))
            else:
                score = SCORE_MIN
            success = score > 0.5

    except Exception as e:
        # Connection failed — still emit valid [END]
        success = False
        score = SCORE_MIN

    log_end(success, steps_completed, score, rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env_url = (os.getenv("OASIS_ENV_URL")
              or os.getenv("GLUCORL_ENV_URL")
              or "http://localhost:8000")

    client_openai = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy",
        timeout=10.0,
    )

    for task_id in [1, 2, 3]:
        run_task(client_openai, env_url, task_id)


if __name__ == "__main__":
    main()