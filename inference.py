"""
GlucoRL Inference Script
========================
Runs a language model agent through all 3 GlucoRL tasks.

Required environment variables:
    API_BASE_URL   LLM API endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace token used as API key

Optional:
    GLUCORL_ENV_URL  Environment server URL (default: http://localhost:8000)
"""

import os
import json
import re
import time
from openai import OpenAI
from client import GlucoEnv
from models import GlucoAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 480                  # full day
INFERENCE_STEP_INTERVAL = 5      # agent acts every 5 steps, holds action between
TEMPERATURE = 0.1
MAX_TOKENS = 100
FALLBACK_ACTION = GlucoAction(basal_rate=1.0, bolus_dose=0.0)

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


def run_task(client_openai: OpenAI, env_url: str, task_id: int) -> dict:
    """
    Run a single task episode and return summary metrics.

    The agent queries the LLM every INFERENCE_STEP_INTERVAL steps and
    holds the last action in between to stay within API rate limits
    and finish under the 20-minute wall clock budget.
    """
    print(f"\n{'='*50}")
    print(f"Running Task {task_id}...")
    print(f"{'='*50}")

    with GlucoEnv(base_url=env_url) as env:
        result = env.reset(task_id=task_id)
        obs = result.observation
        total_reward = 0.0
        steps_completed = 0
        action = FALLBACK_ACTION
        llm_failed = False  # After first failure, skip LLM to avoid timeout

        for step in range(MAX_STEPS):
            if result.done:
                break

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
                    print(f"  LLM call failed at step {step}: {exc}")
                    action = FALLBACK_ACTION
                    llm_failed = True
                    print("  Switching to fallback actions for remaining steps")

            result = env.step(action)
            obs = result.observation
            step_reward = result.reward if result.reward is not None else 0.0
            total_reward += step_reward
            steps_completed += 1

            # Print every ~4 simulated hours
            if step % 48 == 0:
                print(
                    f"  Step {step:3d} | Glucose: {obs.glucose_mg_dl:6.1f} mg/dL "
                    f"| Trend: {obs.glucose_trend:15s} "
                    f"| Reward: {step_reward:+.2f}"
                )

        state = env.state()
        tir = state.tir_current
        print(f"\nTask {task_id} complete:")
        print(f"  Steps: {steps_completed}")
        print(f"  TIR: {tir:.1%}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Hypo events: {state.hypo_events}")
        print(f"  Severe hypo: {state.severe_hypo_events}")

        return {
            "task_id": task_id,
            "tir": tir,
            "total_reward": total_reward,
            "hypo_events": state.hypo_events,
            "severe_hypo_events": state.severe_hypo_events,
            "steps": steps_completed,
        }


def main():
    env_url = os.getenv("GLUCORL_ENV_URL") or "http://localhost:8000"
    print("GlucoRL Inference Script")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {env_url}")

    if not MODEL_NAME:
        print("WARNING: MODEL_NAME not set — LLM calls will fail, using fallback actions")
    if not API_KEY:
        print("WARNING: HF_TOKEN / API_KEY not set — LLM calls will fail, using fallback actions")

    client_openai = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy", timeout=10.0)
    results = []

    start_time = time.time()

    for task_id in [1, 2, 3]:
        try:
            r = run_task(client_openai, env_url, task_id)
            results.append(r)
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            results.append({"task_id": task_id, "tir": 0.0, "total_reward": -999})

    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print("FINAL BASELINE SCORES")
    print(f"{'='*50}")
    for r in results:
        print(
            f"Task {r['task_id']}: TIR={r.get('tir', 0):.1%} | "
            f"Reward={r.get('total_reward', 0):.1f}"
        )
    print(f"\nTotal inference time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
