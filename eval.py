"""
OASIS Evaluation Script
=========================
Runs the PID controller baseline and a constant-basal fallback agent through
all OASIS tasks, printing a comparison table suitable for the README.

Optionally connects to an LLM agent via the inference server if GLUCORL_ENV_URL
is set (requires a running server and LLM credentials).

Usage:
    python eval.py                  # Direct evaluation (no server needed)
    python eval.py --with-server    # Also test via server + client
"""

import argparse
import sys
import time

from models import GlucoAction
from server.glucorl_environment import GlucoRLEnvironment
from server.pid_controller import PIDController
from server.graders import grade, score_task_3_single
from server.constants import TASK3_EVAL_PATIENTS


def run_episode(env: GlucoRLEnvironment, task_id: int, agent_fn, seed: int = 42) -> dict:
    """
    Run a single episode and return metrics.

    Args:
        env: OASIS environment instance.
        task_id: Task to evaluate (1, 2, or 3).
        agent_fn: Callable(glucose: float, obs) -> GlucoAction.
        seed: Random seed for reproducibility.

    Returns:
        Dict with tir, score, hypo_events, severe_hypo_events, hyper_events, steps.
    """
    obs = env.reset(task_id=task_id, seed=seed)
    glucose = obs.glucose_mg_dl

    for _ in range(480):
        action = agent_fn(glucose, obs)
        obs = env.step(action)
        glucose = obs.glucose_mg_dl
        if obs.done:
            break

    state = env.state
    score = grade(task_id, state)

    return {
        "tir": state.tir_current,
        "score": score,
        "hypo_events": state.hypo_events,
        "severe_hypo_events": state.severe_hypo_events,
        "hyper_events": state.hyper_events,
        "steps": state.step,
        "reward": state.episode_reward_total,
        "glucose_min": min(state.glucose_history),
        "glucose_max": max(state.glucose_history),
    }


def run_task3_full(env: GlucoRLEnvironment, agent_fn) -> dict:
    """
    Run the full Task 3 evaluation: 5 episodes with fixed patient set.

    Args:
        env: OASIS environment instance.
        agent_fn: Callable(glucose: float, obs) -> GlucoAction.

    Returns:
        Dict with per-patient scores and average.
    """
    patient_scores = []

    for patient in TASK3_EVAL_PATIENTS:
        # Manually set up the environment for a specific patient
        env._task_id = 3
        env._patient_name = patient
        initial_g = env._patient_mgr.reset(patient)
        env._step_count = 0
        env._done = False
        env._glucose_history = [initial_g]
        env._reward_history = []
        env._action_history = []
        env._episode_reward = 0.0
        env._consecutive_severe_hypo = 0
        env._hypo_events = 0
        env._severe_hypo_events = 0
        env._hyper_events = 0
        env._episode_id = f"eval-{patient}"

        glucose = initial_g
        obs = env._build_observation(glucose)

        for _ in range(480):
            action = agent_fn(glucose, obs)
            obs = env.step(action)
            glucose = obs.glucose_mg_dl
            if obs.done:
                break

        state = env.state
        s = score_task_3_single(state)
        patient_scores.append({
            "patient": patient,
            "score": s,
            "tir": state.tir_current,
            "steps": state.step,
            "severe_hypo": state.severe_hypo_events,
        })

    avg_score = sum(p["score"] for p in patient_scores) / len(patient_scores)

    return {
        "patients": patient_scores,
        "average_score": avg_score,
    }


def pid_agent(glucose: float, obs) -> GlucoAction:
    """PID controller agent — uses global PID instance."""
    return _pid.act(glucose)


def fallback_agent(glucose: float, obs) -> GlucoAction:
    """Constant basal agent — no intelligence."""
    return GlucoAction(basal_rate=1.0, bolus_dose=0.0)


# Global PID instance (reset between tasks)
_pid = PIDController()


def print_separator():
    print(f"{'─' * 72}")


def main():
    parser = argparse.ArgumentParser(description="OASIS Evaluation")
    parser.add_argument(
        "--with-server",
        action="store_true",
        help="Also test via running server (requires uvicorn on port 8000)",
    )
    args = parser.parse_args()

    env = GlucoRLEnvironment()

    agents = {
        "Constant Basal": fallback_agent,
        "PID Controller": pid_agent,
    }

    print()
    print("OASIS Evaluation")
    print("==================")
    print()

    # ──────────────────────────────────────────────────────────────
    # Task 1 & 2: single-episode evaluation
    # ──────────────────────────────────────────────────────────────
    results = {}

    for task_id in [1, 2]:
        print_separator()
        print(f"  Task {task_id}")
        print_separator()

        for agent_name, agent_fn in agents.items():
            if agent_name == "PID Controller":
                _pid.reset()

            r = run_episode(env, task_id, agent_fn, seed=42)
            results[(agent_name, task_id)] = r

            print(
                f"  {agent_name:20s} │ TIR {r['tir']:5.1%} │ "
                f"Score {r['score']:.3f} │ Hypo {r['hypo_events']:3d} │ "
                f"SevHypo {r['severe_hypo_events']:2d} │ "
                f"Hyper {r['hyper_events']:3d} │ "
                f"Steps {r['steps']:3d} │ "
                f"Glucose [{r['glucose_min']:.0f}–{r['glucose_max']:.0f}]"
            )
        print()

    # ──────────────────────────────────────────────────────────────
    # Task 3: multi-patient evaluation
    # ──────────────────────────────────────────────────────────────
    print_separator()
    print("  Task 3 — Cross-Patient Evaluation (5 patients)")
    print_separator()

    for agent_name, agent_fn in agents.items():
        if agent_name == "PID Controller":
            _pid.reset()

        t3 = run_task3_full(env, agent_fn)
        results[(agent_name, 3)] = {"score": t3["average_score"]}

        print(f"  {agent_name}:")
        for p in t3["patients"]:
            status = "✓" if p["steps"] == 480 else f"terminated@{p['steps']}"
            print(
                f"    {p['patient']:18s} │ TIR {p['tir']:5.1%} │ "
                f"Score {p['score']:.3f} │ SevHypo {p['severe_hypo']:2d} │ "
                f"{status}"
            )
        print(f"    {'AVERAGE':18s} │       │ Score {t3['average_score']:.3f}")
        print()

    # ──────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────
    print_separator()
    print("  SUMMARY")
    print_separator()
    print(f"  {'Agent':20s} │ {'Task 1':>8s} │ {'Task 2':>8s} │ {'Task 3':>8s}")
    print(f"  {'─' * 20} │ {'─' * 8} │ {'─' * 8} │ {'─' * 8}")

    for agent_name in agents:
        scores = []
        for task_id in [1, 2, 3]:
            s = results.get((agent_name, task_id), {}).get("score", 0.0)
            scores.append(s)
        print(
            f"  {agent_name:20s} │ {scores[0]:8.3f} │ {scores[1]:8.3f} │ {scores[2]:8.3f}"
        )

    print()
    print("  All scores are in [0.0, 1.0]. Higher is better.")
    print("  Task 3 score is averaged over 5 patients with fixed seed.")
    print()


if __name__ == "__main__":
    main()
