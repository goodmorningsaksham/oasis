"""
GlucoRL Environment Client.

Provides a typed, synchronous client for interacting with the GlucoRL
FastAPI server over WebSocket. Follows the OpenEnv EnvClient pattern
used by the hackathon evaluator and inference scripts.

Example:
    >>> from client import GlucoEnv
    >>> from models import GlucoAction
    >>>
    >>> with GlucoEnv(base_url="http://localhost:8000") as env:
    ...     result = env.reset(task_id=1)
    ...     print(result.observation.glucose_mg_dl)
    ...     while not result.done:
    ...         action = GlucoAction(basal_rate=1.0, bolus_dose=0.0)
    ...         result = env.step(action)
    ...         print(f"Glucose: {result.observation.glucose_mg_dl}")
"""

from typing import Dict, Any

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

from models import GlucoAction, GlucoObservation, GlucoState, GlucoReward


class GlucoEnv(EnvClient[GlucoAction, GlucoObservation, GlucoState]):
    """
    Client for the GlucoRL Environment.

    Maintains a persistent WebSocket connection to the server so that
    environment state is preserved across reset → step → step → state calls.

    Args:
        base_url: Server URL (http:// or ws://). Automatically converted to ws://.
        **kwargs: Forwarded to EnvClient (e.g. message_timeout_s).
    """

    def __init__(self, base_url: str, **kwargs: Any):
        # simglucose stepping is fast, but allow headroom for slow hosts
        kwargs.setdefault("message_timeout_s", 120.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: GlucoAction) -> Dict[str, Any]:
        """Convert a GlucoAction to the JSON dict the server expects."""
        return {
            "basal_rate": action.basal_rate,
            "bolus_dose": action.bolus_dose,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GlucoObservation]:
        """
        Convert the server's JSON response into a typed StepResult.

        The server sends:
            {
                "observation": { ...fields excluding done/reward/metadata... },
                "reward": float | None,
                "done": bool
            }
        """
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        reward_raw = payload.get("reward")

        observation = GlucoObservation(
            glucose_mg_dl=obs_data.get("glucose_mg_dl", 0.0),
            glucose_trend=obs_data.get("glucose_trend", "stable"),
            meal_announced=obs_data.get("meal_announced", False),
            meal_grams_announced=obs_data.get("meal_grams_announced", 0.0),
            time_of_day_hours=obs_data.get("time_of_day_hours", 0.0),
            step=obs_data.get("step", 0),
            patient_id=obs_data.get("patient_id"),
            last_action_basal=obs_data.get("last_action_basal", 1.0),
            last_action_bolus=obs_data.get("last_action_bolus", 0.0),
            true_glucose_mg_dl=obs_data.get("true_glucose_mg_dl"),
            insulin_on_board_units=obs_data.get("insulin_on_board_units", 0.0),
            exercise_intensity=obs_data.get("exercise_intensity", 0.0),
            exercise_announced=obs_data.get("exercise_announced", False),
            glucose_history_window=obs_data.get("glucose_history_window", []),
            illness_active=obs_data.get("illness_active", False),
            done=done,
            reward=reward_raw,
        )

        # Build decomposed reward if the raw value is a number
        reward_obj = None
        if reward_raw is not None:
            reward_obj = GlucoReward(
                tir_contribution=reward_raw if reward_raw > 0 else 0.0,
                hypo_penalty=reward_raw if reward_raw < 0 and obs_data.get("glucose_mg_dl", 999) < 70 else 0.0,
                hyper_penalty=reward_raw if reward_raw < 0 and obs_data.get("glucose_mg_dl", 0) > 180 else 0.0,
                overdose_penalty=0.0,
                step_total=float(reward_raw),
            )

        return StepResult(
            observation=observation,
            reward=reward_raw,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GlucoState:
        """Convert the server's state JSON into a typed GlucoState."""
        return GlucoState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", 1),
            patient_name=payload.get("patient_name", ""),
            step=payload.get("step", 0),
            done=payload.get("done", False),
            glucose_history=payload.get("glucose_history", []),
            reward_history=payload.get("reward_history", []),
            tir_current=payload.get("tir_current", 0.0),
            hypo_events=payload.get("hypo_events", 0),
            severe_hypo_events=payload.get("severe_hypo_events", 0),
            hyper_events=payload.get("hyper_events", 0),
            episode_reward_total=payload.get("episode_reward_total", 0.0),
        )
