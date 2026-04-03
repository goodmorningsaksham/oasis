"""
FastAPI application for the GlucoRL Environment.

Uses OpenEnv's create_app factory to generate standard endpoints:
    - POST /reset   : Reset the environment (accepts task_id in body)
    - POST /step    : Execute an insulin dosing action
    - GET  /state   : Get current environment state
    - GET  /health  : Health check
    - GET  /schema  : Action/observation JSON schemas
    - WS   /ws      : WebSocket for persistent sessions

Additional custom endpoints:
    - GET  /tasks   : List all 3 tasks with descriptions
    - GET  /healthz : Alias health check

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import logging
import sys
import os

# Ensure project root is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server.http_server import create_app

from models import GlucoAction, GlucoObservation
from server.glucorl_environment import GlucoRLEnvironment
from server.graders import grade

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create the OpenEnv-compliant FastAPI app
# ---------------------------------------------------------------------------

app = create_app(
    GlucoRLEnvironment,
    GlucoAction,
    GlucoObservation,
    env_name="glucorl",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# Custom endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["Environment Info"])
async def list_tasks():
    """Return descriptions of all 3 GlucoRL tasks."""
    return [
        {
            "id": 1,
            "name": "Basal Rate Control",
            "difficulty": "easy",
            "description": (
                "Single stable adult patient, no meals. "
                "Optimise basal insulin rate to keep glucose in "
                "70-180 mg/dL for a full 24-hour simulated day."
            ),
        },
        {
            "id": 2,
            "name": "Meal Bolus Timing",
            "difficulty": "medium",
            "description": (
                "Same adult patient with 3 announced daily meals "
                "(breakfast 50g, lunch 70g, dinner 80g). "
                "Deliver correct bolus doses at the right time to "
                "prevent post-meal spikes while avoiding hypoglycemia."
            ),
        },
        {
            "id": 3,
            "name": "Cross-Patient Generalisation",
            "difficulty": "hard",
            "description": (
                "Random patient sampled from 30 profiles "
                "(adolescent, adult, child). Meals are NOT announced. "
                "Develop a policy that generalises across varied "
                "patient physiology without knowing which patient "
                "is being treated."
            ),
        },
    ]


@app.get("/healthz", tags=["Health"])
async def healthz():
    """Quick health check — verifies the environment can be instantiated."""
    try:
        env = GlucoRLEnvironment()
        return {"status": "ok"}
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the GlucoRL server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
