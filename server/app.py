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

from fastapi.responses import HTMLResponse

# Ensure project root is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server.http_server import create_app

from models import GlucoAction, GlucoObservation
from server.glucorl_environment import GlucoRLEnvironment
from server.graders import grade, grade_detailed

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
    """Return descriptions of all 4 GlucoRL tasks."""
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
        {
            "id": 4,
            "name": "Sick Day Management",
            "difficulty": "expert",
            "description": (
                "Random patient with simulated illness causing 1.5-2.5x "
                "insulin resistance starting at an unknown time. "
                "Meals and exercise are unannounced. The agent must "
                "detect rising glucose from resistance and adapt its "
                "dosing strategy without being told illness is occurring."
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


@app.post("/grade", tags=["Evaluation"])
async def grade_episode(task_id: int = 1):
    """
    Grade the current completed episode and return detailed score breakdown.

    Instantiates an environment to access state. Note: for stateful grading,
    use the WebSocket client to run an episode, then call state() and grade
    client-side. This endpoint is provided for convenience and testing.

    Returns 400 if no episode data is available.
    """
    try:
        env = GlucoRLEnvironment()
        state = env.state
        if not state.glucose_history:
            return {"error": "No episode data available. Run an episode first via WebSocket."}
        result = grade_detailed(task_id, state)
        return result
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error("Grade endpoint failed: %s", e, exc_info=True)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Live Glucose Visualisation Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GlucoRL Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }
  h1 { text-align: center; margin-bottom: 20px; color: #38bdf8; font-size: 1.5rem; }
  .stats { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-card { background: #1e293b; border-radius: 10px; padding: 14px 22px;
               text-align: center; min-width: 120px; }
  .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
  .stat-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
  .stat-value.good { color: #4ade80; }
  .stat-value.warn { color: #fbbf24; }
  .stat-value.bad { color: #f87171; }
  .stat-value.neutral { color: #e2e8f0; }
  .chart-container { background: #1e293b; border-radius: 10px; padding: 16px;
                     max-width: 1000px; margin: 0 auto 20px; }
  .controls { display: flex; gap: 10px; justify-content: center; align-items: center; margin-bottom: 20px; }
  select, button { padding: 8px 16px; border-radius: 6px; border: 1px solid #334155;
                   background: #1e293b; color: #e2e8f0; font-size: 0.9rem; cursor: pointer; }
  button { background: #2563eb; border-color: #2563eb; font-weight: 600; }
  button:hover { background: #1d4ed8; }
  .footer { text-align: center; color: #475569; font-size: 0.75rem; margin-top: 12px; }
</style>
</head>
<body>
<h1>GlucoRL — Live Glucose Monitor</h1>

<div class="controls">
  <select id="taskSelect">
    <option value="1">Task 1 — Basal Rate Control</option>
    <option value="2">Task 2 — Meal Bolus Timing</option>
    <option value="3">Task 3 — Cross-Patient</option>
  </select>
  <button onclick="resetEpisode()">Reset Episode</button>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="stat-label">Glucose</div>
    <div class="stat-value neutral" id="glucoseVal">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">TIR</div>
    <div class="stat-value neutral" id="tirVal">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Step</div>
    <div class="stat-value neutral" id="stepVal">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Hypo Events</div>
    <div class="stat-value neutral" id="hypoVal">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Severe Hypo</div>
    <div class="stat-value neutral" id="sevHypoVal">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Reward</div>
    <div class="stat-value neutral" id="rewardVal">--</div>
  </div>
</div>

<div class="chart-container">
  <canvas id="glucoseChart"></canvas>
</div>

<div class="footer">Auto-refreshes every 3 seconds from /state endpoint</div>

<script>
const ctx = document.getElementById('glucoseChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Glucose (mg/dL)',
      data: [],
      borderColor: '#38bdf8',
      backgroundColor: 'rgba(56,189,248,0.1)',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      fill: true
    }]
  },
  options: {
    responsive: true,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: 'Step', color: '#94a3b8' },
        ticks: { color: '#64748b', maxTicksLimit: 20 },
        grid: { color: '#1e293b' }
      },
      y: {
        title: { display: true, text: 'mg/dL', color: '#94a3b8' },
        min: 30, max: 400,
        ticks: { color: '#64748b' },
        grid: { color: '#1e293b' }
      }
    },
    plugins: {
      legend: { labels: { color: '#94a3b8' } },
      annotation: undefined
    }
  },
  plugins: [{
    id: 'zones',
    beforeDraw(chart) {
      const { ctx, chartArea: { left, right, top, bottom }, scales: { y } } = chart;
      function fillZone(yLow, yHigh, color) {
        const pxTop = y.getPixelForValue(Math.min(yHigh, 400));
        const pxBot = y.getPixelForValue(Math.max(yLow, 30));
        ctx.fillStyle = color;
        ctx.fillRect(left, pxTop, right - left, pxBot - pxTop);
      }
      fillZone(0, 54, 'rgba(239,68,68,0.12)');
      fillZone(54, 70, 'rgba(251,191,36,0.08)');
      fillZone(70, 180, 'rgba(74,222,128,0.07)');
      fillZone(180, 250, 'rgba(251,191,36,0.08)');
      fillZone(250, 500, 'rgba(239,68,68,0.10)');
    }
  }]
});

function glucoseClass(g) {
  if (g < 54) return 'bad';
  if (g < 70) return 'warn';
  if (g <= 180) return 'good';
  if (g <= 250) return 'warn';
  return 'bad';
}

async function fetchState() {
  try {
    const r = await fetch('/state');
    const s = await r.json();
    if (!s.glucose_history || s.glucose_history.length === 0) return;
    const gh = s.glucose_history;
    chart.data.labels = gh.map((_, i) => i);
    chart.data.datasets[0].data = gh;
    chart.update();
    const last = gh[gh.length - 1];
    const gEl = document.getElementById('glucoseVal');
    gEl.textContent = last.toFixed(0);
    gEl.className = 'stat-value ' + glucoseClass(last);
    const tirEl = document.getElementById('tirVal');
    const tirPct = (s.tir_current * 100).toFixed(1) + '%';
    tirEl.textContent = tirPct;
    tirEl.className = 'stat-value ' + (s.tir_current >= 0.7 ? 'good' : s.tir_current >= 0.5 ? 'warn' : 'bad');
    document.getElementById('stepVal').textContent = s.step_count + '/480';
    const hypoEl = document.getElementById('hypoVal');
    hypoEl.textContent = s.hypo_events;
    hypoEl.className = 'stat-value ' + (s.hypo_events === 0 ? 'good' : 'bad');
    const shEl = document.getElementById('sevHypoVal');
    shEl.textContent = s.severe_hypo_events;
    shEl.className = 'stat-value ' + (s.severe_hypo_events === 0 ? 'good' : 'bad');
    document.getElementById('rewardVal').textContent = s.episode_reward_total.toFixed(1);
  } catch(e) { console.error('Fetch failed:', e); }
}

async function resetEpisode() {
  const taskId = document.getElementById('taskSelect').value;
  try {
    await fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_id: parseInt(taskId)})
    });
    fetchState();
  } catch(e) { console.error('Reset failed:', e); }
}

fetchState();
setInterval(fetchState, 3000);
</script>
</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse, tags=["UI"])
async def dashboard():
    """Live glucose monitoring dashboard with Chart.js visualisation."""
    return HTMLResponse(content=DASHBOARD_HTML)


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
