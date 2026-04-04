---
title: GlucoRL
emoji: 💉
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

<div align="center">

# GlucoRL

### An OpenEnv Environment for Training AI Agents to Manage Insulin Dosing in Type 1 Diabetes

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.1-blue?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![simglucose](https://img.shields.io/badge/Simulator-UVa%2FPadova%20T1D-green?style=flat-square)](https://github.com/jxx123/simglucose)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

*4 tasks · 8 clinical enhancements · physiologically validated simulator · live dashboard*

</div>

---

## The Problem

Type 1 Diabetes (T1D) affects **over 9 million people worldwide**. These patients produce no insulin and must manage blood glucose through continuous external delivery — a task that is unforgiving in both directions:

**Too little insulin → Hyperglycemia** (glucose > 180 mg/dL). Sustained high blood sugar causes progressive organ damage: retinopathy, nephropathy, neuropathy, and cardiovascular disease. Above 250 mg/dL is considered a medical emergency.

**Too much insulin → Hypoglycemia** (glucose < 70 mg/dL). Acutely life-threatening. Below 54 mg/dL, patients risk seizures, loss of consciousness, and death within minutes. Hypoglycemia is **2–6× more dangerous** than hyperglycemia in the short term.

The clinical gold standard is **Time-in-Range (TIR)**: the percentage of time glucose stays within the safe 70–180 mg/dL window. Clinical guidelines recommend ≥70% TIR. Most patients achieve far less.

### Why Current Systems Fall Short

Commercial insulin pumps use **PID controllers** or Model Predictive Control — rule-based systems with a fundamental limitation: they are tuned for an average patient. No patient is average.

- A child may be 3–5× more insulin-sensitive than an adult
- Exercise increases insulin sensitivity unpredictably by 20–70%
- Illness causes insulin resistance, requiring 1.5–2.5× the normal dose
- Post-meal glucose spikes vary by meal composition, timing, and stress
- CGM sensors lag actual glucose by 5–15 minutes with ±15 mg/dL noise

Static controllers fail silently at all of these. **GlucoRL is an environment for training RL agents that don't.**

---

## Why Reinforcement Learning

An RL agent trained on GlucoRL can learn what rule-based systems cannot:

| Challenge | PID Controller | RL Agent |
|-----------|:--------------:|:--------:|
| Patient personalisation | ✗ Fixed parameters | ✓ Learns per-patient policy |
| Meal bolus timing | ✗ Reactive correction | ✓ Anticipates announced meals |
| Exercise adaptation | ✗ Unaware | ✓ Observes intensity, adjusts |
| Illness/resistance | ✗ Fails dangerously | ✓ Detects from glucose signal |
| Cross-patient transfer | ✗ One size fits none | ✓ Generalises across 30 patients |
| Sensor noise tolerance | ✗ Treats noise as signal | ✓ Learns to filter uncertainty |

---

## Environment Overview

Each episode simulates **one full 24-hour day** (480 steps × 3 minutes per step) using the **FDA-accepted UVa/Padova metabolic simulator** via [simglucose](https://github.com/jxx123/simglucose). The agent interacts with a physiologically realistic virtual patient, observing CGM glucose readings and deciding insulin delivery at every step.

```
Episode: 480 steps × 3 min = 24 hours
Patients: 30 profiles (10 adults, 10 adolescents, 10 children)
Simulator: UVa/Padova T1D (FDA-accepted metabolic model)
CGM: Subcutaneous glucose (Gsub) with ISO 15197 sensor noise
```

---

## Four Tasks — Escalating Real-World Difficulty

| Task | Name | Difficulty | Patient | Meals | Exercise | Illness | Key Challenge |
|:----:|------|:----------:|---------|:-----:|:--------:|:-------:|---------------|
| 1 | Basal Rate Control | 🟢 Easy | adult#001 (fixed) | None | None | None | Maintain stable glucose with basal only |
| 2 | Meal Bolus Timing | 🟡 Medium | adult#001 (fixed) | 3 meals (announced) | Announced | None | Pre-meal bolus timing to prevent spikes |
| 3 | Cross-Patient Generalisation | 🔴 Hard | Random from 30 | 3 meals (unannounced) | Random (unannounced) | None | Adapt to unknown patient physiology |
| 4 | Sick Day Management | ⚫ Expert | Random from 30 | 3 meals (unannounced) | Random (unannounced) | Active (unknown) | Detect and adapt to hidden insulin resistance |

### Task 1 — Basal Rate Control
A single stable adult patient with no meals or exercise. The agent adjusts continuous basal insulin to maintain glucose in the 70–180 mg/dL range for a full simulated day. Establishes the baseline of correct background insulin delivery.

### Task 2 — Meal Bolus Timing
The same adult patient with three daily meals: breakfast (50g CHO, step 100), lunch (70g CHO, step 200), and dinner (80g CHO, step 320). Meals are announced 30 minutes in advance. A moderate exercise event occurs at step 150 (also announced). The agent must learn to pre-dose bolus insulin at the right time and reduce basal during exercise to prevent both post-meal spikes and exercise-induced hypoglycemia.

### Task 3 — Cross-Patient Generalisation
A random patient sampled from all 30 profiles at each reset — adolescents, adults, and children with dramatically different insulin sensitivity. Meals occur at the same schedule but are **not announced**. Random unannounced exercise events occur. The agent receives no patient identifier and must develop a policy that generalises across the full patient population.

### Task 4 — Sick Day Management
The hardest real-world T1D challenge. A random patient develops simulated illness causing **1.5–2.5× insulin resistance** starting at an unknown time (between step 20–100). The agent is never told illness is occurring. It must detect rising glucose despite normal dosing, infer that insulin is less effective than expected, and increase delivery accordingly — without over-correcting into hypoglycemia. PID controllers fail catastrophically on this task.

---

## Observation Space

The agent receives a rich, clinically motivated observation at every step:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `glucose_mg_dl` | float | 20–600 | Current CGM reading with ISO 15197 measurement noise (σ=10 mg/dL) |
| `glucose_trend` | string | 5 values | Rate of change: `rapidly_falling` / `falling` / `stable` / `rising` / `rapidly_rising` |
| `glucose_history_window` | list[float] | 0–600 each | Last 12 CGM readings (36 minutes of history) for temporal reasoning |
| `meal_announced` | bool | — | Meal coming within 30 minutes (Task 2 only) |
| `meal_grams_announced` | float | 0–80 | Carbohydrate grams in upcoming announced meal |
| `exercise_intensity` | float | 0.0–1.0 | Current exercise intensity (0=rest, 1=maximum). Announced in Task 2, unannounced in Task 3/4 |
| `exercise_announced` | bool | — | Exercise event starting within 30 minutes (Task 2 only) |
| `insulin_on_board_units` | float | 0.0–20.0 | Active insulin still in effect from recent boluses (pharmacokinetic decay model) |
| `time_of_day_hours` | float | 0.0–24.0 | Simulated time of day in hours |
| `step` | int | 0–479 | Current step in the episode |
| `patient_id` | string / null | — | Patient identifier (null in Task 3/4 to force generalisation) |
| `last_action_basal` | float | 0.0–5.0 | Basal rate delivered at previous step |
| `last_action_bolus` | float | 0.0–20.0 | Bolus dose delivered at previous step |
| `illness_active` | bool | — | Debug only — always False in normal mode |
| `true_glucose_mg_dl` | float / null | — | Debug only — pre-noise glucose value. None in production |

### Design Notes

**CGM Noise** — Glucose readings simulate real CGM behaviour using the subcutaneous compartment (`Gsub`) from the UVa/Padova model, which naturally lags plasma glucose by 5–15 minutes due to interstitial diffusion. Gaussian measurement noise (σ=10 mg/dL) is applied per the ISO 15197 accuracy standard. Rewards are computed on true glucose; the agent sees the noisy signal.

**Glucose History Window** — 36 minutes of CGM context enables temporal reasoning. An agent can detect trends (sustained rise after a meal vs. transient spike) and compute rate-of-change without requiring an RNN backbone.

**Insulin-on-Board** — Modelled using exponential decay (τ≈4 hours, peak at ~60 minutes). Prevents naive agents from stacking boluses. Commercial artificial pancreas systems (Medtronic MiniMed 780G, Tandem Control-IQ) display IOB as a primary safety signal.

**Exercise Intensity** — Physiologically, exercise increases insulin sensitivity by 20–70% depending on intensity. The agent observes current intensity and must reduce insulin delivery to prevent exercise-induced hypoglycemia.

---

## Action Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `basal_rate` | float | 0.0–5.0 | Continuous background insulin in units/hr |
| `bolus_dose` | float | 0.0–20.0 | One-time meal or correction insulin in units |

The agent delivers insulin in two clinically standard forms. Basal insulin provides continuous background coverage; bolus insulin compensates for meals or corrects elevated glucose. Both are delivered every 3 minutes.

---

## Reward Function

Dense per-step reward provides continuous training signal throughout each episode. The function is clinically motivated: hypoglycemia receives **2–6× heavier penalties** than hyperglycemia because it is acutely life-threatening while hyperglycemia causes slower cumulative damage.

| Condition | Component | Value | Clinical Rationale |
|-----------|-----------|:-----:|--------------------|
| Glucose 70–180 mg/dL | TIR contribution | **+1.0** | In safe target range |
| Glucose 54–70 mg/dL | Hypo penalty | **−1.0** | Mild hypoglycemia — dangerous |
| Glucose < 54 mg/dL | Severe hypo penalty | **−3.0** | Severe hypoglycemia — life-threatening |
| Glucose 180–250 mg/dL | Hyper penalty | **−0.5** | Mild hyperglycemia — long-term damage |
| Glucose > 250 mg/dL | Severe hyper penalty | **−1.5** | Severe hyperglycemia — acute risk |
| < 54 mg/dL + large recent bolus | Overdose penalty | **−3.0** | Bolus-induced crash — prevents reward hacking |
| Hypo corrected within 10 steps | Recovery bonus | **+0.5** | Rewards active correction over passive waiting |
| Hyper corrected within 10 steps | Recovery bonus | **+0.3** | Rewards proactive management |

### Why the Recovery Bonus Matters

Without a recovery signal, an agent can learn to passively wait for glucose to self-correct. The recovery bonus incentivises active clinical management — the agent must actively bring glucose back into range within a clinically reasonable timeframe (30 minutes). This produces more medically sound behaviour and better real-world transferability.

---

## Baseline Scores

Scores are computed by task-specific graders returning values in [0.0, 1.0]. All results are deterministic (seed=42) and fully reproducible via `python eval.py`.

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Task 4 (Expert) |
|-------|:---:|:---:|:---:|:---:|
| Constant Basal (fallback) | 1.000 | 0.000 | 0.345 | ~0.050 |
| PID Controller | 1.000 | 0.736 | 0.206 | ~0.120 |
| **Target: Good RL Agent** | **≥ 0.95** | **≥ 0.70** | **≥ 0.60** | **≥ 0.45** |

### Task 1 — Basal Rate Control
Both agents achieve perfect TIR (100%) — without meals, adult#001's glucose remains stable on any reasonable basal rate. The PID holds tighter (123–139 mg/dL) vs. constant basal (139–175 mg/dL).

### Task 2 — Meal Bolus Timing
The constant basal agent scores 0.000: three uncompensated meals produce sustained hyperglycemia (peak 330 mg/dL, 76% of the day above range). The PID controller achieves 84.6% TIR and score 0.736 through correction boluses, but still incurs post-meal spike penalties from reactive rather than pre-emptive dosing. An RL agent that learns to anticipate announced meals can eliminate these spikes entirely.

### Task 3 — Cross-Patient Generalisation
This task exposes why adaptive control matters. The adult-tuned PID controller causes **fatal hypoglycemia in 4 of 5 evaluation patients** — children and adolescents are far more insulin-sensitive. It scores 0.206 on average. The "dumber" constant basal scores 0.345 because its conservative dosing avoids the most dangerous crashes — though it still fails sensitive patients.

Per-patient PID results (seed=42):

| Patient | TIR | Score | Outcome |
|---------|:---:|:-----:|---------|
| child#001 | 78.0% | 0.030 | Emergency termination at step 50 |
| adolescent#004 | 73.9% | 0.000 | Emergency termination at step 203 |
| adolescent#001 | 100.0% | 1.000 | Full episode — adult-like sensitivity |
| child#004 | 72.9% | 0.000 | Emergency termination at step 59 |
| adolescent#009 | 74.0% | 0.000 | Emergency termination at step 408 |

An RL agent that detects patient insulin sensitivity from early glucose dynamics and adapts its policy accordingly has dramatic room to outperform both baselines.

### Task 4 — Sick Day Management
Hidden insulin resistance makes Task 4 genuinely frontier-level. The PID controller, calibrated for healthy physiology, fails to increase delivery when insulin becomes less effective and scores approximately 0.120. The constant basal agent, paradoxically, performs slightly better in some runs by accidentally providing conservative dosing that avoids catastrophic hypoglycemia during resistance phases. A successful RL agent must learn to detect the illness signature in glucose dynamics and increase delivery proportionally — a problem that directly mirrors real clinical sick-day management protocols.

---

## Clinical Simulation Fidelity

GlucoRL prioritises physiological accuracy over simplicity:

**Simulator** — The [UVa/Padova metabolic model](https://github.com/jxx123/simglucose) is the FDA-accepted gold standard for in-silico T1D research. It models glucose-insulin kinetics, hepatic glucose production, insulin absorption from subcutaneous tissue, and glucose utilisation by peripheral and central tissues for 30 validated virtual patients.

**CGM Model** — The subcutaneous glucose compartment (`Gsub`) provides natural physiological lag matching real CGM behaviour. ISO 15197-calibrated measurement noise (σ=10 mg/dL) is layered on top. Reward computation uses true glucose; the agent operates on the noisy signal — exactly as a real closed-loop system must.

**Insulin Pharmacokinetics** — Insulin-on-board tracking uses an exponential decay model (IOB_STEP_DECAY=0.94 per 3-minute step, peak activity at ~60 minutes, clearance by ~240 minutes). This matches the pharmacokinetic profile of rapid-acting insulin analogues (Humalog, Novolog, Fiasp) used in commercial pumps.

**Exercise Physiology** — Exercise events increase insulin sensitivity by 20–70% depending on intensity, modelled as a multiplier on effective insulin delivery. This reflects the well-documented phenomenon of exercise-enhanced glucose transport in skeletal muscle.

**Insulin Resistance** — Task 4 illness simulation applies a 1.5–2.5× resistance multiplier to effective insulin delivery, modelling the inflammatory response that reduces insulin receptor sensitivity during infection or illness. The onset timing and severity are randomised per episode, requiring the agent to infer resistance from glucose behaviour alone.

---

## Setup and Usage

### Quick Start — Local

```bash
# Clone and install
git clone https://github.com/saksham1771/glucorl.git
cd glucorl
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Open the live dashboard
open http://localhost:8000/dashboard

# Run baseline inference
export GLUCORL_ENV_URL="http://localhost:8000"
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
python inference.py
```

### Docker

```bash
docker build -t glucorl .
docker run -p 8000:8000 glucorl

# Verify all endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'
curl http://localhost:8000/tasks
```

### Python Client

```python
from client import GlucoEnv
from models import GlucoAction

with GlucoEnv(base_url="http://localhost:8000") as env:
    # Run Task 2 — Meal Bolus Timing
    result = env.reset(task_id=2)
    while not result.done:
        obs = result.observation
        # Agent sees: glucose, trend, meal announcement, IOB, exercise, history window
        print(f"Glucose: {obs.glucose_mg_dl:.1f} | "
              f"Trend: {obs.glucose_trend} | "
              f"Meal: {obs.meal_announced} | "
              f"IOB: {obs.insulin_on_board_units:.2f}u | "
              f"Exercise: {obs.exercise_intensity:.1f}")

        action = GlucoAction(basal_rate=1.2, bolus_dose=2.0 if obs.meal_announced else 0.0)
        result = env.step(action)

    state = env.state()
    print(f"\nFinal TIR: {state.tir_current:.1%}")
    print(f"Total reward: {state.episode_reward_total:.1f}")
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Evaluation (PID vs Constant Basal)

```bash
python eval.py
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode. Body: `{"task_id": 1}` (1–4) |
| POST | `/step` | Take action. Body: `{"basal_rate": 1.0, "bolus_dose": 0.0}` |
| GET | `/state` | Full episode state: glucose history, TIR, clinical events |
| GET | `/tasks` | List all 4 tasks with descriptions and difficulty |
| POST | `/grade` | Detailed score breakdown for current episode (decomposed components) |
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| GET | `/dashboard` | Live glucose monitoring web interface with real-time Chart.js visualisation |
| WS | `/ws` | WebSocket for persistent sessions (used by Python client) |

### `/grade` Response Example

```json
{
  "total": 0.68,
  "tir_score": 0.84,
  "tir_readings": 403,
  "total_readings": 479,
  "hypo_penalty": 0.0,
  "post_meal_penalties": {
    "step_100": {"peak": 195.2, "penalty": 0.03},
    "step_200": {"peak": 212.1, "penalty": 0.08},
    "step_320": {"peak": 188.4, "penalty": 0.03}
  },
  "bonus": 0.05,
  "components": {
    "tir": 0.84,
    "no_hypo_bonus": 0.05,
    "post_meal_total": -0.14,
    "severe_hypo_penalty": 0.0
  },
  "clinical_summary": {
    "hypo_events": 0,
    "hyper_events": 76,
    "severe_hypo_events": 0
  }
}
```

---

## Training with RL

GlucoRL is designed for compatibility with modern RL training frameworks:

**GRPO via TRL** — The environment's dense per-step reward signal, continuous 2D action space, and 480-step episodes are well-suited for [TRL's GRPO implementation](https://github.com/huggingface/trl). The four-task curriculum provides natural difficulty scaling for progressive training.

**Observation Design** — The `glucose_history_window` (12 readings = 36 minutes of context) enables standard feedforward agents to reason temporally without requiring RNN or transformer architectures. The full history is also available via the `/state` endpoint for agents that prefer complete episode context.

**Reward Shaping** — The recovery bonus (+0.5/+0.3) and asymmetric hypo/hyper penalties produce clear signal variance across episode types. Successful episodes achieve cumulative rewards of +300 to +480; failed episodes score -200 to -500 depending on the severity and duration of out-of-range events. GRPO requires this variance to compute meaningful advantages.

**Example Training Loop**

```python
from client import GlucoEnv
from models import GlucoAction

env_url = "http://localhost:8000"
rollouts = []

for episode in range(8):  # 8 parallel rollouts for GRPO
    trajectory = []
    with GlucoEnv(base_url=env_url) as env:
        obs = env.reset(task_id=2).observation
        while True:
            # Your policy here
            action = policy(obs)
            result = env.step(action)
            trajectory.append({
                "obs": obs,
                "action": action,
                "reward": result.reward,
            })
            if result.done:
                break
            obs = result.observation
    rollouts.append(trajectory)

# Pass rollouts to GRPO trainer
```

---

## Project Structure

```
glucorl/
├── inference.py                  # Baseline inference script (OpenAI client)
├── models.py                     # Pydantic models: Action, Observation, State, Reward
├── client.py                     # GlucoEnv WebSocket client
├── eval.py                       # PID vs baseline evaluation
├── openenv.yaml                  # OpenEnv spec (4 tasks)
├── Dockerfile                    # HF Spaces deployment (openenv-base)
├── requirements.txt              # Python dependencies
├── server/
│   ├── app.py                    # FastAPI server + /dashboard + /grade endpoints
│   ├── glucorl_environment.py    # Core environment: reset/step/state
│   ├── patient_manager.py        # simglucose wrapper with CGM noise + exercise
│   ├── reward_calculator.py      # Shaped reward with recovery bonus
│   ├── graders.py                # Task graders + grade_detailed()
│   ├── pid_controller.py         # PID baseline with anti-windup
│   └── constants.py              # All thresholds, schedules, pharmacokinetics
└── tests/
    ├── test_environment.py       # reset/step/state/meal/exercise/termination tests
    ├── test_graders.py           # Grader range, determinism, and scoring tests
    └── test_reward.py            # Per-zone reward and recovery bonus tests
```

---

## Technical Implementation Notes

**OpenEnv Compliance** — `GlucoAction`, `GlucoObservation`, and `GlucoState` inherit from OpenEnv base types (`Action`, `Observation`, `State`). The server uses the `create_app` factory from `openenv.core` for fully spec-compliant `/reset`, `/step`, `/state`, `/health`, `/schema`, and `/ws` endpoints.

**simglucose Integration** — Patient physiology uses `T1DPatient.withName()` with `SimAction(insulin=X, CHO=Y)` namedtuples. Each 3-minute environment step runs 3 × 1-minute simulator mini-steps with meal CHO injected on the first mini-step only.

**Glucose Field** — `patient.observation.Gsub` (subcutaneous glucose) is used throughout. This is the correct field for simglucose v0.2.11 and accurately represents what a real CGM sensor measures.

**Determinism** — Task graders are pure functions of episode state with no randomness. The same glucose history always produces the same score. Task 3's 5-patient evaluation uses `random.Random(42).sample()` for a fixed, reproducible patient set.

---

## Acknowledgements

- **simglucose** — FDA-accepted UVa/Padova T1D metabolic simulator by [Jinyu Xie](https://github.com/jxx123/simglucose)
- **OpenEnv** — Open environment specification by [Meta PyTorch team](https://github.com/meta-pytorch/OpenEnv)
- **UVa/Padova Model** — Kovatchev et al., *Journal of Diabetes Science and Technology*, 2009