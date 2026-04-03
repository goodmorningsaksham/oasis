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

# GlucoRL: An OpenEnv Environment for Training AI Agents to Manage Insulin Dosing in Type 1 Diabetes

GlucoRL is a reinforcement learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec that trains AI agents to make real-time insulin dosing decisions for Type 1 Diabetic (T1D) patients. Agents observe continuous glucose monitor (CGM) readings and decide how much insulin to deliver every 3 minutes over a simulated 24-hour day, using the FDA-accepted UVa/Padova metabolic simulator ([simglucose](https://github.com/jxx123/simglucose)).

## The Problem

Type 1 Diabetes affects over 9 million people worldwide. These patients produce no insulin and must manage blood glucose through external delivery — a task that is unforgiving in both directions:

**Too little insulin → Hyperglycemia** (glucose > 180 mg/dL). Sustained high blood sugar causes progressive organ damage: retinopathy, nephropathy, neuropathy, and cardiovascular disease. Over 250 mg/dL is considered severe.

**Too much insulin → Hypoglycemia** (glucose < 70 mg/dL). Acutely dangerous. Below 54 mg/dL, patients risk seizures, loss of consciousness, and death within minutes. Hypoglycemia is 2–4× more dangerous than hyperglycemia in the short term.

The clinical gold standard metric is **Time-in-Range (TIR)**: the percentage of time glucose stays within the safe 70–180 mg/dL window. Clinical guidelines recommend ≥70% TIR.

Current commercial insulin pumps use PID (Proportional-Integral-Derivative) controllers or Model Predictive Control. These rule-based systems cannot adapt to individual patient variability, changing meal patterns, exercise, stress, or illness. They use fixed parameters tuned for an "average" patient — but no patient is average.

## Why Reinforcement Learning

An RL agent can learn what static controllers cannot:

- **Patient personalisation**: Different patients have dramatically different insulin sensitivity. Children can be 3–5× more sensitive than adults. An RL agent can learn patient-specific policies from interaction.
- **Meal adaptation**: Post-meal glucose spikes are the primary source of poor TIR. An RL agent can learn optimal bolus timing and dosing for different meal sizes.
- **Proactive control**: Rather than reacting to glucose readings already out of range, an RL agent can learn to anticipate patterns and act preventively.
- **Safety under uncertainty**: RL with shaped rewards can learn the asymmetry between hypo and hyper risk — that it is far worse to crash low than to drift high.

GlucoRL provides the standardised training and evaluation environment to develop and benchmark such agents.

## Environment Description

Each episode simulates one full 24-hour day (480 steps × 3 minutes per step). The agent interacts with a physiologically realistic virtual patient through the simglucose simulator.

**Observation**: At each step, the agent receives the current CGM glucose reading, a trend indicator, meal announcements (in Task 2), time of day, and its previous action.

**Action**: The agent sets two insulin parameters — a continuous basal rate (background insulin) and a bolus dose (meal/correction insulin).

**Reward**: A shaped per-step reward that provides +1.0 for in-range glucose, with asymmetric penalties for hypoglycemia (−1.0 to −3.0) and hyperglycemia (−0.5 to −1.5), plus an overdose penalty when a large bolus causes a severe crash.

## Tasks

| Task | Name | Difficulty | Patient | Meals | Announced | Objective |
|------|------|-----------|---------|-------|-----------|-----------|
| 1 | Basal Rate Control | Easy | adult#001 (fixed) | None | — | Keep glucose in 70–180 mg/dL using basal only |
| 2 | Meal Bolus Timing | Medium | adult#001 (fixed) | 3 meals (50g, 70g, 80g) | Yes (30 min ahead) | Manage post-meal spikes with bolus dosing |
| 3 | Cross-Patient Generalisation | Hard | Random from 30 | 3 meals (same schedule) | No | Develop a robust policy across diverse patients |

## Action Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `basal_rate` | float | 0.0 – 5.0 | Continuous background insulin delivery in units/hr |
| `bolus_dose` | float | 0.0 – 20.0 | One-time meal or correction insulin in units |

## Observation Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `glucose_mg_dl` | float | ~20 – 600 | Current CGM blood glucose reading in mg/dL |
| `glucose_trend` | string | 5 values | Rate of change: `rapidly_falling`, `falling`, `stable`, `rising`, `rapidly_rising` |
| `meal_announced` | bool | — | Whether a meal is coming within 30 minutes (Task 2 only) |
| `meal_grams_announced` | float | 0 – 80 | Carbohydrate grams in the upcoming meal |
| `time_of_day_hours` | float | 0.0 – 24.0 | Simulated time of day in hours |
| `step` | int | 0 – 479 | Current step number in the episode |
| `patient_id` | string or null | — | Patient identifier (null in Task 3 to prevent memorisation) |
| `last_action_basal` | float | 0.0 – 5.0 | Basal rate from the previous step |
| `last_action_bolus` | float | 0.0 – 20.0 | Bolus dose from the previous step |

## Reward Function

The reward is computed at every step, providing dense signal for RL training:

| Glucose Zone | Reward Component | Value | Rationale |
|-------------|-----------------|-------|-----------|
| 70–180 mg/dL (target) | TIR contribution | +1.0 | In safe range |
| 54–70 mg/dL (mild hypo) | Hypo penalty | −1.0 | Dangerous, needs correction |
| < 54 mg/dL (severe hypo) | Hypo penalty | −3.0 | Life-threatening |
| 180–250 mg/dL (mild hyper) | Hyper penalty | −0.5 | Causes long-term damage |
| > 250 mg/dL (severe hyper) | Hyper penalty | −1.5 | Acute risk |
| < 54 mg/dL + recent large bolus | Overdose penalty | −3.0 | Punishes bolus-induced crashes |

Hypoglycemia is penalised 2–6× more heavily than hyperglycemia because it is acutely life-threatening, while hyperglycemia causes slower, cumulative damage. This asymmetry reflects real clinical risk.

## Baseline Scores

Scores are computed by task-specific graders that return a value in [0.0, 1.0] based on Time-in-Range, post-meal spike penalties, and severe hypoglycemia penalties. All results below are deterministic (seed=42) and fully reproducible via `python eval.py`.

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) |
|-------|:---:|:---:|:---:|
| Constant Basal (no intelligence) | 1.000 | 0.000 | 0.345 |
| PID Controller | 1.000 | 0.736 | 0.206 |
| Target: Good RL Agent | ≥ 0.95 | ≥ 0.70 | ≥ 0.60 |

### Detailed Task 1 — Basal Rate Control

Both agents achieve perfect TIR (100%) because without meals, adult#001's glucose stays in the 70–180 mg/dL range with any reasonable basal rate. The PID controller holds glucose tighter (123–139 mg/dL) compared to constant basal (139–175 mg/dL).

### Detailed Task 2 — Meal Bolus Timing

The constant basal agent scores 0.000 because three uncompensated meals cause sustained hyperglycemia (glucose peaks at 330 mg/dL, 76% of the day spent above range). The PID controller achieves TIR of 84.6% and score 0.736 by using correction boluses when glucose exceeds 200 mg/dL, though it still incurs post-meal spike penalties (peak 205 mg/dL). An RL agent that learns pre-meal bolus timing could eliminate these spikes entirely.

### Detailed Task 3 — Cross-Patient Generalisation

This is where the environment reveals why adaptive control matters. The PID controller, tuned for adults, causes **fatal hypoglycemia in 4 out of 5 evaluation patients** (children and adolescents are far more insulin-sensitive). It scores only 0.206 averaged across the 5-patient evaluation set. Paradoxically, the "dumber" constant basal agent scores better (0.345) because its conservative dosing avoids crashing sensitive patients — though it still fails children who go hypoglycemic even on low basal rates.

Per-patient PID results (seed=42):

| Patient | TIR | Score | Outcome |
|---------|-----|-------|---------|
| child#001 | 78.0% | 0.030 | Emergency termination at step 50 |
| adolescent#004 | 73.9% | 0.000 | Emergency termination at step 203 |
| adolescent#001 | 100.0% | 1.000 | Full episode completed |
| child#004 | 72.9% | 0.000 | Emergency termination at step 59 |
| adolescent#009 | 74.0% | 0.000 | Emergency termination at step 408 |

An RL agent that detects patient insulin sensitivity from early glucose dynamics and adapts its dosing policy accordingly has significant room to outperform both baselines on Task 3.

## Setup and Usage

### Local Development

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/glucorl.git
cd glucorl
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
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

# Verify
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Evaluation

```bash
python eval.py
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start a new episode. Body: `{"task_id": 1}` |
| POST | `/step` | Take an action. Body: `{"basal_rate": 1.0, "bolus_dose": 0.0}` |
| GET | `/state` | Get full episode state with glucose history and metrics |
| GET | `/tasks` | List all 3 tasks with descriptions |
| GET | `/health` | Health check |
| WS | `/ws` | WebSocket for persistent sessions (used by client) |

### Python Client

```python
from client import GlucoEnv
from models import GlucoAction

with GlucoEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id=2)
    while not result.done:
        action = GlucoAction(basal_rate=1.2, bolus_dose=0.0)
        result = env.step(action)
        print(f"Glucose: {result.observation.glucose_mg_dl:.1f} mg/dL")
    state = env.state()
    print(f"TIR: {state.tir_current:.1%}")
```

## Training with RL

GlucoRL is designed to be compatible with GRPO training via [TRL](https://github.com/huggingface/trl). The environment's dense per-step reward signal, continuous action space, and 480-step episodes make it well-suited for policy gradient methods. The three-task curriculum provides natural difficulty scaling for progressive training.

## Acknowledgements

- **simglucose**: FDA-accepted UVa/Padova T1D metabolic simulator by Jinyu Xie
- **OpenEnv**: Open environment specification by Meta PyTorch team
