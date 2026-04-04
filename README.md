---
title: OASIS
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

# 🏥 OASIS

### Optimized Adaptive System for Insulin Scheduling

*An OpenEnv reinforcement learning environment for training AI agents to manage insulin dosing in Type 1 Diabetes*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.1-blue?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![simglucose](https://img.shields.io/badge/Simulator-UVa%2FPadova%20T1D-green?style=flat-square)](https://github.com/jxx123/simglucose)
[![Tests](https://img.shields.io/badge/Tests-120%20passed-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**4 tasks · 19-field observation space · gamma-CDF pharmacokinetics · live interactive dashboard**

[Live Demo](https://huggingface.co/spaces/saksham1771/glucorl) · [API Docs](https://huggingface.co/spaces/saksham1771/glucorl/docs) · [Quick Start](#quick-start)

</div>

---

## Why This Matters

Type 1 Diabetes affects **over 9 million people worldwide**. These patients produce zero insulin and must deliver it externally every few minutes — a decision that is unforgiving in both directions:

| | Condition | Glucose Level | Consequence |
|---|---|---|---|
| ⚠️ | Mild Hyperglycemia | > 180 mg/dL | Progressive organ damage over months |
| 🔴 | Severe Hyperglycemia | > 250 mg/dL | Diabetic ketoacidosis — emergency |
| ⚠️ | Mild Hypoglycemia | < 70 mg/dL | Confusion, tremors, impaired function |
| 💀 | Severe Hypoglycemia | < 54 mg/dL | Seizures, loss of consciousness, **death within minutes** |

The clinical gold standard is **Time-in-Range (TIR)**: the percentage of time glucose stays within 70–180 mg/dL. Guidelines recommend ≥70%. Most patients achieve far less.

**Current commercial insulin pumps use PID controllers** — rule-based systems tuned for an "average" patient. But no patient is average. A child may be 5× more insulin-sensitive than an adult. Exercise changes sensitivity unpredictably. Illness causes resistance. These controllers fail silently, and patients pay the price.

**OASIS exists to train RL agents that adapt where PID controllers cannot.**

---

## What Makes OASIS Different

OASIS is not a toy environment. Every design decision is grounded in clinical physiology:

| Feature | Implementation | Clinical Basis |
|---------|---------------|----------------|
| **FDA-accepted simulator** | UVa/Padova T1D model via simglucose | Gold standard for in-silico T1D research |
| **30 virtual patients** | Adolescents, adults, children with distinct physiology | Real inter-patient variability |
| **CGM noise** | σ=10 mg/dL Gaussian on subcutaneous glucose (Gsub) | ISO 15197 accuracy standard |
| **Gamma-CDF pharmacokinetics** | IOB modelled with gamma distribution (peak 55 min, clear 8 hrs) | Rapid-acting insulin absorption profile (Lispro/Aspart) |
| **Exercise physiology** | 20–70% insulin sensitivity increase during activity | Skeletal muscle glucose transport |
| **Illness simulation** | 1.5–2.5× insulin resistance at unknown onset | Inflammatory insulin receptor downregulation |
| **Asymmetric reward** | Hypo penalised 2–6× heavier than hyper | Acute vs. cumulative clinical risk |
| **Recovery bonus** | +0.5 for hypo correction, +0.3 for hyper within 10 steps | Incentivises active clinical management |

---

## The RL Problem

At each 3-minute step, the agent observes a 19-field clinical state and outputs a 2D continuous action:

```
        ┌─────────────────────────────────────────┐
        │           OBSERVATION (19 fields)        │
        │                                          │
        │  CGM glucose (noisy) ──────── 142.3 mg/dL│
        │  Glucose trend ──────────────── rising   │
        │  12-reading history window ─── [138, ...] │
        │  Meal announced? ──────────────── Yes    │
        │  Meal carbs ───────────────────── 70g    │
        │  Exercise intensity ────────────── 0.0   │
        │  Insulin-on-board (gamma-CDF) ── 2.4 U  │
        │  Time of day ──────────────────── 10.0 h │
        │  ... and 11 more fields                  │
        └────────────────┬────────────────────────┘
                         │
                    ┌────▼────┐
                    │  AGENT  │
                    └────┬────┘
                         │
        ┌────────────────▼────────────────────────┐
        │            ACTION (2 fields)             │
        │                                          │
        │  Basal rate ──────── 1.2 U/hr (0.0–5.0) │
        │  Bolus dose ──────── 5.0 U   (0.0–20.0) │
        └────────────────┬────────────────────────┘
                         │
        ┌────────────────▼────────────────────────┐
        │         REWARD (6 components)            │
        │                                          │
        │  In-range bonus ──────────────── +1.0    │
        │  Hypo penalty ────────── -1.0 to -3.0   │
        │  Hyper penalty ──────── -0.5 to -1.5    │
        │  Overdose penalty ──────────── -3.0      │
        │  Recovery bonus ─────── +0.3 to +0.5    │
        │  Step total ──────────── sum of above    │
        └─────────────────────────────────────────┘
```

---

## Four Tasks — Escalating Real-World Difficulty

| Task | Name | Difficulty | Patient | Meals | Exercise | Illness |
|:----:|------|:----------:|---------|:-----:|:--------:|:-------:|
| 1 | Basal Rate Control | 🟢 Easy | adult#001 | None | None | None |
| 2 | Meal Bolus Timing | 🟡 Medium | adult#001 | 3 announced | Announced | None |
| 3 | Cross-Patient Generalisation | 🔴 Hard | Random/30 | 3 unannounced | Random | None |
| 4 | Sick Day Management | ⚫ Expert | Random/30 | 3 unannounced | Random | 1.5–2.5× resistance |

**Task 1** establishes baseline control — keep glucose stable with basal insulin only.

**Task 2** introduces meal management. Three daily meals (50g/70g/80g CHO) are announced 30 minutes ahead. The agent must learn pre-meal bolus timing. A moderate exercise event at step 150 (also announced) tests exercise-aware dosing.

**Task 3** tests generalisation. A random patient from 30 profiles — children who are 5× more sensitive, adults, adolescents. Meals and exercise are unannounced. Patient identity is hidden. The agent must infer physiology from glucose dynamics alone.

**Task 4** is genuinely frontier-level. A random patient develops illness causing 1.5–2.5× insulin resistance at an unknown time. The agent is never told. It must detect rising glucose despite normal dosing, infer that insulin has become less effective, and increase delivery without over-correcting. PID controllers fail catastrophically on this task — an RL agent that succeeds here would represent a clinically meaningful advance.

---

## Baseline Scores

All results deterministic (seed=42), reproducible via `python eval.py`:

| Agent | Task 1 | Task 2 | Task 3 | Task 4 |
|-------|:------:|:------:|:------:|:------:|
| Constant Basal (no intelligence) | 1.000 | 0.000 | 0.345 | ~0.050 |
| PID Controller (clinical standard) | 1.000 | 0.736 | 0.206 | ~0.120 |
| **Target: Good RL Agent** | **≥ 0.95** | **≥ 0.70** | **≥ 0.60** | **≥ 0.45** |

**Key insight from Task 3**: The PID controller scores *worse* than constant basal (0.206 vs 0.345) because its adult-tuned aggressive corrections cause fatal hypoglycemia in 4 of 5 child/adolescent patients. A "smarter" fixed controller is more dangerous than a conservative one when patient physiology varies. This is exactly why adaptive RL agents are needed.

---

## Observation Space (19 fields)

| Field | Type | Description |
|-------|------|-------------|
| `glucose_mg_dl` | float | CGM reading with ISO 15197 noise (σ=10 mg/dL) |
| `glucose_trend` | string | `rapidly_falling` / `falling` / `stable` / `rising` / `rapidly_rising` |
| `glucose_history_window` | list[float] | Last 12 CGM readings (36 min context) |
| `meal_announced` | bool | Meal within 30 min (Task 2 only) |
| `meal_grams_announced` | float | Carbs in announced meal |
| `exercise_intensity` | float | Current exercise (0=rest, 1=max) |
| `exercise_announced` | bool | Exercise within 30 min (Task 2 only) |
| `insulin_on_board_units` | float | Active insulin via gamma-CDF PK model |
| `time_of_day_hours` | float | Simulated time (0.0–24.0) |
| `step` | int | Current step (0–479) |
| `patient_id` | string/null | Hidden in Task 3/4 |
| `last_action_basal` | float | Previous basal rate |
| `last_action_bolus` | float | Previous bolus dose |
| `true_glucose_mg_dl` | float/null | Pre-noise glucose (research/debug) |
| `illness_active` | bool | Debug only — always False in normal mode |

---

## Reward Function (6 components)

| Glucose Zone | Component | Value | Rationale |
|-------------|-----------|:-----:|-----------|
| 70–180 mg/dL | TIR contribution | +1.0 | Target range |
| 54–70 mg/dL | Hypo penalty | −1.0 | Dangerous |
| < 54 mg/dL | Severe hypo | −3.0 | Life-threatening |
| 180–250 mg/dL | Hyper penalty | −0.5 | Long-term damage |
| > 250 mg/dL | Severe hyper | −1.5 | Acute risk |
| < 54 + recent bolus | Overdose | −3.0 | Prevents reward hacking |
| Hypo corrected ≤10 steps | Recovery bonus | +0.5 | Rewards active correction |
| Hyper corrected ≤10 steps | Recovery bonus | +0.3 | Rewards proactive management |

---

## Insulin-on-Board: Gamma-CDF Pharmacokinetic Model

Unlike simple exponential decay models, OASIS uses a **gamma-distribution cumulative absorption curve** matching the pharmacokinetic profile of rapid-acting insulin (Lispro/Aspart/Fiasp):

```
IOB(t) = Σ insulin_dose[i] × (1 − Fγ(t − t_injection[i]))
```

Where Fγ is the gamma CDF with shape k=2, peak at 55 minutes. The model tracks 160 steps (8 hours) of insulin delivery history — both basal and bolus — and computes the fraction NOT YET absorbed at each time offset. This produces realistic IOB curves: a 10U bolus shows 10.0U immediately, 6.3U at 90 minutes, 2.4U at 4 hours.

Commercial artificial pancreas systems (Medtronic 780G, Tandem Control-IQ, Omnipod 5) display IOB as a primary safety signal to prevent bolus stacking. OASIS gives RL agents the same information.

---

## Quick Start

### Local Development

```bash
git clone https://github.com/saksham1771/glucorl.git
cd glucorl
pip install -r requirements.txt
uvicorn server.app:app --port 8000

# Open the interactive dashboard
open http://localhost:8000
```

### Docker

```bash
docker build -t oasis .
docker run -p 8000:8000 oasis
```

### Python Client

```python
from client import GlucoEnv
from models import GlucoAction

with GlucoEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id=2)
    while not result.done:
        obs = result.observation
        action = GlucoAction(
            basal_rate=1.2,
            bolus_dose=5.0 if obs.meal_announced else 0.0
        )
        result = env.step(action)
    state = env.state()
    print(f"TIR: {state.tir_current:.1%}")
```

### Run Inference

```bash
export GLUCORL_ENV_URL="http://localhost:8000"
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_your_token"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
python inference.py
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive web dashboard (WebSocket-based, real-time) |
| POST | `/reset` | Start episode. Body: `{"task_id": 1}` (1–4) |
| POST | `/step` | Take action. Body: `{"basal_rate": 1.0, "bolus_dose": 0.0}` |
| GET | `/state` | Full episode state with glucose history and metrics |
| GET | `/tasks` | List all 4 tasks with descriptions |
| POST | `/grade` | Detailed decomposed score breakdown |
| GET | `/health` | Health check |
| WS | `/ws` | WebSocket for persistent sessions |
| GET | `/docs` | Swagger API documentation |

---

## Training with RL

OASIS is designed for **GRPO training via TRL**:

- **Dense reward**: every step produces signal (+1.0 to −6.0 range)
- **Continuous action space**: 2D (basal + bolus) — amenable to policy gradient methods
- **480-step episodes**: long enough for meaningful trajectories, short enough for fast iteration
- **4-task curriculum**: natural difficulty progression for progressive training
- **Reward variance**: successful episodes score +300 to +480, failed episodes −200 to −500 — GRPO needs this spread

The `glucose_history_window` (12 readings) enables feedforward agents to reason temporally without RNN architectures. Full history is available via `/state` for agents that prefer complete episode context.

---

## Project Structure

```
oasis/
├── inference.py                  # Baseline inference (OpenAI client)
├── models.py                     # Pydantic: Action, Observation, State, Reward
├── client.py                     # WebSocket client (EnvClient)
├── eval.py                       # PID vs baseline benchmark
├── openenv.yaml                  # OpenEnv spec (4 tasks)
├── Dockerfile                    # HF Spaces (openenv-base)
├── server/
│   ├── app.py                    # FastAPI + interactive dashboard + /grade
│   ├── glucorl_environment.py    # Core: reset/step/state with all 8 enhancements
│   ├── patient_manager.py        # simglucose wrapper + CGM noise + exercise
│   ├── reward_calculator.py      # Shaped reward + recovery bonus
│   ├── graders.py                # 4 task graders + grade_detailed()
│   ├── pid_controller.py         # PID baseline with anti-windup
│   └── constants.py              # Thresholds, PK/PD, meals, exercise, illness
└── tests/                        # 120 tests (environment, graders, reward)
```

---

## Acknowledgements

- **simglucose** — FDA-accepted UVa/Padova T1D simulator by [Jinyu Xie](https://github.com/jxx123/simglucose)
- **OpenEnv** — Open environment specification by [Meta PyTorch](https://github.com/meta-pytorch/OpenEnv)
- **UVa/Padova Model** — Kovatchev et al., *Journal of Diabetes Science and Technology*, 2009
- **Insulin PK/PD** — Gamma-CDF absorption model based on Hovorka et al., 2004