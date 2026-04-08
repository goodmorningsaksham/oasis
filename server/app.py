"""
FastAPI application for the OASIS Environment.

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

# Disable OpenEnv's broken web interface — we serve our own at /
os.environ["ENABLE_WEB_INTERFACE"] = "false"

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
    env_name="oasis",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# Custom endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["Environment Info"])
async def list_tasks():
    """Return descriptions of all 4 OASIS tasks."""
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
# OASIS Web Interface — served at / for HuggingFace Spaces
# ---------------------------------------------------------------------------

APP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OASIS — Insulin Dosing RL Environment</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0f1e;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#0f172a 0%,#1a1f3a 100%);padding:18px 28px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e293b}
.header h1{font-size:1.5rem;color:#38bdf8;letter-spacing:0.5px}
.header h1 span{color:#94a3b8;font-weight:400;font-size:0.85rem;margin-left:8px}
.badges{display:flex;gap:8px;align-items:center}
.badge{padding:4px 12px;border-radius:20px;font-size:0.7rem;font-weight:700;letter-spacing:1px}
.badge-ver{background:#1e3a5f;color:#38bdf8}
.badge-live{background:#064e3b;color:#34d399}
.badge-dot{width:7px;height:7px;background:#34d399;border-radius:50%;display:inline-block;margin-right:4px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.container{max-width:1400px;margin:0 auto;padding:20px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px}
.stat{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:14px 18px}
.stat-label{font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px}
.stat-val{font-size:1.8rem;font-weight:700}
.good{color:#4ade80}.warn{color:#fbbf24}.bad{color:#f87171}.neutral{color:#e2e8f0}.info{color:#38bdf8}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;align-items:center}
.btn{padding:9px 20px;border-radius:8px;border:none;font-size:0.85rem;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:6px;transition:all 0.15s}
.btn-blue{background:#2563eb;color:#fff}.btn-blue:hover{background:#1d4ed8}
.btn-green{background:#059669;color:#fff}.btn-green:hover{background:#047857}
.btn-orange{background:#d97706;color:#fff}.btn-orange:hover{background:#b45309}
.btn-gray{background:#374151;color:#e2e8f0}.btn-gray:hover{background:#4b5563}
.btn-red{background:#dc2626;color:#fff}.btn-red:hover{background:#b91c1c}
.btn:disabled{opacity:0.4;cursor:not-allowed}
select{padding:9px 14px;border-radius:8px;border:1px solid #334155;background:#111827;color:#e2e8f0;font-size:0.85rem}
.ep-status{margin-left:auto;font-size:0.85rem;color:#64748b}
.ep-status b{color:#e2e8f0}
.main-grid{display:grid;grid-template-columns:1fr 320px;gap:16px;margin-bottom:18px}
.panel{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:18px}
.panel-title{font-size:0.95rem;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.chart-wrap{position:relative;height:320px}
.control-group{margin-bottom:16px}
.control-label{font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px}
.control-row{display:flex;align-items:center;gap:10px}
.control-row input[type=range]{flex:1;accent-color:#38bdf8}
.control-val{font-size:0.9rem;font-weight:600;min-width:50px;text-align:right;color:#38bdf8}
.scores-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.score-card{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:14px}
.score-card h3{font-size:0.75rem;color:#64748b;margin-bottom:4px}
.score-card .score-name{font-size:0.85rem;font-weight:600;margin-bottom:8px}
.score-bar{height:6px;background:#1e293b;border-radius:3px;overflow:hidden;margin-bottom:4px}
.score-fill{height:100%;border-radius:3px;transition:width 0.5s}
.score-val{font-size:0.8rem;font-weight:600}
.log{background:#0a0f1e;border:1px solid #1e293b;border-radius:8px;padding:10px;max-height:120px;overflow-y:auto;font-family:'Courier New',monospace;font-size:0.75rem;color:#94a3b8;margin-top:12px}
.log div{padding:1px 0}
@media(max-width:900px){.stats{grid-template-columns:repeat(3,1fr)}.main-grid{grid-template-columns:1fr}.scores-grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="header">
  <h1>🏥 OASIS <span>Optimized Adaptive System for Insulin Scheduling</span></h1>
  <div class="badges">
    <span class="badge badge-ver">OPENENV 1.0</span>
    <span class="badge badge-live"><span class="badge-dot"></span>LIVE</span>
  </div>
</div>
<div class="container">
  <div class="stats">
    <div class="stat"><div class="stat-label">Glucose (mg/dL)</div><div class="stat-val neutral" id="sGluc">—</div></div>
    <div class="stat"><div class="stat-label">True Glucose</div><div class="stat-val neutral" id="sTrue">—</div></div>
    <div class="stat"><div class="stat-label">Time in Range</div><div class="stat-val neutral" id="sTir">—</div></div>
    <div class="stat"><div class="stat-label">Current Step</div><div class="stat-val info" id="sStep">—</div></div>
    <div class="stat"><div class="stat-label">Hypo Events</div><div class="stat-val neutral" id="sHypo">—</div></div>
    <div class="stat"><div class="stat-label">Total Reward</div><div class="stat-val neutral" id="sReward">—</div></div>
  </div>

  <div class="actions">
    <select id="taskSel">
      <option value="1">Task 1 — Basal Rate Control</option>
      <option value="2">Task 2 — Meal Bolus Timing</option>
      <option value="3">Task 3 — Cross-Patient</option>
      <option value="4">Task 4 — Sick Day</option>
    </select>
    <button class="btn btn-blue" onclick="doReset()">🔄 Reset Environment</button>
    <button class="btn btn-green" id="btnRun" onclick="toggleRun()">▶️ Run PID Agent</button>
    <button class="btn btn-orange" onclick="doGrade()">📊 Grade Episode</button>
    <button class="btn btn-gray" onclick="window.open('/docs','_blank')">📋 API Docs</button>
    <div class="ep-status">Episode: <b id="epStatus">Not started</b></div>
  </div>

  <div class="main-grid">
    <div class="panel">
      <div class="panel-title">📈 Glucose Trace</div>
      <div class="chart-wrap"><canvas id="glucChart"></canvas></div>
      <div class="log" id="logBox"><div>Welcome to OASIS. Click Reset to start an episode.</div></div>
    </div>
    <div class="panel">
      <div class="panel-title">⚡ Manual Control</div>
      <div class="control-group">
        <div class="control-label">Basal Rate (U/hr)</div>
        <div class="control-row">
          <input type="range" id="basalSlider" min="0" max="5" step="0.1" value="1.0">
          <div class="control-val" id="basalVal">1.0</div>
        </div>
      </div>
      <div class="control-group">
        <div class="control-label">Bolus Dose (units)</div>
        <div class="control-row">
          <input type="range" id="bolusSlider" min="0" max="20" step="0.5" value="0">
          <div class="control-val" id="bolusVal">0.0</div>
        </div>
      </div>
      <button class="btn btn-blue" style="width:100%;justify-content:center" onclick="doStep()">Step →</button>
      <div style="margin-top:18px">
        <div class="panel-title">📋 Current Observation</div>
        <div id="obsInfo" style="font-size:0.78rem;color:#94a3b8;line-height:1.7">
          Click Reset to begin
        </div>
      </div>
    </div>
  </div>

  <div class="panel-title" style="margin-bottom:10px">🎯 Task Scores</div>
  <div class="scores-grid">
    <div class="score-card"><h3>TASK 1</h3><div class="score-name">Basal Rate Control</div>
      <div class="score-bar"><div class="score-fill" id="s1bar" style="width:0%;background:#4ade80"></div></div>
      <div class="score-val" id="s1val">—</div></div>
    <div class="score-card"><h3>TASK 2</h3><div class="score-name">Meal Bolus Timing</div>
      <div class="score-bar"><div class="score-fill" id="s2bar" style="width:0%;background:#fbbf24"></div></div>
      <div class="score-val" id="s2val">—</div></div>
    <div class="score-card"><h3>TASK 3</h3><div class="score-name">Cross-Patient</div>
      <div class="score-bar"><div class="score-fill" id="s3bar" style="width:0%;background:#f87171"></div></div>
      <div class="score-val" id="s3val">—</div></div>
    <div class="score-card"><h3>TASK 4</h3><div class="score-name">Sick Day</div>
      <div class="score-bar"><div class="score-fill" id="s4bar" style="width:0%;background:#c084fc"></div></div>
      <div class="score-val" id="s4val">—</div></div>
  </div>
</div>

<script>
const ctx = document.getElementById('glucChart').getContext('2d');
const chart = new Chart(ctx, {
  type:'line',
  data:{labels:[],datasets:[
    {label:'CGM Glucose',data:[],borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,0.08)',borderWidth:2,pointRadius:0,tension:0.3,fill:true},
    {label:'True Glucose',data:[],borderColor:'#a78bfa',borderWidth:1,pointRadius:0,tension:0.3,borderDash:[4,3]}
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
    scales:{x:{title:{display:true,text:'Step',color:'#64748b'},ticks:{color:'#475569',maxTicksLimit:20},grid:{color:'#1e293b22'}},
            y:{title:{display:true,text:'mg/dL',color:'#64748b'},min:30,max:400,ticks:{color:'#475569'},grid:{color:'#1e293b44'}}},
    plugins:{legend:{labels:{color:'#94a3b8',usePointStyle:true,pointStyle:'line'}}}},
  plugins:[{id:'zones',beforeDraw(c){
    const{ctx:x,chartArea:{left:l,right:r},scales:{y}}=c;
    function fill(lo,hi,col){x.fillStyle=col;const t=y.getPixelForValue(Math.min(hi,400)),b=y.getPixelForValue(Math.max(lo,30));x.fillRect(l,t,r-l,b-t)}
    fill(0,54,'rgba(239,68,68,0.08)');fill(54,70,'rgba(251,191,36,0.05)');
    fill(70,180,'rgba(74,222,128,0.04)');fill(180,250,'rgba(251,191,36,0.05)');fill(250,500,'rgba(239,68,68,0.06)')}}]
});

let ws=null, running=false, runTimer=null, glucHist=[], trueHist=[], stepN=0, done=true, totalReward=0;
const log=document.getElementById('logBox');

function addLog(msg){const d=document.createElement('div');d.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;log.prepend(d);if(log.children.length>50)log.removeChild(log.lastChild)}
function gClass(g){return g<54?'bad':g<70?'warn':g<=180?'good':g<=250?'warn':'bad'}

document.getElementById('basalSlider').oninput=function(){document.getElementById('basalVal').textContent=parseFloat(this.value).toFixed(1)};
document.getElementById('bolusSlider').oninput=function(){document.getElementById('bolusVal').textContent=parseFloat(this.value).toFixed(1)};

function getWsUrl(){const p=location.protocol==='https:'?'wss:':'ws:';return p+'//'+location.host+'/ws'}

function ensureWs(){
  return new Promise((resolve,reject)=>{
    if(ws&&ws.readyState===WebSocket.OPEN)return resolve(ws);
    if(ws)try{ws.close()}catch(e){}
    ws=new WebSocket(getWsUrl());
    ws.onopen=()=>{addLog('WebSocket connected');resolve(ws)};
    ws.onerror=e=>{addLog('WebSocket error');reject(e)};
    ws.onclose=()=>{addLog('WebSocket closed');ws=null};
  });
}

function wsSend(msg){
  return new Promise(async(resolve,reject)=>{
    try{
      const s=await ensureWs();
      s.onmessage=e=>{
        try{
          const parsed=JSON.parse(e.data);
          resolve(parsed);
        }catch(pe){reject(pe)}
      };
      s.send(JSON.stringify(msg));
    }catch(e){reject(e)}
  });
}

function getObs(r){
  // Handle different response formats from OpenEnv
  if(r&&r.data&&r.data.observation) return r.data.observation;
  if(r&&r.observation) return r.observation;
  if(r&&r.data&&r.data.glucose_mg_dl) return r.data;
  if(r&&r.glucose_mg_dl) return r;
  return null;
}

function getRew(r){
  if(r&&r.data&&r.data.reward!==undefined) return r.data.reward;
  if(r&&r.reward!==undefined) return r.reward;
  return null;
}

function getDone(r){
  if(r&&r.data&&r.data.done!==undefined) return r.data.done;
  if(r&&r.done!==undefined) return r.done;
  return false;
}

async function doReset(){
  stopRun();
  const tid=parseInt(document.getElementById('taskSel').value);
  try{
    const r=await wsSend({type:'reset',data:{task_id:tid}});
    addLog('Response: '+JSON.stringify(r).substring(0,200));
    const obs=getObs(r);
    if(!obs){addLog('Reset: no observation in response');return}
    glucHist=[obs.glucose_mg_dl];trueHist=[obs.true_glucose_mg_dl||obs.glucose_mg_dl];
    stepN=0;done=false;totalReward=0;
    updateChart();updateStats(obs,null);
    document.getElementById('epStatus').textContent='Running (Task '+tid+')';
    addLog('Episode reset — Task '+tid+', Glucose: '+obs.glucose_mg_dl.toFixed(1)+' mg/dL');
  }catch(e){addLog('Reset failed: '+e)}
}

async function doStep(){
  if(done){addLog('Episode is done. Click Reset first.');return}
  const basal=parseFloat(document.getElementById('basalSlider').value);
  const bolus=parseFloat(document.getElementById('bolusSlider').value);
  try{
    const r=await wsSend({type:'step',data:{basal_rate:basal,bolus_dose:bolus}});
    const obs=getObs(r);const rew=getRew(r);
    if(!obs){addLog('Step: no observation in response');return}
    stepN=obs.step;done=getDone(r);
    glucHist.push(obs.glucose_mg_dl);trueHist.push(obs.true_glucose_mg_dl||obs.glucose_mg_dl);
    if(rew!==null)totalReward+=rew;
    updateChart();updateStats(obs,rew);
    if(done){document.getElementById('epStatus').textContent='Done';addLog('Episode complete at step '+stepN)}
  }catch(e){addLog('Step failed: '+e)}
}

function pidAction(g){
  let basal=1.0+(g-120)*0.02;
  basal=Math.max(0,Math.min(5,basal));
  let bolus=g>200?Math.max(0,(g-200)/50):0;
  if(g<90){basal*=0.3;bolus=0}
  if(g<70){basal=0;bolus=0}
  return{basal:Math.round(basal*100)/100,bolus:Math.round(bolus*100)/100};
}

function toggleRun(){
  if(running){stopRun();return}
  if(done){addLog('Reset first before running agent.');return}
  running=true;
  document.getElementById('btnRun').textContent='⏹️ Stop Agent';
  document.getElementById('btnRun').className='btn btn-red';
  runStep();
}
function stopRun(){
  running=false;if(runTimer)clearTimeout(runTimer);runTimer=null;
  document.getElementById('btnRun').textContent='▶️ Run PID Agent';
  document.getElementById('btnRun').className='btn btn-green';
}
async function runStep(){
  if(!running||done){stopRun();return}
  const g=glucHist[glucHist.length-1];
  const a=pidAction(g);
  document.getElementById('basalSlider').value=a.basal;document.getElementById('basalVal').textContent=a.basal.toFixed(1);
  document.getElementById('bolusSlider').value=a.bolus;document.getElementById('bolusVal').textContent=a.bolus.toFixed(1);
  try{
    const r=await wsSend({type:'step',data:{basal_rate:a.basal,bolus_dose:a.bolus}});
    const obs=getObs(r);const rew=getRew(r);
    if(!obs){addLog('Agent step: no observation');stopRun();return}
    stepN=obs.step;done=getDone(r);
    glucHist.push(obs.glucose_mg_dl);trueHist.push(obs.true_glucose_mg_dl||obs.glucose_mg_dl);
    if(rew!==null)totalReward+=rew;
    updateChart();updateStats(obs,rew);
    if(stepN%48===0)addLog('Step '+stepN+' | Glucose: '+obs.glucose_mg_dl.toFixed(1)+' | Basal: '+a.basal+' | Bolus: '+a.bolus);
    if(done){addLog('Episode complete — step '+stepN);stopRun();document.getElementById('epStatus').textContent='Done'}
    else{runTimer=setTimeout(runStep,20)}
  }catch(e){addLog('Agent step failed: '+e);stopRun()}
}

async function doGrade(){
  try{
    const r=await wsSend({type:'state'});
    const s=r.data||r;
    if(!s.glucose_history||s.glucose_history.length<2){addLog('No episode data to grade.');return}
    // Fetch grade from HTTP endpoint for each task (uses state from WS session won't work, so compute client-side)
    const tir=s.tir_current;
    const score=Math.max(0,Math.min(1,tir+(s.severe_hypo_events===0?0.05:0)-s.severe_hypo_events*0.1));
    const tid=s.task_id;
    const el=document.getElementById('s'+tid+'val');
    const bar=document.getElementById('s'+tid+'bar');
    if(el){el.textContent=score.toFixed(3);bar.style.width=(score*100)+'%'}
    addLog('Task '+tid+' graded: TIR='+((tir*100).toFixed(1))+'% Score='+score.toFixed(3));
  }catch(e){addLog('Grade failed: '+e)}
}

function updateChart(){
  chart.data.labels=glucHist.map((_,i)=>i);
  chart.data.datasets[0].data=[...glucHist];
  chart.data.datasets[1].data=[...trueHist];
  chart.update();
}

function updateStats(obs,rew){
  const g=obs.glucose_mg_dl;
  const el=document.getElementById('sGluc');el.textContent=g.toFixed(0);el.className='stat-val '+gClass(g);
  const tEl=document.getElementById('sTrue');
  if(obs.true_glucose_mg_dl){tEl.textContent=obs.true_glucose_mg_dl.toFixed(0);tEl.className='stat-val '+gClass(obs.true_glucose_mg_dl)}
  document.getElementById('sStep').textContent=stepN+'/480';
  document.getElementById('sReward').textContent=totalReward.toFixed(1);

  // Compute running TIR from glucHist (skip first)
  if(glucHist.length>1){
    const readings=trueHist.slice(1);
    const inRange=readings.filter(g=>g>=70&&g<=180).length;
    const tir=(inRange/readings.length*100).toFixed(1);
    const tirEl=document.getElementById('sTir');tirEl.textContent=tir+'%';
    tirEl.className='stat-val '+(parseFloat(tir)>=70?'good':parseFloat(tir)>=50?'warn':'bad');
  }
  // Count hypo from trueHist
  const hypoN=trueHist.slice(1).filter(g=>g<70).length;
  const hEl=document.getElementById('sHypo');hEl.textContent=hypoN;hEl.className='stat-val '+(hypoN===0?'good':'bad');

  document.getElementById('obsInfo').innerHTML=
    '<b>Trend:</b> '+obs.glucose_trend+'<br>'+
    '<b>Time:</b> '+(obs.time_of_day_hours||0).toFixed(1)+' hrs<br>'+
    '<b>Meal announced:</b> '+(obs.meal_announced?obs.meal_grams_announced+'g':'No')+'<br>'+
    '<b>Exercise:</b> '+(obs.exercise_intensity>0?((obs.exercise_intensity*100).toFixed(0))+'%':'Rest')+'<br>'+
    '<b>IOB:</b> '+(obs.insulin_on_board_units||0).toFixed(2)+' U<br>'+
    '<b>Patient:</b> '+(obs.patient_id||'Hidden')+'<br>'+
    (rew!==null?'<b>Last reward:</b> '+(rew>=0?'+':'')+rew.toFixed(2):'');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, tags=["UI"],
         include_in_schema=False)
async def root_ui():
    """OASIS web interface — served at root for HuggingFace Spaces."""
    return HTMLResponse(content=APP_HTML)


@app.get("/dashboard", response_class=HTMLResponse, tags=["UI"])
async def dashboard():
    """Alias for the main web interface."""
    return HTMLResponse(content=APP_HTML)


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the OASIS server."""
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