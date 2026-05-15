#!/bin/bash
set -euo pipefail
python3 << 'PYEOF'
import sys, re, json, os, subprocess, math
from datetime import datetime

container = "kimi_k25_eagle3_v2_phase1"
dir_path = "/home/danyzhan/Lumen-RL/dashboards/SDDD/Kimi_K25_SDDD_MI350"
dashboard = os.path.join(dir_path, "phase1.html")
data_file = os.path.join(dir_path, "phase1_data.json")
total_steps = 111012

# --- Load existing ---
existing = {}
max_existing_step = -1
if os.path.exists(data_file):
    try:
        with open(data_file) as f:
            existing = json.load(f)
        if existing.get("steps"):
            max_existing_step = max(existing["steps"])
    except Exception:
        existing = {}

fields = ["steps","grad_norms","losses","lrs",
          "step_0_acc","step_1_acc","step_2_acc","step_3_acc",
          "step_0_loss","step_1_loss","step_2_loss","step_3_loss",
          "step_times","teacher_times","train_times"]
for f in fields:
    if f not in existing:
        existing[f] = []

# --- Parse docker logs ---
# Grep the JSON log file directly (much faster than `docker logs` pipe
# for containers with massive Mooncake C++ log spam — 71GB+ log files).
container_running = True
try:
    tmp = os.path.join(dir_path, ".step_logs.tmp")
    log_path_result = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.LogPath}}"],
        capture_output=True, text=True, timeout=10
    )
    log_path = log_path_result.stdout.strip()
    if log_path and os.path.exists(log_path):
        os.system(f"sudo grep -F 'callbacks: step=' {log_path} > {tmp} 2>/dev/null")
    elif max_existing_step > 0:
        os.system(f"docker logs --since 40m {container} 2>&1 | grep -F 'callbacks: step=' > {tmp}")
    else:
        os.system(f"docker logs --tail 200000 {container} 2>&1 | grep -F 'callbacks: step=' > {tmp}")
    with open(tmp) as f:
        raw = f.read()
    try:
        os.remove(tmp)
    except Exception:
        pass
except Exception:
    raw = ""
try:
    ps = subprocess.run(["docker","ps","--filter",f"name={container}","--format","{{.Names}}"],
                        capture_output=True, text=True, timeout=10)
    container_running = container.strip() in ps.stdout.strip()
except Exception:
    container_running = False

pattern = re.compile(
    r'step=(\d+)\s+grad_norm=([^\s]+)\s+loss=([^\s]+)\s+lr=([^\s]+)\s+'
    r'seq/max_len=([^\s]+)\s+'
    r'step_0_acc=([^\s]+)\s+step_0_loss=([^\s]+)\s+'
    r'step_1_acc=([^\s]+)\s+step_1_loss=([^\s]+)\s+'
    r'step_2_acc=([^\s]+)\s+step_2_loss=([^\s]+)\s+'
    r'step_3_acc=([^\s]+)\s+step_3_loss=([^\s]+)\s+'
    r'timing/step_s=([^\s]+)\s+timing/teacher_s=([^\s]+)\s+timing/train_s=([^\s]+)')

new_count = 0
for m in pattern.finditer(raw):
    step = int(m.group(1))
    if m.group(3) == 'nan' or step <= max_existing_step:
        continue
    existing["steps"].append(step)
    existing["grad_norms"].append(float(m.group(2)))
    existing["losses"].append(float(m.group(3)))
    existing["lrs"].append(float(m.group(4)))
    existing["step_0_acc"].append(float(m.group(6)))
    existing["step_0_loss"].append(float(m.group(7)))
    existing["step_1_acc"].append(float(m.group(8)))
    existing["step_1_loss"].append(float(m.group(9)))
    existing["step_2_acc"].append(float(m.group(10)))
    existing["step_2_loss"].append(float(m.group(11)))
    existing["step_3_acc"].append(float(m.group(12)))
    existing["step_3_loss"].append(float(m.group(13)))
    existing["step_times"].append(float(m.group(14)))
    existing["teacher_times"].append(float(m.group(15)))
    existing["train_times"].append(float(m.group(16)))
    new_count += 1

if not existing["steps"]:
    print("No valid steps found"); sys.exit(0)

with open(data_file, 'w') as f:
    json.dump(existing, f)

steps=existing["steps"]; losses=existing["losses"]; grad_norms=existing["grad_norms"]
lrs=existing["lrs"]; step_times=existing["step_times"]
s0a=existing["step_0_acc"]; s1a=existing["step_1_acc"]
s2a=existing["step_2_acc"]; s3a=existing["step_3_acc"]
s0l=existing["step_0_loss"]; s1l=existing["step_1_loss"]
s2l=existing["step_2_loss"]; s3l=existing["step_3_loss"]
n = len(steps)

# --- Moving average ---
def ma(arr, w=200):
    out, s = [], 0.0
    for i, v in enumerate(arr):
        s += v
        if i >= w: s -= arr[i-w]
        out.append(s / min(i+1, w))
    return out

teacher_times=existing["teacher_times"]; train_times=existing["train_times"]
ml=ma(losses); mg=ma(grad_norms); mst=ma(step_times)
mtt=ma(teacher_times); mtr=ma(train_times)
ma0=ma(s0a); ma1=ma(s1a); ma2=ma(s2a); ma3=ma(s3a)
ml0=ma(s0l); ml1=ma(s1l); ml2=ma(s2l); ml3=ma(s3l)

# --- Subsample (max 2000) ---
stride = max(1, n // 2000)
idx = list(range(0, n, stride))
if idx[-1] != n-1: idx.append(n-1)
S = lambda a: [a[i] for i in idx]

ps=S(steps)
# raw
rl=S(losses); rg=S(grad_norms); rst=S(step_times)
rteach=S(teacher_times); rtrain=S(train_times)
ra0=S(s0a); ra1=S(s1a); ra2=S(s2a); ra3=S(s3a)
rl0=S(s0l); rl1=S(s1l); rl2=S(s2l); rl3=S(s3l)
# MA
pl=S(ml); pg=S(mg); pt=S(mst); plr=S(lrs)
ptt=S(mtt); ptr=S(mtr)
pa0=S(ma0); pa1=S(ma1); pa2=S(ma2); pa3=S(ma3)
pl0=S(ml0); pl1=S(ml1); pl2=S(ml2); pl3=S(ml3)

# --- Stats ---
cur = steps[-1]; pct = cur/total_steps*100
w = min(1000, n)
al = sum(losses[-w:])/w; ast = sum(step_times[-w:])/w
eta = (total_steps-cur)*ast/3600
es = cur*ast; eh=int(es//3600); em=int((es%3600)//60)
aa0=sum(s0a[-w:])/w*100; aa1=sum(s1a[-w:])/w*100
aa2=sum(s2a[-w:])/w*100; aa3=sum(s3a[-w:])/w*100
status = "Training" if container_running else ("Completed" if cur>=total_steps-100 else "Stopped")
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kimi K2.5 Eagle3 SDDD v2 - Phase 1</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:16px}}
.header{{text-align:center;padding:16px 0}}
h1{{color:#58a6ff;margin:0 0 4px 0;font-size:22px;font-weight:600}}
.sub{{color:#8b949e;font-size:13px;margin:0}}
.st{{font-size:14px;margin:6px 0;font-weight:600}}
.st-training{{color:#3fb950}}.st-completed{{color:#58a6ff}}.st-stopped{{color:#f85149}}
.stats{{display:flex;justify-content:center;gap:16px;margin:14px 0;flex-wrap:wrap}}
.s{{background:#161b22;border:1px solid #21262d;padding:10px 18px;border-radius:6px;text-align:center}}
.sv{{font-size:20px;font-weight:600;color:#58a6ff}}
.sl{{font-size:11px;color:#8b949e;margin-top:3px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:1600px;margin:0 auto}}
.ch{{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:8px;height:380px}}
@media(max-width:1100px){{.charts{{grid-template-columns:1fr}}}}
.up{{text-align:center;color:#8b949e;font-size:11px;margin-top:12px}}
</style>
</head><body>
<div class="header">
<h1>Kimi K2.5 Eagle3 SDDD v2 - Phase 1</h1>
<p class="sub">perfectblend | lr=5e-5 | spec_length=4 | bfloat16 | 8x MI350 (FSDP2 + vLLM)</p>
<p class="st st-{status.lower()}">{status}</p>
<div class="stats">
<div class="s"><div class="sv">{cur:,} / {total_steps:,}</div><div class="sl">Step ({pct:.1f}%)</div></div>
<div class="s"><div class="sv">{eh}h {em}m</div><div class="sl">Elapsed</div></div>
<div class="s"><div class="sv">{al:.4f}</div><div class="sl">Avg Loss (last 1K)</div></div>
<div class="s"><div class="sv">{aa0:.1f}% / {aa1:.1f}% / {aa2:.1f}% / {aa3:.1f}%</div><div class="sl">Acc pos 0/1/2/3</div></div>
<div class="s"><div class="sv">{ast*1000:.0f} ms</div><div class="sl">Step Time</div></div>
<div class="s"><div class="sv">{eta:.1f} h</div><div class="sl">ETA</div></div>
</div></div>
<div class="charts">
<div class="ch" id="c1"></div>
<div class="ch" id="c2"></div>
<div class="ch" id="c3"></div>
<div class="ch" id="c4"></div>
<div class="ch" id="c5"></div>
<div class="ch" id="c6"></div>
</div>
<script>
var dark={{paper_bgcolor:'#161b22',plot_bgcolor:'#0d1117',font:{{color:'#c9d1d9',size:11}},
margin:{{l:55,r:20,t:40,b:40}},
legend:{{x:1,y:1,xanchor:'right',orientation:'h',font:{{size:10}},bgcolor:'rgba(0,0,0,0)'}},
xaxis:{{gridcolor:'#21262d',title:'Step'}},yaxis:{{gridcolor:'#21262d'}}}};

var s={json.dumps(ps)};
// raw
var rloss={json.dumps(S(losses))};
var rgrad={json.dumps(S(grad_norms))};
var rst={json.dumps(S(step_times))};
var ra0={json.dumps(S(s0a))},ra1={json.dumps(S(s1a))},ra2={json.dumps(S(s2a))},ra3={json.dumps(S(s3a))};
var rl0={json.dumps(S(s0l))},rl1={json.dumps(S(s1l))},rl2={json.dumps(S(s2l))},rl3={json.dumps(S(s3l))};
// smoothed
var loss={json.dumps(pl)};
var grad={json.dumps(pg)};
var tm={json.dumps(pt)};
var lr={json.dumps(plr)};
var rteach={json.dumps(rteach)};
var rtrain={json.dumps(rtrain)};
var teach={json.dumps(ptt)};
var train={json.dumps(ptr)};
var a0={json.dumps(pa0)},a1={json.dumps(pa1)},a2={json.dumps(pa2)},a3={json.dumps(pa3)};
var l0={json.dumps(pl0)},l1={json.dumps(pl1)},l2={json.dumps(pl2)},l3={json.dumps(pl3)};
var C=['#58a6ff','#3fb950','#d29922','#f778ba'];

function L(id,traces,title,extra){{
  var layout=JSON.parse(JSON.stringify(dark));
  layout.title={{text:title,font:{{size:13,color:'#c9d1d9'}}}};
  if(extra){{for(var k in extra){{if(k==='yaxis')Object.assign(layout.yaxis,extra[k]);else layout[k]=extra[k];}}}};
  Plotly.newPlot(id,traces,layout,{{responsive:true}});
}}
function raw(x,y,c){{return {{x:x,y:y,mode:'lines',line:{{color:c,width:1}},opacity:0.35,showlegend:false,hoverinfo:'skip'}};}}

L('c1',[
  raw(s,rloss,'#f85149'),
  {{x:s,y:loss,mode:'lines',name:'Loss',line:{{color:'#f85149',width:2}}}},
],'Training Loss');

L('c2',[
  raw(s,ra0.map(v=>v*100),C[0]),raw(s,ra1.map(v=>v*100),C[1]),raw(s,ra2.map(v=>v*100),C[2]),raw(s,ra3.map(v=>v*100),C[3]),
  {{x:s,y:a0.map(v=>v*100),mode:'lines',name:'Pos 0',line:{{color:C[0],width:2}}}},
  {{x:s,y:a1.map(v=>v*100),mode:'lines',name:'Pos 1',line:{{color:C[1],width:2}}}},
  {{x:s,y:a2.map(v=>v*100),mode:'lines',name:'Pos 2',line:{{color:C[2],width:2}}}},
  {{x:s,y:a3.map(v=>v*100),mode:'lines',name:'Pos 3',line:{{color:C[3],width:2}}}},
],'Accuracy by Position (%)',{{yaxis:{{range:[0,105],title:'%'}},legend:{{x:0.99,y:0.01,xanchor:'right',yanchor:'bottom'}}}});

L('c3',[
  raw(s,rgrad,'#d29922'),
  {{x:s,y:grad,mode:'lines',name:'Grad Norm',line:{{color:'#d29922',width:2}}}},
],'Gradient Norm');

L('c4',[
  raw(s,rst.map(v=>v*1000),'#8b949e'),
  raw(s,rteach.map(v=>v*1000),'#58a6ff'),
  raw(s,rtrain.map(v=>v*1000),'#3fb950'),
  {{x:s,y:tm.map(v=>v*1000),mode:'lines',name:'Total',line:{{color:'#8b949e',width:2}}}},
  {{x:s,y:teach.map(v=>v*1000),mode:'lines',name:'Teacher',line:{{color:'#58a6ff',width:2}}}},
  {{x:s,y:train.map(v=>v*1000),mode:'lines',name:'Train',line:{{color:'#3fb950',width:2}}}},
],'Step Time (ms)',{{yaxis:{{title:'ms'}},legend:{{x:0.99,y:0.99,xanchor:'right',yanchor:'top'}}}});

L('c5',[
  raw(s,rl0,C[0]),raw(s,rl1,C[1]),raw(s,rl2,C[2]),raw(s,rl3,C[3]),
  {{x:s,y:l0,mode:'lines',name:'Pos 0',line:{{color:C[0],width:2}}}},
  {{x:s,y:l1,mode:'lines',name:'Pos 1',line:{{color:C[1],width:2}}}},
  {{x:s,y:l2,mode:'lines',name:'Pos 2',line:{{color:C[2],width:2}}}},
  {{x:s,y:l3,mode:'lines',name:'Pos 3',line:{{color:C[3],width:2}}}},
],'Loss by Position',{{legend:{{x:0.99,y:0.99,xanchor:'right',yanchor:'top'}}}});

L('c6',[{{x:s,y:lr,mode:'lines',name:'LR',line:{{color:'#a5d6ff',width:2}}}}],
  'Learning Rate');
</script>
<p class="up">Updated: {now} | {n:,} points</p>
</body></html>"""

with open(dashboard, 'w') as f:
    f.write(html)
print(f"Dashboard updated: {n} total ({new_count} new), step {cur}, status: {status}")
PYEOF
