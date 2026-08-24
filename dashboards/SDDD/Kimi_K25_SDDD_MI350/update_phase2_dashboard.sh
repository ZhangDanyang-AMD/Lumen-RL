#!/bin/bash
set -euo pipefail
python3 << 'PYEOF'
import sys, re, json, os, subprocess, math
from datetime import datetime

container = "kimi_k25_eagle3_v2_phase2_atom"
dir_path = "/home/danyzhan/Lumen-RL/dashboards/SDDD/Kimi_K25_SDDD_MI350"
dashboard = os.path.join(dir_path, "phase2.html")
data_file = os.path.join(dir_path, "phase2_data.json")
total_steps = 22609

# --- Build data from scratch each run (all sources merged) ---
existing = {}
fields = ["steps","grad_norms","losses","lrs",
          "step_0_acc","step_1_acc","step_2_acc","step_3_acc",
          "step_0_loss","step_1_loss","step_2_loss","step_3_loss",
          "step_times","teacher_times","train_times"]
eval_fields = ["eval_steps","eval_losses","eval_acc_lens",
               "eval_step_0_acc","eval_step_1_acc","eval_step_2_acc","eval_step_3_acc",
               "eval_step_0_loss","eval_step_1_loss","eval_step_2_loss","eval_step_3_loss"]
for f in fields + eval_fields:
    existing[f] = []

# --- Parse logs from persistent file + docker logs ---
persistent_log = os.path.join(dir_path, "logs", "phase2_atom_training.log")
container_running = True
try:
    tmp = os.path.join(dir_path, ".step_logs.tmp")
    tmp_eval = os.path.join(dir_path, ".eval_logs.tmp")

    # Collect training step lines from all sources
    all_step_lines = {}
    all_eval_lines = {}

    # Source 1: persistent log files (contains historical + tailed data)
    persistent_eval_log = os.path.join(dir_path, "logs", "phase2_atom_eval.log")
    if os.path.exists(persistent_log):
        with open(persistent_log) as f:
            for line in f:
                if "callbacks: step=" in line and "eval step=" not in line:
                    m = re.search(r'step=(\d+)', line)
                    if m:
                        all_step_lines[int(m.group(1))] = line
    if os.path.exists(persistent_eval_log):
        with open(persistent_eval_log) as f:
            for line in f:
                if "callbacks: eval step=" in line:
                    m = re.search(r'eval step=(\d+)', line)
                    if m:
                        all_eval_lines[int(m.group(1))] = line

    # Source 2: docker logs (may have newer data not yet in persistent log)
    try:
        result = subprocess.run(
            ["docker", "logs", container],
            capture_output=True, text=True, timeout=30
        )
        docker_output = result.stdout + result.stderr
        for line in docker_output.splitlines():
            if "callbacks: step=" in line and "eval step=" not in line:
                m = re.search(r'step=(\d+)', line)
                if m:
                    all_step_lines[int(m.group(1))] = line
            elif "callbacks: eval step=" in line:
                m = re.search(r'eval step=(\d+)', line)
                if m:
                    all_eval_lines[int(m.group(1))] = line
    except Exception:
        pass

    # Source 3: docker log files via sudo (current + rotated)
    try:
        log_path_result = subprocess.run(
            ["docker", "inspect", container, "--format", "{{.LogPath}}"],
            capture_output=True, text=True, timeout=10
        )
        log_path = log_path_result.stdout.strip()
        if log_path:
            log_files = [log_path]
            for i in range(1, 5):
                rotated = f"{log_path}.{i}"
                if os.path.exists(rotated):
                    log_files.append(rotated)
            for lf in log_files:
                try:
                    result = subprocess.run(
                        ["sudo", "grep", "-F", "callbacks:", lf],
                        capture_output=True, text=True, timeout=60
                    )
                    for line in result.stdout.splitlines():
                        try:
                            parsed = json.loads(line).get("log", line)
                        except Exception:
                            parsed = line
                        if "eval step=" not in parsed and "callbacks: step=" in parsed:
                            m = re.search(r'step=(\d+)', parsed)
                            if m:
                                all_step_lines[int(m.group(1))] = parsed
                        elif "callbacks: eval step=" in parsed:
                            m = re.search(r'eval step=(\d+)', parsed)
                            if m:
                                all_eval_lines[int(m.group(1))] = parsed
                except Exception:
                    pass
    except Exception:
        pass

    # Combine sorted by step
    raw = "\n".join(all_step_lines[k] for k in sorted(all_step_lines))
    raw_eval = "\n".join(all_eval_lines[k] for k in sorted(all_eval_lines))

except Exception:
    raw = ""
    raw_eval = ""
try:
    ps = subprocess.run(["docker","ps","--filter",f"name={container}","--format","{{.Names}}"],
                        capture_output=True, text=True, timeout=10)
    container_running = container.strip() in ps.stdout.strip()
except Exception:
    container_running = False

# --- Parse training steps ---
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
    if m.group(3) == 'nan':
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

# --- Parse eval steps ---
eval_pattern = re.compile(
    r'eval step=(\d+)\s+eval/loss=([^\s]+)\s+eval/simulated_acc_len=([^\s]+)\s+'
    r'eval/step_0_acc=([^\s]+)\s+eval/step_0_loss=([^\s]+)\s+'
    r'eval/step_1_acc=([^\s]+)\s+eval/step_1_loss=([^\s]+)\s+'
    r'eval/step_2_acc=([^\s]+)\s+eval/step_2_loss=([^\s]+)\s+'
    r'eval/step_3_acc=([^\s]+)\s+eval/step_3_loss=([^\s]+)')

eval_new = 0
for m in eval_pattern.finditer(raw_eval):
    step = int(m.group(1))
    accs = [float(m.group(4)), float(m.group(6)), float(m.group(8)), float(m.group(10))]
    # Corrected simulated accept length using cumulative product
    cum_prod = 1.0
    corrected_acc_len = 1.0
    for a in accs:
        cum_prod *= a
        corrected_acc_len += cum_prod

    existing["eval_steps"].append(step)
    existing["eval_losses"].append(float(m.group(2)))
    existing["eval_acc_lens"].append(corrected_acc_len)
    existing["eval_step_0_acc"].append(accs[0])
    existing["eval_step_0_loss"].append(float(m.group(5)))
    existing["eval_step_1_acc"].append(accs[1])
    existing["eval_step_1_loss"].append(float(m.group(7)))
    existing["eval_step_2_acc"].append(accs[2])
    existing["eval_step_2_loss"].append(float(m.group(9)))
    existing["eval_step_3_acc"].append(accs[3])
    existing["eval_step_3_loss"].append(float(m.group(11)))
    eval_new += 1

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

# Eval data
e_steps=existing["eval_steps"]; e_losses=existing["eval_losses"]
e_acc_lens=existing["eval_acc_lens"]
e0a=existing["eval_step_0_acc"]; e1a=existing["eval_step_1_acc"]
e2a=existing["eval_step_2_acc"]; e3a=existing["eval_step_3_acc"]
e0l=existing["eval_step_0_loss"]; e1l=existing["eval_step_1_loss"]
e2l=existing["eval_step_2_loss"]; e3l=existing["eval_step_3_loss"]
ne = len(e_steps)

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

# Train-based accept length from MA'd accuracies
train_acc_len = []
for i in range(n):
    al_val = 1.0
    cp = 1.0
    for a_arr in [ma0, ma1, ma2, ma3]:
        cp *= a_arr[i]
        al_val += cp
    train_acc_len.append(al_val)

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
ptal=S(train_acc_len)

# --- Stats ---
cur = steps[-1]; pct = cur/total_steps*100
w = min(1000, n)
al = sum(losses[-w:])/w
w_eta = min(250, n)
ast = sum(step_times[-w_eta:])/w_eta
eta = (total_steps-cur)*ast/3600
es = cur*ast; eh=int(es//3600); em=int((es%3600)//60)
aa0=sum(s0a[-w:])/w*100; aa1=sum(s1a[-w:])/w*100
aa2=sum(s2a[-w:])/w*100; aa3=sum(s3a[-w:])/w*100
status = "Training" if container_running else ("Completed" if cur>=total_steps-100 else "Stopped")
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# Latest eval accept length
latest_acc_len = e_acc_lens[-1] if e_acc_lens else 0.0
latest_eval_loss = e_losses[-1] if e_losses else 0.0
latest_train_acc_len = train_acc_len[-1] if train_acc_len else 0.0

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kimi K2.5 Eagle3 SDDD v2 - Phase 2 (ATOM)</title>
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
<h1>Kimi K2.5 Eagle3 SDDD v2 - Phase 2 (ATOM)</h1>
<p class="sub">mixed (VL+CN+tool+agent+writing) | lr=5e-5 | bs=8 | spec_length=4 | 8x MI350 (FSDP2 + ATOM MXFP4) | RoPE=50000+YaRN | loss_mask=auto | VLM parse</p>
<p class="st st-{status.lower()}">{status}</p>
<div class="stats">
<div class="s"><div class="sv">{cur:,} / {total_steps:,}</div><div class="sl">Step ({pct:.1f}%)</div></div>
<div class="s"><div class="sv">{eh}h {em}m</div><div class="sl">Elapsed</div></div>
<div class="s"><div class="sv">{al:.4f}</div><div class="sl">Avg Loss (last 1K)</div></div>
<div class="s"><div class="sv">{aa0:.1f}% / {aa1:.1f}% / {aa2:.1f}% / {aa3:.1f}%</div><div class="sl">Acc pos 0/1/2/3</div></div>
<div class="s"><div class="sv">{ast*1000:.0f} ms</div><div class="sl">Step Time</div></div>
<div class="s"><div class="sv">{eta:.1f} h</div><div class="sl">ETA</div></div>
<div class="s"><div class="sv">{latest_train_acc_len:.4f} / {latest_acc_len:.4f}</div><div class="sl">Accept Len (Train / Eval)</div></div>
<div class="s"><div class="sv">{latest_eval_loss:.4f}</div><div class="sl">Eval Loss</div></div>
</div></div>
<div class="charts">
<div class="ch" id="c1"></div>
<div class="ch" id="c2"></div>
<div class="ch" id="c7"></div>
<div class="ch" id="c8"></div>
<div class="ch" id="c3"></div>
<div class="ch" id="c4"></div>
<div class="ch" id="c5"></div>
<div class="ch" id="c9"></div>
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

// Train accept length (from MA'd train accuracy)
var trainAccLen={json.dumps(ptal)};

// Eval data
var es={json.dumps(e_steps)};
var eloss={json.dumps(e_losses)};
var eacclen={json.dumps(e_acc_lens)};
var ea0={json.dumps(e0a)},ea1={json.dumps(e1a)},ea2={json.dumps(e2a)},ea3={json.dumps(e3a)};
var el0={json.dumps(e0l)},el1={json.dumps(e1l)},el2={json.dumps(e2l)},el3={json.dumps(e3l)};

var C=['#58a6ff','#3fb950','#d29922','#f778ba'];

function L(id,traces,title,extra){{
  var layout=JSON.parse(JSON.stringify(dark));
  layout.title={{text:title,font:{{size:13,color:'#c9d1d9'}}}};
  if(extra){{for(var k in extra){{if(k==='yaxis')Object.assign(layout.yaxis,extra[k]);else layout[k]=extra[k];}}}};
  Plotly.newPlot(id,traces,layout,{{responsive:true}});
}}
function raw(x,y,c){{return {{x:x,y:y,mode:'lines',line:{{color:c,width:1}},opacity:0.35,showlegend:false,hoverinfo:'skip'}};}}
function evaltr(x,y,name,c){{return {{x:x,y:y,mode:'lines+markers',name:name,line:{{color:c,width:2}},marker:{{size:5,color:c}}}};}}

L('c1',[
  raw(s,rloss,'#f85149'),
  {{x:s,y:loss,mode:'lines',name:'Train Loss (MA)',line:{{color:'#f85149',width:2}}}},
  evaltr(es,eloss,'Eval Loss','#58a6ff'),
],'Training & Eval Loss');

L('c2',[
  raw(s,ra0.map(v=>v*100),C[0]),raw(s,ra1.map(v=>v*100),C[1]),raw(s,ra2.map(v=>v*100),C[2]),raw(s,ra3.map(v=>v*100),C[3]),
  {{x:s,y:a0.map(v=>v*100),mode:'lines',name:'Pos 0',line:{{color:C[0],width:2}}}},
  {{x:s,y:a1.map(v=>v*100),mode:'lines',name:'Pos 1',line:{{color:C[1],width:2}}}},
  {{x:s,y:a2.map(v=>v*100),mode:'lines',name:'Pos 2',line:{{color:C[2],width:2}}}},
  {{x:s,y:a3.map(v=>v*100),mode:'lines',name:'Pos 3',line:{{color:C[3],width:2}}}},
],'Train Accuracy by Position (%)',{{yaxis:{{range:[0,105],title:'%'}},legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

L('c7',[
  {{x:s,y:trainAccLen,mode:'lines',name:'Train (MA)',line:{{color:'#d29922',width:2}}}},
  evaltr(es,eacclen,'Eval','#3fb950'),
],'Accept Length (Train vs Eval)',{{yaxis:{{title:'Accept Length'}},legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

L('c8',[
  evaltr(es,ea0.map(v=>v*100),'Pos 0',C[0]),
  evaltr(es,ea1.map(v=>v*100),'Pos 1',C[1]),
  evaltr(es,ea2.map(v=>v*100),'Pos 2',C[2]),
  evaltr(es,ea3.map(v=>v*100),'Pos 3',C[3]),
],'Eval Accuracy by Position (%)',{{yaxis:{{range:[0,105],title:'%'}},legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

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
],'Train Loss by Position',{{legend:{{x:0.01,y:0.01,xanchor:'left',yanchor:'bottom'}}}});

L('c9',[
  evaltr(es,el0,'Pos 0',C[0]),
  evaltr(es,el1,'Pos 1',C[1]),
  evaltr(es,el2,'Pos 2',C[2]),
  evaltr(es,el3,'Pos 3',C[3]),
],'Eval Loss by Position',{{legend:{{x:0.99,y:0.99,xanchor:'right',yanchor:'top'}}}});

L('c6',[{{x:s,y:lr,mode:'lines',name:'LR',line:{{color:'#a5d6ff',width:2}}}}],
  'Learning Rate');
</script>
<p class="up">Updated: {now} | {n:,} train + {ne} eval points</p>
</body></html>"""

with open(dashboard, 'w') as f:
    f.write(html)
print(f"Dashboard updated: {n} train ({new_count} new) + {ne} eval ({eval_new} new), step {cur}, acc_len={latest_acc_len:.4f}, status: {status}")
PYEOF
