#!/bin/bash
set -euo pipefail
python3 << 'PYEOF'
import sys, re, json, os, subprocess, math
from datetime import datetime

container = "kimi_k3_dspark_v1"
dir_path = "/home/danyzhan/Lumen-RL/dashboards/SDDD/Kimi_K3_SDDD_MI350"
dashboard = os.path.join(dir_path, "dashboard.html")
data_file = os.path.join(dir_path, "data.json")
total_steps = 59630
cache_batches = 200

# --- Load existing ---
existing = {}
max_existing_step = -1
max_existing_eval_step = -1
if os.path.exists(data_file):
    try:
        with open(data_file) as f:
            existing = json.load(f)
        if existing.get("steps"):
            max_existing_step = max(existing["steps"])
        if existing.get("eval_steps"):
            max_existing_eval_step = max(existing["eval_steps"])
    except Exception:
        existing = {}

# 7 positions (block_size=7), plus DSpark-specific loss components
fields = ["steps","grad_norms","losses","lrs",
          "step_0_acc","step_1_acc","step_2_acc","step_3_acc",
          "step_4_acc","step_5_acc","step_6_acc",
          "step_0_loss","step_1_loss","step_2_loss","step_3_loss",
          "step_4_loss","step_5_loss","step_6_loss",
          "ce_losses","tv_losses","conf_losses",
          "step_times"]
eval_fields = ["eval_steps","eval_losses","eval_acc_lens",
               "eval_step_0_acc","eval_step_1_acc","eval_step_2_acc","eval_step_3_acc",
               "eval_step_4_acc","eval_step_5_acc","eval_step_6_acc",
               "eval_step_0_loss","eval_step_1_loss","eval_step_2_loss","eval_step_3_loss",
               "eval_step_4_loss","eval_step_5_loss","eval_step_6_loss"]
for f in fields + eval_fields:
    if f not in existing:
        existing[f] = []

# --- Parse docker logs ---
container_running = True
try:
    tmp = os.path.join(dir_path, ".step_logs.tmp")
    tmp_eval = os.path.join(dir_path, ".eval_logs.tmp")
    log_path_result = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.LogPath}}"],
        capture_output=True, text=True, timeout=10
    )
    log_path = log_path_result.stdout.strip()
    log_exists = log_path and subprocess.run(["sudo", "test", "-f", log_path], capture_output=True, timeout=5).returncode == 0
    if log_exists:
        os.system(f"sudo grep -F 'callbacks: step=' {log_path} > {tmp}.raw 2>/dev/null")
        os.system(f"sudo grep -F 'callbacks: eval step=' {log_path} > {tmp_eval}.raw 2>/dev/null")
        for src, dst in [(f"{tmp}.raw", tmp), (f"{tmp_eval}.raw", tmp_eval)]:
            with open(src) as _rf, open(dst, "w") as _wf:
                for _line in _rf:
                    try:
                        _wf.write(json.loads(_line)["log"])
                    except Exception:
                        _wf.write(_line)
            try:
                os.remove(src)
            except Exception:
                pass
    else:
        os.system(f"docker logs {container} 2>&1 | grep -F 'callbacks: step=' > {tmp}")
        os.system(f"docker logs {container} 2>&1 | grep -F 'callbacks: eval step=' > {tmp_eval}")
    with open(tmp) as f:
        raw = f.read()
    with open(tmp_eval) as f:
        raw_eval = f.read()
    for fp in [tmp, tmp_eval]:
        try:
            os.remove(fp)
        except Exception:
            pass
except Exception:
    raw = ""
    raw_eval = ""
try:
    ps = subprocess.run(["docker","ps","--filter",f"name={container}","--format","{{.Names}}"],
                        capture_output=True, text=True, timeout=10)
    container_running = container.strip() in ps.stdout.strip()
except Exception:
    container_running = False

# --- Parse training steps (7 positions + DSpark losses) ---
pattern = re.compile(
    r'step=(\d+)\s+grad_norm=([^\s]+)\s+loss=([^\s]+)\s+lr=([^\s]+)\s+'
    r'(?:ce_loss=([^\s]+)\s+tv_loss=([^\s]+)\s+conf_loss=([^\s]+)\s+)?'
    r'(?:seq/max_len=([^\s]+)\s+)?'
    r'step_0_acc=([^\s]+)\s+step_0_loss=([^\s]+)\s+'
    r'step_1_acc=([^\s]+)\s+step_1_loss=([^\s]+)\s+'
    r'step_2_acc=([^\s]+)\s+step_2_loss=([^\s]+)\s+'
    r'step_3_acc=([^\s]+)\s+step_3_loss=([^\s]+)\s+'
    r'step_4_acc=([^\s]+)\s+step_4_loss=([^\s]+)\s+'
    r'step_5_acc=([^\s]+)\s+step_5_loss=([^\s]+)\s+'
    r'step_6_acc=([^\s]+)\s+step_6_loss=([^\s]+)\s+'
    r'timing/step_s=([^\s]+)')

new_count = 0
for m in pattern.finditer(raw):
    step = int(m.group(1))
    if m.group(3) == 'nan' or step <= max_existing_step:
        continue
    existing["steps"].append(step)
    existing["grad_norms"].append(float(m.group(2)))
    existing["losses"].append(float(m.group(3)))
    existing["lrs"].append(float(m.group(4)))
    existing["ce_losses"].append(float(m.group(5) or 0))
    existing["tv_losses"].append(float(m.group(6) or 0))
    existing["conf_losses"].append(float(m.group(7) or 0))
    for i in range(7):
        existing[f"step_{i}_acc"].append(float(m.group(9 + i*2)))
        existing[f"step_{i}_loss"].append(float(m.group(10 + i*2)))
    existing["step_times"].append(float(m.group(23)))
    new_count += 1

# --- Parse eval steps ---
eval_pattern = re.compile(
    r'eval step=(\d+)\s+eval/loss=([^\s]+)\s+eval/simulated_acc_len=([^\s]+)\s+'
    r'eval/step_0_acc=([^\s]+)\s+eval/step_0_loss=([^\s]+)\s+'
    r'eval/step_1_acc=([^\s]+)\s+eval/step_1_loss=([^\s]+)\s+'
    r'eval/step_2_acc=([^\s]+)\s+eval/step_2_loss=([^\s]+)\s+'
    r'eval/step_3_acc=([^\s]+)\s+eval/step_3_loss=([^\s]+)\s+'
    r'eval/step_4_acc=([^\s]+)\s+eval/step_4_loss=([^\s]+)\s+'
    r'eval/step_5_acc=([^\s]+)\s+eval/step_5_loss=([^\s]+)\s+'
    r'eval/step_6_acc=([^\s]+)\s+eval/step_6_loss=([^\s]+)')

eval_new = 0
for m in eval_pattern.finditer(raw_eval):
    step = int(m.group(1))
    if step <= max_existing_eval_step:
        continue
    accs = [float(m.group(4+i*2)) for i in range(7)]
    cum_prod = 1.0
    corrected_acc_len = 1.0
    for a in accs:
        cum_prod *= a
        corrected_acc_len += cum_prod

    existing["eval_steps"].append(step)
    existing["eval_losses"].append(float(m.group(2)))
    existing["eval_acc_lens"].append(corrected_acc_len)
    for i in range(7):
        existing[f"eval_step_{i}_acc"].append(float(m.group(4+i*2)))
        existing[f"eval_step_{i}_loss"].append(float(m.group(5+i*2)))
    eval_new += 1

if not existing["steps"]:
    print("No valid steps found"); sys.exit(0)

with open(data_file, 'w') as f:
    json.dump(existing, f)

steps=existing["steps"]; losses=existing["losses"]; grad_norms=existing["grad_norms"]
lrs=existing["lrs"]; step_times=existing["step_times"]
ce_losses=existing["ce_losses"]; tv_losses=existing["tv_losses"]; conf_losses=existing["conf_losses"]
sa = [existing[f"step_{i}_acc"] for i in range(7)]
sl = [existing[f"step_{i}_loss"] for i in range(7)]
n = len(steps)

# Eval data
e_steps=existing["eval_steps"]; e_losses=existing["eval_losses"]
e_acc_lens=existing["eval_acc_lens"]
ea = [existing[f"eval_step_{i}_acc"] for i in range(7)]
el = [existing[f"eval_step_{i}_loss"] for i in range(7)]
ne = len(e_steps)

# --- Moving average ---
def ma(arr, w=200):
    out, s = [], 0.0
    for i, v in enumerate(arr):
        s += v
        if i >= w: s -= arr[i-w]
        out.append(s / min(i+1, w))
    return out

ml=ma(losses); mg=ma(grad_norms); mst=ma(step_times)
mce=ma(ce_losses); mtv=ma(tv_losses); mcf=ma(conf_losses)
msa = [ma(sa[i]) for i in range(7)]

# Train-based accept length from MA'd accuracies (7 positions)
train_acc_len = []
for i in range(n):
    al_val = 1.0
    cp = 1.0
    for j in range(7):
        cp *= msa[j][i]
        al_val += cp
    train_acc_len.append(al_val)

# --- Subsample (max 2000) ---
stride = max(1, n // 2000)
idx = list(range(0, n, stride))
if idx[-1] != n-1: idx.append(n-1)
S = lambda a: [a[i] for i in idx]

ps=S(steps)
rl=S(losses); rg=S(grad_norms); rst=S(step_times)
pl=S(ml); pg=S(mg); pt=S(mst); plr=S(lrs)
pce=S(mce); ptv=S(mtv); pcf=S(mcf)
rce=S(ce_losses); rtv=S(tv_losses); rcf=S(conf_losses)
psa = [S(msa[i]) for i in range(7)]
rsa = [S(sa[i]) for i in range(7)]
psl = [S(ma(sl[i])) for i in range(7)]
rsl = [S(sl[i]) for i in range(7)]
ptal = S(train_acc_len)

# Round boundaries (every cache_batches steps)
round_boundaries = list(range(0, int(steps[-1])+1, cache_batches))

# --- Stats ---
cur = steps[-1]; pct = cur/total_steps*100
w = min(1000, n)
al = sum(losses[-w:])/w
w_eta = min(250, n)
ast = sum(step_times[-w_eta:])/w_eta
eta = (total_steps-cur)*ast/3600
es_time = cur*ast; eh=int(es_time//3600); em=int((es_time%3600)//60)
aa = [sum(sa[i][-w:])/w*100 for i in range(7)]
status = "Training" if container_running else ("Completed" if cur>=total_steps-100 else "Stopped")
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

latest_acc_len = e_acc_lens[-1] if e_acc_lens else 0.0
latest_eval_loss = e_losses[-1] if e_losses else 0.0
latest_train_acc_len = train_acc_len[-1] if train_acc_len else 0.0
current_round = cur // cache_batches

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kimi K3 DSpark SDDD — MI350</title>
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
<h1>Kimi K3 DSpark SDDD — MI350</h1>
<p class="sub">batch-alternating | lr=6e-4 | bs=8 | block_size=7 | cache_batches={cache_batches} | 8x MI350 (Replicate + vLLM TP=8)</p>
<p class="st st-{status.lower()}">{status} (Round {current_round})</p>
<div class="stats">
<div class="s"><div class="sv">{cur:,} / {total_steps:,}</div><div class="sl">Step ({pct:.1f}%)</div></div>
<div class="s"><div class="sv">{eh}h {em}m</div><div class="sl">Elapsed</div></div>
<div class="s"><div class="sv">{al:.4f}</div><div class="sl">Avg Loss (last 1K)</div></div>
<div class="s"><div class="sv">{aa[0]:.1f} / {aa[1]:.1f} / {aa[2]:.1f} / {aa[3]:.1f} / {aa[4]:.1f} / {aa[5]:.1f} / {aa[6]:.1f}</div><div class="sl">Acc pos 0-6 (%)</div></div>
<div class="s"><div class="sv">{ast*1000:.0f} ms</div><div class="sl">Step Time</div></div>
<div class="s"><div class="sv">{eta:.1f} h</div><div class="sl">ETA</div></div>
<div class="s"><div class="sv">{latest_train_acc_len:.4f} / {latest_acc_len:.4f}</div><div class="sl">Accept Len (Train / Eval)</div></div>
<div class="s"><div class="sv">{latest_eval_loss:.4f}</div><div class="sl">Eval Loss</div></div>
</div></div>
<div class="charts">
<div class="ch" id="c1"></div>
<div class="ch" id="c2"></div>
<div class="ch" id="c7"></div>
<div class="ch" id="c10"></div>
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
var rloss={json.dumps(rl)};
var rgrad={json.dumps(rg)};
var rst_t={json.dumps(rst)};
var rce={json.dumps(rce)},rtv={json.dumps(rtv)},rcf={json.dumps(rcf)};
var loss={json.dumps(pl)};
var grad={json.dumps(pg)};
var tm={json.dumps(pt)};
var lr={json.dumps(plr)};
var pce={json.dumps(pce)},ptv={json.dumps(ptv)},pcf={json.dumps(pcf)};
var trainAccLen={json.dumps(ptal)};

// Per-position accuracy (7 positions)
var ra=[{','.join(json.dumps(rsa[i]) for i in range(7))}];
var pa=[{','.join(json.dumps(psa[i]) for i in range(7))}];
var rl_pos=[{','.join(json.dumps(rsl[i]) for i in range(7))}];
var pl_pos=[{','.join(json.dumps(psl[i]) for i in range(7))}];

// Eval data
var es_e={json.dumps(e_steps)};
var eloss={json.dumps(e_losses)};
var eacclen={json.dumps(e_acc_lens)};
var ea=[{','.join(json.dumps(ea[i]) for i in range(7))}];
var el_e=[{','.join(json.dumps(el[i]) for i in range(7))}];

// Round boundaries
var roundBounds={json.dumps(round_boundaries)};

var C=['#58a6ff','#3fb950','#d29922','#f778ba','#a371f7','#ff7b72','#79c0ff'];

function L(id,traces,title,extra){{
  var layout=JSON.parse(JSON.stringify(dark));
  layout.title={{text:title,font:{{size:13,color:'#c9d1d9'}}}};
  // Add round boundary annotations
  layout.shapes=roundBounds.map(function(x){{return {{type:'line',x0:x,x1:x,y0:0,y1:1,yref:'paper',line:{{color:'#30363d',width:1,dash:'dot'}}}}}});
  if(extra){{for(var k in extra){{if(k==='yaxis')Object.assign(layout.yaxis,extra[k]);else layout[k]=extra[k];}}}};
  Plotly.newPlot(id,traces,layout,{{responsive:true}});
}}
function raw(x,y,c){{return {{x:x,y:y,mode:'lines',line:{{color:c,width:1}},opacity:0.35,showlegend:false,hoverinfo:'skip'}};}}
function evaltr(x,y,name,c){{return {{x:x,y:y,mode:'lines+markers',name:name,line:{{color:c,width:2}},marker:{{size:5,color:c}}}};}}

L('c1',[
  raw(s,rloss,'#f85149'),
  {{x:s,y:loss,mode:'lines',name:'Train Loss (MA)',line:{{color:'#f85149',width:2}}}},
  evaltr(es_e,eloss,'Eval Loss','#58a6ff'),
],'Training & Eval Loss');

L('c2',[
  ...Array.from({{length:7}},(_,i)=>raw(s,ra[i].map(v=>v*100),C[i])),
  ...Array.from({{length:7}},(_,i)=>({{x:s,y:pa[i].map(v=>v*100),mode:'lines',name:'Pos '+i,line:{{color:C[i],width:2}}}})),
],'Train Accuracy by Position (%)',{{yaxis:{{range:[0,105],title:'%'}},legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

L('c7',[
  {{x:s,y:trainAccLen,mode:'lines',name:'Train (MA)',line:{{color:'#d29922',width:2}}}},
  evaltr(es_e,eacclen,'Eval','#3fb950'),
],'Accept Length (Train vs Eval)',{{yaxis:{{title:'Accept Length'}},legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

L('c10',[
  raw(s,rce,'#58a6ff'),raw(s,rtv,'#3fb950'),raw(s,rcf,'#d29922'),
  {{x:s,y:pce,mode:'lines',name:'CE Loss',line:{{color:'#58a6ff',width:2}}}},
  {{x:s,y:ptv,mode:'lines',name:'TV Loss',line:{{color:'#3fb950',width:2}}}},
  {{x:s,y:pcf,mode:'lines',name:'Conf Loss',line:{{color:'#d29922',width:2}}}},
],'DSpark Loss Components',{{legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

L('c8',[
  ...Array.from({{length:7}},(_,i)=>evaltr(es_e,ea[i].map(v=>v*100),'Pos '+i,C[i])),
],'Eval Accuracy by Position (%)',{{yaxis:{{range:[0,105],title:'%'}},legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

L('c3',[
  raw(s,rgrad,'#d29922'),
  {{x:s,y:grad,mode:'lines',name:'Grad Norm',line:{{color:'#d29922',width:2}}}},
],'Gradient Norm');

L('c4',[
  raw(s,rst_t.map(v=>v*1000),'#8b949e'),
  {{x:s,y:tm.map(v=>v*1000),mode:'lines',name:'Total',line:{{color:'#8b949e',width:2}}}},
],'Step Time (ms)',{{yaxis:{{title:'ms'}}}});

L('c5',[
  ...Array.from({{length:7}},(_,i)=>raw(s,rl_pos[i],C[i])),
  ...Array.from({{length:7}},(_,i)=>({{x:s,y:pl_pos[i],mode:'lines',name:'Pos '+i,line:{{color:C[i],width:2}}}})),
],'Train Loss by Position',{{legend:{{x:0.01,y:0.01,xanchor:'left',yanchor:'bottom'}}}});

L('c9',[
  ...Array.from({{length:7}},(_,i)=>evaltr(es_e,el_e[i],'Pos '+i,C[i])),
],'Eval Loss by Position',{{legend:{{x:0.99,y:0.99,xanchor:'right',yanchor:'top'}}}});

L('c6',[{{x:s,y:lr,mode:'lines',name:'LR',line:{{color:'#a5d6ff',width:2}}}}],
  'Learning Rate');
</script>
<p class="up">Updated: {now} | {n:,} train + {ne} eval points | Round boundaries every {cache_batches} steps</p>
</body></html>"""

with open(dashboard, 'w') as f:
    f.write(html)
print(f"Dashboard updated: {n} train ({new_count} new) + {ne} eval ({eval_new} new), step {cur}, acc_len={latest_acc_len:.4f}, status: {status}")
PYEOF
