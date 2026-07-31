#!/bin/bash
# Kimi K3 DSpark SDDD dashboard generator.
#
# Parses the training log directly (no docker/sudo needed), so it runs on the
# login node against the NFS-visible log.
#
#   K3_LOG=~/train9k.log bash update_dashboard.sh
#
# Env overrides: K3_LOG, K3_DIR, K3_TOTAL_STEPS, K3_CACHE_BATCHES, K3_MA_WINDOW
# K3_RESET=1 discards accumulated history and reparses from scratch.
set -euo pipefail
export K3_DIR="${K3_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
python3 << 'PYEOF'
import json
import os
import re
import sys
from datetime import datetime, timezone

LOG = os.environ.get("K3_LOG", os.path.expanduser("~/train9k.log"))
DIR = os.environ["K3_DIR"]
MA_WINDOW = int(os.environ.get("K3_MA_WINDOW", "200"))

dashboard = os.path.join(DIR, "dashboard.html")
data_file = os.path.join(DIR, "data.json")

NPOS = 7
KV = re.compile(r"([a-zA-Z0-9_/]+)=([0-9.eE+-]+)")
TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

FIELDS = (["steps", "ts", "grad_norms", "losses", "lrs", "step_times",
           "ce_losses", "tv_losses", "conf_losses"]
          + [f"step_{i}_acc" for i in range(NPOS)]
          + [f"step_{i}_loss" for i in range(NPOS)])
EVAL_FIELDS = (["eval_steps", "eval_losses", "eval_acc_lens"]
               + [f"eval_step_{i}_acc" for i in range(NPOS)]
               + [f"eval_step_{i}_loss" for i in range(NPOS)])


def parse_ts(line):
    m = TS.match(line)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


# --- Run fingerprint: reset accumulated data when a new run starts ---------
# The log is truncated on every launch, so identify the run by its banner.
run_id = ""
cfg = {}
if not os.path.exists(LOG):
    print(f"Log not found: {LOG}")
    sys.exit(0)

with open(LOG, errors="ignore") as fh:
    head = fh.read(2_000_000)

m = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*LumenRL starting:.*?steps=(\d+)",
              head, re.M)
if m:
    cfg["start_time"], cfg["total_steps"] = m.group(1), int(m.group(2))
else:
    cfg["start_time"] = "?"
run_id = f"{LOG}|{cfg.get('start_time')}|{cfg.get('total_steps')}"

existing, offset = {}, 0
if os.path.exists(data_file) and os.environ.get("K3_RESET", "") != "1":
    try:
        with open(data_file) as fh:
            saved = json.load(fh)
        existing = saved
        # Freshly detected values win; saved ones only fill gaps.
        for k, v in saved.get("cfg", {}).items():
            cfg.setdefault(k, v)
        if saved.get("run_id") == run_id:
            offset = saved.get("log_offset", 0)
        # Otherwise the log was truncated by a relaunch: re-read from the top
        # but keep the history and append only steps we have not seen, so a
        # container restart does not lose the curve. Delete data.json (or set
        # K3_RESET=1) to start a genuinely new experiment.
    except Exception:
        existing = {}
for f in FIELDS + EVAL_FIELDS:
    existing.setdefault(f, [])

max_step = max(existing["steps"]) if existing["steps"] else -1
max_eval_step = max(existing["eval_steps"]) if existing["eval_steps"] else -1


# Config lines can sit anywhere in the log (vLLM worker noise pushes them far
# past any fixed-size head read), so pick them up while streaming.
CFG_PATTERNS = [
    ("Batch-alternating mode",
     re.compile(r"total_steps=(\d+), cache_batches=(\d+), num_rounds=(\d+)"),
     lambda m: {"total_steps": int(m.group(1)), "cache_batches": int(m.group(2)),
                "num_rounds": int(m.group(3))}),
    ("cached samples", re.compile(r"Loaded (\d+) cached samples"),
     lambda m: {"dataset_size": int(m.group(1))}),
    ("BF16Optimizer",
     re.compile(r"(\d+) trainable params.*?lr=([0-9.eE+-]+), decay=(\w+), "
                r"warmup=(\d+)/(\d+)"),
     lambda m: {"params": int(m.group(1)), "lr": float(m.group(2)),
                "decay": m.group(3), "warmup": int(m.group(4))}),
    ("train_global_batch_size", re.compile(r"train_global_batch_size=(\d+)"),
     lambda m: {"batch_size": int(m.group(1))}),
    ("max_total_sequence_length", re.compile(r"max_total_sequence_length=(\d+)"),
     lambda m: {"seq_len": int(m.group(1))}),
    ("anchor_num", re.compile(r"anchor_num=(\d+)"),
     lambda m: {"anchor_num": int(m.group(1))}),
]


def scan_config(line):
    for needle, pat, build in CFG_PATTERNS:
        if needle in line:
            m = pat.search(line)
            if m:
                cfg.update(build(m))

# --- Incremental parse ------------------------------------------------------
new_train = new_eval = 0
with open(LOG, errors="ignore") as fh:
    fh.seek(offset)
    for line in fh:
        if "callbacks: eval step=" in line:
            d = dict(KV.findall(line[line.index("eval step="):]))
            if "eval/loss" not in d or int(d["step"]) <= max_eval_step:
                continue
            try:
                accs = [float(d[f"eval/step_{i}_acc"]) for i in range(NPOS)]
            except KeyError:
                continue
            # Accept length = 1 + sum_k prod_{j<=k} acc_j. The leading 1 is the
            # token the target model emits itself, so a useless draft scores 1.0.
            # eval/simulated_acc_len in the log omits that 1.
            cum, acc_len = 1.0, 1.0
            for a in accs:
                cum *= a
                acc_len += cum
            existing["eval_steps"].append(int(d["step"]))
            existing["eval_losses"].append(float(d["eval/loss"]))
            existing["eval_acc_lens"].append(acc_len)
            for i in range(NPOS):
                existing[f"eval_step_{i}_acc"].append(accs[i])
                existing[f"eval_step_{i}_loss"].append(
                    float(d.get(f"eval/step_{i}_loss", 0)))
            new_eval += 1
        elif "callbacks: step=" in line:
            d = dict(KV.findall(line[line.index("step="):]))
            if "grad_norm" not in d or "step_0_acc" not in d:
                continue
            if int(d["step"]) <= max_step:
                continue
            loss = float(d.get("loss", "nan"))
            if loss != loss:  # NaN
                continue
            existing["steps"].append(int(d["step"]))
            existing["ts"].append(parse_ts(line) or 0.0)
            existing["grad_norms"].append(float(d["grad_norm"]))
            existing["losses"].append(loss)
            existing["lrs"].append(float(d.get("lr", 0)))
            existing["step_times"].append(float(d.get("timing/step_s", 0)))
            existing["ce_losses"].append(float(d.get("ce_loss", 0)))
            existing["tv_losses"].append(float(d.get("tv_loss", 0)))
            existing["conf_losses"].append(float(d.get("conf_loss", 0)))
            for i in range(NPOS):
                existing[f"step_{i}_acc"].append(float(d.get(f"step_{i}_acc", 0)))
                existing[f"step_{i}_loss"].append(float(d.get(f"step_{i}_loss", 0)))
            new_train += 1
        else:
            scan_config(line)
    new_offset = fh.tell()

total_steps = int(os.environ.get("K3_TOTAL_STEPS", cfg.get("total_steps", 36000)))
cache_batches = int(os.environ.get("K3_CACHE_BATCHES", cfg.get("cache_batches", 500)))
batch_size = cfg.get("batch_size", 8)
dataset_size = cfg.get("dataset_size", 0)
cfg.setdefault("anchor_num", 512)

if not existing["steps"]:
    print(f"No training steps in {LOG} yet (run started {cfg.get('start_time')}, "
          f"total_steps={total_steps}). Phase A probably still prefilling.")
    sys.exit(0)

existing["run_id"] = run_id
existing["log_offset"] = new_offset
existing["cfg"] = cfg
with open(data_file, "w") as fh:
    json.dump(existing, fh)

steps = existing["steps"]
losses = existing["losses"]
grad_norms = existing["grad_norms"]
lrs = existing["lrs"]
step_times = existing["step_times"]
ts = existing["ts"]
ce_losses = existing["ce_losses"]
tv_losses = existing["tv_losses"]
conf_losses = existing["conf_losses"]
sa = [existing[f"step_{i}_acc"] for i in range(NPOS)]
sl = [existing[f"step_{i}_loss"] for i in range(NPOS)]
n = len(steps)

e_steps = existing["eval_steps"]
e_losses = existing["eval_losses"]
e_acc_lens = existing["eval_acc_lens"]
ea = [existing[f"eval_step_{i}_acc"] for i in range(NPOS)]
el = [existing[f"eval_step_{i}_loss"] for i in range(NPOS)]
ne = len(e_steps)


def ma(arr, w=MA_WINDOW):
    out, s = [], 0.0
    for i, v in enumerate(arr):
        s += v
        if i >= w:
            s -= arr[i - w]
        out.append(s / min(i + 1, w))
    return out


ml, mg, mst = ma(losses), ma(grad_norms), ma(step_times)
mce, mtv, mcf = ma(ce_losses), ma(tv_losses), ma(conf_losses)
msa = [ma(sa[i]) for i in range(NPOS)]

train_acc_len = []
for i in range(n):
    val, cp = 1.0, 1.0
    for j in range(NPOS):
        cp *= msa[j][i]
        val += cp
    train_acc_len.append(val)

# --- Subsample to keep the page light --------------------------------------
stride = max(1, n // 2000)
idx = list(range(0, n, stride))
if idx[-1] != n - 1:
    idx.append(n - 1)
S = lambda a: [a[i] for i in idx]

ps = S(steps)
rl, rg, rst = S(losses), S(grad_norms), S(step_times)
pl, pg, pt, plr = S(ml), S(mg), S(mst), S(lrs)
pce, ptv, pcf = S(mce), S(mtv), S(mcf)
rce, rtv, rcf = S(ce_losses), S(tv_losses), S(conf_losses)
psa = [S(msa[i]) for i in range(NPOS)]
rsa = [S(sa[i]) for i in range(NPOS)]
psl = [S(ma(sl[i])) for i in range(NPOS)]
rsl = [S(sl[i]) for i in range(NPOS)]
ptal = S(train_acc_len)

round_boundaries = list(range(0, int(steps[-1]) + 1, cache_batches))

# --- Stats ------------------------------------------------------------------
cur = steps[-1]
pct = cur / total_steps * 100
w = min(1000, n)
avg_loss = sum(losses[-w:]) / w
avg_step_t = sum(step_times[-min(250, n):]) / min(250, n)

# Wall-clock rate includes Phase A prefill and vLLM restarts, unlike
# timing/step_s which only covers the Phase B optimizer step.
valid_ts = [(s, t) for s, t in zip(steps, ts) if t > 0]
if len(valid_ts) >= 2 and valid_ts[-1][0] > valid_ts[0][0]:
    span = valid_ts[-1][1] - valid_ts[0][1]
    wall_per_step = span / (valid_ts[-1][0] - valid_ts[0][0])
else:
    span, wall_per_step = 0.0, avg_step_t
eta_h = (total_steps - cur) * wall_per_step / 3600
eh, em = int(span // 3600), int((span % 3600) // 60)

aa = [sum(sa[i][-w:]) / w * 100 for i in range(NPOS)]
samples_seen = (cur + 1) * batch_size
epochs = samples_seen / dataset_size if dataset_size else 0.0

# The container may not be visible from the login node; fall back to log
# freshness (a stalled log for >20 min means the run is not producing steps).
age = datetime.now(timezone.utc).timestamp() - os.path.getmtime(LOG)
if cur >= total_steps - 1:
    status = "Completed"
elif age < 1200:
    status = "Training"
else:
    status = "Stopped"

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
latest_acc_len = e_acc_lens[-1] if e_acc_lens else 0.0
latest_eval_loss = e_losses[-1] if e_losses else 0.0
latest_train_acc_len = train_acc_len[-1] if train_acc_len else 0.0
current_round = cur // cache_batches
subtitle = (f"batch-alternating | lr={cfg.get('lr', 0):.1e} | bs={batch_size} | "
            f"T={cfg.get('seq_len', '?')} | block_size=7 | anchor={cfg.get('anchor_num')} | "
            f"cache_batches={cache_batches} | 8x MI355X (Replicate + vLLM TP=8)")

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
<p class="sub">{subtitle}</p>
<p class="st st-{status.lower()}">{status} (Round {current_round}) — started {cfg.get('start_time')} UTC</p>
<div class="stats">
<div class="s"><div class="sv">{cur:,} / {total_steps:,}</div><div class="sl">Step ({pct:.1f}%)</div></div>
<div class="s"><div class="sv">{samples_seen:,}</div><div class="sl">Samples seen ({epochs:.2f} epoch of {dataset_size:,})</div></div>
<div class="s"><div class="sv">{eh}h {em}m</div><div class="sl">Elapsed</div></div>
<div class="s"><div class="sv">{avg_loss:.4f}</div><div class="sl">Avg Loss (last {w})</div></div>
<div class="s"><div class="sv">{aa[0]:.1f} / {aa[1]:.1f} / {aa[2]:.1f} / {aa[3]:.1f} / {aa[4]:.1f} / {aa[5]:.1f} / {aa[6]:.1f}</div><div class="sl">Acc pos 0-6 (%)</div></div>
<div class="s"><div class="sv">{avg_step_t*1000:.0f} ms / {wall_per_step:.1f} s</div><div class="sl">Phase B step / wall per step</div></div>
<div class="s"><div class="sv">{eta_h:.1f} h</div><div class="sl">ETA</div></div>
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

var ra=[{','.join(json.dumps(rsa[i]) for i in range(NPOS))}];
var pa=[{','.join(json.dumps(psa[i]) for i in range(NPOS))}];
var rl_pos=[{','.join(json.dumps(rsl[i]) for i in range(NPOS))}];
var pl_pos=[{','.join(json.dumps(psl[i]) for i in range(NPOS))}];

var es_e={json.dumps(e_steps)};
var eloss={json.dumps(e_losses)};
var eacclen={json.dumps(e_acc_lens)};
var ea=[{','.join(json.dumps(ea[i]) for i in range(NPOS))}];
var el_e=[{','.join(json.dumps(el[i]) for i in range(NPOS))}];

var roundBounds={json.dumps(round_boundaries)};

var C=['#58a6ff','#3fb950','#d29922','#f778ba','#a371f7','#ff7b72','#79c0ff'];

function L(id,traces,title,extra){{
  var layout=JSON.parse(JSON.stringify(dark));
  layout.title={{text:title,font:{{size:13,color:'#c9d1d9'}}}};
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
],'Accept Length = 1 + &Sigma;&prod;acc (target &ge; 3.0)',{{yaxis:{{title:'Accept Length'}},legend:{{x:0.01,y:0.99,xanchor:'left',yanchor:'top'}}}});

L('c10',[
  raw(s,rce,'#58a6ff'),raw(s,rtv,'#3fb950'),raw(s,rcf,'#d29922'),
  {{x:s,y:pce,mode:'lines',name:'CE Loss',line:{{color:'#58a6ff',width:2}}}},
  {{x:s,y:ptv,mode:'lines',name:'TV/L1 Loss',line:{{color:'#3fb950',width:2}}}},
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
  {{x:s,y:tm.map(v=>v*1000),mode:'lines',name:'Phase B step',line:{{color:'#8b949e',width:2}}}},
],'Step Time (ms, Phase B only)',{{yaxis:{{title:'ms'}}}});

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
<p class="up">Updated: {now} | {n:,} train + {ne} eval points | source: {LOG} | round boundaries every {cache_batches} steps</p>
</body></html>"""

with open(dashboard, "w") as fh:
    fh.write(html)
print(f"Dashboard updated: {n} train ({new_train} new) + {ne} eval ({new_eval} new), "
      f"step {cur}/{total_steps}, accept_len={latest_acc_len:.4f}, status={status}")
print(f"  -> {dashboard}")
PYEOF
