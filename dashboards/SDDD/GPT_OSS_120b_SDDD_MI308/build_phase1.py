#!/usr/bin/env python3
"""Build phase1.html for gpt-oss-120b Eagle3 SDDD run from training logs."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent.parent.parent / "output" / "GPT_OSS_120b_SDDD" / "LumenRL"
LOGS = [
    LOG_DIR / "host_launch_20260528_082051.log",   # steps 0-5000 (crash run)
    LOG_DIR / "gpt-oss-120b-eagle3-vllm-mi308.log",  # steps 5000-21000 (resumed)
]

PAT = re.compile(
    r"lumenrl\.trainer\.callbacks: step=(?P<step>\d+) "
    r"grad_norm=(?P<grad_norm>[\d.eE+-]+) "
    r"loss=(?P<loss>[\d.eE+-]+) "
    r"lr=(?P<lr>[\d.eE+-]+) "
    r"seq/max_len=(?P<seq_len>\d+) "
    r"step_0_acc=(?P<s0a>[\d.eE+-]+) step_0_loss=(?P<s0l>[\d.eE+-]+) "
    r"step_1_acc=(?P<s1a>[\d.eE+-]+) step_1_loss=(?P<s1l>[\d.eE+-]+) "
    r"step_2_acc=(?P<s2a>[\d.eE+-]+) step_2_loss=(?P<s2l>[\d.eE+-]+) "
    r"step_3_acc=(?P<s3a>[\d.eE+-]+) step_3_loss=(?P<s3l>[\d.eE+-]+) "
    r"timing/step_s=(?P<t_step>[\d.eE+-]+) "
    r"timing/teacher_s=(?P<t_teacher>[\d.eE+-]+) "
    r"timing/train_s=(?P<t_train>[\d.eE+-]+)"
)

records: dict[int, dict] = {}
for log in LOGS:
    with log.open() as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            d = m.groupdict()
            step = int(d["step"])
            records[step] = {
                "step": step,
                "grad_norm": float(d["grad_norm"]),
                "loss": float(d["loss"]),
                "lr": float(d["lr"]),
                "seq_len": int(d["seq_len"]),
                "s0a": float(d["s0a"]), "s0l": float(d["s0l"]),
                "s1a": float(d["s1a"]), "s1l": float(d["s1l"]),
                "s2a": float(d["s2a"]), "s2l": float(d["s2l"]),
                "s3a": float(d["s3a"]), "s3l": float(d["s3l"]),
                "t_step": float(d["t_step"]),
                "t_teacher": float(d["t_teacher"]),
                "t_train": float(d["t_train"]),
            }

rows = [records[k] for k in sorted(records.keys())]
print(f"parsed {len(rows)} unique steps (range {rows[0]['step']}..{rows[-1]['step']})")

steps   = [r["step"]   for r in rows]
loss    = [r["loss"]    for r in rows]
grad    = [r["grad_norm"] for r in rows]
lr      = [r["lr"]      for r in rows]
seq_len = [r["seq_len"] for r in rows]
s0a = [r["s0a"] for r in rows]; s1a = [r["s1a"] for r in rows]
s2a = [r["s2a"] for r in rows]; s3a = [r["s3a"] for r in rows]
s0l = [r["s0l"] for r in rows]; s1l = [r["s1l"] for r in rows]
s2l = [r["s2l"] for r in rows]; s3l = [r["s3l"] for r in rows]
t_step = [r["t_step"] for r in rows]
t_teacher = [r["t_teacher"] for r in rows]
t_train = [r["t_train"] for r in rows]

def ema(xs, alpha=0.02):
    out = []
    s = xs[0]
    for x in xs:
        s = alpha * x + (1 - alpha) * s
        out.append(s)
    return out

# Summary stats
tail = rows[-1000:] if len(rows) >= 1000 else rows
avg_loss_tail = sum(r["loss"] for r in tail) / len(tail)
avg_s0a_tail  = sum(r["s0a"]  for r in tail) / len(tail)
avg_s1a_tail  = sum(r["s1a"]  for r in tail) / len(tail)
avg_s2a_tail  = sum(r["s2a"]  for r in tail) / len(tail)
avg_s3a_tail  = sum(r["s3a"]  for r in tail) / len(tail)
total_steps   = rows[-1]["step"] + 1
target_steps  = 21000

data = {
    "steps": steps,
    "loss": loss,
    "loss_ema": ema(loss),
    "grad": grad,
    "lr": lr,
    "seq_len": seq_len,
    "s0a": s0a, "s1a": s1a, "s2a": s2a, "s3a": s3a,
    "s0l": s0l, "s1l": s1l, "s2l": s2l, "s3l": s3l,
    "t_step": t_step, "t_teacher": t_teacher, "t_train": t_train,
}
(ROOT / "phase1_data.json").write_text(json.dumps(data))

stat_html = (
    f'<div class="s"><div class="sv">{total_steps:,} / {target_steps:,}</div>'
    f'<div class="sl">Step (100%)</div></div>'
    f'<div class="s"><div class="sv">{avg_loss_tail:.4f}</div>'
    f'<div class="sl">Avg Loss (last 1K)</div></div>'
    f'<div class="s"><div class="sv">'
    f'{avg_s0a_tail*100:.1f}% / {avg_s1a_tail*100:.1f}% / {avg_s2a_tail*100:.1f}% / {avg_s3a_tail*100:.1f}%'
    f'</div><div class="sl">Acc pos 0/1/2/3 (last 1K)</div></div>'
)

DATA_JSON = json.dumps(data)

HTML = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT-OSS-120B Eagle3 SDDD - Phase 1</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:16px}}
.header{{text-align:center;padding:16px 0}}
h1{{color:#58a6ff;margin:0 0 4px 0;font-size:22px;font-weight:600}}
.sub{{color:#8b949e;font-size:13px;margin:0}}
.st{{font-size:14px;margin:6px 0;font-weight:600}}
.st-completed{{color:#58a6ff}}
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
<h1>GPT-OSS-120B Eagle3 SDDD - Phase 1</h1>
<p class="sub">ultrachat_200k | lr=5e-5 | spec_length=4 | bfloat16 | 8x MI308 (FSDP2 + vLLM TP=4 MXFP4)</p>
<p class="st st-completed">Completed</p>
<div class="stats">
{stat_html}
</div>
</div>
<div class="charts">
<div class="ch" id="ch_loss"></div>
<div class="ch" id="ch_acc"></div>
<div class="ch" id="ch_step_loss"></div>
<div class="ch" id="ch_grad"></div>
<div class="ch" id="ch_lr"></div>
<div class="ch" id="ch_timing"></div>
</div>
<p class="up">Data: 0&ndash;5000 from host_launch_20260528_082051.log, 5000&ndash;21000 from gpt-oss-120b-eagle3-vllm-mi308.log</p>
<script>
const D = {DATA_JSON};
const dark = {{
  paper_bgcolor:'#161b22', plot_bgcolor:'#0d1117',
  font:{{color:'#c9d1d9', size:11}},
  margin:{{l:50,r:20,t:36,b:40}},
  xaxis:{{gridcolor:'#21262d', zerolinecolor:'#30363d', title:{{text:'Step'}}}},
  yaxis:{{gridcolor:'#21262d', zerolinecolor:'#30363d'}},
  legend:{{orientation:'h', y:-0.18, x:0.5, xanchor:'center', font:{{size:10}}}},
  hovermode:'x unified'
}};
function layout(title, ytitle, extra) {{
  const L = JSON.parse(JSON.stringify(dark));
  L.title = {{text:title, font:{{size:13, color:'#58a6ff'}}}};
  L.yaxis.title = {{text:ytitle}};
  if (extra) Object.assign(L, extra);
  return L;
}}
const cfg = {{displayModeBar:false, responsive:true}};

Plotly.newPlot('ch_loss', [
  {{x:D.steps, y:D.loss, mode:'lines', name:'loss', line:{{color:'#8b949e', width:1}}}},
  {{x:D.steps, y:D.loss_ema, mode:'lines', name:'EMA(α=0.02)', line:{{color:'#58a6ff', width:2}}}}
], layout('Total Loss', 'loss'), cfg);

Plotly.newPlot('ch_acc', [
  {{x:D.steps, y:D.s0a, mode:'lines', name:'pos 0', line:{{color:'#3fb950', width:1.2}}}},
  {{x:D.steps, y:D.s1a, mode:'lines', name:'pos 1', line:{{color:'#58a6ff', width:1.2}}}},
  {{x:D.steps, y:D.s2a, mode:'lines', name:'pos 2', line:{{color:'#d29922', width:1.2}}}},
  {{x:D.steps, y:D.s3a, mode:'lines', name:'pos 3', line:{{color:'#f85149', width:1.2}}}}
], layout('Draft Token Acceptance Rate', 'acc'), cfg);

Plotly.newPlot('ch_step_loss', [
  {{x:D.steps, y:D.s0l, mode:'lines', name:'pos 0', line:{{color:'#3fb950', width:1.2}}}},
  {{x:D.steps, y:D.s1l, mode:'lines', name:'pos 1', line:{{color:'#58a6ff', width:1.2}}}},
  {{x:D.steps, y:D.s2l, mode:'lines', name:'pos 2', line:{{color:'#d29922', width:1.2}}}},
  {{x:D.steps, y:D.s3l, mode:'lines', name:'pos 3', line:{{color:'#f85149', width:1.2}}}}
], layout('Per-Position Loss', 'loss'), cfg);

Plotly.newPlot('ch_grad', [
  {{x:D.steps, y:D.grad, mode:'lines', name:'grad_norm', line:{{color:'#bc8cff', width:1}}}}
], layout('Gradient Norm', '||g||'), cfg);

Plotly.newPlot('ch_lr', [
  {{x:D.steps, y:D.lr, mode:'lines', name:'lr', line:{{color:'#39c5cf', width:1.5}}}}
], layout('Learning Rate', 'lr'), cfg);

Plotly.newPlot('ch_timing', [
  {{x:D.steps, y:D.t_step, mode:'lines', name:'step_s', line:{{color:'#58a6ff', width:0.8}}, opacity:0.7}},
  {{x:D.steps, y:D.t_teacher, mode:'lines', name:'teacher_s', line:{{color:'#d29922', width:0.8}}, opacity:0.7}},
  {{x:D.steps, y:D.t_train, mode:'lines', name:'train_s', line:{{color:'#3fb950', width:0.8}}, opacity:0.7}}
], layout('Per-Step Timing (s)', 'seconds'), cfg);
</script>
</body></html>
"""

OUT = ROOT / "phase1.html"
OUT.write_text(HTML)
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
