#!/usr/bin/env python3
"""Parse async training log and generate an HTML dashboard."""
import re
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
LOG_FILE = os.path.join(REPO_ROOT, "output/DAPO/1a-bf16-async/1a-bf16-async-full.log")
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboards")
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "1a-bf16-async.html")


def parse_log(log_path):
    steps = []
    nan_grad_info = []
    dapo_info = []

    step_re = re.compile(
        r"step=(\d+)\s+"
        r"(?:async/param_version=[\d.]+\s+)?"
        r"loss=([\d.eE+-]+|nan)\s+"
        r"loss_pg=([\d.eE+-]+|nan)\s+"
        r"loss_total=([\d.eE+-]+|nan)\s+"
        r"timing/gen_s=([\d.]+)\s+"
        r"timing/step_s=([\d.]+)"
    )
    nan_re = re.compile(
        r"\[step (\d+)\] Zeroed NaN grads in (\d+)/(\d+) params, grad_norm=([\d.]+)"
    )
    dapo_re = re.compile(
        r"DAPO advantages: active_frac=([\d.]+).*?"
        r"rewards min=([\d.eE+-]+) max=([\d.eE+-]+) mean=([\d.eE+-]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            m = step_re.search(line)
            if m:
                steps.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)) if m.group(2) != "nan" else None,
                    "gen_s": float(m.group(5)),
                    "step_s": float(m.group(6)),
                })

            m = nan_re.search(line)
            if m:
                nan_grad_info.append({
                    "step": int(m.group(1)),
                    "nan_params": int(m.group(2)),
                    "total_params": int(m.group(3)),
                    "grad_norm": float(m.group(4)),
                })

            m = dapo_re.search(line)
            if m:
                dapo_info.append({
                    "active_frac": float(m.group(1)),
                    "reward_mean": float(m.group(4)),
                })

    return steps, nan_grad_info, dapo_info


def generate_dashboard(steps, nan_grad_info, dapo_info):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    valid_losses = [s["loss"] for s in steps if s["loss"] is not None]
    step_times = [s["step_s"] for s in steps]
    gen_times = [s["gen_s"] for s in steps]
    step_nums = [s["step"] for s in steps]
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    remaining = max(0, 275 - len(steps)) * avg_step_time / 3600 if steps else 0
    reward_means = [d["reward_mean"] for d in dapo_info]
    active_fracs = [d["active_frac"] for d in dapo_info]

    loss_color = "green" if valid_losses else "amber"
    loss_str = f"{valid_losses[-1]:.4f}" if valid_losses else "N/A"
    nan_str = f"{nan_grad_info[-1]['nan_params']}/{nan_grad_info[-1]['total_params']}" if nan_grad_info else "0"
    nan_color = "red" if nan_grad_info else "green"
    rew_str = f"{reward_means[-1]:.3f}" if reward_means else "N/A"
    af_str = f"{active_fracs[-1]:.0%}" if active_fracs else "N/A"
    avg_gen = sum(gen_times) / len(gen_times) if gen_times else 0

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>LumenRL 1A-Async Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e1e4e8}}
.hdr{{background:linear-gradient(135deg,#1a1e2e,#252a3a);padding:24px 32px;border-bottom:1px solid #30363d}}
.hdr h1{{font-size:1.6rem;font-weight:600}}.hdr .s{{color:#8b949e;font-size:.9rem;margin-top:4px}}
.cds{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;padding:20px 32px}}
.cd{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.cd .l{{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}}
.cd .v{{font-size:1.4rem;font-weight:600;margin-top:4px}}
.cd .v.g{{color:#3fb950}}.cd .v.a{{color:#d29922}}.cd .v.r{{color:#f85149}}
.ch{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 32px 32px}}
.cb{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
.cb h3{{font-size:.9rem;margin-bottom:12px;color:#c9d1d9}}
@media(max-width:900px){{.ch{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="hdr"><h1>LumenRL 1A-Async &#8212; BF16 Fully Async Training</h1>
<div class="s">Qwen3-8B-Base &#183; 8x MI350X &#183; DAPO &#183; Updated {now}</div></div>
<div class="cds">
<div class="cd"><div class="l">Steps</div><div class="v g">{len(steps)} / 275</div></div>
<div class="cd"><div class="l">Latest Loss</div><div class="v {loss_color}">{loss_str}</div></div>
<div class="cd"><div class="l">Avg Step</div><div class="v">{avg_step_time:.1f}s</div></div>
<div class="cd"><div class="l">ETA</div><div class="v a">{remaining:.1f}h</div></div>
<div class="cd"><div class="l">Avg Gen</div><div class="v">{avg_gen:.1f}s</div></div>
<div class="cd"><div class="l">NaN Grads</div><div class="v {nan_color}">{nan_str}</div></div>
<div class="cd"><div class="l">Reward</div><div class="v">{rew_str}</div></div>
<div class="cd"><div class="l">Active Frac</div><div class="v">{af_str}</div></div>
</div>
<div class="ch">
<div class="cb"><h3>Training Loss</h3><canvas id="c1"></canvas></div>
<div class="cb"><h3>Step / Gen Time (s)</h3><canvas id="c2"></canvas></div>
<div class="cb"><h3>Reward Mean</h3><canvas id="c3"></canvas></div>
<div class="cb"><h3>Grad Norm (post NaN-zero)</h3><canvas id="c4"></canvas></div>
</div>
<script>
const dg={{color:'rgba(255,255,255,.06)'}},dt={{color:'#8b949e'}};
const co={{responsive:true,plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},scales:{{x:{{grid:dg,ticks:dt}},y:{{grid:dg,ticks:dt}}}}}};
const co0={{responsive:true,plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},scales:{{x:{{grid:dg,ticks:dt}},y:{{grid:dg,ticks:dt,beginAtZero:true}}}}}};
new Chart(document.getElementById('c1'),{{type:'line',data:{{labels:{json.dumps(step_nums)},datasets:[{{label:'Loss',data:{json.dumps([s['loss'] for s in steps])},borderColor:'#58a6ff',borderWidth:2,pointRadius:2,tension:.3,spanGaps:true}}]}},options:co}});
new Chart(document.getElementById('c2'),{{type:'line',data:{{labels:{json.dumps(step_nums)},datasets:[{{label:'Step',data:{json.dumps(step_times)},borderColor:'#d29922',borderWidth:2,pointRadius:1,tension:.3}},{{label:'Gen',data:{json.dumps(gen_times)},borderColor:'#3fb950',borderWidth:2,pointRadius:1,tension:.3}}]}},options:co}});
new Chart(document.getElementById('c3'),{{type:'line',data:{{labels:{json.dumps(list(range(len(reward_means))))},datasets:[{{label:'Reward',data:{json.dumps(reward_means)},borderColor:'#f0883e',borderWidth:2,pointRadius:1,tension:.3}}]}},options:co}});
new Chart(document.getElementById('c4'),{{type:'line',data:{{labels:{json.dumps([n['step'] for n in nan_grad_info])},datasets:[{{label:'Grad Norm',data:{json.dumps([n['grad_norm'] for n in nan_grad_info])},borderColor:'#bc8cff',borderWidth:2,pointRadius:2,tension:.3}},{{label:'NaN Params (÷10)',data:{json.dumps([n['nan_params']/10 for n in nan_grad_info])},borderColor:'rgba(248,81,73,.6)',borderWidth:2,pointRadius:2,tension:.3}}]}},options:co0}});
</script></body></html>"""
    return html


def main():
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        sys.exit(1)
    steps, nan_grad_info, dapo_info = parse_log(LOG_FILE)
    html = generate_dashboard(steps, nan_grad_info, dapo_info)
    os.makedirs(os.path.dirname(DASHBOARD_FILE), exist_ok=True)
    with open(DASHBOARD_FILE, "w") as f:
        f.write(html)
    print(f"Dashboard updated: {len(steps)} steps -> {DASHBOARD_FILE}")


if __name__ == "__main__":
    main()
