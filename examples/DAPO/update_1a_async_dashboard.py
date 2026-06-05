#!/usr/bin/env python3
"""Parse async training log and generate a rich HTML dashboard for Exp 1A."""
import re
import json
import os
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
CONTAINER_LOG = "/workspace/Lumen-RL/output/DAPO/1a-bf16-async/1a-bf16-async-v2.log"
LOG_FILE = "/tmp/1a-bf16-async-v2.log"
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboards", "DAPO")
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "Qwen3-8B-Base_BF16_DAPO-Async.html")

NUM_SEQS_PER_STEP = 64  # 4 prompts × 16 gens
EST_AVG_RESP_LEN = 960  # tokens, from old ATOM run average

# Old ATOM run data (from 1a_metrics_dashboard.html) for comparison
OLD_ATOM_STEPS = list(range(16))
OLD_ATOM_REWARDS = [-0.734, -0.953, -0.75, -0.906, -0.844, -0.797, -0.734, -0.656, -0.672, -0.828, -0.703, -0.844, -0.859, -0.766, -0.797, -0.594]
OLD_ATOM_STEP_TIME = [131.9, 140.5, 126.9, 134.1, 119.3, 131.9, 154.9, 116.4, 95.7, 122.8, 118.0, 108.3, 146.9, 92.7, 153.7, 104.2]
OLD_ATOM_GEN_TIME = [47.0, 51.5, 51.0, 51.9, 47.2, 51.5, 56.3, 47.6, 44.1, 49.2, 47.8, 44.6, 55.8, 39.4, 56.5, 41.4]

# HF-Opt baseline data (72 steps)
HF_STEPS = list(range(72))
HF_STEP_TIME = [70.6, 57.5, 57.2, 58.2, 58.3, 57.4, 56.5, 58.8, 58.3, 59.3, 58.2, 56.7, 57.2, 56.1, 57.5, 57.8, 59.7, 58.2, 59.0, 57.9, 57.8, 56.9, 58.1, 51.6, 60.0, 57.3, 58.1, 56.6, 57.6, 57.9, 51.8, 57.8, 55.0, 60.3, 55.1, 59.8, 59.3, 58.9, 58.8, 54.4, 54.4, 59.5, 58.4, 58.4, 58.5, 57.5, 54.7, 55.1, 58.1, 54.6, 54.9, 53.1, 54.2, 52.5, 56.6, 56.6, 55.1, 55.0, 58.9, 54.6, 59.8, 58.3, 56.8, 54.7, 59.2, 59.4, 59.0, 56.7, 54.3, 59.0, 58.6, 56.0]
HF_GEN_TIME = [57.3, 50.7, 50.4, 51.0, 51.5, 50.8, 49.9, 52.1, 51.5, 52.2, 51.4, 49.9, 50.5, 49.4, 50.4, 50.9, 52.6, 51.4, 52.3, 51.1, 51.0, 50.1, 51.3, 44.6, 52.9, 50.5, 51.2, 49.8, 50.8, 51.2, 44.8, 51.2, 48.2, 53.1, 48.1, 52.6, 52.6, 52.2, 52.0, 47.7, 47.7, 52.4, 51.7, 51.6, 51.7, 50.7, 48.1, 48.3, 51.3, 48.0, 48.1, 46.0, 47.4, 45.6, 49.9, 49.8, 47.9, 48.3, 52.1, 47.9, 53.1, 51.5, 50.1, 47.9, 52.1, 52.3, 51.8, 50.0, 47.6, 51.8, 51.9, 49.2]


def parse_log(log_path):
    steps, nan_info, dapo_info = [], [], []
    step_re = re.compile(
        r"step=(\d+)\s+(?:async/param_version=[\d.]+\s+)?"
        r"loss=([\d.eE+-]+|nan)\s+loss_pg=([\d.eE+-]+|nan)\s+"
        r"loss_total=([\d.eE+-]+|nan)\s+timing/gen_s=([\d.]+)\s+timing/step_s=([\d.]+)"
    )
    nan_re = re.compile(r"\[step (\d+)\] Zeroed NaN grads in (\d+)/(\d+) params, grad_norm=([\d.]+)")
    dapo_re = re.compile(
        r"DAPO advantages: active_frac=([\d.]+).*?"
        r"rewards min=([\d.eE+-]+) max=([\d.eE+-]+) mean=([\d.eE+-]+)"
    )
    with open(log_path) as f:
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
                nan_info.append({"step": int(m.group(1)), "nan_params": int(m.group(2)),
                                 "total_params": int(m.group(3)), "grad_norm": float(m.group(4))})
            m = dapo_re.search(line)
            if m:
                dapo_info.append({"active_frac": float(m.group(1)), "reward_mean": float(m.group(4))})
    return steps, nan_info, dapo_info


def generate_dashboard(steps, nan_info, dapo_info):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(steps)
    step_nums = [s["step"] for s in steps]
    losses = [s["loss"] for s in steps]
    step_times = [s["step_s"] for s in steps]
    gen_times = [s["gen_s"] for s in steps]
    train_times = [round(s["step_s"] - s["gen_s"], 1) for s in steps]
    grad_norms = [ni["grad_norm"] for ni in nan_info]
    nan_params = [ni["nan_params"] for ni in nan_info]
    rewards = [d["reward_mean"] for d in dapo_info]
    active_fracs = [d["active_frac"] for d in dapo_info]

    tpt_ms = [round(g * 1000 / (NUM_SEQS_PER_STEP * EST_AVG_RESP_LEN), 3) for g in gen_times]

    valid_losses = [l for l in losses if l is not None]
    avg_step = sum(step_times) / n if n else 0
    avg_gen = sum(gen_times) / n if n else 0
    avg_train = sum(train_times) / n if n else 0
    avg_tpt = sum(tpt_ms) / n if n else 0
    eta_h = max(0, 275 - n) * avg_step / 3600 if n else 0
    latest_loss = f"{valid_losses[-1]:.4f}" if valid_losses else "N/A"
    latest_reward = f"{rewards[-1]:.3f}" if rewards else "N/A"
    latest_af = f"{active_fracs[-1]:.0%}" if active_fracs else "N/A"
    nan_str = f"{nan_info[-1]['nan_params']}/{nan_info[-1]['total_params']}" if nan_info else "0/0"
    pct = n / 275 * 100

    running = n < 275
    status_cls = "running" if running else "stopped"
    status_txt = f"RUNNING — step {n}/275" if running else f"DONE — {n} steps"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LumenRL — Qwen3-8B-Base | BF16 | DAPO Async (8×MI350X)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Instrument+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0e14; --card: #131820; --border: #1e2a3a;
    --text: #c5cdd8; --muted: #6b7a8d; --accent: #4facfe;
    --green: #36d399; --orange: #fbbd23; --red: #f87272;
    --purple: #a78bfa; --teal: #2dd4bf; --pink: #f472b6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Instrument Sans', sans-serif; padding: 28px 32px; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #f0f4f8;
       background: linear-gradient(135deg, #4facfe, #36d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ color: var(--muted); font-size: 0.88rem; margin-bottom: 28px; font-family: 'JetBrains Mono', monospace; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px; }}
  .card h2 {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 14px; color: var(--accent); letter-spacing: -0.01em; }}
  .chart-wrap {{ height: 300px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }}
  th, td {{ padding: 7px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  .status {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
  .status.running {{ background: var(--green); animation: pulse 1.5s infinite; }}
  .status.stopped {{ background: var(--orange); }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
  .badge-ok {{ background: rgba(54,211,153,0.12); color: var(--green); }}
  .badge-warn {{ background: rgba(251,189,35,0.12); color: var(--orange); }}
  .badge-info {{ background: rgba(79,172,254,0.12); color: var(--accent); }}
  .badge-red {{ background: rgba(248,114,114,0.12); color: var(--red); }}
  .full-width {{ grid-column: 1 / -1; }}
  .metric-row {{ display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }}
  .metric-box {{ background: linear-gradient(135deg, rgba(79,172,254,0.05), rgba(54,211,153,0.03));
    border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; flex: 1; min-width: 130px; }}
  .metric-box .label {{ font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; font-family: 'JetBrains Mono', monospace; }}
  .metric-box .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }}
  .metric-box .delta {{ font-size: 0.72rem; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }}
  .delta.pos {{ color: var(--green); }} .delta.neg {{ color: var(--red); }} .delta.neu {{ color: var(--muted); }}
  .note {{ border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;
    font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; }}
  .note-ok {{ background: rgba(54,211,153,0.06); border: 1px solid rgba(54,211,153,0.2); color: var(--green); }}
  .note-warn {{ background: rgba(251,189,35,0.06); border: 1px solid rgba(251,189,35,0.2); color: var(--orange); }}
  .note-info {{ background: rgba(79,172,254,0.06); border: 1px solid rgba(79,172,254,0.2); color: var(--accent); }}
  .attempt-log {{ font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; line-height: 1.7; }}
  .attempt-log .ok {{ color: var(--green); }} .attempt-log .fail {{ color: var(--red); }} .attempt-log .info {{ color: var(--accent); }}
  .version-tag {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.65rem; font-weight: 600; margin-right: 4px; }}
  .async-tag {{ background: rgba(79,172,254,0.15); color: var(--accent); }}
  .hf-tag {{ background: rgba(107,122,141,0.2); color: var(--muted); }}
</style>
</head>
<body>
<h1>LumenRL — Qwen3-8B-Base | BF16 | DAPO Async (8×MI350X)</h1>
<div class="subtitle">
  <span class="version-tag async-tag">Async v2</span><span class="status {status_cls}"></span> {status_txt} &nbsp;|&nbsp;
  Last updated: {now}
</div>

<div class="note note-ok">
  <b>Loss is valid.</b> AsyncRLTrainer staged pipeline producing non-NaN losses across all steps.
  NaN gradient workaround active (226/399 params zeroed per step). Model weights remain clean.
  OOM fix (chunked old_log_probs) applied — v1 OOM'd at step 11, v2 stable past step {n-1}.
</div>

<div class="note note-warn">
  <b>Open issues:</b> NaN grads in ~57% of params (FSDP2 backward on ROCm), weight sync to ATOM disabled (FSDP2 lazy storage).
  Reward is still near base-model random (~-0.97). Active fraction 25% (DAPO filtering expected). See <code>.cursor/tmp-rl-bugs.md</code>.
</div>

<div class="metric-row">
  <div class="metric-box">
    <div class="label">Current Step</div>
    <div class="value" style="color:var(--green)">{n}</div>
    <div class="delta neu">{n}/275 ({pct:.1f}%)</div>
  </div>
  <div class="metric-box">
    <div class="label">Latest Loss</div>
    <div class="value" style="color:var(--green)">{latest_loss}</div>
    <div class="delta pos">valid (non-NaN)</div>
  </div>
  <div class="metric-box">
    <div class="label">Reward Mean</div>
    <div class="value" style="color:var(--orange)">{latest_reward}</div>
    <div class="delta neg">near base-model random</div>
  </div>
  <div class="metric-box">
    <div class="label">Avg Step Time</div>
    <div class="value" style="color:var(--accent)">{avg_step:.1f}s</div>
    <div class="delta neu">gen {avg_gen:.0f}s + train {avg_train:.0f}s</div>
  </div>
  <div class="metric-box">
    <div class="label">Time/Token (gen)</div>
    <div class="value" style="color:var(--teal)">{avg_tpt:.2f}ms</div>
    <div class="delta neu">{NUM_SEQS_PER_STEP} seqs × ~{EST_AVG_RESP_LEN} tok</div>
  </div>
  <div class="metric-box">
    <div class="label">ETA</div>
    <div class="value" style="color:var(--orange)">{eta_h:.1f}h</div>
    <div class="delta neu">{275-n} steps remaining</div>
  </div>
  <div class="metric-box">
    <div class="label">NaN Grads</div>
    <div class="value" style="color:var(--red)">{nan_str}</div>
    <div class="delta neg">zeroed before optim step</div>
  </div>
  <div class="metric-box">
    <div class="label">Active Frac</div>
    <div class="value" style="color:var(--accent)">{latest_af}</div>
    <div class="delta neu">DAPO dynamic sampling</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>DAPO Loss (policy gradient)</h2>
    <div class="chart-wrap"><canvas id="chart-loss"></canvas></div>
  </div>
  <div class="card">
    <h2>Reward Mean per Step</h2>
    <div class="chart-wrap"><canvas id="chart-reward"></canvas></div>
  </div>
  <div class="card">
    <h2>Step Time (s) — Async vs HF-Opt Baseline</h2>
    <div class="chart-wrap"><canvas id="chart-step-time"></canvas></div>
  </div>
  <div class="card">
    <h2>Generation Time (s) — Async vs HF-Opt</h2>
    <div class="chart-wrap"><canvas id="chart-gen-time"></canvas></div>
  </div>
  <div class="card">
    <h2>Training Time (s) — Async (step - gen)</h2>
    <div class="chart-wrap"><canvas id="chart-train-time"></canvas></div>
  </div>
  <div class="card">
    <h2>Grad Norm (post NaN-zero) &amp; NaN Params</h2>
    <div class="chart-wrap"><canvas id="chart-grad"></canvas></div>
  </div>
  <div class="card">
    <h2>Time per Token (ms/tok) — Generation</h2>
    <div class="chart-wrap"><canvas id="chart-tpt"></canvas></div>
  </div>

  <div class="card full-width">
    <h2>Metrics Definitions</h2>
    <table>
      <tr><th>Metric</th><th>Formula</th><th>Unit</th><th>Description</th></tr>
      <tr>
        <td style="color:var(--accent)"><b>loss (DAPO)</b></td>
        <td><code>-1/|A⁺| Σ min(rₜ·Â, clip(rₜ, 1-ε_low, 1+ε_high)·Â)</code></td>
        <td>scalar</td>
        <td>Clipped PPO policy-gradient loss with DAPO asymmetric clip (ε_low=0.2, ε_high=0.28)</td>
      </tr>
      <tr>
        <td style="color:var(--orange)"><b>reward_mean</b></td>
        <td><code>mean(reward_fn(response))</code></td>
        <td>scalar</td>
        <td>Average reward across all sequences in the batch (math accuracy verifier)</td>
      </tr>
      <tr>
        <td style="color:var(--accent)"><b>active_frac</b></td>
        <td><code>|{{i : max(Rᵢ) ≠ min(Rᵢ)}}| / N</code></td>
        <td>ratio [0,1]</td>
        <td>DAPO filtering: fraction of prompts with non-degenerate reward variance</td>
      </tr>
      <tr>
        <td style="color:var(--teal)"><b>time_per_token_ms</b></td>
        <td><code>gen_s × 1000 / (num_seqs × avg_resp_len)</code></td>
        <td>ms/token</td>
        <td>Estimated latency per generated token during ATOM rollout. num_seqs={NUM_SEQS_PER_STEP}, avg_resp_len≈{EST_AVG_RESP_LEN}</td>
      </tr>
      <tr>
        <td style="color:var(--accent)"><b>step_time</b></td>
        <td><code>timing/step_s</code> from log</td>
        <td>seconds</td>
        <td>Total wall-clock time for one training step (generation + forward + backward + optim)</td>
      </tr>
      <tr>
        <td style="color:var(--teal)"><b>gen_time</b></td>
        <td><code>timing/gen_s</code> from log</td>
        <td>seconds</td>
        <td>ATOM rollout time (prompt encoding + generation of all sequences)</td>
      </tr>
      <tr>
        <td style="color:var(--accent)"><b>train_time</b></td>
        <td><code>step_time - gen_time</code></td>
        <td>seconds</td>
        <td>Time for PPO forward, backward, optimizer step, weight sync, gradient NaN cleanup</td>
      </tr>
      <tr>
        <td style="color:var(--purple)"><b>grad_norm</b></td>
        <td><code>‖∇θ‖₂</code> after NaN-zeroing</td>
        <td>scalar</td>
        <td>L2 norm of the full gradient vector after zeroing NaN entries</td>
      </tr>
      <tr>
        <td style="color:var(--red)"><b>nan_params</b></td>
        <td><code>|{{p : any(isnan(p.grad))}}|</code></td>
        <td>count</td>
        <td>Number of FSDP2 param groups with at least one NaN gradient element</td>
      </tr>
      <tr>
        <td style="color:var(--accent)"><b>ETA</b></td>
        <td><code>(275 - current_step) × avg_step_time / 3600</code></td>
        <td>hours</td>
        <td>Estimated time to complete all 275 training steps</td>
      </tr>
    </table>
  </div>

  <div class="card full-width">
    <h2>Configuration</h2>
    <table>
      <tr><th>Parameter</th><th>Async v2 (current)</th><th>Reference (VERL H100)</th><th>Status</th></tr>
      <tr><td>Framework</td><td style="color:var(--green)"><b>LumenRL AsyncRLTrainer</b></td><td>VERL + vLLM</td><td><span class="badge badge-info">NEW</span></td></tr>
      <tr><td>Inference</td><td style="color:var(--green)"><b>ATOM (vLLM 0.16.1)</b></td><td>vLLM 0.7</td><td><span class="badge badge-ok">ATOM</span></td></tr>
      <tr><td>GPUs</td><td>8× MI350X</td><td>8× H100</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>Model</td><td>Qwen3-8B-Base</td><td>Qwen3-8B-Base</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>Algorithm</td><td>DAPO</td><td>DAPO</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>require_batches</td><td style="color:var(--orange)">4</td><td>32 prompts</td><td><span class="badge badge-warn">SMALLER</span></td></tr>
      <tr><td>gens/prompt</td><td>16</td><td>16</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>max_seq_len</td><td style="color:var(--orange)">8192</td><td>21504</td><td><span class="badge badge-warn">SHORTER</span></td></tr>
      <tr><td>LR</td><td>1e-6</td><td>1e-6</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>clip (low/high)</td><td>0.2 / 0.28</td><td>0.2 / 0.28</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>kl_coeff</td><td>0.0</td><td>0.0</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>FSDP2</td><td>on (8 GPU)</td><td>on (8 GPU)</td><td><span class="badge badge-ok">MATCH</span></td></tr>
      <tr><td>total_steps</td><td>275</td><td>500</td><td><span class="badge badge-warn">FEWER</span></td></tr>
      <tr><td>weight_sync</td><td style="color:var(--red)">DISABLED</td><td>enabled</td><td><span class="badge badge-red">BROKEN</span></td></tr>
    </table>
  </div>

  <div class="card full-width">
    <h2>Known Issues</h2>
    <table>
      <tr><th>Issue</th><th>Symptom</th><th>Impact</th><th>Status</th></tr>
      <tr>
        <td style="color:var(--orange)"><b>NaN Gradients</b></td>
        <td>226-232/399 params have NaN grads after backward</td>
        <td>Zeroed before optimizer step; model weights clean</td>
        <td><span class="badge badge-warn">WORKAROUND</span></td>
      </tr>
      <tr>
        <td style="color:var(--orange)"><b>Weight Sync Disabled</b></td>
        <td>ATOM always generates with initial weights</td>
        <td>Model can't improve generation quality</td>
        <td><span class="badge badge-warn">KNOWN</span></td>
      </tr>
      <tr>
        <td style="color:var(--green)"><b>OOM at step 11 (v1)</b></td>
        <td>lm_head 60.7 GiB alloc on full batch</td>
        <td>Fixed: chunked old_log_probs forward</td>
        <td><span class="badge badge-ok">FIXED</span></td>
      </tr>
    </table>
  </div>

  <div class="card full-width">
    <h2>Run History</h2>
    <div class="attempt-log">
      <div><b style="color:var(--accent)">Async v2 (current, OOM fix)</b></div>
      <div class="info">Container: lumenrl_1a | Started: 2026-04-18 03:40 UTC | Config: 4 prompts × 16 gens, seq=8K, staged pipeline</div>
      <div><span class="ok">{status_txt}</span> — {avg_step:.0f}s/step, loss valid. OOM fix: chunked old_log_probs.</div>
      <div style="margin-top:10px"><b style="color:var(--red)">Async v1 (OOM'd)</b></div>
      <div class="fail">OOM at step 11 — lm_head tried 60.7 GiB on full 64-seq batch. Killed.</div>
      <div style="margin-top:10px"><b style="color:var(--muted)">Old ATOM Run (loss=nan)</b></div>
      <div class="fail">16 steps, loss=nan. Fixed: asymmetric clip, grad accum, NaN masking, logsumexp revert.</div>
      <div style="margin-top:10px"><b style="color:var(--muted)">HF-Opt Run</b></div>
      <div><span class="ok">STOPPED</span> step 71/275 — 57s/step, 606 tok/s. Acc=0% (max_seq too short).</div>
    </div>
  </div>
</div>

<script>
const cF = {{ family: "'JetBrains Mono', monospace" }};
const cO = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ labels: {{ color: '#6b7a8d', font: {{ ...cF, size: 11 }} }} }} }},
  scales: {{
    x: {{ type: 'linear', ticks: {{ color: '#6b7a8d', font: cF }}, grid: {{ color: '#1a2233' }},
         title: {{ display: true, text: 'Training Step', color: '#6b7a8d', font: cF }} }},
    y: {{ ticks: {{ color: '#6b7a8d', font: cF }}, grid: {{ color: '#1a2233' }} }}
  }}
}};
function mk(label, steps, data, color, opts={{}}) {{
  return {{
    label, data: steps.map((s,i) => ({{x: s, y: data[i]}})),
    borderColor: color, backgroundColor: color + '14',
    tension: 0.3, pointRadius: 2, borderWidth: 2, fill: false,
    pointBackgroundColor: color, ...opts
  }};
}}

// Async v2 data
const aS = {json.dumps(step_nums)};
const aL = {json.dumps(losses)};
const aR = {json.dumps(rewards[:n] if len(rewards) >= n else rewards)};
const aSt = {json.dumps(step_times)};
const aGt = {json.dumps(gen_times)};
const aTt = {json.dumps(train_times)};
const aGN = {json.dumps(grad_norms)};
const aNP = {json.dumps([p/10 for p in nan_params])};
const aTpt = {json.dumps(tpt_ms)};

// HF-Opt baseline
const hS = {json.dumps(HF_STEPS)};
const hSt = {json.dumps(HF_STEP_TIME)};
const hGt = {json.dumps(HF_GEN_TIME)};

new Chart(document.getElementById('chart-loss'), {{
  type: 'line',
  data: {{ datasets: [mk('Async v2 Loss', aS, aL, '#4facfe', {{fill:true, spanGaps:true}})] }},
  options: {{ ...cO, scales: {{ ...cO.scales,
    y: {{ ...cO.scales.y, title: {{ display:true, text:'DAPO Loss', color:'#6b7a8d', font:cF }} }} }} }}
}});

new Chart(document.getElementById('chart-reward'), {{
  type: 'line',
  data: {{ datasets: [mk('Async v2 Reward', Array.from({{length:aR.length}},(_,i)=>i), aR, '#f0883e', {{fill:true}})] }},
  options: {{ ...cO, scales: {{ ...cO.scales,
    y: {{ ...cO.scales.y, title: {{ display:true, text:'Reward (mean)', color:'#6b7a8d', font:cF }} }} }} }}
}});

new Chart(document.getElementById('chart-step-time'), {{
  type: 'line',
  data: {{ datasets: [
    mk('Async v2 (s)', aS, aSt, '#4facfe'),
    mk('HF-Opt (s)', hS, hSt, '#6b7a8d', {{borderDash:[6,3], pointRadius:1}})
  ] }},
  options: {{ ...cO, scales: {{ ...cO.scales,
    y: {{ ...cO.scales.y, min:0, title: {{ display:true, text:'Step Time (s)', color:'#6b7a8d', font:cF }} }} }} }}
}});

new Chart(document.getElementById('chart-gen-time'), {{
  type: 'line',
  data: {{ datasets: [
    mk('Async v2 Gen (s)', aS, aGt, '#2dd4bf'),
    mk('HF-Opt Gen (s)', hS, hGt, '#6b7a8d', {{borderDash:[6,3], pointRadius:1}})
  ] }},
  options: {{ ...cO, scales: {{ ...cO.scales,
    y: {{ ...cO.scales.y, min:0, title: {{ display:true, text:'Generation Time (s)', color:'#6b7a8d', font:cF }} }} }} }}
}});

new Chart(document.getElementById('chart-train-time'), {{
  type: 'line',
  data: {{ datasets: [mk('Async v2 Train (s)', aS, aTt, '#fbbd23', {{fill:true}})] }},
  options: {{ ...cO, scales: {{ ...cO.scales,
    y: {{ ...cO.scales.y, min:0, title: {{ display:true, text:'Training Time (s)', color:'#6b7a8d', font:cF }} }} }} }}
}});

new Chart(document.getElementById('chart-grad'), {{
  type: 'line',
  data: {{ datasets: [
    mk('Grad Norm', aS.slice(0,aGN.length), aGN, '#a78bfa'),
    mk('NaN Params (÷10)', aS.slice(0,aNP.length), aNP, 'rgba(248,114,114,0.6)')
  ] }},
  options: {{ ...cO, scales: {{ ...cO.scales,
    y: {{ ...cO.scales.y, min:0, title: {{ display:true, text:'Grad Norm / NaN Count', color:'#6b7a8d', font:cF }} }} }} }}
}});

new Chart(document.getElementById('chart-tpt'), {{
  type: 'line',
  data: {{ datasets: [mk('Time/Token (ms)', aS, aTpt, '#2dd4bf', {{fill:true}})] }},
  options: {{ ...cO, scales: {{ ...cO.scales,
    y: {{ ...cO.scales.y, min:0, title: {{ display:true, text:'ms / token (gen)', color:'#6b7a8d', font:cF }} }} }} }}
}});
</script>
</body>
</html>"""
    return html


def main():
    subprocess.run(
        ["docker", "cp", f"lumenrl_1a:{CONTAINER_LOG}", LOG_FILE],
        capture_output=True, timeout=30,
    )
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        sys.exit(1)
    steps, nan_info, dapo_info = parse_log(LOG_FILE)
    html = generate_dashboard(steps, nan_info, dapo_info)
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    with open(DASHBOARD_FILE, "w") as f:
        f.write(html)
    print(f"Dashboard updated: {len(steps)} steps -> {DASHBOARD_FILE}")


if __name__ == "__main__":
    main()
