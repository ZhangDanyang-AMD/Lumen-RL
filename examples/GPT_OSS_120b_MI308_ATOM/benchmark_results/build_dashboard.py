#!/usr/bin/env python3
"""Build an HTML dashboard for a bench_eagle3_vllm.py results JSON.

Usage:
    python3 build_dashboard.py <results.json>
        -> writes <results>.html next to it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <results.json>", file=sys.stderr)
    sys.exit(2)

src = Path(sys.argv[1]).resolve()
data = json.loads(src.read_text())
dst = src.with_suffix(".html")

title = src.stem  # e.g. phase1_v2_vllm_results_20260528_085745

DATA_JSON = json.dumps(data)

HTML = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT-OSS-120B Eagle3 Benchmark - {title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:16px}}
.header{{text-align:center;padding:12px 0}}
h1{{color:#58a6ff;margin:0 0 4px 0;font-size:20px;font-weight:600}}
.sub{{color:#8b949e;font-size:12px;margin:0}}
.stats{{display:flex;justify-content:center;gap:16px;margin:14px 0;flex-wrap:wrap}}
.s{{background:#161b22;border:1px solid #21262d;padding:10px 18px;border-radius:6px;text-align:center;min-width:140px}}
.sv{{font-size:20px;font-weight:600;color:#58a6ff}}
.sl{{font-size:11px;color:#8b949e;margin-top:3px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:1600px;margin:0 auto}}
.ch{{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:8px;height:380px}}
.tbl-wrap{{max-width:1600px;margin:14px auto;background:#161b22;border:1px solid #21262d;border-radius:6px;padding:8px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#58a6ff;font-weight:600;background:#0d1117;position:sticky;top:0}}
td.name,th.name{{text-align:left;font-family:ui-monospace,Menlo,monospace}}
tr:hover td{{background:#0d1117}}
.bad{{color:#f85149}}.warn{{color:#d29922}}.ok{{color:#3fb950}}
.up{{text-align:center;color:#8b949e;font-size:11px;margin-top:12px}}
@media(max-width:1100px){{.charts{{grid-template-columns:1fr}}}}
</style>
</head><body>
<div class="header">
<h1>GPT-OSS-120B Eagle3 vLLM Benchmark</h1>
<p class="sub">{title} | base=gpt-oss-120b (MXFP4, 8x MI308 TP=8) | draft=Eagle3 (1-layer, num_spec=4)</p>
</div>
<div class="stats" id="stats"></div>
<div class="charts">
<div class="ch" id="ch_accept"></div>
<div class="ch" id="ch_accept_rate"></div>
<div class="ch" id="ch_tput"></div>
<div class="ch" id="ch_errors"></div>
</div>
<div class="tbl-wrap">
<table id="tbl">
<thead><tr>
  <th class="name">benchmark</th>
  <th>num_samples</th>
  <th>errors</th>
  <th>accept_length</th>
  <th>acceptance_rate %</th>
  <th>throughput tok/s</th>
  <th>latency s</th>
  <th>total_output_tokens</th>
  <th>num_drafts</th>
  <th>num_accepted</th>
</tr></thead>
<tbody></tbody>
</table>
</div>
<p class="up">Generated from {src.name}. accept_length = 1 + accepted/drafts; higher is better (1.0 = no draft accepted).</p>
<script>
const D = {DATA_JSON};
const names = Object.keys(D);
const accept   = names.map(n => D[n].accept_length);
const acceptR  = names.map(n => D[n].acceptance_rate);
const tput     = names.map(n => D[n].throughput_tps);
const errors   = names.map(n => D[n].errors);
const ns       = names.map(n => D[n].num_samples);

// Summary stats
const totalSamples = ns.reduce((a,b)=>a+b,0);
const totalErrors  = errors.reduce((a,b)=>a+b,0);
const totalTokens  = names.reduce((a,n)=>a+D[n].total_output_tokens,0);
const totalLatency = names.reduce((a,n)=>a+D[n].latency_s,0);
const totalDrafts    = names.reduce((a,n)=>a+(D[n].num_drafts||0),0);
const totalAccepted  = names.reduce((a,n)=>a+(D[n].num_accepted||0),0);
const overallAccept  = totalDrafts ? (1 + totalAccepted/totalDrafts) : 1.0;
const overallTput    = totalLatency ? (totalTokens/totalLatency) : 0;

const stats = document.getElementById('stats');
function stat(v,l){{return `<div class="s"><div class="sv">${{v}}</div><div class="sl">${{l}}</div></div>`;}}
stats.innerHTML =
  stat(names.length, 'Benchmarks') +
  stat(totalSamples.toLocaleString(), 'Total samples') +
  stat(totalErrors, 'Total errors') +
  stat(overallAccept.toFixed(3), 'Overall accept_length') +
  stat(overallTput.toFixed(0)+' tok/s', 'Overall throughput') +
  stat((totalLatency/60).toFixed(1)+' min', 'Total wall time');

const dark = {{
  paper_bgcolor:'#161b22', plot_bgcolor:'#0d1117',
  font:{{color:'#c9d1d9', size:11}},
  margin:{{l:60,r:20,t:36,b:120}},
  xaxis:{{gridcolor:'#21262d', zerolinecolor:'#30363d', tickangle:-30}},
  yaxis:{{gridcolor:'#21262d', zerolinecolor:'#30363d'}},
  hovermode:'x'
}};
function layout(title, ytitle){{
  const L = JSON.parse(JSON.stringify(dark));
  L.title = {{text:title, font:{{size:13, color:'#58a6ff'}}}};
  L.yaxis.title = {{text:ytitle}};
  return L;
}}
const cfg = {{displayModeBar:false, responsive:true}};

Plotly.newPlot('ch_accept', [{{
  x:names, y:accept, type:'bar',
  marker:{{color:accept.map(v => v >= 1.3 ? '#3fb950' : (v >= 1.15 ? '#d29922' : '#f85149'))}},
  text:accept.map(v => v.toFixed(3)), textposition:'outside'
}}], layout('Mean Accept Length (1.0 = no draft accepted; higher = better)', 'accept_length'), cfg);

Plotly.newPlot('ch_accept_rate', [{{
  x:names, y:acceptR, type:'bar',
  marker:{{color:acceptR.map(v => v >= 30 ? '#3fb950' : (v >= 10 ? '#d29922' : '#f85149'))}},
  text:acceptR.map(v => v.toFixed(1)+'%'), textposition:'outside'
}}], layout('Acceptance Rate (% of draft tokens accepted)', '%'), cfg);

Plotly.newPlot('ch_tput', [{{
  x:names, y:tput, type:'bar', marker:{{color:'#58a6ff'}},
  text:tput.map(v => v.toFixed(0)), textposition:'outside'
}}], layout('Throughput (output tokens / sec, end-to-end)', 'tok/s'), cfg);

Plotly.newPlot('ch_errors', [
  {{x:names, y:ns,     type:'bar', name:'samples', marker:{{color:'#30363d'}}}},
  {{x:names, y:errors, type:'bar', name:'errors',  marker:{{color:'#f85149'}}}}
], Object.assign(layout('Samples vs. Errors', 'count'), {{barmode:'overlay'}}), cfg);

// Table
const tbody = document.querySelector('#tbl tbody');
for (const n of names) {{
  const r = D[n];
  const errP = r.num_samples ? (r.errors / r.num_samples * 100) : 0;
  const errCls = errP >= 50 ? 'bad' : (errP > 0 ? 'warn' : 'ok');
  const acCls  = r.accept_length >= 1.3 ? 'ok' : (r.accept_length >= 1.15 ? 'warn' : 'bad');
  tbody.insertAdjacentHTML('beforeend', `
    <tr>
      <td class="name">${{n}}</td>
      <td>${{r.num_samples}}</td>
      <td class="${{errCls}}">${{r.errors}}</td>
      <td class="${{acCls}}">${{r.accept_length.toFixed(3)}}</td>
      <td>${{r.acceptance_rate.toFixed(1)}}</td>
      <td>${{r.throughput_tps.toFixed(1)}}</td>
      <td>${{r.latency_s.toFixed(1)}}</td>
      <td>${{r.total_output_tokens.toLocaleString()}}</td>
      <td>${{(r.num_drafts||0).toLocaleString()}}</td>
      <td>${{(r.num_accepted||0).toLocaleString()}}</td>
    </tr>`);
}}
</script>
</body></html>
"""

dst.write_text(HTML)
print(f"wrote {dst} ({dst.stat().st_size/1024:.1f} KB)")
