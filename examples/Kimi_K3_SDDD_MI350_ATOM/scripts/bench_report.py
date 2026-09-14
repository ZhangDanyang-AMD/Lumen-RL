#!/usr/bin/env python3
"""把一个 label 的 13 套 benchmark 结果和官方 draft、gen-5 并排列出来。

分母只能用 results/bench_official-t0-*.json —— 同机、同镜像、同协议测的官方
draft。GEN4_RUNBOOK 7.4：不要拿 ATOM 上的数字去除 vLLM 的 model card 数字，
同一个官方 draft 在本机相对它自己 card 的偏差从 +1.5% 到 -23.8%，和数据集强
相关，不存在可外推的统一系数。
"""
import json
import os
import sys

R = os.environ.get("BENCH_RESULTS", "/home/jimguo12/k3-bench/results")
SETS = ["mtbench", "aime2026", "gsm8k", "humaneval", "math500", "mbpp",
        "speed_coding", "speed_multilingual", "speed_qa", "speed_rag",
        "speed_writing", "swebench_pro", "32k-speed_throughput16k"]


def load(label, s):
    p = os.path.join(R, f"bench_{label}-{s}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "gen6-t0"
    # -32k 那套的 label 带后缀，题目集名不带；文件名拼起来刚好一致
    print(f"{'set':26}{'官方':>9}{'gen5':>9}{'gen6':>9}"
          f"{'gen6/官方':>11}{'vs gen5':>10}{'n':>7}")
    rows = []
    for s in SETS:
        o, g5 = load("official-t0", s), load("gen5-t0", s)
        g6 = load(tag, s)
        if o is None:
            continue
        ov = o["tok_per_fwd"]
        g5v = g5["tok_per_fwd"] if g5 else None
        if g6 is None:
            print(f"{s:26}{ov:>9.4f}"
                  f"{(f'{g5v:.4f}' if g5v else '—'):>9}{'跑着':>9}")
            continue
        g6v = g6["tok_per_fwd"]
        rows.append((s, ov, g5v, g6v))
        d = f"{100*(g6v/g5v-1):+.1f}%" if g5v else "—"
        print(f"{s:26}{ov:>9.4f}{(f'{g5v:.4f}' if g5v else '—'):>9}"
              f"{g6v:>9.4f}{100*g6v/ov:>10.1f}%{d:>10}{g6['n']:>7}")

    if not rows:
        return
    n = len(rows)
    mo = sum(r[1] for r in rows) / n
    m6 = sum(r[3] for r in rows) / n
    print(f"\n  {n}/13 套完成")
    print(f"  {'均值 gen6/官方':22}{m6:.4f} / {mo:.4f} = {100*m6/mo:.1f}%")
    have5 = [r for r in rows if r[2]]
    if have5:
        m5 = sum(r[2] for r in have5) / len(have5)
        mo5 = sum(r[1] for r in have5) / len(have5)
        print(f"  {'同样这几套 gen5':22}{m5:.4f} / {mo5:.4f} = {100*m5/mo5:.1f}%")
    if n == 13:
        print("\n  gen-5 全 13 套是 91.1%，短板是 32k-speed_throughput16k（41.7%）"
              "和 swebench_pro（84.9%）。")


if __name__ == "__main__":
    main()
