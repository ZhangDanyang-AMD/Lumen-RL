#!/usr/bin/env python3
"""组装 gen-6 训练集：新生成的长样本 + 按类重新配平的旧语料。

两个要解决的问题：

1. **gen-5 的多语种占了 74.8%**，于是 75% 的梯度步花在多语种上。后果不在服务端
   （英文 benchmark 都涨了），而在评测尺子：训练侧 eval 取训练集尾部，量的分布和
   目标 benchmark 不重合，AL 涨了 37% 而 MT-Bench 只涨 4%。这一轮要把多语种压下去。

2. **各类的池子大小差了两个数量级**，所以"均衡"不可能是字面意义的等量：
   code 去重后只剩 23,083 行（唯一 prompt 的物理上限），而 multilingual 有 400 万。
   做法是给每类设一个上限，稀缺的类全取，充裕的类截断。

选取只从 gen-5 那份已经验证过的 536 万行里做，不重新生成 —— 那些行的 token 回环
校验早就过了。
"""
import argparse
import json
import random
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="gen-5 的 train.jsonl，带 source 字段")
    ap.add_argument("--new", nargs="*", default=[], help="新生成并转换好的 jsonl")
    ap.add_argument("--old-total", type=int, default=1_000_000)
    ap.add_argument("--multilingual-share", type=float, default=0.15,
                    help="多语种在旧语料里的占比上限（gen-5 是 0.748）")
    ap.add_argument("--cap", action="append", default=[], metavar="类名=行数",
                    help="给某一类单独设上限，可重复。腾出来的配额会被水位法自动"
                         "分给还有余量的类。gen-5 合并时 kimi-mtp 没写 source，"
                         "它的类名是 '?'，用 --cap '?=122546' 把它减半。")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default="")
    ap.add_argument("--seed", type=int, default=20260911)
    a = ap.parse_args()

    # 一遍扫出每类的行号，不把正文读进内存（536 万行正文有几十 GB）
    by_cat = defaultdict(list)
    with open(a.old, errors="ignore") as f:
        for i, line in enumerate(f):
            try:
                s = json.loads(line).get("source") or "?"
            except Exception:
                continue
            by_cat[s].append(i)
    print(f"旧语料 {sum(len(v) for v in by_cat.values()):,} 行，{len(by_cat)} 类")
    for s, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {s:<22} {len(v):>9,}")

    ml = [s for s in by_cat if s.startswith("multilingual_") or s == "aya"]
    en = [s for s in by_cat if s not in ml]
    ml_budget = int(a.old_total * a.multilingual_share)
    en_budget = a.old_total - ml_budget

    caps = {}
    for spec in a.cap:
        k, _, v = spec.partition("=")
        caps[k] = int(v)

    def allocate(cats, budget):
        """水位法：给每类同一个上限，池子小于上限的全取，剩下的预算摊给还有余量的类。
        这样稀缺类（code）不会被按比例压到更小，充裕类也不会独吞。
        --cap 相当于把那一类的池子调小，腾出来的配额自然流向别的类。"""
        pick, remaining = {}, budget
        pool = {c: min(len(by_cat[c]), caps.get(c, 1 << 62)) for c in cats}
        while pool and remaining > 0:
            cap = remaining // len(pool)
            if cap == 0:
                break
            done = [c for c, n in pool.items() if n <= cap]
            if not done:
                for c in pool:
                    pick[c] = cap
                remaining = 0
                break
            for c in done:
                pick[c] = pool[c]
                remaining -= pool[c]
                del pool[c]
        return pick

    plan = {**allocate(en, en_budget), **allocate(ml, ml_budget)}
    rng = random.Random(a.seed)
    keep = set()
    for c, k in plan.items():
        keep.update(rng.sample(by_cat[c], min(k, len(by_cat[c]))))
    print(f"\n旧语料选 {len(keep):,} 行（目标 {a.old_total:,}）：")
    for c, k in sorted(plan.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<22} {k:>9,}  （池子 {len(by_cat[c]):,}，取 {100*k/len(by_cat[c]):.0f}%）")

    n_out = Counter()
    with open(a.out, "w") as g:
        with open(a.old, errors="ignore") as f:
            for i, line in enumerate(f):
                if i in keep:
                    g.write(line if line.endswith("\n") else line + "\n")
                    try:
                        n_out[json.loads(line).get("source") or "?"] += 1
                    except Exception:
                        pass
        for p in a.new:
            with open(p, errors="ignore") as f:
                for line in f:
                    if line.strip():
                        g.write(line if line.endswith("\n") else line + "\n")
                        try:
                            n_out[json.loads(line).get("source") or "new"] += 1
                        except Exception:
                            n_out["new"] += 1

    total = sum(n_out.values())
    print(f"\n合计 {total:,} 行 -> {a.out}")
    mlc = sum(v for k, v in n_out.items() if k.startswith("multilingual_") or k == "aya")
    print(f"  多语种占比 {100*mlc/total:.1f}%（gen-5 是 74.8%）")
    for s, c in sorted(n_out.items(), key=lambda kv: -kv[1]):
        print(f"    {s:<22} {c:>9,}  {100*c/total:5.1f}%")
    if a.stats_out:
        json.dump({"total": total, "by_source": dict(n_out), "plan": plan},
                  open(a.stats_out, "w"), indent=2)
    print("\n/!\\ 这个文件还要过 selfcheck/preprocess_dataset.py（32768 口径）"
          "才能定 num_training_steps，别用行数直接除 128。")


if __name__ == "__main__":
    main()
