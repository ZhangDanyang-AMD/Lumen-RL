#!/usr/bin/env python3
"""证明「按行取 + 按 micro-batch collate」与旧的「pad 到全宽再切回来」逐元素相同。

为什么值得单独写这个：改的是喂给 draft 的张量本身，而这条链路上错了不会报错 ——
gen-1 的 RoPE 约定和 gen-2 的导出张量名都是「训练/服务两边各自自洽，跑完才发现」。
所以不靠 smoke test 跑通来推断，直接比对张量。

纯张量，不需要 GPU、Mooncake 或 manifest。
"""
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from lumenrl.trainer.spec_distill_trainer import SpecDistillTrainer

HIDDEN = 64          # 真实是 7168，这里缩小只为跑得快；形状逻辑与维度无关
AUX = 5
WIDTH_H = AUX * HIDDEN


def make_rows(lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i, n in enumerate(lens):
        rows.append({
            # 用可辨识的内容而不是纯随机：错位一格能直接看出来
            "hidden_states": (torch.arange(n * WIDTH_H, dtype=torch.float32)
                              .reshape(n, WIDTH_H) + i * 1e6).to(torch.bfloat16),
            "input_ids": torch.randint(0, 163840, (n,), generator=g, dtype=torch.int64),
            "last_hidden_states": (torch.arange(n * HIDDEN, dtype=torch.float32)
                                   .reshape(n, HIDDEN) - i * 1e5).to(torch.bfloat16),
        })
    return rows


def old_path(rows, total_len, micro_batch):
    """旧实现：每行 pad 到 total_len -> stack -> 按 micro-batch 切回 actual_len。"""
    hidden = torch.stack([
        F.pad(r["hidden_states"], (0, 0, 0, total_len - r["input_ids"].shape[0]))
        for r in rows])
    ids = torch.stack([
        F.pad(r["input_ids"], (0, total_len - r["input_ids"].shape[0])) for r in rows])
    last = torch.stack([
        F.pad(r["last_hidden_states"], (0, 0, 0, total_len - r["input_ids"].shape[0]))
        for r in rows])
    data = {"hidden_states": hidden, "input_ids": ids, "last_hidden_states": last}

    out = []
    for lo in range(0, len(rows), micro_batch):
        hi = min(lo + micro_batch, len(rows))
        mb = {k: v[lo:hi] for k, v in data.items()}
        # 旧代码用 attention_mask 的行和求 actual_len；这里等价地用真实行长，
        # 因为 mask 的行和就是行长（不一致会在 trainer 里被那条 warning 抓住）
        actual = max(r["input_ids"].shape[0] for r in rows[lo:hi])
        out.append({
            k: (v[:, :actual].contiguous() if v.dim() >= 2 else v)
            for k, v in mb.items()
        })
    return out


def new_path(rows, micro_batch):
    out = []
    for lo in range(0, len(rows), micro_batch):
        hi = min(lo + micro_batch, len(rows))
        out.append(SpecDistillTrainer._collate_disagg_rows(rows[lo:hi]))
    return out


def main():
    cases = [
        ("micro_batch=1 变长（正式配置）", [17, 5, 4096, 1, 233, 8191, 64, 900], 1),
        ("micro_batch=2 行长不等（需要 pad）", [17, 5, 4096, 1, 233, 811, 64, 900], 2),
        ("micro_batch=4", [100, 200, 300, 400, 50, 60, 70, 80], 4),
        ("全等长", [128] * 8, 1),
        ("单行", [321], 1),
    ]
    bad = 0
    for name, lens, mb in cases:
        rows = make_rows(lens, seed=len(lens))
        total_len = max(lens)
        old = old_path(rows, total_len, mb)
        new = new_path(rows, mb)
        assert len(old) == len(new), f"{name}: micro-batch 数量不同"
        for i, (o, n) in enumerate(zip(old, new)):
            assert set(o) == set(n), f"{name}: 键不同 {set(o)} vs {set(n)}"
            for k in o:
                if o[k].shape != n[k].shape:
                    print(f"  !! {name} mb{i} {k}: 形状 {o[k].shape} vs {n[k].shape}")
                    bad += 1
                elif not torch.equal(o[k], n[k]):
                    print(f"  !! {name} mb{i} {k}: 数值不等")
                    bad += 1
        print(f"  {'OK  ' if not bad else 'FAIL'} {name}: {len(new)} 个 micro-batch，"
              f"宽度 {[int(x['input_ids'].shape[1]) for x in new][:6]}")

    # token_embeds 必须不再出现：它是 1.75 GiB/rank/step 的纯浪费
    keys = set(new_path(make_rows([10]), 1)[0])
    assert keys == {"hidden_states", "input_ids", "last_hidden_states"}, keys
    print("  OK   collate 不再产出 token_embeds")

    # eval 走的那条：显式给 width，形状必须和旧实现一致
    rows = make_rows([17, 5, 900, 1])
    got = SpecDistillTrainer._collate_disagg_rows(rows, 900)
    exp = old_path(rows, 900, 4)[0]
    assert got["hidden_states"].shape == exp["hidden_states"].shape
    assert torch.equal(got["input_ids"], exp["input_ids"])
    print("  OK   显式 width（eval 路径）与旧实现同形状同值")

    # 越界必须报错，不能静默截断
    try:
        SpecDistillTrainer._collate_disagg_rows(make_rows([100]), 50)
    except ValueError:
        print("  OK   行长超过 width 时报错而不是静默截断")
    else:
        print("  !!   行长超过 width 竟然没报错")
        bad += 1

    print("\nFAIL" if bad else "\nPASS: 新旧路径逐元素等价")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
