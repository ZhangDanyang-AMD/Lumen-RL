#!/usr/bin/env python3
"""按 bf16_optimizer.LRSchedulerWithWarmup 的原样公式推 gen-7 的 cosine 参数。

为什么要单独算：cosine 下 `learning_rate` 是**入口和收尾绑死**的 —— 从 49,000 步
切进去时，实际 lr 是 `min_lr + coeff*(max_lr-min_lr)`，而 coeff 由 step 在总步数
里的位置决定，和"我想从多少开始"没有直接关系。gen-6 用 WSD 就是为了绕开这一点。
现在改回 cosine，就必须反解 max_lr 让入口落在想要的值上。

目标：入口 lr 接上 gen-6 的终点 1.0e-5（WSD 退火到 min_lr 的那个值），不制造
re-warming 冲击；收尾到一个非零的 min_lr，给 gen-8 留续训入场券。
"""
import argparse
import math

RESUME_STEP = 49000       # checkpoint_49000，scheduler.last_epoch 会被设成这个
GEN6_EXIT_LR = 1.0e-5     # gen-6 实际收尾的 lr


def lr_at(step, max_lr, min_lr, total, warmup_ratio=0.04, init_lr=0.0):
    warmup = int(warmup_ratio * total)
    if warmup > 0 and step <= warmup:
        return init_lr + (max_lr - init_lr) * step / warmup
    if step > total:
        return min_lr
    dr = max(0.0, min(1.0, (step - warmup) / max(total - warmup, 1)))
    coeff = 0.5 * (math.cos(math.pi * dr) + 1.0)
    return min_lr + coeff * (max_lr - min_lr)


def solve_max_lr(entry_lr, min_lr, total, warmup_ratio=0.04):
    """反解：要让 step=RESUME_STEP 处的 lr 等于 entry_lr，max_lr 该是多少。"""
    warmup = int(warmup_ratio * total)
    dr = (RESUME_STEP - warmup) / (total - warmup)
    coeff = 0.5 * (math.cos(math.pi * dr) + 1.0)
    return min_lr + (entry_lr - min_lr) / coeff, coeff, warmup


def main():
    ap = argparse.ArgumentParser()
    # /!\ 用预处理报出来的 num_training_steps，不要用 行数//bs。后者不扣 eval
    # 尾部，会差一步；runbook 5.3 的原则是这个数必须实测。
    ap.add_argument("--steps", type=int, required=True,
                    help="预处理报出的一个 epoch 的步数")
    ap.add_argument("--min-lr", type=float, default=2.0e-6)
    ap.add_argument("--entry-lr", type=float, default=GEN6_EXIT_LR)
    a = ap.parse_args()

    epoch_steps = a.steps
    total = RESUME_STEP + epoch_steps
    max_lr, coeff, warmup = solve_max_lr(a.entry_lr, a.min_lr, total)

    print(f"一个 epoch {epoch_steps:,} 步（预处理实测）")
    print(f"num_training_steps = {RESUME_STEP:,} + {epoch_steps:,} = {total:,}")
    print(f"warmup_steps = int(0.04*{total}) = {warmup:,}"
          f"  ({'远小于' if warmup < RESUME_STEP else '/!\\ 大于'} 续训起点，"
          f"{'不会重新 warmup' if warmup < RESUME_STEP else '会重新 warmup，危险'})")
    print(f"入口处 decay_ratio = {(RESUME_STEP-warmup)/(total-warmup):.5f}"
          f"   coeff = {coeff:.5f}")
    print(f"\n反解得 learning_rate = {max_lr:.4e}"
          f"  （取整写 {round(max_lr/1e-6)*1e-6:.1e}）")

    # 取整后复核整条曲线
    peak = round(max_lr / 1e-7) * 1e-7
    print(f"\n用 learning_rate={peak:.3e}  min_lr={a.min_lr:.1e} 的实际曲线:")
    marks = [RESUME_STEP, RESUME_STEP + 1]
    marks += [RESUME_STEP + int(epoch_steps * q) for q in (0.25, 0.5, 0.75, 1.0)]
    for s in marks:
        print(f"  step {s:>7,}   lr = {lr_at(s, peak, a.min_lr, total):.4e}")

    mean = sum(lr_at(s, peak, a.min_lr, total)
               for s in range(RESUME_STEP + 1, total + 1)) / epoch_steps
    print(f"\n  本轮平均 lr {mean:.3e}")
    print(f"  gen-6 平均 lr 约 4.0e-05 / 9,878 步；"
          f"本轮 {mean:.2e} / {epoch_steps:,} 步 = "
          f"学习量约 gen-6 的 {100*mean*epoch_steps/(4.0e-5*9878):.0f}%")
    print(f"\n  /!\\ learning_rate={peak:.3e} 这个峰值永远不会被走到（入口就在 "
          f"{100*coeff:.1f}% 处）。这份配置只能用于从 {RESUME_STEP:,} 步续训，"
          f"从零训会真的用到峰值。")


if __name__ == "__main__":
    main()
