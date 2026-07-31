"""Render the 4-panel DSpark training figure from LumenRL logs.

Mirrors the reference dashboard layout: loss, train accuracy per position,
accept length, eval accuracy per position.

Accept length here is ``1 + sum_k prod_{j<=k} acc_j`` -- the leading 1 is the
token the target model emits itself, so a draft that accepts nothing still
scores 1.0. Note ``eval/simulated_acc_len`` in the trainer logs omits that 1,
which is why raw log values look one lower than the dashboard's.

Usage: python3 plot_progress.py out.png log1 [log2 ...]
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPOS = 7
MA_WINDOW = 200
KV = re.compile(r"([a-zA-Z0-9_/]+)=([0-9.eE+-]+)")


def parse(paths):
    train, ev = [], []
    for path in paths:
        with open(path, errors="ignore") as fh:
            for line in fh:
                if "callbacks: eval step=" in line:
                    d = dict(KV.findall(line[line.index("eval step="):]))
                    if "eval/loss" in d:
                        ev.append(d)
                elif "callbacks: step=" in line:
                    d = dict(KV.findall(line[line.index("step="):]))
                    if "grad_norm" in d and "step_0_acc" in d:
                        train.append(d)
    return train, ev


def num(d, key, default=0.0):
    try:
        return float(d[key])
    except (KeyError, ValueError):
        return default


def moving_avg(values, window=MA_WINDOW):
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def accept_length(acc_per_pos):
    """1 + cumulative-product sum, matching the reference dashboard scale."""
    total, cum = 1.0, 1.0
    for a in acc_per_pos:
        cum *= a
        total += cum
    return total


def main():
    out_png, logs = sys.argv[1], sys.argv[2:]
    train, ev = parse(logs)
    if not train:
        sys.exit("no training lines parsed")

    steps = [num(d, "step") for d in train]
    tr_loss = [num(d, "ce_loss") for d in train]
    tr_acc = [[num(d, f"step_{i}_acc") for d in train] for i in range(NPOS)]
    tr_acc_ma = [moving_avg(a) for a in tr_acc]
    tr_len = [accept_length([tr_acc_ma[i][j] for i in range(NPOS)])
              for j in range(len(train))]

    e_steps = [num(d, "step") for d in ev]
    e_loss = [num(d, "eval/loss") for d in ev]
    e_acc = [[num(d, f"eval/step_{i}_acc") for d in ev] for i in range(NPOS)]
    e_len = [accept_length([e_acc[i][j] for i in range(NPOS)])
             for j in range(len(ev))]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(2, 2, figsize=(14, 7.5), dpi=110)
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f778ba",
              "#a371f7", "#ff7b72", "#79c0ff"]

    a = ax[0][0]
    a.plot(steps, tr_loss, color="#ff7b72", alpha=0.25, lw=0.5)
    a.plot(steps, moving_avg(tr_loss), color="#ff7b72", lw=1.5,
           label="Train CE Loss (MA)")
    if ev:
        a.plot(e_steps, e_loss, color="#58a6ff", lw=1.5, label="Eval Loss")
    a.set_title("Training & Eval Loss")
    a.set_xlabel("Step")
    a.legend(fontsize=8)

    a = ax[0][1]
    for i in range(NPOS):
        a.plot(steps, [v * 100 for v in tr_acc_ma[i]], color=colors[i],
               lw=1.2, label=f"Pos {i}")
    a.set_title("Train Accuracy by Position (%)")
    a.set_xlabel("Step")
    a.set_ylabel("%")
    a.legend(fontsize=7, ncol=2)

    a = ax[1][0]
    a.plot(steps, tr_len, color="#d29922", lw=1.5, label="Train (MA)")
    if ev:
        a.plot(e_steps, e_len, color="#3fb950", lw=1.8, marker="o",
               ms=2.5, label="Eval")
    a.set_title("Accept Length (Train vs Eval)")
    a.set_xlabel("Step")
    a.set_ylabel("Accept Length")
    a.legend(fontsize=8)

    a = ax[1][1]
    for i in range(NPOS):
        a.plot(e_steps, [v * 100 for v in e_acc[i]], color=colors[i],
               lw=1.5, marker="o", ms=2, label=f"Pos {i}")
    a.set_title("Eval Accuracy by Position (%)")
    a.set_xlabel("Step")
    a.set_ylabel("%")
    a.legend(fontsize=7, ncol=2)

    for row in ax:
        for cell in row:
            cell.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())

    print(f"steps parsed : {len(train)} train / {len(ev)} eval")
    print(f"last step    : {int(steps[-1])}")
    print(f"accept length: train {tr_len[-1]:.4f}" +
          (f" / eval {e_len[-1]:.4f}" if ev else ""))
    if ev:
        pos = " / ".join(f"{e_acc[i][-1]*100:.1f}" for i in range(NPOS))
        print(f"eval acc pos : {pos}")
    print(f"wrote        : {out_png}")


if __name__ == "__main__":
    main()
