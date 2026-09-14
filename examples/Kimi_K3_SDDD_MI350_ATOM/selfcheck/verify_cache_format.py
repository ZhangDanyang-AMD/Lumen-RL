#!/usr/bin/env python3
"""核对预处理缓存的存储格式与常驻内存，并把 int32 路径与旧的 list[int] 路径对齐。

要证明三件事：
  1. input_ids 存的是 int32 ndarray，不是 list[int]（否则内存补丁没生效）
  2. torch.as_tensor(..., long) 取出来的张量和把同一行按旧路径 list->tensor
     取出来的逐元素相同（否则是静默的数据损坏，正是这个项目栽过两次的那类）
  3. 每行的常驻字节数，用来外推 550 万行 x 8 rank 的总量
"""
import argparse
import sys

import numpy as np
import torch


def rss_kib():
    """进程常驻内存。

    不用 sys.getsizeof 递归量：torch.load 出来的 ndarray 是同一个 pickle 缓冲区的
    视图（obj.base 不是 None），getsizeof 只返回约 112 字节的对象头，把真正的
    数据算漏了 —— 第一版就是这么量出「0.4 KiB/行」而 ids 本身 7.1 KiB 的自相矛盾。
    RSS 是没有歧义的那个数。
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extrapolate-rows", type=int, default=5_000_000)
    ap.add_argument("--ranks", type=int, default=8)
    a = ap.parse_args()

    rss0 = rss_kib()
    data = torch.load(a.cache, weights_only=False)
    rss1 = rss_kib()
    n = len(data)
    print(f"缓存 {a.cache}")
    print(f"行数 {n:,}")

    item = data[0]
    ids = item["input_ids"]
    print(f"\ninput_ids 类型 : {type(ids).__name__}")
    print(f"input_ids dtype: {getattr(ids, 'dtype', '(无)')}")
    if not isinstance(ids, np.ndarray) or ids.dtype != np.int32:
        sys.exit("!! 不是 int32 ndarray —— 内存补丁没生效")
    print("键            :", sorted(item.keys()))

    # ---- 2. 新旧两条路径逐元素比对 ----
    bad = 0
    for i in range(min(n, 2000)):
        arr = data[i]["input_ids"]
        new = torch.as_tensor(arr, dtype=torch.long)
        old = torch.tensor(arr.tolist(), dtype=torch.long)   # 旧代码那条路
        if new.dtype != torch.long or not torch.equal(new, old):
            bad += 1
    print(f"\n新旧路径比对 : 2000 行里 {bad} 行不一致")
    if bad:
        sys.exit("!! as_tensor 路径与 list 路径不等价，停止")

    # 词表边界：int32 无损的前提
    mx = max(int(data[i]["input_ids"].max()) for i in range(min(n, 5000)))
    mn = min(int(data[i]["input_ids"].min()) for i in range(min(n, 5000)))
    print(f"token id 范围: [{mn}, {mx}]（int32 上限 2147483647，无损）")

    # ---- 3. 内存 ----
    lens = [len(data[i]["input_ids"]) for i in range(n)]
    mean_len = sum(lens) / n
    held = (rss1 - rss0) * 1024 / n            # 字节/行
    print(f"\n序列长度 mean {mean_len:.0f}")
    print(f"torch.load 带来的 RSS 增量 {(rss1-rss0)/1024:.0f} MiB "
          f"= {held/1024:.1f} KiB/行")
    print(f"  其中纯 ids   {4*mean_len/1024:.1f} KiB/行（int32）")

    gib = held * a.extrapolate_rows / 2**30
    # 旧路径：Python int 每个 28 字节的对象 + list 里 8 字节的槽位
    old = (held - 4 * mean_len + 36 * mean_len) * a.extrapolate_rows / 2**30
    print(f"\n外推 {a.extrapolate_rows:,} 行（节点内存 2.8 TB = 2867 GiB）:")
    print(f"  int32 路径   单 rank {gib:6.0f} GiB   x{a.ranks} = {gib*a.ranks:6.0f} GiB")
    print(f"  旧 list 路径 单 rank {old:6.0f} GiB   x{a.ranks} = {old*a.ranks:6.0f} GiB")


if __name__ == "__main__":
    main()
