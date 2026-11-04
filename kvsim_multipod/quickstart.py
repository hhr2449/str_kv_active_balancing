#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SRT 快速上手：加载 full 包数据，复现三个关键结论。

用法：
    python srt_quickstart.py <解压后的 full 包目录>

示例：
    tar -xzf experiments/offline_modeling_dataset_full.tar.gz -C /tmp/kvprefetch_full
    python ft_local/srt_quickstart.py /tmp/kvprefetch_full

输出三块（跑通了就是对的）：
    [1] transfer 失败率   —— 预期成功率 ≈1.6%，失败全因 source cache does not cover
    [2] 命中率天花板      —— 预期目标 ~86.2% vs 实际 ~56%
    [3] kvsim 基线        —— AFF/LEAST/REQ 的 hit/skew（块=64）

依赖：同目录的 kvsim_multipod.py；纯标准库，无 GPU。
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kvsim_multipod as kv  # noqa: E402


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iso_ms(iso):
    """ISO 时间字符串 -> epoch 毫秒（整数）。"""
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def stat_transfer(data_dir):
    """[1] 复现 transfer 失败率。"""
    rows = load_jsonl(os.path.join(data_dir, "transfer.jsonl"))
    ok = rej = 0
    by_reason = defaultdict(lambda: [0, 0])  # reason -> [ok, rejected]
    fail_msgs = defaultdict(int)
    ok_tokens = []

    for t in rows:
        reason = t.get("reason", "?")
        if t.get("ok"):
            ok += 1
            by_reason[reason][0] += 1
            if t.get("transferred_tokens"):
                ok_tokens.append(t["transferred_tokens"])
        else:
            rej += 1
            by_reason[reason][1] += 1
            msg = t.get("fail_message", "")
            if "source cache does not cover" in msg:
                fail_msgs["source cache does not cover (device=0 host=0)"] += 1
            elif "source residency probe failed" in msg:
                fail_msgs["source residency probe failed"] += 1
            else:
                fail_msgs[msg[:60]] += 1

    total = ok + rej
    print("=" * 60)
    print("[1] transfer 失败率")
    print(f"  ok={ok}  rejected={rej}  成功率={ok / total * 100:.2f}%")
    for reason in sorted(by_reason):
        o, r = by_reason[reason]
        n = o + r
        print(f"    reason={reason:12s} ok={o:4d} rejected={r:4d}  成功率 {o / n * 100:.1f}%")
    print("  失败原因分布:")
    for msg, c in sorted(fail_msgs.items(), key=lambda x: -x[1]):
        print(f"    - {msg}: {c}")
    if ok_tokens:
        ok_tokens.sort()
        print(f"  成功 transfer 的 token 数: min={ok_tokens[0]} "
              f"p50={ok_tokens[len(ok_tokens) // 2]} max={ok_tokens[-1]}")


def stat_ceiling(data_dir):
    """[2] 标定命中率天花板。"""
    trace = load_jsonl(os.path.join(data_dir, "trace.jsonl"))
    n_tokens = sum(int(r["input_length"]) for r in trace)
    target = sum(int(r["cached_target"]) for r in trace)

    per_req = load_jsonl(os.path.join(data_dir, "per_request.jsonl"))
    prompt = sum(int(r["prompt_tokens"]) for r in per_req)
    cached = sum(int(r["cached_tokens"]) for r in per_req)

    target_rate = target / n_tokens * 100
    actual_rate = cached / prompt * 100
    print("=" * 60)
    print("[2] 命中率天花板")
    print(f"  目标命中率 (cached_target/n_tokens) = {target_rate:.2f}%")
    print(f"  实际命中率 (cached/prompt)         = {actual_rate:.2f}%")
    print(f"  潜在空间   (目标 - 实际)           = {target_rate - actual_rate:.2f} 个百分点")


def run_baseline(data_dir):
    """[3] kvsim 基线（块=64）。"""
    trace = load_jsonl(os.path.join(data_dir, "trace.jsonl"))
    base = min(_iso_ms(r["timestamp"]) for r in trace)
    reqs = [
        (_iso_ms(r["timestamp"]) - base,
         int(r["input_length"]),
         int(r.get("output_length", 0)),
         tuple(r["hash_ids"]))
        for r in trace
    ]
    reqs.sort()

    # 关键：本项目 trace.jsonl 的块 = 64 token（mamba chunk），不是 kvsim 默认的 512。
    kv.BLOCK = 64
    # 只用前 50 条请求做演示：完整 548 条 + 大容量会因 kvsim 的 _evict 是 O(size) 朴素实现
    # 而变得极慢（块=64 时块数是块=512 的 8 倍）。任务 2 加存储分层时建议顺带把淘汰改成
    # O(log size)（用堆/有序结构维护最老叶子）。
    reqs = reqs[:50]
    total_ms = reqs[-1][0]
    # N=6 对齐真实环境 6 prefill；C 是单 pod 的 host 容量（块数）。
    # 真实环境单 pod host ≈ 905K token / 64 ≈ 14146 块；这里取 2000 只为快速演示。
    N, C = 6, 2000

    print("=" * 60)
    print(f"[3] kvsim 基线（块=64, 前 50 条请求, N={N}, C={C}，子集演示）")
    print("    预期：AFF 命中高但 skew 大（失衡），LEAST 均衡但命中低，REQ 兼顾两者")
    for routing in ("AFF", "LEAST", "REQ"):
        r = kv.run(reqs, N, C, routing=routing, eval_start=0, eval_end=total_ms)
        print(f"  {routing:6s} hit={r['token_hit'] * 100:5.2f}%  "
              f"skew={r['skew']:5.2f}  saved={r['saved'] / 1e6:.3f}M")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvprefetch_full"
    if not os.path.isdir(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("用法: python srt_quickstart.py <解压后的 full 包目录>")
        sys.exit(1)
    stat_transfer(data_dir)
    stat_ceiling(data_dir)
    run_baseline(data_dir)
