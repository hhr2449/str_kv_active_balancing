#!/usr/bin/env python3
"""阶段 1：统计传输成败、失败信息、成功长度及缓存命中率。"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "kvsim_multipod/offline_modeling_dataset_full",
    )
    data_dir = parser.parse_args().data_dir
    transfers = load_jsonl(data_dir / "transfer.jsonl")
    requests = load_jsonl(data_dir / "per_request.jsonl")
    with (data_dir / "request_meta.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    print("[1] 按 reason 统计传输成败")
    print("reason\ttotal\tok\trejected\tsuccess_rate")
    counts = Counter((row["reason"], row["ok"]) for row in transfers)
    for reason in sorted({row["reason"] for row in transfers}):
        ok = counts[reason, True]
        rejected = counts[reason, False]
        total = ok + rejected
        print(f"{reason}\t{total}\t{ok}\t{rejected}\t{ok / total:.2%}")
    ok = sum(row["ok"] for row in transfers)
    total = len(transfers)
    print(f"合计\t{total}\t{ok}\t{total - ok}\t{ok / total:.2%}")

    print("\n[2] fail_message 分布")
    failures = Counter(row["fail_message"] for row in transfers if not row["ok"])
    for message, count in failures.most_common():
        print(f"{count} 次\t{message}")

    print("\n[3] 成功传输的 token 长度分布")
    print("requested_tokens\ttransferred_tokens\tcount")
    lengths = Counter(
        (row["requested_tokens"], row["transferred_tokens"])
        for row in transfers if row["ok"]
    )
    for (requested, transferred), count in sorted(lengths.items()):
        print(f"{requested}\t{transferred}\t{count}")

    print("\n[4] 目标命中率与实际命中率")
    n_tokens = sum(row["n_tokens"] for row in metadata)
    target = sum(row["cached_target"] for row in metadata)
    prompt = sum(row["prompt_tokens"] for row in requests)
    cached = sum(row["cached_tokens"] for row in requests)
    target_rate = target / n_tokens
    actual_rate = cached / prompt
    print(f"元数据请求数={len(metadata)}，实测请求数={len(requests)}")
    print(f"Σn_tokens={n_tokens}，Σcached_target={target}")
    print(f"Σprompt_tokens={prompt}，Σcached_tokens={cached}")
    print(f"目标命中率（Σcached_target / Σn_tokens）={target_rate:.2%}")
    print(f"实际命中率（Σcached_tokens / Σprompt_tokens）={actual_rate:.2%}")
    print(f"目标与实际差值={(target_rate - actual_rate) * 100:.2f} 个百分点")


if __name__ == "__main__":
    main()
