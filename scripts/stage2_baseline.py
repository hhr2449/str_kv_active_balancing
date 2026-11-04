#!/usr/bin/env python3
"""阶段 2：完整 Trace 的 base / idea2 / idea3，各配置运行一次。"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kvsim_multipod"))
import kvsim_multipod as kv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path,
                        default=ROOT / "kvsim_multipod/offline_modeling_dataset_full/trace.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/stage2")
    parser.add_argument("--pods", type=int, default=6)
    parser.add_argument("--capacity-blocks", type=int, default=2000)
    args = parser.parse_args()

    raw_trace = args.trace.read_bytes()
    trace = [json.loads(line) for line in raw_trace.splitlines() if line.strip()]
    times = [int(datetime.fromisoformat(row["timestamp"]).timestamp() * 1000)
             for row in trace]
    start = min(times)
    reqs = [(t - start, row["input_length"], row["output_length"], tuple(row["hash_ids"]))
            for t, row in zip(times, trace)]
    reqs.sort(key=lambda row: row[0])
    kv.BLOCK = 64
    eval_end = reqs[-1][0] + 1
    common = dict(theta=2.0, W=300000, infinite=False,
                  eval_start=0, eval_end=eval_end, cap2=2, depth_cap=None,
                  budget_bytes=None, stable_broadcast=0,
                  demand_weight_depth=True, scan_limit=1500)
    cases = [
        ("base", "AFF", dict(routing="AFF")),
        ("base", "LEAST", dict(routing="LEAST")),
        ("base", "REQ", dict(routing="REQ")),
        ("base", "REQ theta=3", dict(routing="REQ", theta=3.0)),
        ("idea2", "AFF + broadcast1", dict(routing="AFF", stable_broadcast=1)),
        ("idea2", "REQ + broadcast1", dict(routing="REQ", stable_broadcast=1)),
        ("idea2", "REQ + broadcast4", dict(routing="REQ", stable_broadcast=4)),
        ("idea3", "REQ + OracleCap2 dc=8",
         dict(routing="REQ", policy="Oracle", cap2=2, depth_cap=8)),
    ]
    simulator = ROOT / "kvsim_multipod/kvsim_multipod.py"
    config = dict(
        trace=str(args.trace.resolve()),
        trace_sha256=hashlib.sha256(raw_trace).hexdigest(),
        simulator_sha256=hashlib.sha256(simulator.read_bytes()).hexdigest(),
        requests=len(reqs), input_tokens=sum(row[1] for row in reqs),
        pods=args.pods, capacity_blocks=args.capacity_blocks, block_tokens=kv.BLOCK,
        page_bytes=kv.PAGE_BYTES, wire_accounting="legacy_14_MiB_per_block",
        time_unit="ms", equal_timestamp_order="trace_file_order",
        eval_start=0, eval_end=eval_end,
        random_gini_samples=200, random_gini_seed=20260911,
        common_parameters=common,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = dict(config=config, results=[])
    output_path = args.output_dir / "results.json"
    print(json.dumps(config, ensure_ascii=False), flush=True)
    print("mode\tcase\tsaved\thit_pct\tskew\twire_GiB\tpwire_GiB\tseconds", flush=True)
    for mode, tag, overrides in cases:
        parameters = common | overrides
        began = perf_counter()
        result = kv.run(reqs, args.pods, args.capacity_blocks, **parameters)
        elapsed = perf_counter() - began
        output["results"].append(dict(mode=mode, case=tag, parameters=parameters,
                                      elapsed_seconds=elapsed, metrics=result))
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"{mode}\t{tag}\t{result['saved']}\t{result['token_hit'] * 100:.4f}\t"
              f"{result['skew']:.4f}\t{result['wire_GiB']:.6f}\t"
              f"{result['proactive_wire_GiB']:.6f}\t{elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
