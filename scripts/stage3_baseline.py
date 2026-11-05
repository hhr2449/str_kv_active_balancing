#!/usr/bin/env python3
"""阶段 3：新版分层模拟器的 base / idea2 / idea3，与阶段 2 使用相同回放口径。"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kvsim_multipod"))
import kvsim_multipod_revised as kv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path,
                        default=ROOT / "kvsim_multipod/offline_modeling_dataset_full/trace.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/stage3")
    parser.add_argument("--pods", type=int, default=6)
    parser.add_argument("--capacity-blocks", type=int, default=2000, help="每个 Pod 的 Host 容量（block）")
    # 新版单独配置 Device 和事件发布周期，主动评估仍固定为 5 秒。
    parser.add_argument("--device-cap", type=int, help="Device 容量（block），默认同 Host")
    parser.add_argument("--event-publish-interval-ms", type=int, default=1000,
                        help="KV Event 发布周期（毫秒），默认 1000")
    args = parser.parse_args()
    if args.device_cap is not None and args.device_cap < 0:
        parser.error("--device-cap 必须 >= 0")
    if args.event_publish_interval_ms <= 0:
        parser.error("--event-publish-interval-ms 必须 > 0")

    # Trace 读取、稳定排序及评价窗口与 stage2_baseline.py 保持一致。
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
                  demand_weight_depth=True, scan_limit=1500,
                  device_cap=args.device_cap,
                  event_publish_interval_ms=args.event_publish_interval_ms)
    # 策略及参数与阶段 2 相同，仅由新版模拟器实现分层和周期决策。
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
    simulator = ROOT / "kvsim_multipod/kvsim_multipod_revised.py"
    config = dict(
        trace=str(args.trace.resolve()),
        trace_sha256=hashlib.sha256(raw_trace).hexdigest(),
        simulator_sha256=hashlib.sha256(simulator.read_bytes()).hexdigest(),
        requests=len(reqs), input_tokens=sum(row[1] for row in reqs),
        pods=args.pods, capacity_blocks=args.capacity_blocks, block_tokens=kv.BLOCK,
        cache_model="device_host", transfer_mode="device_only",
        device_capacity_blocks=args.capacity_blocks if args.device_cap is None else args.device_cap,
        event_publish_interval_ms=args.event_publish_interval_ms,
        proactive_interval_ms=kv.PROACTIVE_INTERVAL_MS,
        transfer_stats_scope=dict(transfer_stats="full_trace", transfer_stats_eval="eval_window"),
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
    for case_index, (mode, tag, overrides) in enumerate(cases, 1):
        parameters = common | overrides
        began = perf_counter()
        result = kv.run(reqs, args.pods, args.capacity_blocks, **parameters)
        elapsed = perf_counter() - began
        # 明细按配置单独保存 JSONL，汇总保留全程和评价窗统计。
        events = result.pop("transfer_events")
        event_dir = args.output_dir / "transfers"
        event_dir.mkdir(parents=True, exist_ok=True)
        event_path = event_dir / f"{case_index:02d}.jsonl"
        with event_path.open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        output["results"].append(dict(mode=mode, case=tag, parameters=parameters,
                                      elapsed_seconds=elapsed, metrics=result,
                                      transfer_events_file=str(event_path.relative_to(args.output_dir))))
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"{mode}\t{tag}\t{result['saved']}\t{result['token_hit'] * 100:.4f}\t"
              f"{result['skew']:.4f}\t{result['wire_GiB']:.6f}\t"
              f"{result['proactive_wire_GiB']:.6f}\t{elapsed:.3f}", flush=True)
        # 主输出使用评价窗，只打印启用策略，没有尝试时成功率显示 --。
        for strategy, stats in result["transfer_stats_eval"].items():
            rate = "--" if stats["success_rate"] is None else f"{stats['success_rate'] * 100:.2f}"
            print(f"# {tag} / {strategy} [eval]: skipped={stats['skipped']} attempts={stats['attempts']} "
                  f"ok={stats['ok']} failed={stats['failed']} success_pct={rate} "
                  f"failure_reasons={json.dumps(stats['failure_reasons'])}", flush=True)


if __name__ == "__main__":
    main()
